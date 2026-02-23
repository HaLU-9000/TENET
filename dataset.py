import os
from pathlib import Path
import random
from typing import Any
import tifffile
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode as I
from torch.utils.data import Dataset
from scipy.stats import lognorm, truncnorm
from fft_conv_pytorch import fft_conv

from utils import mask_, surround_mask_, tifpath_to_tensor, load_anything
from model import ImagingProcess

def gen_indices(I, low, high):
    return np.random.randint(low, high, (I,))

def gen_coords(I, size, cropsize, scale=None, label=None):
    zcoord = np.random.randint(0, size[0]-cropsize[0], (I,))
    xcoord = np.random.randint(0, size[1]-cropsize[1], (I,))
    ycoord = np.random.randint(0, size[2]-cropsize[2], (I,))
    return np.array([zcoord, xcoord, ycoord])

def gen_image_label_coords(I, size, cropsize, scale):
    zcoord = np.random.randint(0, size[0]-cropsize[0], (I,))
    xcoord = np.random.randint(0, size[1]-cropsize[1], (I,))
    ycoord = np.random.randint(0, size[2]-cropsize[2], (I,))
    return np.array([zcoord, xcoord, ycoord]), np.array([zcoord // scale, xcoord, ycoord])

def apply_mask( mask, image, mask_size, mask_num):
    if mask:
        image = mask_(image, mask_size, mask_num)
    return image

def apply_surround_mask(surround, image, surround_size):
    if surround:
        image = surround_mask_(image, surround_size)
    return image

def sample_truncnorm(low, high, loc, scale):
    a = (low  - loc) / scale
    b = (high - loc) / scale
    return truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)

def gen_imaging_parameters(params_ranges:dict
                           )->dict:
    """
    :params params_ranges
    ## input exapmle\n
    {"mu_z"   : [0,   1, 0.2  ,  0.5 ],\n
     "sig_z"  : [0,   1, 0.2  ,  0.5 ],\n
     "bet_z"  : [0,  50,  25  , 12.5 ],\n
     "bet_xy" : [0,  20,   1. ,  5.  ],\n
     "alpha"  : [0, 100,  10  ,  5.  ],\n
     "sig_eps": [0, 0.3, 0.15 ,  0.05],\n
     "scale"  : [1, 2, 4, 8, 12      ] \n
     } 
    """
    params = {}
    for param in params_ranges:
        if param == "scale":
            params[param] = np.random.choice((params_ranges[param]))
        else:
            params[param] = sample_truncnorm(*params_ranges[param])
    return params

def sequentialflip(image, i):
        options = [[-3], [-3,-2], [-3,-2,-1], [-3,-1],
                   [-2], [-2,-1], [-1], [-4]]
        option = options[i%8]
        return torch.flip(image, dims=option)

class Rotate:
    def __init__(self, i=None, j=None):
        if i is not None:
            self.i = i
        else:
            self.i = random.choice([0,2])
        if j is not None:
            self.j = j
        else:
            self.j = random.choice([0,1,2,3])
    def __call__(self, x):
        return torch.rot90(torch.rot90(x, self.i, [1, 2]), self.j, [2, 3]), self.i, self.j


class Crop:
    def __init__(self, coord:list, cropsize:list):
        self.coord = coord
        self.csize = cropsize
    def __call__(self,x:torch.Tensor):
        x_is_4d = False
        if len(x.shape) == 4:
            x_is_4d = True
            x = x.unsqueeze(0)
        x = x[:,  :,  self.coord[0] : self.coord[0] + self.csize[0],
                      self.coord[1] : self.coord[1] + self.csize[1],
                      self.coord[2] : self.coord[2] + self.csize[2]].detach().clone()
        if x_is_4d:
            x = x.squeeze(0)
        return x


class Blur(nn.Module):
    def __init__(self, scale, z, x, y, mu_z, sig_z, bet_xy, bet_z, alpha, sig_eps, device):
        super().__init__()
        self.scale    = scale
        self.z        = z
        self.x        = x
        self.y        = y
        self.mu_z     = mu_z
        self.sig_z    = sig_z
        self.bet_xy   = bet_xy
        self.bet_z    = bet_z
        self.alpha    = alpha
        self.sig_eps  = sig_eps
        self.zd,      \
        self.xd,      \
        self.yd       = self.distance(z, x, y)
        self.alf      = self.gen_alf().to(device)
        self.sum_alf  = torch.sum(self.alf)
        self.logn_ppf = lognorm.ppf([0.99], 1, loc=mu_z, scale=sig_z)[0] # normalize by init value
        self.theomax  = self.sum_alf * self.logn_ppf

    def distance(self, z, x, y):
        [zd, xd, yd] = [torch.zeros(1, 1, z, x, y,) for _ in range(3)]
        for k in range(-z // 2, z // 2 + 1):
            zd[:, :, k + z // 2, :, :,] = k ** 2
        for i in range(-x // 2, x // 2 + 1):
            xd[:, :, :, i + x // 2, :,] = i ** 2
        for j in range(-y // 2, y // 2 + 1):
            yd[:, :, :, :, j + y // 2,] = j ** 2
        return zd, xd, yd

    def gen_alf(self):
        d_2 = self.zd / self.bet_z ** 2 + (self.xd + self.yd) / self.bet_xy ** 2
        alf = torch.exp(-d_2 / 2)
        normterm = (self.bet_z * self.bet_xy ** 2) * (torch.pi * 2) ** 1.5
        alf = alf / normterm 
        alf  = torch.ones_like(alf) - torch.exp(-self.alpha * alf)
        return alf

    def forward(self, inp):
        inp  = inp.unsqueeze(0)
        pz0  = dist.LogNormal(loc   = self.mu_z  * torch.ones_like(inp),
                              scale = self.sig_z * torch.ones_like(inp),)       
        rec  = inp * pz0.sample() # E[z0|mu_z, sig_z]
        #z0   = z0 * torch.ones_like(inp, requires_grad=True)
        rec  = rec * 3.3
        #rec  = torch.clip(rec, min=0, max=self.logn_ppf)
        rec  = F.conv3d(input   = rec                               ,
                        weight  = self.alf                          ,
                        stride  = (self.scale, 1, 1)                       ,
                        padding = ((self.z - self.scale + 1) // 2  , 
                                   (self.x) // 2                    , 
                                   (self.y) // 2                    ,),)
        rec  = rec / self.theomax
        #rec  = (rec - rec.min()) / (rec.max() - rec.min())
        prec = dist.Normal(loc   = rec         ,
                           scale = self.sig_eps,)
        rec  = prec.sample()
        rec  = rec.squeeze(0)
        rec  = torch.clip(rec, min=0, max=1)
        rec  = rec.squeeze(0)
        return rec

    
class Augmentation():
    def __init__(self, params):
        self.mask          = params["mask"]
        self.mask_size     = params["mask_size"]
        self.mask_num      = params["mask_num"]
        self.surround      = params["surround"]
        self.surround_size = params["surround_size"]
        self.original_size = params["original_size"]
        self.cropsize      = params["cropsize"]

    def apply_mask(self, mask, image, mask_size, mask_num):
        if mask:
            image = mask_(image, mask_size, mask_num)
        return image

    def apply_surround_mask(self, surround, image, surround_size):
        if surround:
            image = surround_mask_(image, surround_size)
        return image

    def __call__(self, image):
        image = self.apply_mask(self.mask, image, self.mask_size, self.mask_num)
        image = self.apply_surround_mask(self.surround, image, self.surround_size)
        return image

    def crop(self, image, label):
        scale = label.shape[2] // image.shape[2]
        scaled_cropsize  = [self.cropsize[0] // scale, *self.cropsize[1:]]
        lcoords, icoords = gen_image_label_coords(1, self.original_size, self.cropsize, scale)
        lcoords, icoords = lcoords[:, 0], icoords[:, 0]
        
        label = Crop(lcoords, self.cropsize)(label)
        image = Crop(icoords, scaled_cropsize)(image)
        
        return image, label
    
class Mask():
    def __init__(self):
        pass

    def apply_mask(self, mask, image, mask_size, mask_num):
        if mask:
            image = mask_(image, mask_size, mask_num)
        return image


class RandomCutDataset(Dataset):
    '''
    input  : 4d torch.tensor (large (like 768**3) size) (image and label)
    output : 4d torch.tensor (small (like 128**3) size)
             ([channels, z_size, x_size, y_size]) 
             of randomly cropped/rotated image and label
    folderpath : large data path ("randomdata" in this repo)
    imagename : "0001***.pt" `s "**" part. (e.g. "_x1")
    labelname : "0001***.pt" `s "**" part. (e.g. "_label")
    I : sample size. Returns I samples. (e.g. 200)
    low, high : use [low]th ~ [high]th files in folderpath as data.
    scale: scale (should be same as [imagename]'s int part.)
    '''
    def __init__(self,
                 folderpath:str,
                 size:list,
                 cropsize:list,
                 I:int, 
                 low:int,
                 high:int,
                 scale:int,
                 train=True,
                 mask=True,
                 mask_size=[10, 10, 10],
                 mask_num=1,
                 surround=True,
                 surround_size=[64, 8, 8],
                 seed=904):
        self.I             = I
        self.low           = low
        self.high          = high
        self.scale         = scale
        self.size          = size
        self.labelxs       = [folderpath+"/"+file
                              for file in sorted(os.listdir(folderpath))
                              if "labelx" in file]
        self.labelzs       = [folderpath+"/"+file
                              for file in sorted(os.listdir(folderpath))
                              if "labelz" in file]
        
        self.csize         = cropsize
        self.ssize         = [cropsize[0]//scale, cropsize[1], cropsize[2]]
        self.train         = train
        self.mask          = mask
        self.mask_size     = mask_size
        self.mask_num      = mask_num
        self.surround      = surround
        self.surround_size = surround_size
        self.options       = [[-3], [-3,-2], [-3,-2,-1], [-3,-1],
                              [-2], [-2,-1], [-1], [-4]]

        if train == False:
            np.random.seed(seed)
            self.indiceslist = gen_indices(I, low, high)
            self.coordslist  = self.gen_coords(I, size, cropsize, scale)
    
    def couple_randomflip(self, image, label):
        option = self.options[np.random.choice([i for i in range(8)])]
        return image.flip(dims=option), label.flip(dims=option)

    def gen_coords(self, I, size, cropsize, scale):
        zcoord = np.random.randint(0, size[0]-cropsize[0], (I,))
        xcoord = np.random.randint(0, size[1]-cropsize[1], (I,))
        ycoord = np.random.randint(0, size[2]-cropsize[2], (I,))
        return np.array([zcoord, xcoord, ycoord]), np.array([zcoord // scale, xcoord, ycoord])

    def apply_mask(self, mask, image, mask_size, mask_num):
        if mask:
            image = mask_(image, mask_size, mask_num)
        return image

    def apply_surround_mask(self, surround, image, surround_size):
        if surround:
            image = surround_mask_(image, surround_size)
        return image

    def __getitem__(self, idx):
        if self.train:
            idx              = gen_indices(1, self.low, self.high).item()#;print('idx ', idx)
            lcoords, icoords = self.gen_coords(1,self.size,self.csize,
                                               self.scale)
            lcoords, icoords = lcoords[:, 0], icoords[:, 0]
            labelx, i, j = Rotate(    )(Crop(lcoords, self.csize
                                      )(load_anything(self.labelxs[idx])))
            labelz, _, _ = Rotate(i, j)(Crop(lcoords, self.csize
                                      )(load_anything(self.labelzs[idx])))
            labelx, labelz = self.couple_randomflip(labelx, labelz)
        else:
            _idx    = self.indiceslist[idx]  # convert idx to [low] ~[high] number
            lcoords = self.coordslist[0][:, idx]
            labelx  = Crop(lcoords, self.csize
                           )(load_anything(self.labelxs[_idx]))
            labelz  = Crop(lcoords, self.csize
                           )(load_anything(self.labelzs[_idx]))
            labelx, labelz = self.couple_randomflip(labelx, labelz)
        return {"labelx" : labelx,
                "labelz" : labelz}

    def __len__(self):
        return self.I
    

class RealDensityDataset(Dataset):
    '''
    input  : 4d torch.tensor (large (like 768**3) size) (image)
    output : 4d torch.tensor (small (like 128**3) size)
             ([channels, z_size, x_size, y_size]) 
             of randomly cropped/rotated image
    folderpath : large data path ("randomdata" in this repo)
    imagename : "0001***.pt" `s "**" part. (e.g. "_x1")
    I : sample size. Returns I samples. (e.g. 200)
    low, high : use [low]th ~ [high]th files in folderpath as data.
    scale: scale (should be same as [imagename]'s int part.)

    algorithm
    (init)
    1. calculate score by conv filter
    2. normalize score to [0, 1]
    (__getitem__)
    3. r ~ uniform(0,1)
    4. accept | if r < score
       reject | else
    '''
    def __init__(self, folderpath:str, scorefolderpath:str, imagename:str,
                 size:list, cropsize:list, I:int, low:int, high:int, scale:int,
                 train=True, mask=True, score=None, score_saving=True,
                 mask_size=[10, 10, 10], mask_num=1,
                 surround=True, surround_size=[64, 8, 8],
                 seed=904):
        self.I             = I
        self.low           = low
        self.high          = high
        self.scale         = scale
        self.size          = size
        self.ssize         = [size[0]//scale, size[1], size[2]]
        self.imagename     = imagename
        self.images        = list(sorted(Path(folderpath).glob(f'*{imagename}.pt')))
        self.csize         = cropsize
        self.scsize        = [cropsize[0]//scale, cropsize[1], cropsize[2]]
        self.train         = train
        self.mask          = mask
        self.mask_num      = mask_num
        self.mask_size     = mask_size
        self.surround      = surround
        self.surround_size = surround_size
        self.icoords_size  = [self.ssize[0] - self.scsize[0] + 1,
                              self.ssize[1] - self.scsize[1] + 1,
                              self.ssize[2] - self.scsize[2] + 1,]
        self.options       = [[-3], [-3,-2], [-3,-2,-1], [-3,-1],
                              [-2], [-2,-1], [-1], [-4]]
        
        if score is None:
            self.scores = self.gen_scores(self.images, self.icoords_size, self.scsize)
            if score_saving:
                self.save_scores(self.scores, scorefolderpath)
        else:
            self.scores = score
        if train == False:
            np.random.seed(seed)
            self.indiceslist = self.gen_indices(I * 10000, low, high)
            self.coordslist  = self.gen_coords(I * 10000, self.icoords_size)

    def gen_scores(self, images, icoords_size, scsize):
        _scores = torch.zeros((len(images), 1, *icoords_size))
        for n, i in enumerate(images):
            print(f'(init) calcurating the score...({n+1}/{len(images)})')
            _score = fft_conv(signal  = torch.load(i)        ,
                              kernel  = torch.ones(1, 1, *scsize),
                              stride  = 1                      ,
                              padding = 0                      ,)
            _scores[n] = _score
        _fscores = _scores.flatten()
        _scores = (_scores - torch.min(_fscores))            \
                / ((torch.max(_fscores) - torch.min(_fscores)))
        return _scores

    def save_scores(self, scores, scorefolderpath):
        torch.save(scores, scorefolderpath+f'/{self.imagename}_score.pt')

    def gen_indices(self, I, low, high):
        return np.random.randint(low, high, (I,))
    
    def gen_coords(self, I, icoords_size):
        zcoord = np.random.randint(0, icoords_size[0], (I,))
        xcoord = np.random.randint(0, icoords_size[1], (I,))
        ycoord = np.random.randint(0, icoords_size[2], (I,))
        return np.array([zcoord, xcoord, ycoord])

    def apply_mask(self, mask, image, mask_size, mask_num):
        if mask:
            image = mask_(image, mask_size, mask_num)
        return image

    def apply_surround_mask(self, surround, image, surround_size):
        if surround:
            image = surround_mask_(image, surround_size)
        return image

    def randomflip(self, image):
        option = self.options[np.random.choice([i for i in range(8)])]
        return image.flip(dims=option)

    def __getitem__(self, idx):
        if self.train:
            r, s = 1, 0
            while not r < s:
                idx     = self.gen_indices(1, self.low, self.high).item()
                icoords = self.gen_coords( 1, self.icoords_size)
                icoords = icoords[:, 0]
                z, x, y = icoords
                r = np.random.uniform(0, 1)
                s = self.scores[idx, 0, z, x, y]
            image, _, _      = Rotate(    )(Crop(icoords, self.scsize
                                                )(torch.load(self.images[idx])))
            image = self.randomflip(image)
            image = self.apply_mask(self.mask, image, self.mask_size, self.mask_num)
            image = self.apply_surround_mask(self.surround, image, self.surround_size)
        else:
            r, s = 1, 0
            c = 0
            while not r < s:
                c += 1
                _idx     = self.indiceslist[c]
                icoords = self.coordslist[:, c]
                z, x, y = icoords
                r = np.random.uniform(0, 1)
                s = self.scores[_idx, 0, z, x, y]
            image   = Crop(icoords, self.scsize)(torch.load(self.images[_idx]))
            image = self.apply_surround_mask(self.surround, image, self.surround_size)
        return {"image": image}

    def __len__(self):
        return self.I
    
class DensityDataset(Dataset):
    '''
    input  : 4d (CZXY) array [.npy, .pt, .tif] are allowed.
    output : 4d torch.tensor (small (like 128**3) size)
             ([channels, z_size, x_size, y_size]) 
             of randomly cropped/rotated image
    folderpath : large data path ("randomdata" in this repo)
    validation_list : list of the name of validation data "xxxx_**.tif" `s "**" parts. (e.g. "_x1")[list]
    I : sample size. Returns I samples. (e.g. 200)
    low, high : use [low]th ~ [high]th files in folderpath as data.
    scale: scale (should be same as [imagename]'s int part.)

    algorithm
    (init)
    1. score = mean of the array.
    2. r ~ uniform(0,1)
    4. accept | if r < score
       reject | else
    '''
    def __init__(self, folderpath:str,
                 size:list, cropsize:list, I:int, scale:int,
                 train=True, mask=True, train_data_rate=0.8,
                 mask_size=[10, 10, 10], mask_num=1, test_tuple=('-1',),
                 surround=True, surround_size=[64, 8, 8],
                 seed=904):
        self.I             = I
        self.scale         = scale
        self.size          = size
        self.ssize         = [size[0]//scale, size[1], size[2]]
        self.images        = [folderpath+"/"+file
                              for file in sorted(
                                  os.listdir(
                                      folderpath)
                                      ) if not file.startswith('_')\
                and not file.split(".")[0].endswith(test_tuple)]
        self.high          = int(len(self.images) * train_data_rate)
        self.num_images    = len(self.images)
        self.csize         = cropsize
        self.scsize        = [cropsize[0]//scale, cropsize[1], cropsize[2]]
        self.train         = train
        self.mask          = mask
        self.mask_num      = mask_num
        self.mask_size     = mask_size
        self.surround      = surround
        self.surround_size = surround_size
        self.icoords_size  = [self.ssize[0] - self.scsize[0],
                              self.ssize[1] - self.scsize[1],
                              self.ssize[2] - self.scsize[2],]
        self.options       = [[-3], [-3,-2], [-3,-2,-1], [-3,-1],
                              [-2], [-2,-1], [-1], [-4]]
        if train:
            print("num of image:" , self.num_images)
            print("num of training image:" , self.high)
            print("num of validation image:" , self.num_images - self.high)
        
        if train == False:
            np.random.seed(seed)
            self.indiceslist = self.gen_indices(
                I * 100000, self.high, self.num_images)
            self.coordslist  = self.gen_coords(
                I * 100000, self.icoords_size)
        
    def gen_indices(self, I, low, high):
        return np.random.randint(low, high, (I,))
    
    def gen_coords(self, I, icoords_size):
        zcoord = np.random.randint(0, icoords_size[0], (I,))
        xcoord = np.random.randint(0, icoords_size[1], (I,))
        ycoord = np.random.randint(0, icoords_size[2], (I,))
        return np.array([zcoord, xcoord, ycoord])

    def apply_mask(self, mask, image, mask_size, mask_num):
        if mask:
            image = mask_(image, mask_size, mask_num)
        return image

    def apply_surround_mask(self, surround, image, surround_size):
        if surround:
            image = surround_mask_(image, surround_size)
        return image

    def randomflip(self, image):
        option = self.options[np.random.choice([i for i in range(8)])]
        return image.flip(dims=option)

    def __getitem__(self, idx):
        if self.train:
            r, s = 0.0001, 0
            valid = False
            while not r < s:
                while not valid:
                    idx     = self.gen_indices(1, 0, self.high).item()
                    icoords = self.gen_coords(1, self.icoords_size)
                    icoords = icoords[:, 0]
                    r = np.random.uniform(0, 0.0001)
                    image  = (Crop(icoords, self.scsize
                                  )(load_anything(self.images[idx])))
                    valid = (1, *self.scsize) == image.shape
                s = image.mean().item() + 0.1
            image, _, _ = Rotate()(image)
            image = self.randomflip(image)
            image = self.apply_mask(
                self.mask, image, self.mask_size, self.mask_num)
            image = self.apply_surround_mask(
                self.surround, image, self.surround_size)
        else:
            r, s = 0.0001, 0
            valid = False
            while not r < s:
                while not valid:
                    _idx     = self.indiceslist[idx]
                    icoords = self.coordslist[:, idx]
                    r     = np.random.uniform(0, 0.0001)
                    image = Crop(icoords, self.scsize
                                 )(load_anything(self.images[_idx]))
                    valid = (1, *self.scsize) == image.shape
                s = image.mean().item() + 1
            image = self.apply_surround_mask(
                self.surround, image, self.surround_size)
        return {"image": image}

    def __len__(self):
        return self.I
    
class RealSeveralDataset(Dataset):
    '''
    input  : 4d tif image path (large (like 768**3) size) (image)
    output : 4d torch.tensor (small (like 128**3) size)
             ([channels, z_size, x_size, y_size]) 
             of randomly cropped/rotated image
    folderpath : large data path ("randomdata" in this repo)
    imagename : "0001***.pt" `s "**" part. (e.g. "_x1")
    I : sample size. Returns I samples. (e.g. 200)
    low, high : use [low]th ~ [high]th files in folderpath as data.
    scale: scale (should be same as [imagename]'s int part.)

    algorithm
    (init)
    1. calculate score by conv filter
    2. normalize score to [0, 1]
    (__getitem__)
    3. r ~ uniform(0,1)
    4. accept | if r < score
       reject | else
    '''
    def __init__(self, folderpath:str, imagename:str,
                 cropsize:list, I:int, low:int, high,
                 train=True, mask=True, mask_size=[10, 10, 10], mask_num=1,
                 surround=True, surround_size=[64, 8, 8], preprocess=False, seed=904):
        self.I             = I
        self.images        = list(sorted(Path(folderpath).glob(f'*{imagename}*.tif')))
        self.low           = low
        if high is not None:
            self.high = high
        else:
            self.high = len(self.images) - 1
        self.csize         = cropsize
        self.train         = train
        self.mask          = mask
        self.mask_num      = mask_num
        self.mask_size     = mask_size
        self.surround      = surround
        self.surround_size = surround_size
        self.options       = [[-2], [-2,-1], [-1], [-4]]
        self.preprocess    = preprocess
        if train == False:
            np.random.seed(seed)

    def gen_indices(self, I, low, high):
        return np.random.randint(low, high, (I,))
    
    def gen_coords(self, I, icoords_size):
        zcoord = np.random.randint(0, icoords_size[0], (I,))
        xcoord = np.random.randint(0, icoords_size[1], (I,))
        ycoord = np.random.randint(0, icoords_size[2], (I,))
        return np.array([zcoord, xcoord, ycoord])

    def apply_mask(self, mask, image, mask_size, mask_num):
        if mask:
            image = mask_(image, mask_size, mask_num)
        return image

    def apply_surround_mask(self, surround, image, surround_size):
        if surround:
            image = surround_mask_(image, surround_size)
        return image

    def randomflip(self, image):
        option = self.options[np.random.choice([i for i in range(len(self.options))])]
        return image.flip(dims=option)

    def temporal_solution(self, name):
        name = str(name)
        if "2_Spine" in name:
            scale = 2   
        if "3_2-"     in name:
            scale = 8    
        if "Beads"    in name:
            scale = 10     
        if "MD"       in name:
            scale = 12   
        if "1_Spine" in name:
            scale = 6  

        return scale

    def __getitem__(self, idx):
        r, s = 1, 0
        while not r < s:
            idx = self.gen_indices(1, self.low, self.high).item()
            image = tifpath_to_tensor(self.images[idx], preprocess=self.preprocess)
            scale = self.temporal_solution(self.images[idx])
            ssize = [image.size(1), image.size(2), image.size(3)]
            icoords_size  = [ssize[0] - self.csize[0]//scale + 1,
                             ssize[1] - self.csize[1]        + 1,
                             ssize[2] - self.csize[2]        + 1,]
            scsize        = [self.csize[0]//scale ,
                             self.csize[1]        ,
                             self.csize[2]        ,]
            icoords = self.gen_coords(1, icoords_size)
            icoords = icoords[:, 0]
            image = Crop(icoords, scsize)(image)
            s = torch.sum(image) / (self.csize[0]//scale * self.csize[1] * self.csize[2])
            r = np.random.uniform(0, 0.1)
        image, _, _ = Rotate()(image)
        image = self.randomflip(image)
        if self.train:
            image = self.apply_mask(self.mask, image, self.mask_size, self.mask_num)
        image = self.apply_surround_mask(self.surround, image, self.surround_size)
        return image, scale

    def __len__(self):
        return self.I
    
class RandomBlurDataset(Dataset):
    '''
    input  : 4d torch.tensor (large (like 768**3) size) (label)
    output : 4d torch.tensor (small (like 128**3) size)
             ([channels, z_size, x_size, y_size]) 
             of randomly blurred/cropped/rotated image and label
    :param folderpath : large data path ("randomdata" in this repo)
    labelname : "0001***.pt" `s "**" part. (e.g. "_label")
    I : sample size. Returns I samples. (e.g. 200)
    low, high : use [low]th ~ [high]th files in folderpath as data.
    '''
    def __init__(self, folderpath:str,
                 size:list, cropsize:list, I:int, low:int, high:int,
                 z, x, y,
                 imaging_params_range:dict,
                 validation_params:dict,
                 device, is_train=True, mask=True, 
                 mask_size=[10, 10, 10], mask_num=1,
                 surround=True, surround_size=[72, 8, 8],
                 seed=523):
        
        self.z             = z
        self.x             = x
        self.y             = y
        self.I             = I
        self.low           = low
        self.high          = high
        self.size          = size
        self.labels        = list(sorted(Path(folderpath).glob(f'*_label.npy')))
        self.csize         = cropsize
        self.is_train      = is_train
        self.mask          = mask
        self.mask_size     = mask_size
        self.mask_num      = mask_num
        self.surround      = surround
        self.surround_size = surround_size
        self.params_range  = imaging_params_range
        self.imaging       = ImagingProcess(device, validation_params,
                                            z=z, x=x, y=y, mode="dataset")
        self.validation_scale = validation_params["scale"]
        self.device           = device
        if is_train == False:
            np.random.seed(seed)
            self.indiceslist = gen_indices(I, low, high)
            self.coordslist  = gen_coords(I, size, cropsize)

    def __getitem__(self, idx):
        if self.is_train:
            idx     = gen_indices(1, self.low, self.high).item()
            lcoords = gen_coords(1, self.size, self.csize,)
            lcoords = lcoords[:, 0]
            label, _, _      = Rotate()(
                Crop(lcoords, self.csize)(
                torch.from_numpy(np.load(self.labels[idx]))).float()
                .to(self.device))
            params = gen_imaging_parameters(self.params_range)
            imaging = ImagingProcess(self.device, params,
                                     z=self.z, x=self.x, y=self.y,
                                     mode="dataset")
            with torch.no_grad():
                image = imaging(label)
            image = apply_mask(self.mask, image, self.mask_size, self.mask_num)
            surround_size = [self.surround_size[0] // params["scale"],
                             self.surround_size[1],
                             self.surround_size[2],]
            image = apply_surround_mask(self.surround, image, surround_size)
        else:
            _idx    = self.indiceslist[idx]  # convert idx to [low] ~[high] number
            lcoords = self.coordslist[0][:, idx]
            label   = Crop(lcoords, self.csize)(
                      torch.from_numpy(np.load(self.labels[_idx])).float()
                      .to(self.device))
            with torch.no_grad():
                image = self.imaging(label, self.is_train)
            surround_size = [self.surround_size[0] // self.validation_scale,
                             self.surround_size[1],
                             self.surround_size[2],]
            image   = apply_surround_mask(self.surround, image,
                                          surround_size)

        return image, label, params

    def __len__(self):
        return self.I


class RandomBlurbyModelDataset(Dataset):
    '''
    input  : 4d torch.tensor (large (like 768**3) size) (label)
    output : 4d torch.tensor (small (like 128**3) size)
             ([channels, z_size, x_size, y_size]) 
             of randomly blurred/cropped/rotated image and label
    :param folderpath : large data path ("randomdata" in this repo)
    labelname : "0001***.pt" `s "**" part. (e.g. "_label")
    I : sample size. Returns I samples. (e.g. 200)
    low, high : use [low]th ~ [high]th files in folderpath as data.
    '''
    def __init__(self, folderpath:str,
                 size:list, cropsize:list, I:int, low:int, high:int,
                 imaging_function,
                 imaging_params_range:dict,
                 validation_params:dict,
                 device, is_train=True, mask=True, 
                 mask_size=[10, 10, 10], mask_num=1,
                 surround=True, surround_size=[72, 8, 8],
                 seed=523):
        self.I             = I
        self.low           = low
        self.high          = high
        self.size          = size
        self.labels        = list(sorted(Path(folderpath).glob(f'*_label.npy')))
        self.csize         = cropsize
        self.is_train      = is_train
        self.mask          = mask
        self.mask_size     = mask_size
        self.mask_num      = mask_num
        self.surround      = surround
        self.surround_size = surround_size
        self.params_range  = imaging_params_range
        self.imaging       = imaging_function
        self.valid_params = validation_params
        self.device           = device
        if is_train == False:
            np.random.seed(seed)
            self.indiceslist = gen_indices(I, low, high)
            self.coordslist  = gen_coords(I, size, cropsize)

    def __getitem__(self, idx):
        if self.is_train:
            idx     = gen_indices(1, self.low, self.high).item()
            lcoords = gen_coords(1, self.size, self.csize,)
            lcoords = lcoords[:, 0]
            label, _, _      = Rotate()(
                Crop(lcoords, self.csize)(
                torch.from_numpy(np.load(self.labels[idx]))).float()
                .to(self.device))
            params = gen_imaging_parameters(self.params_range)
            with torch.no_grad():
                image = self.imaging.sample_from_params(label, params)
            image = apply_mask(self.mask, image, self.mask_size, self.mask_num)
            surround_size = [self.surround_size[0] // params["scale"],
                             self.surround_size[1],
                             self.surround_size[2],]
            image = apply_surround_mask(self.surround, image, surround_size)
        else:
            _idx    = self.indiceslist[idx]  # convert idx to [low] ~[high] number
            lcoords = self.coordslist[0][:, idx]
            label   = Crop(lcoords, self.csize)(
                      torch.from_numpy(np.load(self.labels[_idx])).float()
                      .to(self.device))
            with torch.no_grad():
                image = self.imaging.sample_from_params(label, self.valid_params)
            surround_size = [self.surround_size[0] // self.valid_params["scale"],
                             self.surround_size[1],
                             self.surround_size[2],]
            image   = apply_surround_mask(self.surround, image,
                                          surround_size)

        return image, label, params

    def __len__(self):
        return self.I


class LabelandBlurParamsDataset(Dataset):
    '''
    input  : 4d torch.tensor (large (like 768**3) size) (label)
    output : 4d torch.tensor (small (like 128**3) size)
             ([channels, z_size, x_size, y_size]) 
             of randomly blurred/cropped/rotated image and label
    :param folderpath : large data path ("randomdata" in this repo)
    labelname : "0001***.pt" `s "**" part. (e.g. "_label")
    I : sample size. Returns I samples. (e.g. 200)
    low, high : use [low]th ~ [high]th files in folderpath as data.
    '''
    def __init__(self, folderpath:str,
                 size:list, cropsize:list, I:int, low:int, high:int,
                 imaging_function,
                 imaging_params_range:dict,
                 validation_params:dict,
                 device, is_train=True, mask=True, 
                 mask_size=[10, 10, 10], mask_num=1,
                 surround=True, surround_size=[72, 8, 8],
                 seed=523):
        self.I             = I
        self.low           = low
        self.high          = high
        self.size          = size
        self.labels        = list(sorted(Path(folderpath).glob(f'*_label.npy')))
        self.csize         = cropsize
        self.is_train      = is_train
        self.mask          = mask
        self.mask_size     = mask_size
        self.mask_num      = mask_num
        self.surround      = surround
        self.surround_size = surround_size
        self.params_range  = imaging_params_range
        self.imaging       = imaging_function
        self.valid_params = validation_params
        self.device           = device
        if is_train == False:
            np.random.seed(seed)
            self.indiceslist = gen_indices(I, low, high)
            self.coordslist  = gen_coords(I, size, cropsize)

    def __getitem__(self, idx):
        if self.is_train:
            idx     = gen_indices(1, self.low, self.high).item()
            lcoords = gen_coords(1, self.size, self.csize,)
            lcoords = lcoords[:, 0]
            label, _, _      = Rotate()(
                Crop(lcoords, self.csize)(
                torch.from_numpy(np.load(self.labels[idx]))).float())
            params = gen_imaging_parameters(self.params_range)
        else:
            _idx    = self.indiceslist[idx]  # convert idx to [low] ~[high] number
            lcoords = self.coordslist
            lcoords = self.coordslist[:, idx]
            label   = Crop(lcoords, self.csize)(
                      torch.from_numpy(np.load(self.labels[_idx])).float())
            params  = self.valid_params

        return label, params

    def __len__(self):
        return self.I


class ParamScaler():
    def __init__(self, scales):
        self.scales = scales
    def normalize(self, params):
        for (k, v), (sk, sv) in zip(params.items(), self.scales.items()):
            if k == sk:
                v = v / sv
                d = {k: v}
                params.update(d)
            else:
                print(sk, " unmatched with ", k)
        return params
    
    def denormalize(self, params):
        for (k, v), (sk, sv) in zip(params.items(), self.scales.items()):
            if k == sk:
                v = v * sv
                d = {k: v}
                params.update(d)
            else:
                print(sk, " unmatched with ", k)
        return params


import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode as I


class Vibrate():
    def __init__(self, vibration_params:dict):
        self.max_step      = vibration_params["max_step"            ]
        self.b_sigma       = vibration_params["b_sigma"             ]
        self.amp_alpha     = vibration_params["amp_alpha"           ]
        self.omega_alpha   = vibration_params["omega_alpha"         ]
        self.inv_var_alpha = vibration_params["noise_inv_var_alpha" ]

        self.num_step = 0

    def __call__(self, img, determined=False):
        img = img.clone().detach()
        if img.dim() == 5:
            for i in range(img.size(0)):
                if determined:
                    params_y = self._gen_deterministic_params(self.num_step)
                    params_x = self._gen_deterministic_params(self.num_step)
                else:
                    params_x = self._gen_params(self.num_step)
                    params_y = self._gen_params(self.num_step)
                len_t = img.size(-3)
                vib_y = self._vibration(len_t, *params_y)
                vib_x = self._vibration(len_t, *params_x)
                shift = np.stack([vib_y, vib_x], 1)
                img[i] = self._shift3d(img[i], shift)

        return img
    
    def step(self):
        self.num_step += 1
        return self.num_step
    
    def set_arbitrary_step(self, num):
        self.num_step = num

    def _gen_params(self, num_step):
        b_sigma = self._hyper_param_linear_schedule(
            num_step = num_step       ,
            param    = self.b_sigma   ,
            option   = "up"           ,)
        
        amp_alpha  = self._hyper_param_linear_schedule(
            num_step = num_step       ,
            param    = self.amp_alpha ,
            option   = "up"           ,)
        
        omega_alpha = self._hyper_param_linear_schedule(
            num_step = num_step         ,
            param    = self.omega_alpha ,
            option   = "up"             ,)
        
        inv_var_alpha = self._hyper_param_linear_schedule(
            num_step = num_step           ,
            param    = self.inv_var_alpha ,
            option   = "down"             ,)
        
        amp           = np.random.gamma(amp_alpha, 1)
        b             = np.random.normal(0, b_sigma)
        c             = 0
        omega         = np.random.gamma(omega_alpha, 1)
        phi           = np.random.uniform(0, 2 * np.pi)
        noise_inv_var = np.random.gamma(inv_var_alpha, 1)
        
        return amp, b, c, omega, phi, noise_inv_var
    
    def _gen_deterministic_params(self, num_step):
        
        amp_alpha  = self._hyper_param_linear_schedule(
            num_step = num_step       ,
            param    = self.amp_alpha ,
            option   = "up"           ,)
        
        omega_alpha = self._hyper_param_linear_schedule(
            num_step = num_step         ,
            param    = self.omega_alpha ,
            option   = "up"             ,)
        
        inv_var_alpha = self._hyper_param_linear_schedule(
            num_step = num_step           ,
            param    = self.inv_var_alpha ,
            option   = "down"             ,)
        
        amp           = amp_alpha
        b             = 0
        c             = 0
        omega         = omega_alpha
        phi           = 0
        noise_inv_var = np.random.gamma(inv_var_alpha, 1)
        
        return amp, b, c, omega, phi, noise_inv_var
    
    def _hyper_param_linear_schedule(
            self, num_step, param:dict, option)->float:
        
        if option == "up":
            org = param["min"]
            fin = param["max"]

        elif option == "down":
            org = param["max"]
            fin = param["min"]

        else:
            raise ValueError(f"option must be 'up' or 'down'\
                             (current option is '{option}')")
    
        rate = min(num_step / self.max_step, 1.)
        return org + (fin - org) * rate


    def _vibration(self, len_t, amp, b, c, omega, phi, noise_inv_var):
        t = np.arange(0, len_t, 1)
        x = b * t \
          + amp * np.exp( - c * t) * np.cos(omega * t + phi) \
          + np.random.randn(len_t) / noise_inv_var
        x = x - x.mean()
        return x

    def _shift2d(self, img, shift, interp) -> torch.Tensor:
        h, w = list(img.shape[-2:])
        _ch = (h - 1) / (h ** 2)
        _cw = (w - 1) / (w ** 2)
        ph, pw = shift
        disp = torch.cat((torch.full((1, h, w, 1), _ch * ph),
                          torch.full((1, h, w, 1), _cw * pw)), dim=3)
        _shifted_img = T.functional.elastic_transform(img.unsqueeze(0),
                                                      disp,
                                                      interpolation=interp)
        return _shifted_img.squeeze(0)

    def _shift3d(self, img, shift, interp=I.BILINEAR) -> torch.Tensor:
        img_dim = 5
        if img.dim() == 4:
            img_dim = 4
            img = img.unsqueeze(0)
        z_num = img.size(2)
        shifted_img = img.clone()
        for z in range(z_num):
            shifted_img[:, 0, z] = self._shift2d(img[:, 0, z].clone(),
                                                shift[z],
                                                interp)
        if img_dim == 4:
            shifted_img = shifted_img.squeeze(0)
        return shifted_img

if __name__ == "__main__":
    device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))


    params_ranges = {"mu_z"   : [0,   1, 0.2  ,  0.5 ],
                     "sig_z"  : [0,   1, 0.2  ,  0.5 ],
                     "bet_z"  : [0,  50,  25  , 12.5 ],
                     "bet_xy" : [0,  20,   1. ,  5.  ],
                     "alpha"  : [0, 100,  10  ,  5.  ],
                     "sig_eps": [0, 0.3, 0.15 ,  0.05],
                     "scale"  : [1, 2, 4, 8, 12      ]
                     }
    validation_params = gen_imaging_parameters(params_ranges)
    imageprocess = ImagingProcess(device, validation_params, z=71, x=3, y=3, mode="dataset")
    train_dataset = RandomBlurbyModelDataset(folderpath = "newrandomdataset",
                                      size              = (1200, 500, 500)  ,
                                      cropsize          = ( 240, 112, 112)  ,
                                      I                 =  10               ,
                                      low               =   0               ,
                                      high              =  19               ,
                                      imaging_function  = imageprocess      ,
                                      imaging_params_range = params_ranges  ,
                                      validation_params = validation_params,
                                      device = device
                                      )
    for i in range(10):
        print(train_dataset[i][0].mean(),
              train_dataset[i][0].shape,
              train_dataset[i][2])