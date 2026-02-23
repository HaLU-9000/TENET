import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import tifffile
import nd2
from unfoldNd import UnfoldNd

import model

class EarlyStopping():
    """
    path[str]: path you want to save your model
    name[str]: model name
    patience[int]: default = 10 
    window_size[int]: size of the moving window
    mode[int]: 1 for minimizing, -1 for maximizing
    metric[str]: 'mean' for moving mean, 'median' for moving median
    verbose[bool]: whether to print messages
    """
    def __init__(self, path, name, patience=10, window_size=5, mode=1, metric='mean', verbose=True):
        self.patience = patience
        self.window_size = window_size
        self.verbose = verbose
        self.mode = mode
        self.metric = metric
        self.counter = 0
        self.best_stat = None
        self.early_stop = False
        self.val_losses = []
        self.path = path
        self.name = name

    def __call__(self, val_loss, model, optimizer, condition=False):
        self.val_losses.append(val_loss)

        window = min(len(self.val_losses) , self.window_size)
        if self.metric == 'mean':
            moving_stat = np.mean(self.val_losses[-window:])
        elif self.metric == 'median':
            moving_stat = np.median(self.val_losses[-window:])
        else:
            raise ValueError("Unsupported metric. Use 'mean' or 'median'.")
        
        if self.best_stat is None:
            self.best_stat = moving_stat
            self.checkpoint(moving_stat, model, optimizer)
        
        elif not condition:
            self.checkpoint(moving_stat, model, optimizer)
            self.best_stat = moving_stat
            self.counter = 0

        else:
            if (self.mode * moving_stat > self.mode * self.best_stat or f'{moving_stat:.6f}' == 'nan'):
                self.counter += 1
                if self.verbose:
                    print(f' Current Loss ({moving_stat:.6f}) EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    print('EarlyStopping!')
            else:
                self.checkpoint(moving_stat, model, optimizer)
                self.best_stat = moving_stat
                self.counter = 0

    def checkpoint(self, moving_stat, model, optimizer):
        if self.verbose:
            print(f'Moving {self.metric.capitalize()} Loss ({self.best_stat:.6f}) -> Current Loss ({moving_stat:.6f}). Saving models...')
        torch.save(model.state_dict(), f'{self.path}/{self.name}.pt')
        torch.save(optimizer.state_dict(), f'{self.path}/{self.name}_optim.pt')


def path_blur():
    return 0

def save_dataset(model, folderpath, outfolderpath, labelname, outlabelname, scale, device, I=0):
    flist = list(sorted(Path(folderpath).glob(f'*{labelname}.npy')))
    model = model.to(device)
    for i, label in enumerate(flist[I:]):
        label = torch.from_numpy(np.load(label))
#        if not Path(f'{outfolderpath}/{str(i+I).zfill(4)}{outlabelname}.pt').is_file():
        torch.save(label.float(),  f'{outfolderpath}/{str(i+I).zfill(4)}{outlabelname}.pt')
        label = label.unsqueeze(0)
        blur = model.sample(label.to(device))
        blur = blur.detach().to('cpu').squeeze(0)#.numpy()
        #blur = torch.from_numpy(blur) #2022/12/10 changed
        torch.save(blur, f'{outfolderpath}/{str(i+I).zfill(4)}_x{scale}.pt')

def save_label(folderpath, outfolderpath, labelname, outlabelname, I=0):
    flist = list(sorted(Path(folderpath).glob(f'*{labelname}.npy')))
    for i, label in enumerate(flist[I:]):
        label = torch.from_numpy(np.load(label))
        torch.save(label.float(),  f'{outfolderpath}/{str(i+I).zfill(4)}{outlabelname}.pt')
        
def create_mask_(h, w, center=None, radius=None):
    if center is None:
        center = (int(w/2), int(h/2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist_from_center <= radius
    mask = mask * 1.0
    return mask

def gen_bcelist(model, model_name, val_dataset, device, partial=None):
    model.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
    model.eval()
    bce = nn.BCELoss()
    bces = []
    for i in range(len(val_dataset)):
        image, label = val_dataset[i]
        if partial is not None:
            label = label[:, partial[0]:partial[1], :, :].detach()
        image   = image.to(device=device).unsqueeze(0)
        pred, _ = model(image)
        pred    = pred.to(device='cpu').squeeze(0)
        if partial is not None:
            pred = pred[:, partial[0]:partial[1], :, :].detach()
        bces.append(bce(pred, label).to('cpu').item())
    return bces

def gen_bcecontrol(val_dataset, partial=None):
    bce = nn.BCELoss()
    bces = []
    for i in range(len(val_dataset)):
        _, label = val_dataset[i]
        if partial is not None:
            label = label[:, partial[0]:partial[1], :, :].detach() * 1.0
        bces.append(bce(label, torch.ones_like(label) * torch.mean(label)).item())
    return bces

def torch_log2(x):
    return torch.clip(torch.log2(x), min=-100, max=100)

def bcelosswithlog2(inp, target):
    bcelg2 = -torch.mean(target * torch_log2(inp) + (1.0 - target) * torch_log2(1.0 - inp)).to('cpu')
    return bcelg2.item()

def gen_bcelg2list(model, model_name, val_dataset, device, partial=None):
    model.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
    model.eval()
    bces = []
    for i in range(len(val_dataset)):
        image, label = val_dataset[i]
        if partial is not None:
            label = label[:, partial[0]:partial[1], :, :].detach()
        image   = image.to(device=device).unsqueeze(0)
        pred, _ = model(image)
        pred    = pred.to(device='cpu').squeeze(0)
        if partial is not None:
            pred = pred[:, partial[0]:partial[1], :, :].detach()
        bces.append(bcelosswithlog2(pred, label).to('cpu').item())
    return bces

def gen_bcelg2control(val_dataset, partial=None):
    bces = []
    for i in range(len(val_dataset)):
        _, label = val_dataset[i]
        if partial is not None:
            label = label[:, partial[0]:partial[1], :, :].detach() * 1.0
        bces.append(bcelosswithlog2(torch.ones_like(label) * torch.mean(label), label).item())
    return bces

def gen_bcelg2lists_ctrls(model, model_names, val_datasets, device, taus, partials=[]):
    """
    returns bcess (0:control, 1~:model evals)
    """
    bcess = []
    ctrls = []
    for model_name, val_dataset, partial, tau in zip(model_names, val_datasets, partials, taus):
        model.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
        model.eval()
        model.set_tau(tau)
        bces = []
        ctrl = []
        for i in range(len(val_dataset)):
            image, label = val_dataset[i]
            if partial is not None:
                label = label[:, partial[0]:partial[1], :, :].detach()
            image   = image.to(device=device).unsqueeze(0)
            pred, _ = model(image)
            pred    = pred.to(device='cpu').squeeze(0)
            if partial is not None:
                pred = pred[:, partial[0]:partial[1], :, :].detach()
            bces.append(bcelosswithlog2(pred, label))
            ctrl.append(bcelosswithlog2(torch.ones_like(label) * torch.mean(label), label))
        bcess.append(bces)
        ctrls.append(ctrl)
    ctrl = [item for ctrl in ctrls for item in ctrl] # flatten control list
    bcess.insert(0, ctrl)
    return bcess

def mask_(image, mask_size, mask_num):
    """
    image     : 4d/5d tensor
    mask_size : list with 3 elements (z, x, y)
    mask_num  : number of masks (default=1)
    out       : 4d/5d tensor (randomly masked)
    """
    for i in range(mask_num):
        image_is_4d = False
        if len(image.shape) == 4:
            image_is_4d = True
            image = image.unsqueeze(0)
        _b, _c, _z, _x, _y = image.shape
        mask = torch.zeros((_b, _c, *mask_size)).to(device=image.device)
        _, _, mz, mx, my = mask.shape
        z = np.random.randint(0, _z)
        x = np.random.randint(0, _x)
        y = np.random.randint(0, _y)
        z_max = min(z + mz, _z)
        x_max = min(x + mx, _x)
        y_max = min(y + my, _y)
        image[:, :, z : z + mz, x : x + mx, y : y + my] \
        = mask[:, :, 0 : z_max - z, 0 : x_max - x, 0 : y_max - y]
        if image_is_4d:
            image = image.squeeze(0)
    return image

def surround_mask_(image, surround_size):
    """
    image     : 4d/5d tensor
    mask_size : list with 3 elements (z, x, y > 0)
    out       : 4d/5d tensor (surround masked)
    """
    image_is_4d = False
    if len(image.shape) == 4:
        image_is_4d = True
        image = image.unsqueeze(0)
    z, x, y = surround_size
    image = image[:, :, z : -z, x : -x, y : -y]
    image = F.pad(image, (y, y, x, x, z, z)) # see https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    if image_is_4d:
            image = image.squeeze(0)
    return image

def tt(x):
    """
    returns torch cuda tensor of x
    """
    return torch.tensor(x, requires_grad=False, device="cuda")

def split_time(load_folderpath, tifpath, save_folderpath):
    tiff = tifffile.imread(os.path.join(load_folderpath, tifpath))
    tifpath = tifpath[:-4]
    channels = ["C1-", "C2-", "C3-"]
    for channel in channels:
        if channel in tifpath:
            tifpath = tifpath.replace(channel,"")+ channel[:-1]
    if len(tiff.shape) == 3:
        tifffile.imwrite(save_folderpath+tifpath+f"-C{1}-T{1}.tif", tiff[None, :])
    else:
        for t in range(tiff.shape[0]):
            tifffile.imwrite(save_folderpath+tifpath+f"-T{t+1}.tif", tiff[t][None, :])

def tifpath_to_tensor(tifpath, preprocess=False):
    tiff   = tifffile.imread(tifpath).astype('float32')
    tensor = torch.from_numpy(tiff)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    if preprocess:
        tensor = torch.clip(tensor, min=torch.tensor(0.1), max=torch.tensor(1.))
        tensor = (tensor - 0.1) / 0.9
    return tensor

def array_to_tif(path, array):
    tifffile.imwrite(path, array)

def load_anything(image_name):
    """
    input: image_name(str)
    output: torch.tensor
    """
    if image_name[-3:] == "tif" or image_name[-4:] == "tiff":
        image = tifffile.imread(image_name).astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        image = torch.tensor(image)
    elif image_name[-3:] == "npy":
        image = np.load(image_name, allow_pickle=True).astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        image = torch.tensor(image)
    elif image_name[-3:] == ".pt":
        image = torch.load(image_name)
        image = (image - image.min()) / (image.max() - image.min())
    elif image_name[-3:] == "nd2":

        image = nd2.imread(image_name).astype(np.float32)
        image = image.transpose(1,0,2,3)[0:1]
        image = image / (2**12-1)
        image = torch.tensor(image)
    else:
        print(f"YOUR FILE IS NOT AVAILABLE. ({image_name})\
              Use 4D(CZXY, C=1) array of `.pt`, `.npy` or `.tif`")
        
    if image.dim() != 4:
        if image.dim() == 3:
            image = image[None, :].clone()
        elif image.dim() == 5:
            image = image[0].clone()
        else:
            print(f"Your data must be 3d or 5d(TCZXY or CTZXY with C=1, T=1), but it is {image.dim()}d now!")
    return image

def find_best_int_n_bit(bit):
    thresholds = [0, 16, 32, 64, 128, 256]
    for threshold in thresholds:
        if bit < threshold:
            return threshold
    return 256

def convert_tensor_to__8_bit_ndarray(tensor):
    arr = tensor.detach().cpu().numpy() * (2 ** 8  - 1)
    return arr.astype(np.uint8)

def convert_tensor_to_16_bit_ndarray(tensor):
    arr = tensor.detach().cpu().numpy() * (2 ** 16 - 1)
    return arr.astype(np.uint16)

def convert_tensor_to_32_bit_ndarray(tensor):
    arr = tensor.detach().cpu().numpy() * (2 ** 32 - 1)
    return arr.astype(np.uint32)

def convert_tensor_to_64_bit_ndarray(tensor):
    arr = tensor.detach().cpu().numpy() * (2 ** 64 - 1)
    return arr.astype(np.uint64)

def save_ndarray_in_any_format(
        array, file, format):
    
    file = file + "." + format
    if format == "tif":
        tifffile.imwrite(file, array)
    elif format == "npy":
        np.save(file=file, arr=array)
    else:
        print(f"YOUR FORMAT `({format})` IS NOT AVAILABLE. \
              Use 4D(CZXY, C=1) array of `npy` or `tif`")
        
def init_model(params, is_finetuning):
    if is_finetuning:
        params["apply_vq"]        = True
        params["use_x_quantized"] = True
    net = model.JNet(params)
    return net

def init_dna(params):
    dna = model.DeepAlignNet(params)
    return dna

def load_model_weight(model, model_name):
    model.load_state_dict(torch.load(f'model/{model_name}.pt',
                                     map_location="cpu"),
                          strict=False,
                         )

def mount_model_to_device(model, configs):
    model.to(configs["params"]["device"])

def get_basename(pathlike:str):
    return os.path.splitext(os.path.basename(pathlike))[0]

def get_extention(pathlike:str):
    return os.path.splitext(os.path.basename(pathlike))[1]


class ImageProcessing():
    def __init__(self, image=None, image_name=None):
        if self._exist(image_name) and self._exist(image):
            raise ValueError("Both 'image' and 'image_name' cannot\
                             be provided simultaneously.")

        if self._exist(image_name):
            self.image = load_anything(image_name)
            self.image_name = get_basename(image_name)

        elif self._exist(image):
            self.image = image
        
        self.original_shape = image.squeeze(0).shape
        self.processed_image = None
        self.short_progress_bar="{l_bar}{bar:10}{r_bar}{bar:-10b}"
        self.deconv_model = None
        self.align_model  = None

    def apply_both(
            self         ,
            align_model  ,
            deconv_model ,
            align_params ,
            params       ,
            chunk_shape  ,
            overlap      ,
            file_align   ,
            file_deconv  ,
            format       ,
            bit          ,
            ):
        self.process_image(
            model       = align_model     ,
            params      = align_params    ,
            chunk_shape = chunk_shape     ,
            type        = "aligned_image" ,
            overlap     = overlap         ,
            apply_hill  = True
                           )
        self.save_processed_image(
            file   = file_align  ,
            format = format      ,
            bit    = bit         ,
            )
        self.image = self.processed_image
        self.process_image(
            model       = deconv_model    ,
            params      = params          ,
            chunk_shape = chunk_shape     ,
            type        = "enhanced_image",
            overlap     = overlap         ,
            apply_hill  = False
                           )
        self.save_processed_image(
            file   = file_deconv   ,
            format = format        ,
            bit    = bit           ,
            )

    def process_image(
            self               ,
            model              ,
            params             ,
            chunk_shape        ,
            type               ,
            overlap            ,
            margin             ,
            apply_hill  = True ,
            ):
        print("[1/3] making chunks...")
        chunks = self._make_chunks(self.image, chunk_shape, overlap)
        processed_chunks = []
        print("[2/3] processing chunks...")
        for chunk in tqdm(
            chunks, bar_format=self.short_progress_bar):
            processed_chunk = self._process(
                chunk, model, params, margin, apply_hill)
            processed_chunks.append(processed_chunk)
        print("[3/3] reconstrusting image...")
        processed_image = self._reconstruct_images(
            processed_chunks, params, type, overlap)
        self.processed_image = processed_image
        return processed_image
    
    def save_processed_image(self, file, format, bit=16):
        if   bit == 8:
            bit = 8
            array = convert_tensor_to__8_bit_ndarray(self.processed_image)
        elif bit <= 16:
            bit = 16
            array = convert_tensor_to_16_bit_ndarray(self.processed_image)
        elif bit <= 32:
            bit = 32
            array = convert_tensor_to_32_bit_ndarray(self.processed_image)
        elif bit <= 64:
            bit = 64
            array = convert_tensor_to_64_bit_ndarray(self.processed_image)
        else:
            raise ValueError(f"bit must be between 8 and 64, Current: {bit}.")

        save_ndarray_in_any_format(array, file, format)
        print(f"your processed image was saved in {file} with {bit} bit")

    def _exist(self, obj):
        if obj is not None:
            return True
        else:
            return False
        
    def _make_chunks(self, image, chunk_shape, overlap):
        shape = self.original_shape
        strides = self._get_strides(chunk_shape, overlap)
        chunks = []
        for z in range(0, shape[0], strides[0]):
            for x in range(0, shape[1], strides[1]):
                for y in range(0, shape[2], strides[2]):
                    chunk = self._make_chunk(image, z, x, y, chunk_shape)
                    if self._is_not_desired_shape(chunk, chunk_shape):
                        chunk = self._add_zero_padding(chunk, chunk_shape)
                    chunks.append(chunk)

        return chunks
    
    def _get_strides(self, chunk_shape, overlap):
        strides = []
        for c, o in zip(chunk_shape, overlap):
            stride = self._get_stride(c, o)
            strides.append(stride)
        return strides

    def _get_stride(self, chunk_length, overlap_length):
        stride = chunk_length - overlap_length
        if stride <= 0:
            raise ValueError(f"stride must be larger than 1.\
                              current stride is ({stride})")
        else:
            return stride
        
    def _make_chunk(self, image, i, j, k , chunk_shape):
        return image[
            :                     ,
            i : i + chunk_shape[0],
            j : j + chunk_shape[1],
            k : k + chunk_shape[2],
            ]

    def _is_not_desired_shape(self, chunk, shape):
        return chunk.squeeze(0).shape != shape

    def _add_zero_padding(self, chunk, shape):
        padded = torch.zeros(1, *shape)
        padded[
            :,
            :chunk.shape[1],
            :chunk.shape[2],
            :chunk.shape[3],]\
            += chunk
        
        return padded
    
    def _remove_margin(self, chunk, margin):
        padded = torch.zeros(chunk.shape)
        z, x, y = margin.shape
        padded[
            :,
            z : -z ,
            x : -x ,
            y : -y ,
               ] += chunk[     
            :,
            z : -z ,
            x : -x ,
            y : -y ,
               ]
        
        return padded

    
    def _process(self, chunk, model, params, margin,apply_hill):  
        chunk = self._make_5d(chunk)
        chunk = self._to_model_device(chunk, params)
        
        if apply_hill:
            chunk = self.deconv_model.image.hill.sample(chunk)
        chunk = model(chunk)
        enhanced_image = chunk["enhanced_image" ]\
            .squeeze(0).detach().clone().cpu()
        estim_luminance = chunk["estim_luminance"]\
            .squeeze(0).detach().clone().cpu()
        reconstruction = chunk["reconstruction" ]\
            .squeeze(0).detach().clone().cpu()
        
        enhanced_image  = self._remove_margin(enhanced_image , margin)
        estim_luminance = self._remove_margin(estim_luminance, margin)
        reconstruction  = self._remove_margin(reconstruction , margin)

        return {
            "enhanced_image" : enhanced_image ,
            "estim_luminance": estim_luminance,
            "reconstruction" : reconstruction ,
        }
        


    def _make_5d(self, chunk):
        return chunk.unsqueeze(0)
    
    def _to_model_device(self, chunk, params):
        return chunk.to(params["device"])
        
    def _reconstruct_images(self, chunks, params, type:str, overlap):
        shape       = self._get_image_shape(type, params)
        reconstruct = torch.zeros(1, *shape)
        _overlap    = torch.zeros(1, *shape)
        chunk_shape = self._get_processed_chunk_shape(chunks, type)
        overlap     = self._adjust_overlap_shape(overlap, type, params)
        strides     = self._get_strides(chunk_shape, overlap)
        idx         = 0

        for z in range(0, shape[0], strides[0]):
            for x in range(0, shape[1], strides[1]):
                for y in range(0, shape[2], strides[2]):
                    
                    tmp_shape = self._get_cropped_zeros_tensor_shape(
                        reconstruct, z, x, y, chunk_shape)
                    
                    reconstruct = self._insert_chunk_into_zeros_tensor(
                        reconstruct, chunks[idx][type], z, x, y, 
                        chunk_shape, tmp_shape)
                    
                    _overlap = self._insert_chunk_into_zeros_tensor(
                        _overlap, torch.ones_like(chunks[idx][type]), z, x, y, 
                        chunk_shape, tmp_shape)
                    
                    idx += 1

        reconstruct = self._resolve_overlap(reconstruct, _overlap)
        return reconstruct

    def _get_image_shape(self, type, params): 
        if type == "enhanced_image" or type == "estim_luminance":
            scale = params["scale"]
            z, x, y = self.original_shape
            shape = [z * scale, x, y]
        elif type == "reconstruction" or type =="aligned_image":
            shape = self.original_shape
        return shape
    
    def _adjust_overlap_shape(self, overlap, type, params):
        z, x, y = overlap 
        if type == "enhanced_image" or type == "estim_luminance":
            scale = params["scale"]
            overlap = [z * scale, x, y]
        elif type == "reconstruction" or type == "aligned_image":
            overlap = overlap
        return overlap
        
    def _get_processed_chunk_shape(self, chunks, type:str):
        return chunks[0][type].shape[1:]

    def _get_cropped_zeros_tensor_shape(self, image, i, j, k , chunk_shape):
        return image[
            :                     ,
            i : i + chunk_shape[0],
            j : j + chunk_shape[1],
            k : k + chunk_shape[2],
            ].shape
    
    def _insert_chunk_into_zeros_tensor(
            self, zeros, chunks, i, j, k, chunk_shape, tmp_shape):
        zeros[
            :                     ,
            i : i + chunk_shape[0],
            j : j + chunk_shape[1],
            k : k + chunk_shape[2],
            ] \
        += chunks[
            :,
            :tmp_shape[1],
            :tmp_shape[2],
            :tmp_shape[3],
            ]
        return zeros
    
    def _resolve_overlap(self, image, overlap):
        return image / overlap


class MRFLoss():
    """
    First number means the pixel value that will be evaluated and
    second one means the neighbor pixel.
    For example, l_10 means loss weight for pixel value 1 when the neighbors 
    values are 0.
    usage \n
    ```
    mrf_loss = MRFLoss(dims=2, order=1)
    x = F.sigmoid(torch.randn(1000).view(1,1,10,10,10)
    mrf_loss(x)
    >> torch.tensor(1.4376)
    ```
    """
    def __init__(self, dims, order, weights, dilation):
        self.dims = dims
        kernel_size = [order * 2 + 1] * 3
        self.unfoldnd = UnfoldNd(
            kernel_size = kernel_size,
            dilation    = dilation   ,
            padding     = "same"     ,
            stride      = 1          ,
            )
        euclid_vector = self._get_euclid_vector(kernel_size)
        self.weight_vector = self._inverse_euclid_with_zero_handling(
            euclid_vector)
        self.l_00 = weights["l_00"]
        self.l_01 = weights["l_01"]
        self.l_10 = weights["l_10"]
        self.l_11 = weights["l_11"]

    def __call__(self, x):
        x_unfolded_to_kernels = self.unfoldnd(x)
        energy = self._calc_batched_mrf_energy(x, x_unfolded_to_kernels)
        loss   = energy * self.weight_vector.to(energy.device)
        return loss.mean()

    def _calc_batched_mrf_energy(self, x, neighbors):
        x = x.flatten(-self.dims)
        loss = self.l_00 * (1 - x) * (1 - neighbors)              \
             + self.l_01 * (1 - x) *  neighbors                   \
             + self.l_10 *  x      * (1 - neighbors).prod(dim=1)  \
             + self.l_11 *  x      * (1 - (1 - neighbors).prod(dim=1))
        return loss

    def _get_euclid_vector(self, kernel_size:list):
        distance = self._get_euclid_distance_from_center_3d(kernel_size)
        return self._get_vector_from_3d_distance(distance)

    def _get_euclid_distance_from_center_3d(self, kernel_size:list):
        center0 = kernel_size[0] // 2
        center1 = kernel_size[1] // 2
        center2 = kernel_size[2] // 2
        dim_wise_distance = torch.meshgrid(
            torch.arange(kernel_size[0]) - center0,
            torch.arange(kernel_size[1]) - center1,
            torch.arange(kernel_size[2]) - center2,
            indexing="ij")
        distance_pow_2 = torch.zeros(kernel_size)
        for d in dim_wise_distance:
            distance_pow_2 += d ** 2
        distance = torch.sqrt(distance_pow_2)
        return distance

    def _get_vector_from_3d_distance(self, distance):
        vec = distance.flatten()[None]
        return vec[..., None]

    def _inverse_euclid_with_zero_handling(self, vector):
        return torch.where(vector == 0., 0, 1 / vector,)
    
class OldMRFLoss():
    """
    usage \n
    ```
    mrf_loss = MRFLoss(dims=2, mode="all")
    x = torch.randn(100).view(1,1,10,10)
    mrf_loss(x)
    >> torch.tensor(1.4376)
    ```
    """
    def __init__(self, dims, order=1, mode="orthogonal"):
        if dims == 1:
            self.axis = [-1]
        if dims == 2:
            self.axis = [-2, -1]
        if dims == 3:
            self.axis = [-3, -2, -1]
        self.dims = dims
        self.order = order
        if mode == "orthogonal":
            self.shifts = self._get_orthogonal_shift_list(self.order)
        elif mode == "all":
            self.shifts = self._get_shift_list(self.order)
        else:
            raise ValueError(f"mode '{mode}' is not implemented." +\
                 " Try 'all' or 'orthogonal'. ")

    def __call__(self, x):
        energy = 0
        for shift in self.shifts:
            euclid = self._calc_euclid_distance(shift)
            markov = self._markov_difference(x, shift)
            energy = energy + self._criterion(diff=markov, dist=euclid)
        return energy

    def _criterion(self, diff, dist, mode="mean", loss_type="squared"):
        if loss_type == "gaussian":
            loss = self._gaussian_loss(diff, dist)
        elif loss_type == "squared":
            loss = self._squared_loss(diff, dist)
        else:
            raise ValueError(f"loss_type '{loss_type}' is not implemented." +\
                             " Try 'gaussian' or 'squared'. ")
        if mode == "mean":
            return loss.mean()
        elif mode == "sum":
            return loss.sum()
        else:
            raise ValueError(f"mode '{mode}' is not implemented." +\
                             " Try 'mean' or 'sum'. ")

    def _gaussian_loss(self, diff, dist):
        coeffs = 1 / (dist * (2 * 3.14) ** 1/2)
        loss = coeffs * torch.exp((-1/2)*(diff / dist) ** 2)
        return loss

    def _squared_loss(self, diff, dist):
        return (diff / dist) ** 2

    def _get_orthogonal_shift_list(self, order):
        shifts = []
        distance = [o for o in range(-order, order+1)]
        org = [0 for _ in range(self.dims)]
        for dim in range(self.dims):
            for dist in distance:
                shift      = org.copy()
                shift[dim] = dist
                shifts.append(shift)
        shifts.sort()
        shifts = self._get_unique_list(shifts)
        shifts = self._remove_zeros(shifts)
        return shifts

    def _get_shift_list(self, order):
        shifts = []
        distance = [o for o in range(-order, order + 1)]
        org = [0 for _ in range(self.dims)]
        shifts.append(org)
        for dim in range(self.dims):
            _shifts = []
            for shift in shifts:
                for dist in distance:
                    shift = shift.copy()
                    shift[dim] = dist
                    _shifts.append(shift)
            shifts = _shifts
        shifts = self._remove_zeros(shifts)
        return shifts

    def _remove_zeros(self, seq):
        return [s for s in seq if not all(e == 0 for e in s)]

    def _get_unique_list(self, seq):
        seen = []
        return [x for x in seq if x not in seen and not seen.append(x)]

    def _calc_euclid_distance(self, shift):
        d = np.array(shift) ** 2
        return np.sqrt(d.sum())

    def _markov_difference(self, x, shift):
        return x - self._modified_roll(x, shift)

    def _modified_roll(self, x, shift):
        org = torch.roll(x, shift, dims=self.axis)
        mask = self._get_mask(x, shift)
        return org * mask

    def _get_mask(self, arr, shift):
        mask = torch.ones_like(arr)
        _shift = shift.copy()
        while len(_shift) < arr.dim():
            _shift.insert(0, 0)
        for dim, s in enumerate(_shift):
            if s >= 0:
                mask = mask.narrow(dim, 0, arr.size(dim) - s)
                mask = torch.cat([
                    torch.zeros_like(mask).narrow(dim, 0, s),
                    mask], 
                                 dim=dim)
            else:
                s = -s
                mask = mask.narrow(dim, s, arr.size(dim) - s)
                mask = torch.cat([
                    mask, 
                    torch.zeros_like(mask).narrow(dim, 0, s)],
                                 dim=dim)
        return mask

# load functions needed (like preprocessing)
def path_of(base, fig, sub, idx=0):
    """get the first file of folder"""
    dir  = os.path.join(base, fig, sub)
    name = sorted(os.listdir(dir))[idx]
    return os.path.join(dir, name)

def norm(x, bit=16):
    return x / (2 ** bit - 1)

def float_to_uint16(x):
    x *= (2 ** 16 - 1)
    return x.astype(np.uint16)

def care_align(x):
    return np.flip(np.rot90(x, axes=(-1,-2)), axis=-1)

def quantile_image(x,  q_max):
    max = np.quantile(x, q = q_max)
    return np.clip(x, a_min=0, a_max=max)

def quantile_image_minmax(x,  q_min, q_max):
    max = np.quantile(x, q = q_max)
    min = np.quantile(x, q = q_min)
    return np.clip(x, a_min=min, a_max=max)

def hill(x):
    return 2 * x ** 0.5 / (1 + x ** 0.5)

def mip(x, sec, start=0, range=10):
    if sec == "xz":
        _x = x[:, :, start:start+range]
        return np.max(_x, axis=2)

    elif sec == "yz":
        _x = x[:, start:start+range, :]
        return np.max(_x, axis=1)
    
    elif sec == "xy":
        _x = x[start:start+range, :, :]
        return np.max(_x, axis=0)

def upsample3d(image, scale=3):
    if len(image.shape) != 3:
        raise ValueError("shape len must be 3")
    image = F.interpolate(
        torch.tensor(image[None, None]*1.0),
        scale_factor=(scale,1,1),
        mode = "nearest",
        ).detach().numpy()
    return image[0, 0]

def iou(input, target, threshold=0.5, smooth=1e-6):
    """numpy.ndarray->float"""
    input  = input  > threshold
    target = target > 0.5
    input  = input.flatten() * 1.0
    target = target.flatten() * 1.0
    intersection = (input * target).sum()
    total = np.sum(input + target)
    union = total - intersection
    iou   =  (intersection + smooth) / (union + smooth)
    return iou

def iou_cuda(input, target, threshold=0.5, smooth=1e-6, device="cuda:0"):
    """numpy.ndarray->float"""
    input  = torch.tensor(input , device=device)
    target = torch.tensor(target, device=device)
    input  = input.flatten()
    target = target.flatten()
    input  = (input  > threshold) * 1.0
    intersection = (input * target).sum()
    total = torch.sum(input + target)
    union = total - intersection
    iou   =  ((intersection + smooth) / (union + smooth)).detach().cpu().numpy()
    return iou

def get_best_iou(input, target, bins=100, smooth=1e-6, device="cuda:0"):
    """numpy.ndarray->float"""
    with torch.no_grad():
        _input = torch.tensor(input  , device=device)
        target = torch.tensor(target , device=device)
        _input = _input.flatten()
        target = target.flatten()
        ths = np.linspace(0, 1, bins)
        iou = np.zeros_like(ths)
        for n in range(len(ths)):
            input        = (_input > ths[n]) * 1.0
            intersection = ( input * target).sum()
            total        = torch.sum(input + target)
            union        = total - intersection
            iou[n]       = ((intersection + smooth) / (union + smooth)
                            ).detach().cpu().numpy()
    del(_input)
    del(target)
    torch.cuda.empty_cache()
    return np.max(iou), np.argmax(iou) / bins