import os
import argparse
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import RandomCutDataset, DensityDataset
import model as model
from train_loop import imagen_instantblur
from dataset import Vibrate
from utils import array_to_tif, load_anything

class SimulationInference():
    def __init__(self, model_name, is_finetuning, is_vibrate, with_align):
        config = open(os.path.join("experiments/configs", f"{model_name}.json"))
        self.configs               = json.load(config)
        self.model_name            = model_name
        self.pre_model_name        = self.configs["pretrained_model"]
        self.params                = self.configs["params"]
        self.params["reconstruct"] = True
        self.device = self.params["device"]
        if is_finetuning:
            self.params["apply_vq"]        = True
            self.params["use_x_quantized"] = True
        val_dataset_params = self.configs["pretrain_val_dataset"]
        self.is_finetuning = is_finetuning
        self.is_vibrate    = is_vibrate
        self.vibrate = Vibrate(vibration_params=self.configs["vibration"])
        self.vibrate.set_arbitrary_step(100)
        if with_align:
            align_params   = self.configs["align_params"]
            deep_align_net = model.DeepAlignNet(align_params)
            self.deep_align_net = deep_align_net.to(device=self.device)
            self.deep_align_net.load_state_dict(
                torch.load(f'model/{self.configs["align_model"]}.pt'),
                strict=False)
            self.with_align = True
        else:
            self.with_align = False
        JNet = model.JNet(self.params)
        self.JNet = JNet.to(device = self.device)
        self.psf_pretrain = self.JNet.image.blur.show_psf_3d()
        self.JNet.load_state_dict(
            torch.load(f'model/{self.pre_model_name}.pt'),
            strict=False)
        self.psf_pretrain = self.JNet.image.blur.show_psf_3d()
        if self.is_finetuning:
            self.JNet.load_state_dict(
                torch.load(f'model/{self.model_name}.pt'),
                strict=False)
        else:
            self.model_name = self.pre_model_name
        self.psf_post = self.JNet.image.blur.show_psf_3d()
        self.JNet.eval()

        val_dataset   = RandomCutDataset(
            folderpath    = val_dataset_params["folderpath"]   ,
            size          = val_dataset_params["size"]         ,
            cropsize      = val_dataset_params["cropsize"]     , 
            I             = val_dataset_params["I"]            ,
            low           = val_dataset_params["low"]          ,
            high          = val_dataset_params["high"]         ,
            scale         = val_dataset_params["scale"]        , ## scale
            mask          = val_dataset_params["mask"]         ,
            mask_size     = val_dataset_params["mask_size"]    ,
            mask_num      = val_dataset_params["mask_num"]     , #( 1% of image)
            surround      = val_dataset_params["surround"]     ,
            surround_size = val_dataset_params["surround_size"],
            seed          = val_dataset_params["seed"]         ,
                                        )
        self.val_loader  = DataLoader(
            val_dataset                   ,
            batch_size  = 1               ,
            shuffle     = False           ,
            pin_memory  = False           ,
            num_workers = 0#os.cpu_count()  ,
                         )
    
    def get_result(self, num_results)->list:
        with torch.no_grad():
            results = []
            for n, val_data in enumerate(self.val_loader):
                if n >= num_results:
                    break
                if self.is_finetuning:
                    self.JNet.image.load_state_dict(
                        torch.load(
                            f"model/{self.pre_model_name}.pt"), 
                            strict=False)
                labelx = val_data["labelx"].to(device = self.device)
                labelz = val_data["labelz"].to(device = self.device)
                image  = imagen_instantblur(
                    model  = self.JNet  ,
                    label  = labelz     ,
                    device = self.device, 
                    params = self.params,)
                _image = self.JNet.image.hill.sample(image).detach().clone()
                if self.is_finetuning:            
                    self.JNet.image.load_state_dict(
                        torch.load(
                            f"model/{self.model_name}.pt"), 
                            strict=False)
                if self.is_vibrate:
                    image = self.vibrate(_image)
                else:
                    image = _image.detach().clone()
                if self.with_align:
                    outdic_a = self.deep_align_net(image)
                    image_a  = outdic_a["aligned_image"]
                else:
                    image_a = image.detach().clone()
                outdict  = self.JNet(image_a)
                outputx  = outdict["enhanced_image"]
                outputz  = outdict["estim_luminance"]
                reconst  = outdict["reconstruction"]
                reconst  = self.JNet.image.hill.sample(reconst).detach().clone()
                qloss    = outdict["quantized_loss"]
                qloss    = qloss.item() if qloss is not None else 0

                image    = image[0].detach().cpu().numpy()
                _image   = _image[0].detach().cpu().numpy()
                image_a  = image_a[0].detach().cpu().numpy()
                labelx   = labelx[0].detach().cpu().numpy()
                labelz   = labelz[0].detach().cpu().numpy()
                outputx  = outputx[0].detach().cpu().numpy()
                outputz  = outputz[0].detach().cpu().numpy()
                reconst  = reconst[0].detach().cpu().numpy()

                results.append({
                    "image"  : image   ,
                    "_image" : _image  ,
                    "imagea" : image_a ,
                    "outputx": outputx ,
                    "outputz": outputz ,
                    "reconst": reconst ,
                    "labelx" : labelx  ,
                    "labelz" : labelz  ,
                    "qloss"  : qloss   
                                })
        return results
        
    def evaluate(self, results)->list:
        msesx = []
        bcesx = []
        msesz = []
        bcesz = []
        qlosses = []
        for d in results:
            msex = np.mean(((d["labelx"] - d["outputx"]) ** 2).flatten())
            msez = np.mean(((d["labelz"] - d["outputz"]) ** 2).flatten())
            bcex = np.mean(-(d["labelx"]*np.log(d["outputx"]) + (1. - d["labelx"])*np.log(1. - d["outputx"])).flatten())
            bcez = np.mean(-(d["labelz"]*np.log(d["outputz"]) + (1. - d["labelz"])*np.log(1. - d["outputz"])).flatten())
            msesx.append(msex)
            msesz.append(msez)
            bcesx.append(bcex)
            bcesz.append(bcez)
            qlosses.append(d["qloss"])
        return {"MSEx" : msesx  ,
                "BCEx" : bcesx  ,
                "MSEz" : msesz  ,
                "BCEz" : bcesz  ,
                "qloss": qlosses,}
        
    def visualize(self, results):
        for n, d in enumerate(results):
            path = self.configs["visualization"]["path"] 
            j   = self.configs["visualization"]["z_stack"]
            j_s = j // self.params["scale"]
            i   = self.configs["visualization"]["x_slice"]
            mip = self.configs["visualization"]["mip"]
            mip_s = mip // self.params["scale"]

            image_xy   = np.max(d["image"  ][0, j_s:j_s+mip_s, :      , :], axis=0)
            _image_xy  = np.max(d["_image" ][0, j_s:j_s+mip_s, :      , :], axis=0)
            imagea_xy  = np.max(d["imagea" ][0, j_s:j_s+mip_s, :      , :], axis=0)
            outputx_xy = np.max(d["outputx"][0, j  :j+mip    , :      , :], axis=0)
            outputz_xy = np.max(d["outputz"][0, j  :j+mip    , :      , :], axis=0)
            labelx_xy  = np.max(d["labelx" ][0, j  :j+mip    , :      , :], axis=0)
            labelz_xy  = np.max(d["labelz" ][0, j  :j+mip    , :      , :], axis=0)
            reconst_xy = np.max(d["reconst"][0, j  :j+mip    , :      , :], axis=0)
            image_z    = np.max(d["image"  ][0, :            , i:i+mip, :], axis=1)
            _image_z   = np.max(d["_image" ][0, :            , i:i+mip, :], axis=1)
            imagea_z   = np.max(d["imagea" ][0, :            , i:i+mip, :], axis=1)
            outputx_z  = np.max(d["outputx"][0, :            , i:i+mip, :], axis=1)
            outputz_z  = np.max(d["outputz"][0, :            , i:i+mip, :], axis=1)
            labelx_z   = np.max(d["labelx" ][0, :            , i:i+mip, :], axis=1)
            labelz_z   = np.max(d["labelz" ][0, :            , i:i+mip, :], axis=1)
            reconst_z  = np.max(d["reconst"][0, :            , i:i+mip, :], axis=1)

            images_info = [ # [image, name, aspect]
                [image_xy  , "original_plane", 1],
                [_image_xy , "novibrate_plane" , 1],
                [imagea_xy , "aligned_plane" , 1],
                [outputx_xy, "outputx_plane" , 1],
                [outputz_xy, "outputz_plane" , 1],
                [labelx_xy , "labelx_plane"  , 1],
                [labelz_xy , "labelz_plane"  , 1],
                [reconst_xy, "reconst_plane" , 1],
                [(reconst_xy - imagea_xy + 1) / 2,
                 "heatmap_plane" , 1],
                [image_z   , "original_depth", self.params["scale"]],
                [_image_z  , "novibrate_depth", self.params["scale"]],
                [imagea_z  , "aligned_depth" , self.params["scale"]],
                [outputx_z , "outputx_depth" , 1],
                [outputz_z , "outputz_depth" , 1],
                [labelx_z  , "labelx_depth"  , 1],
                [labelz_z  , "labelz_depth"  , 1],
                [reconst_z , "reconst_depth" , self.params["scale"]],
                [(reconst_z - imagea_z + 1) / 2,
                 "heatmap_depth", self.params["scale"]],
            ]
            for (image, name, aspect) in images_info:
                #print(name,"\t", image.max(), image.min(), image.mean())
                plt.clf()
                plt.close()
                plt.axis("off")
                plt.imshow(image, cmap='gray', vmin=0.0,vmax=1.0,aspect=aspect)
                plt.savefig(path + f'/{self.model_name}_{n}_{name}.png',
                            format='png',dpi=250,bbox_inches='tight',
                            pad_inches=0)
            plt.clf()
            plt.close()

    def psf_visualize(self):
        psfpre  = self.psf_pretrain.detach().cpu().numpy()
        psfpost = self.psf_post.detach().cpu().numpy()
        #array_to_tif(f"_result_cbias/{self.model_name}_psfpre.tif", psfpre)
        #array_to_tif(f"_result_cbias/{self.model_name}_psfpost.tif", psfpost)

        plt.clf()
        plt.close()
        plt.axis("off")
        plt.imshow(psfpre[:, self.params["size_x"]//2, :], aspect=self.params["scale"])
        plt.savefig(f'./{self.configs["visualization"]["path"]}/{self.model_name}_psf_pre.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
        plt.clf()
        plt.close()
        plt.axis("off")
        plt.imshow(psfpost[:, self.params["size_x"]//2, :], aspect=self.params["scale"])
        plt.savefig(f'./{self.configs["visualization"]["path"]}/{self.model_name}_psf_post.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
    
    def del_model(self):
        self.JNet = self.JNet.cpu()
        del self.JNet

class MicrogliaInference():
    def __init__(self, model_name, is_finetuning, with_align=True):
        
        config = open(os.path.join("experiments/configs", f"{model_name}.json"))
        self.configs               = json.load(config)
        self.model_name            = model_name
        self.pre_model_name        = self.configs["pretrained_model"]
        self.params                = self.configs["params"]
        self.params["reconstruct"] = True
        self.device = self.params["device"]
        if is_finetuning:
            self.params["apply_vq"]        = True
            self.params["use_x_quantized"] = True
        val_dataset_params         = self.configs["val_dataset"]
        self.is_finetuning         = is_finetuning
        self.vibrate = Vibrate(vibration_params=self.configs["vibration"])
        if with_align:
            align_params   = self.configs["align_params"]
            deep_align_net = model.DeepAlignNet(align_params)
            self.deep_align_net = deep_align_net.to(device=self.device)
            self.deep_align_net.load_state_dict(
                torch.load(f'model/{self.configs["align_model"]}.pt'),
                strict=False)
            self.with_align = True
        else:
            self.with_align = False
        JNet = model.JNet(self.params)
        self.JNet = JNet.to(device = self.device)
        self.JNet.load_state_dict(torch.load(f'model/{self.pre_model_name}.pt'),
                                      strict=False)
        self.psf_pretrain = self.JNet.image.blur.show_psf_3d()
        if is_finetuning:
            self.JNet.load_state_dict(torch.load(f'model/{self.model_name}.pt'),
                                          strict=False)
        else:
            self.model_name = self.pre_model_name
        self.psf_post = self.JNet.image.blur.show_psf_3d()
        self.JNet.eval()

        val_dataset   = DensityDataset(
            folderpath      = val_dataset_params["folderpath"   ] ,
            size            = val_dataset_params["size"         ] , # size after segmentation
            cropsize        = val_dataset_params["cropsize"     ] ,
            I               = val_dataset_params["I"            ] ,
            scale           = val_dataset_params["scale"        ] ,
            train           = val_dataset_params["train"        ] ,
            mask            = val_dataset_params["mask"         ] ,
            mask_size       = val_dataset_params["mask_size"    ] ,
            mask_num        = val_dataset_params["mask_num"     ] ,
            surround        = val_dataset_params["surround"     ] ,
            surround_size   = val_dataset_params["surround_size"] ,
            seed            = val_dataset_params["seed"         ] ,
            )
        
        self.val_loader  = DataLoader(
            val_dataset                   ,
            batch_size  = 1               ,
            shuffle     = False           ,
            pin_memory  = False           ,
            num_workers = 0#os.cpu_count()  ,
                         )
    
    def get_result(self, num_results)->list:
        with torch.no_grad():
            results = []
            for n, val_data in enumerate(self.val_loader):
                if n >= num_results:
                    break
                image   = val_data["image"].to(device = self.device)
                image   = self.JNet.image.hill.sample(image)
                if self.with_align:
                    outdict_a  = self.deep_align_net(image)
                    imagea = outdict_a["aligned_image"]
                else:
                    imagea = image
                outdict = self.JNet(imagea)
                outputx = outdict["enhanced_image" ]
                outputz = outdict["estim_luminance"]
                reconst = outdict["reconstruction" ]
                reconst = self.JNet.image.hill.sample(reconst)
                qloss   = outdict["quantized_loss" ]
                qloss   = qloss.item() if qloss is not None else 0
                image   = image[0].detach().cpu().numpy()
                imagea  = imagea[0].detach().cpu().numpy()
                outputx = outputx[0].detach().cpu().numpy()
                outputz = outputz[0].detach().cpu().numpy()
                reconst = reconst[0].detach().cpu().numpy()
                results.append({
                    "image"  : image  ,
                    "imagea" : imagea ,
                    "outputx": outputx,
                    "outputz": outputz,
                    "reconst": reconst,
                    "qloss  ": qloss  
                                })
        return results
        
    def evaluate(self, results)->list:
        qlosses = []
        
        for d in results:
            qlosses.append(d["qloss"])

    def visualize(self, results):
        for n, d in enumerate(results):
            path = self.configs["visualization"]["path"] 
            j   = self.configs["visualization"]["z_stack"]
            j_s = j // self.params["scale"]
            i   = self.configs["visualization"]["x_slice"]
            mip = self.configs["visualization"]["mip"]
            mip_s = mip // self.params["scale"]

            image_xy   = np.max(d["image"  ][0, j_s:j_s+mip_s, :      , :], axis=0)
            imagea_xy  = np.max(d["imagea" ][0, j_s:j_s+mip_s, :      , :], axis=0)
            outputx_xy = np.max(d["outputx"][0, j  :j+mip    , :      , :], axis=0)
            outputz_xy = np.max(d["outputz"][0, j  :j+mip    , :      , :], axis=0)
            reconst_xy = np.max(d["reconst"][0, j  :j+mip    , :      , :], axis=0)
            image_z    = np.max(d["image"  ][0, :            , i:i+mip, :], axis=1)
            imagea_z   = np.max(d["imagea" ][0, :            , i:i+mip, :], axis=1)
            outputx_z  = np.max(d["outputx"][0, :            , i:i+mip, :], axis=1)
            outputz_z  = np.max(d["outputz"][0, :            , i:i+mip, :], axis=1)
            reconst_z  = np.max(d["reconst"][0, :            , i:i+mip, :], axis=1)

            images_info = [ # [image, name, aspect]
                [image_xy  , "original_plane", 1],
                [imagea_xy , "aligned_plane" , 1],
                [outputx_xy, "outputx_plane" , 1],
                [outputz_xy, "outputz_plane" , 1],
                [reconst_xy, "reconst_plane" , 1],
                [(reconst_xy - imagea_xy + 1) / 2,
                 "heatmap_plane" , 1],
                [image_z   , "original_depth", self.params["scale"]],
                [imagea_z  , "aligned_depth" , self.params["scale"]],
                [outputx_z , "outputx_depth" , 1],
                [outputz_z , "outputz_depth" , 1],
                [reconst_z , "reconst_depth" , self.params["scale"]],
                [(reconst_z - imagea_z + 1) / 2,
                 "heatmap_depth", self.params["scale"]],
            ]
            for (image, name, aspect) in images_info:
                #print(name,"\t", image.max(), image.min(), image.mean())
                plt.clf()
                plt.close()
                plt.axis("off")
                plt.imshow(image, cmap='gray', vmin=0.0,aspect=aspect)
                plt.savefig(path + f'/{self.model_name}_microglia_{n}_{name}.png',
                            format='png',dpi=250,bbox_inches='tight',
                            pad_inches=0)
            plt.clf()
            plt.close()

    def psf_visualize(self):
        psfpre  = self.psf_pretrain.detach().cpu().numpy()
        psfpost = self.psf_post.detach().cpu().numpy()
        #array_to_tif(f"_result_cbias/{self.model_name}_psfpre.tif", psfpre)
        #array_to_tif(f"_result_cbias/{self.model_name}_psfpost.tif", psfpost)

        plt.clf()
        plt.close()
        plt.axis("off")
        plt.imshow(psfpre[:, self.params["size_x"]//2, :], aspect=self.params["scale"])
        plt.savefig(f'./{self.configs["visualization"]["path"]}/{self.model_name}_psf_pre.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
        plt.clf()
        plt.close()
        plt.axis("off")
        plt.imshow(psfpost[:, self.params["size_x"]//2, :], aspect=self.params["scale"])
        plt.savefig(f'./{self.configs["visualization"]["path"]}/{self.model_name}_psf_post.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
    
    def del_model(self):
        self.JNet = self.JNet.cpu()
        del self.JNet

class BeadsInference():
    def __init__(self, model_name, cv=0, pretrain=True, threshold=-1):
        self.device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
        config = open(os.path.join("experiments/configs", f"{model_name}.json"))
        self.configs = json.load(config)
        self.params  = self.configs["params"]
        self.params["reconstruct"]     = True
        self.params["apply_vq"]        = True
        #if pretrain == False:
        #    self.params["use_x_quantized"] = True
        if threshold != -1:
            self.params["threshold"] = threshold
        JNet = model.JNet(self.params)
        self.JNet = JNet.to(device = self.device)
        self.JNet.load_state_dict(torch.load(f'model/{self.configs["pretrained_model"]}.pt'),
                                      strict=False)
        self.psf_pretrain = self.JNet.image.blur.show_psf_3d()
        self.model_name = self.configs["pretrained_model"] if pretrain else model_name
        if not pretrain:
            self.JNet.load_state_dict(
                torch.load(
                    f'model/{self.model_name}_cv_{cv}.pt',
                    map_location="cuda:0"), strict=False)

        self.psf_post = self.JNet.image.blur.show_psf_3d()
        self.JNet.eval()
        #self.JNet.tau = 0.1
    
    def get_result(self, datapath="_20231208_tsuji_beads_roi_stackreged"):
        self.images  = [os.path.join(datapath, f) for f in sorted(os.listdir(datapath))]
        self.datapath = datapath
        results  = []
        for image_name in self.images:
            if image_name[-3:] == ".pt":
                image   = torch.load(image_name,
                                 map_location=self.device).to(torch.float32)
            else:
                image = load_anything(image_name).to(self.device).to(torch.float32)
            with torch.no_grad():
                outdict = self.JNet(image.to(self.device).unsqueeze(0))
                output  = outdict["enhanced_image"]
                output  = output.squeeze(0).detach().cpu().numpy()
                qloss   = outdict["quantized_loss"]
                qloss   = qloss.item() if qloss is not None else 0
                reconst = outdict["reconstruction"]
                reconst = reconst.squeeze(0).detach().cpu().numpy()
                image   = image.detach().cpu().numpy()
                results.append([image, output, reconst, qloss])
        return results
    
    def evaluate(self, results):
        volumes = []
        mses    = []
        qlosses = []
        for [image, output, rec, qloss] in results:
            if self.params["threshold"] != -1:
                output = (output > self.params["threshold"]) * 1.0
            volume = np.sum(output).item() * \
                (self.params["res_lateral"] ** 3)
            e = 1e-7
            cov = np.mean((rec - np.mean(rec)) * (image - np.mean(image)))
            var = (np.mean((rec - np.mean(rec)) ** 2))
            beta  = (cov + e) / (var + e)
            alpha = np.mean(image) - beta * np.mean(rec)
            mse = np.mean(((image - (alpha + beta * rec)) ** 2).flatten())
            volumes.append(volume)
            mses.append(mse)
            qlosses.append(qloss)
        mean = np.mean(np.array(volumes))
        sd = np.std(np.array(volumes))
        
        return {"volume": volumes,
                "MSE"   : mses   ,
                "qloss" : qlosses,
                "mean"  : mean   ,
                "sd"    : sd     ,
                }
            
    def visualize(self, results):
        for n, [image, output, reconst, qloss] in enumerate(results):
            rec = reconst
            e = 1e-7
            cov = np.mean((rec - np.mean(rec)) * (image - np.mean(image)))
            var = (np.mean((rec - np.mean(rec)) ** 2))
            beta  = (cov + e) / (var + e)
            alpha = np.mean(image) - beta * np.mean(rec)
            reconst = alpha + beta * rec
            path = self.configs["visualization"]["path"] 
            j    = self.configs["visualization"]["z_stack"]
            i    = self.configs["visualization"]["x_slice"]
            mip  = self.configs["visualization"]["mip"]
            image_z   = np.max(image [0,  : , :, i:i+mip], axis=2)
            output_z  = np.max(output[0,  : , :, i:i+mip], axis=2)
            reconst_z = np.max(reconst[0, : , :, i:i+mip], axis=2)
            plt.axis("off")
            plt.imshow(image_z, cmap='gray', vmin=0.0, aspect=self.params["scale"])
            plt.savefig(path + f'/{self.model_name}_{self.images[n][len(self.datapath)+1:-3]}_original_depth.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
            plt.clf()
            plt.close()

            plt.axis("off")
            plt.imshow(output_z, cmap='gray', vmin=0.0, aspect=1)
            plt.savefig(path + f'/{self.model_name}_{self.images[n][len(self.datapath)+1:-3]}_output_depth.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
            plt.clf()
            plt.close()

            plt.axis("off")
            plt.imshow(reconst_z, cmap='gray', vmin=0.0, aspect=self.params["scale"])
            plt.savefig(path + f'/{self.model_name}_{self.images[n][len(self.datapath)+1:-3]}_reconst_depth.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
            plt.clf()
            plt.close()

            plt.axis("off")
            plt.imshow((reconst_z - image_z + 1) / 2,  vmin=0.0,  vmax=1.0, aspect=self.params["scale"], cmap='seismic')
            plt.savefig(path + f'/{self.model_name}_{self.images[n][len(self.datapath)+1:-3]}_heatmap_depth.png',
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
            plt.clf()
            plt.close()

    def psf_visualize(self):
        psfpre  = self.psf_pretrain.detach().cpu().numpy()
        psfpost = self.psf_post.detach().cpu().numpy()
        array_to_tif(f"_result_cbias/{self.model_name}_psfpre.tif", psfpre)
        array_to_tif(f"_result_cbias/{self.model_name}_psfpost.tif", psfpost)

        plt.clf()
        plt.close()
        plt.axis("off")
        plt.imshow(psfpre[:, self.params["size_x"]//2, :], aspect=10)
        plt.savefig(f'./{self.configs["visualization"]["path"]}/{self.model_name}_psf_pre.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
        plt.clf()
        plt.close()
        plt.axis("off")
        plt.imshow(psfpost[:, self.params["size_x"]//2, :], aspect=10)
        plt.savefig(f'./{self.configs["visualization"]["path"]}/{self.model_name}_psf_post.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)


class Inference():
    def __init__(self, model_name, is_finetuning):
        self.device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
        
        config = open(os.path.join("experiments/configs", f"{model_name}.json"))
        self.configs               = json.load(config)
        self.model_name            = model_name

        self._set_model(is_finetuning, use_trained_psf_to_gen_simulation = True) # set pretrain model or finetuning model
        self._set_align_model()
        self._set_data(val_dataset_params) # 

        self.pre_model_name        = self.configs["pretrained_model"]
        self.params                = self.configs["params"]
        self.params["reconstruct"] = True
        if is_finetuning:
            self.params["apply_vq"]    = True
            self.params["use_x_quantized"] = True
        val_dataset_params         = self.configs["pretrain_val_dataset"]
        self.is_finetuning         = is_finetuning
        self.vibrate = Vibrate(vibration_params=self.configs["vibration"])

        JNet = model.JNet(self.params)
        self.JNet = JNet.to(device = self.device)
        self.psf_pretrain = self.JNet.image.blur.show_psf_3d()
        if self.is_finetuning:
            torch.save(
                self.JNet.image.state_dict(), 
                f'model/{self.model_name}_image_tmp.pt')
        self.JNet.load_state_dict(
            torch.load(f'model/{self.pre_model_name}.pt'),
            strict=False)
        self.psf_post = self.JNet.image.blur.show_psf_3d()
        
        if is_finetuning:
            self.JNet.load_state_dict(
                torch.load(f'model/{self.model_name}.pt'),
                strict=False)
            self.JNet.image.load_state_dict(
                torch.load(f'model/{self.model_name}_image_tmp.pt'),
                strict=False)
        else:
            self.model_name = self.pre_model_name
        self.psf_post = self.JNet.image.blur.show_psf_3d()
        self.JNet.eval()

    
        
    def _set_data(self, val_dataset_params, dataset_class):
        
        val_dataset = dataset_class(
            folderpath    = val_dataset_params["folderpath"]   ,
            size          = val_dataset_params["size"]         ,
            cropsize      = val_dataset_params["cropsize"]     , 
            I             = val_dataset_params["I"]            ,
            low           = val_dataset_params["low"]          ,
            high          = val_dataset_params["high"]         ,
            scale         = val_dataset_params["scale"]        , ## scale
            mask          = val_dataset_params["mask"]         ,
            mask_size     = val_dataset_params["mask_size"]    ,
            mask_num      = val_dataset_params["mask_num"]     , #( 1% of image)
            surround      = val_dataset_params["surround"]     ,
            surround_size = val_dataset_params["surround_size"],
            seed          = val_dataset_params["seed"]         ,
                                    )
        self.val_loader  = DataLoader(
            val_dataset                   ,
            batch_size  = 1               ,
            shuffle     = False           ,
            pin_memory  = False           ,
            num_workers = 0#os.cpu_count()  ,
                         )
    
    def get_result(self, num_results)->list:
        with torch.no_grad():
            results = []
            for n, val_data in enumerate(self.val_loader):
                if n >= num_results:
                    break
                if self.is_finetuning:
                    self.JNet.image.load_state_dict(
                        torch.load(
                            f"model/{self.pre_model_name}.pt"), 
                            strict=False)
                labelx = val_data["labelx"].to(device = self.device)
                labelz = val_data["labelz"].to(device = self.device)
                image = imagen_instantblur(model  = self.JNet  ,
                                           label  = labelz     ,
                                           device = self.device, 
                                           params = self.params,)
                image  = self.JNet.image.hill.sample(image)
                if self.is_finetuning:            
                    self.JNet.image.load_state_dict(
                        torch.load(
                            f"model/{self.model_name}.pt"), 
                            strict=False)
                if self.configs["pretrain_loop"]["is_vibrate"]:
                    image   = self.vibrate(image).detach().clone()
                image    = self.JNet.image.hill.sample(image)
                outdict  = self.JNet(image)
                outputx  = outdict["enhanced_image"]
                outputz  = outdict["estim_luminance"]
                reconst  = outdict["reconstruction"]
                qloss    = outdict["quantized_loss"]
                qloss    = qloss.item() if qloss is not None else 0
                image    = image[0].detach().cpu().numpy()
                labelx   = labelx[0].detach().cpu().numpy()
                labelz   = labelz[0].detach().cpu().numpy()
                outputx  = outputx[0].detach().cpu().numpy()
                outputz  = outputz[0].detach().cpu().numpy()
                reconst  = reconst[0].detach().cpu().numpy()
                results.append([image, outputx, outputz, reconst,
                                labelx, labelz, qloss])
        return results
        
    def evaluate(self, results)->list:
        msesx = []
        bcesx = []
        msesz = []
        bcesz = []
        qlosses = []
        for n, [image, outputx, outputz, reconst,
                labelx, labelz, qloss] in enumerate(results):
            msex = np.mean(((labelx - outputx) ** 2).flatten())
            msez = np.mean(((labelz - outputz) ** 2).flatten())
            bcex = np.mean(-(labelx*np.log(outputx) + (1. - labelx)*np.log(1. - outputx)).flatten())
            bcez = np.mean(-(labelz*np.log(outputz) + (1. - labelz)*np.log(1. - outputz)).flatten())
            msesx.append(msex)
            msesz.append(msez)
            bcesx.append(bcex)
            bcesz.append(bcez)
            qlosses.append(qloss)
        return {"MSEx" : msesx  ,
                "BCEx" : bcesx  ,
                "MSEz" : msesz  ,
                "BCEz" : bcesz  ,
                "qloss": qlosses,}
        
    def visualize(self, results):
        for n, [image, outputx, outputz, reconst,
                labelx, labelz, qloss] in enumerate(results):
            path = self.configs["visualization"]["path"] 
            j   = self.configs["visualization"]["z_stack"]
            j_s = j // self.params["scale"]
            i   = self.configs["visualization"]["x_slice"]
            mip = self.configs["visualization"]["mip"]
            mip_s = mip // self.params["scale"]

            image_xy   = np.max(image  [0, j_s:j_s+mip_s, :      , :], axis=0)
            outputx_xy = np.max(outputx[0, j  :j+mip    , :      , :], axis=0)
            outputz_xy = np.max(outputz[0, j  :j+mip    , :      , :], axis=0)
            labelx_xy  = np.max(labelx [0, j  :j+mip    , :      , :], axis=0)
            labelz_xy  = np.max(labelz [0, j  :j+mip    , :      , :], axis=0)
            reconst_xy = np.max(reconst[0, j  :j+mip    , :      , :], axis=0)
            image_z    = np.max(image  [0, :            , i:i+mip, :], axis=1)
            outputx_z  = np.max(outputx[0, :            , i:i+mip, :], axis=1)
            outputz_z  = np.max(outputz[0, :            , i:i+mip, :], axis=1)
            labelx_z   = np.max(labelx [0, :            , i:i+mip, :], axis=1)
            labelz_z   = np.max(labelz [0, :            , i:i+mip, :], axis=1)
            reconst_z  = np.max(reconst[0, :            , i:i+mip, :], axis=1)

            images_info = [ # [image, name, aspect]
                [image_xy  , "original_plane", 1],
                [outputx_xy, "outputx_plane" , 1],
                [outputz_xy, "outputz_plane" , 1],
                [labelx_xy , "labelx_plane" , 1],
                [labelz_xy , "labelz_plane" , 1],
                [reconst_xy, "reconst_plane" , 1],
                [(reconst_xy - image_xy + 1) / 2,
                 "heatmap_plane" , 1],
                [image_z   , "original_depth", self.params["scale"]],
                [outputx_z , "outputx_depth" , 1],
                [outputz_z , "outputz_depth" , 1],
                [labelx_z  , "labelx_depth"  , 1],
                [labelz_z  , "labelz_depth"  , 1],
                [reconst_z , "reconst_depth" , self.params["scale"]],
                [(reconst_z - image_z + 1) / 2,
                 "heatmap_depth", self.params["scale"]],
            ]
            for (image, name, aspect) in images_info:
                #print(name,"\t", image.max(), image.min(), image.mean())
                plt.clf()
                plt.close()
                plt.axis("off")
                plt.imshow(image, cmap='gray', vmin=0.0,aspect=aspect)
                plt.savefig(path + f'/{self.model_name}_{n}_{name}.png',
                            format='png',dpi=250,bbox_inches='tight',
                            pad_inches=0)
            plt.clf()
            plt.close()

    def visualize_oldversion(self, results, path='result', verbose=False,
                             threshold=-1):
        j   = self.configs["visualization"]["z_stack"]
        j_s = j // self.params["scale"]
        i   = self.configs["visualization"]["x_slice"]
        for n, [image, output, label, qloss] in enumerate(results):
            if threshold != -1:
                output = output >= threshold
            fig = plt.figure(figsize=(25, 15))
            ax1 = fig.add_subplot(231)
            ax2 = fig.add_subplot(232)
            ax3 = fig.add_subplot(233)
            ax4 = fig.add_subplot(234)
            ax5 = fig.add_subplot(235)
            ax6 = fig.add_subplot(236)
            ax1.set_axis_off()
            ax2.set_axis_off()
            ax3.set_axis_off()
            ax4.set_axis_off()
            ax5.set_axis_off()
            ax6.set_axis_off()
            plt.subplots_adjust(hspace=-0.0)
            ax1.imshow(image[0, j_s, :, :],
                cmap='gray', vmin=0.0, aspect=1)
            ax2.imshow(output[0, j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
            ax3.imshow(label[0, j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
            ax4.imshow(image[0, :, i, :],
                cmap='gray', vmin=0.0, aspect= self.params["scale"])
            ax5.imshow(output[0, :, i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
            ax6.imshow(label[0, :, i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
            plt.savefig(path + f'/{self.model_name}_result_{n}.png',
                format='png', dpi=250)
            if verbose:
                plt.show()

    def psf_visualize(self):
        psfpre  = self.psf_pretrain.detach().cpu().numpy()
        psfpost = self.psf_post.detach().cpu().numpy()

        plt.clf()
        plt.close()
        plt.axis("off")
        plt.imshow(psfpre[:, self.params["size_x"]//2, :], aspect=self.params["scale"])
        plt.savefig(f'./{self.configs["visualization"]["path"]}/{self.model_name}_psf_pre.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
        plt.clf()
        plt.close()
        plt.axis("off")
        plt.imshow(psfpost[:, self.params["size_x"]//2, :], aspect=self.params["scale"])
        plt.savefig(f'./{self.configs["visualization"]["path"]}/{self.model_name}_psf_post.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
    
    def del_model(self):
        self.JNet = self.JNet.cpu()
        del self.JNet

