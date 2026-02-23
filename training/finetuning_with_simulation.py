import os
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import timm.scheduler

import model as model
from dataset import RandomCutDataset
from train_loop import finetuning_with_simulation_loop, ElasticWeightConsolidation

parser = argparse.ArgumentParser(description='Pretraining model.')
parser.add_argument('model_name')
parser.add_argument('--train_with_align', action="store_true")
args   = parser.parse_args()

configs = open(os.path.join("experiments/configs",f"{args.model_name}.json"))
configs              = json.load(configs)
params               = configs["params"]
train_dataset_params = configs["pretrain_dataset"]
ewc_dataset_params   = configs["pretrain_dataset"]
val_dataset_params   = configs["pretrain_val_dataset"]
train_loop_params    = configs["train_loop"]
without_noise        = configs["pretrain_loop"]["without_noise"]
vibration_params     = configs["vibration"        ]
device = params["device"]

train_dataset = RandomCutDataset(
    folderpath      = train_dataset_params["folderpath"]    ,
    size            = train_dataset_params["size"]          , # size after segmentation
    cropsize        = train_dataset_params["cropsize"]      , # size after segmentation
    I               = train_dataset_params["I"]             ,
    low             = train_dataset_params["low"]           ,
    high            = train_dataset_params["high"]          ,
    scale           = train_dataset_params["scale"]         , ## scale
    train           = True                                  ,
    mask            = train_dataset_params["mask"]          ,
    mask_num        = train_dataset_params["mask_num"]      ,
    mask_size       = train_dataset_params["mask_size"]     ,
    surround        = train_dataset_params["surround"]      ,
    surround_size   = train_dataset_params["surround_size"] ,
    )

val_dataset   = RandomCutDataset(
    folderpath      = val_dataset_params["folderpath"]      ,
    size            = val_dataset_params["size"]            , # size after segmentation
    cropsize        = val_dataset_params["cropsize"]        ,
    I               = val_dataset_params["I"]               ,
    low             = train_dataset_params["low"]           ,
    high            = train_dataset_params["high"]          ,
    scale           = val_dataset_params["scale"]           ,
    train           = False                                 ,
    mask            = val_dataset_params["mask"]            ,
    mask_size       = val_dataset_params["mask_size"]       ,
    mask_num        = val_dataset_params["mask_num"]        ,
    surround        = val_dataset_params["surround"]        ,
    surround_size   = val_dataset_params["surround_size"]   ,
    seed            = val_dataset_params["seed"]            ,
    )

train_data  = DataLoader(
    train_dataset                                 ,
    batch_size  = train_loop_params["batch_size"] ,
    shuffle     = True                            ,
    pin_memory  = True                            ,
    num_workers = 0#os.cpu_count()                  ,
    )

val_data    = DataLoader(
    val_dataset                                   ,
    batch_size  = train_loop_params["batch_size"] ,
    shuffle     = False                           ,
    pin_memory  = True                            ,
    num_workers = 0#os.cpu_count()                  ,
    )

params["reconstruct"]     = True
params["apply_vq"]        = True
params["use_x_quantized"] = True

JNet = model.JNet(params)
JNet = JNet.to(device = device)

JNet.load_state_dict(torch.load(f'model/{configs["pretrained_model"]}.pt'),
                     strict=False)

train_params = JNet.parameters()

lr = train_loop_params["lr"]

optimizer = optim.Adam(filter(lambda p: p.requires_grad, JNet.parameters()),
                       lr = lr)
scheduler = timm.scheduler.PlateauLRScheduler(
    optimizer      = optimizer   ,
    patience_t     = 10          ,
    warmup_lr_init = lr * 0.1    ,
    warmup_t       = 10          ,)

ewc_dataset   = RandomCutDataset(
    folderpath    = ewc_dataset_params["folderpath"]   ,
    size          = ewc_dataset_params["size"]         ,
    cropsize      = ewc_dataset_params["cropsize"]     , 
    I             = ewc_dataset_params["I"]            ,
    low           = ewc_dataset_params["low"]          ,
    high          = ewc_dataset_params["high"]         ,
    scale         = ewc_dataset_params["scale"]        ,  ## scale
    mask          = ewc_dataset_params["mask"]         ,
    mask_size     = ewc_dataset_params["mask_size"]    ,
    mask_num      = ewc_dataset_params["mask_num"]     ,  #( 1% of image)
    surround      = ewc_dataset_params["surround"]     ,
    surround_size = ewc_dataset_params["surround_size"],
    )

ewc_data    = DataLoader(
    ewc_dataset                   ,
    batch_size  = 1               ,
    shuffle     = True            ,
    pin_memory  = True            ,
    num_workers = 0#os.cpu_count()  ,
    )

if  train_loop_params["ewc"] != None:
    ewc = ElasticWeightConsolidation(
        model              = JNet                                       ,
        params             = params                                     ,
        vibration_params   = vibration_params                           ,
        prev_dataloader    = ewc_data                                   ,
        loss_fnx           = eval(configs["pretrain_loop"]["loss_fnx"]) ,
        loss_fnz           = eval(configs["pretrain_loop"]["loss_fnz"]) ,
        wx                 = configs["pretrain_loop"]["weight_x"]       ,
        wz                 = configs["pretrain_loop"]["weight_z"]       ,
        ewc_dataset_params = ewc_dataset_params                         ,
        init_num_batch     = 100                                        ,
        is_vibrate         = True                                       ,
        device             = device                                     ,
        without_noise      = without_noise
        )
else:
    ewc = None


print(f"============= model {args.model_name} train started =============")
if args.train_with_align:
    finetuning_with_simulation_loop(
        optimizer            = optimizer                            ,
        model                = JNet                                 ,
        train_loader         = train_data                           ,
        val_loader           = val_data                             ,
        device               = device                               ,
        model_name           = args.model_name                      ,
        params               = params                               ,
        ewc                  = ewc                                  ,
        train_dataset_params = train_dataset_params                 ,
        train_loop_params    = train_loop_params                    ,
        vibration_params     = vibration_params                     ,
        scheduler            = scheduler                            ,
    )
else:
    finetuning_with_simulation_loop(
        optimizer            = optimizer                            ,
        model                = JNet                                 ,
        train_loader         = train_data                           ,
        val_loader           = val_data                             ,
        device               = device                               ,
        model_name           = args.model_name                      ,
        params               = params                               ,
        ewc                  = ewc                                  ,
        train_dataset_params = train_dataset_params                 ,
        train_loop_params    = train_loop_params                    ,
        vibration_params     = vibration_params                     ,
        scheduler            = scheduler                            ,
    )

