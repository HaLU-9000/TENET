import os
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import timm.scheduler

import model as model
from dataset import RandomCutDataset
from train_loop import pretrain_loop, deep_align_net_train_loop

parser = argparse.ArgumentParser(description='Pretraining model.')
parser.add_argument('model_name')
parser.add_argument('--train_align', action="store_true")
args = parser.parse_args()

configs = open(os.path.join("experiments/configs", f"{args.model_name}.json"))
configs              = json.load(configs)
params               = configs["params"              ]
train_dataset_params = configs["pretrain_dataset"    ]
val_dataset_params   = configs["pretrain_val_dataset"]
train_loop_params    = configs["pretrain_loop"       ]
vibration_params     = configs["vibration"           ]

device = params["device"]
print(f"Training on device {device}.")
JNet = model.JNet(params)
JNet = JNet.to(device = device)

if args.train_align:
    align_params   = configs["align_params"]
    deep_align_net = model.DeepAlignNet(align_params)
    deep_align_net = deep_align_net.to(device=device)
    lr             = align_params["lr"]
    train_params   = deep_align_net.parameters()
    optimizer_a    = optim.Adam(train_params, lr = align_params["lr"])
    scheduler      = timm.scheduler.PlateauLRScheduler(
        optimizer      = optimizer_a ,
        patience_t     = 20          ,
        warmup_lr_init = lr * 0.1    ,
        warmup_t       = 10          ,
        )
    JNet.load_state_dict(torch.load(f'model/{configs["pretrained_model"]}.pt'),
                         strict=False)
else:
    train_params = JNet.parameters()
    lr = train_loop_params['lr']
    optimizer = optim.Adam(train_params, lr = lr)
    scheduler = timm.scheduler.PlateauLRScheduler(
        optimizer      = optimizer   ,
        patience_t     = 20          ,
        warmup_lr_init = lr * 0.1    ,
        warmup_t       = 10          ,
        )
    

train_dataset = RandomCutDataset(
    folderpath    = train_dataset_params["folderpath"]   ,
    size          = train_dataset_params["size"]         ,
    cropsize      = train_dataset_params["cropsize"]     , 
    I             = train_dataset_params["I"]            ,
    low           = train_dataset_params["low"]          ,
    high          = train_dataset_params["high"]         ,
    scale         = train_dataset_params["scale"]        ,  ## scale
    train         = True                                 ,
    mask          = train_dataset_params["mask"]         ,
    mask_size     = train_dataset_params["mask_size"]    ,
    mask_num      = train_dataset_params["mask_num"]     ,  #( 1% of image)
    surround      = train_dataset_params["surround"]     ,
    surround_size = train_dataset_params["surround_size"],
    )

val_dataset   = RandomCutDataset(
    folderpath    = val_dataset_params["folderpath"]   ,
    size          = val_dataset_params["size"]         ,
    cropsize      = val_dataset_params["cropsize"]     , 
    I             = val_dataset_params["I"]            ,
    low           = val_dataset_params["low"]          ,
    high          = val_dataset_params["high"]         ,
    scale         = val_dataset_params["scale"]        ,  ## scale
    train         = False                              ,
    mask          = val_dataset_params["mask"]         ,
    mask_size     = val_dataset_params["mask_size"]    ,
    mask_num      = val_dataset_params["mask_num"]     ,  #( 1% of image)
    surround      = val_dataset_params["surround"]     ,
    surround_size = val_dataset_params["surround_size"],
    seed          = val_dataset_params["seed"]         ,
    )       

train_data  = DataLoader(
    train_dataset                                ,
    batch_size  = train_loop_params["batch_size"],
    shuffle     = True                           ,
    pin_memory  = True                           ,
    num_workers = 0#os.cpu_count()                 ,
    )

val_data    = DataLoader(
    val_dataset                                  ,
    batch_size  = train_loop_params["batch_size"],
    shuffle     = False                          ,
    pin_memory  = True                           ,
    num_workers = 0#os.cpu_count()                 ,
    )

model_path = 'model'


JNet.image.blur.neuripsf.trainer()
print(f'========= model {configs["pretrained_model"]} train started =========')
pretrain_loop(
    optimizer            = optimizer                   ,
    model                = JNet                        ,
    train_loader         = train_data                  ,
    val_loader           = val_data                    ,
    model_name           = configs["pretrained_model"] ,
    params               = params                      ,
    train_loop_params    = train_loop_params           ,
    train_dataset_params = train_dataset_params        ,
    vibration_params     = vibration_params            ,
    scheduler            = scheduler                   ,
    without_noise        = train_loop_params["without_noise"]
    )