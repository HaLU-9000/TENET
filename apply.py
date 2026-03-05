import os
import argparse
import json

import utils
import sys
import os

# get args
parser = argparse.ArgumentParser(description='')
parser.add_argument('model_name')
parser.add_argument('-image_name')
parser.add_argument('-org_folder')
parser.add_argument('-keyword'   )
parser.add_argument('-dna')
parser.add_argument("--pretrain", action="store_true")
parser.add_argument('-t', '--train_mode')
parser.add_argument('-s','--shape',
                    default=[20, 112, 112], nargs="*", type=float) 
parser.add_argument('-o','--overlap',
                    default=[0, 0, 0], nargs="*", type=float)
parser.add_argument('-c', '--omit_margin',
                    default=[0, 0, 0], nargs="*", type=float)

args = parser.parse_args()
configs = open(os.path.join("experiments/configs", f"{args.model_name}.json"))
configs = json.load(configs)
params  = configs["params"]
shape        = [int(n) for n in args.shape      ]
overlap      = [int(n) for n in args.overlap    ]
omit_margin  = [int(n) for n in args.omit_margin]

if args.image_name is not None:
    images = [args.image_name]
else:
    items = os.listdir(args.org_folder)
    images = []
    for item in items:
        if args.keyword in item:
            images.append(os.path.join(args.org_folder,item))
    images.sort()

# model load
if args.pretrain:
    model = utils.init_model(params, is_finetuning = False)
    utils.load_model_weight(model, model_name = configs["pretrained_model"])
else:
    model = utils.init_model(params, is_finetuning = True)
    utils.load_model_weight(model, model_name = args.model_name)

utils.mount_model_to_device(model, configs = configs)
model.eval()
for image in images:
    print("processing ",image)
    image_basename = utils.get_basename(image)
    image_org  = utils.load_anything(image)
    image      = utils.ImageProcessing(image_org)
    image.deconv_model = model

    image.process_image(
        model       = model            ,
        params      = params           ,
        chunk_shape = shape            ,
        type        = "enhanced_image" ,
        overlap     = overlap          ,
        margin      = omit_margin      ,
        apply_hill  = True
                        )
    if args.pretrain:
        os.makedirs(f"_apply_{configs['pretrained_model']}", exist_ok=True)
        image.save_processed_image(
            file   = f"_apply_{configs['pretrained_model']}/{image_basename}",
            format = "tif",
            bit    = 8)
    else:
        os.makedirs(f"_apply_{args.model_name}", exist_ok=True)
        image.save_processed_image(
            file   = f"_apply_{args.model_name}/{image_basename}",
            format = "tif",
            bit    = 8)
# example usage:
# python3 apply.py  /home/haruhiko/Downloads/Set_03/MDA15_20230915.nd2 JNet_510
# python3 apply.py  _wakelabdata_processed/1_Spine_structure_AD_175-11w-D3-xyz6-020C2-T1.tif JNet_510
# python3 apply.py /home/morita/home/Downloads/Set_03_processed/MDA34_20231110_C_0_no_satu.tif JNet_544