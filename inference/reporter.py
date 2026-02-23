import os
import codecs
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from mdutils.mdutils import MdUtils
import pandas as pd
import model as model
import inference

parser = argparse.ArgumentParser(description='generates report')
parser.add_argument('model_name')
parser.add_argument('-filename', default=None)
parser.add_argument('-show', action='append', default=None)
args   = parser.parse_args() 
configs = open(os.path.join("experiments/configs", f"{args.model_name}.json"))
configs = json.load(configs)
if args.filename is not None:
    md = MdUtils(file_name=f'./experiments/reports/{args.filename}.md')
else:
    md = MdUtils(file_name=f'./experiments/reports/{args.model_name}.md')
###########
## Title ##
###########
md.new_header(level=1, title=f"{args.model_name} Report")
md.new_line(configs["explanation"])
md.new_line(f'pretrained model : {configs["pretrained_model"]}')
################
## Parameters ##
################
md.new_header(level=2, title="Model Parameters")
md.new_line()
params_list = ["Parameter", "Value", "Comment"]
n = 0
for param in configs["params"]:
    if "$" not in param:
        if ("$"+param) in configs["params"]:
            comment = configs["params"]["$"+param]
        else:
            comment = ""
        params_list.extend([param, configs["params"][param], comment])
        n += 1
md.new_table(columns=3, rows=n+1, text=params_list, text_align="left")
##############
## Datasets ##
##############
md.new_header(level=2, title="Datasets and other training details")
for name in [
                "simulation_data_generation",
                "pretrain_dataset"          ,
                "pretrain_val_dataset"      ,
                "train_dataset"             , 
                "val_dataset"               , 
                "pretrain_loop"             ,
                "train_loop"                ,
              ]:
    default_list = ["Parameter", "Value"]
    for n, param in enumerate(configs[name]):
        default_list.extend([param, configs[name][param]])
    md.new_header(level=3, title=name)
    md.new_table(columns=2, rows=len(default_list)//2, text=default_list, text_align="left")
#####################
## Training Curves ##
#####################
if os.path.isfile(f'experiments/traincurves/{args.model_name}.csv'):
    md.new_header(level=2, title="Training Curves")
    md.new_line()
    md.new_header(level=3, title="Pretraining")
    df = pd.read_csv(f'experiments/traincurves/{configs["pretrained_model"]}.csv')
    df.plot()
    plt.xlabel("epoch")
    #plt.ylabel(configs["pretrain_loop"]["loss_fnx"] + " + " + configs["pretrain_loop"]["loss_fnx"])
    path = f'./experiments/tmp/{configs["pretrained_model"]}_train.png'
    plt.savefig(path)
    md.new_line(md.new_reference_image(text="pretrained_model", path=path[1:]))
    md.new_header(level=3, title="Finetuning")
    df = pd.read_csv(f'experiments/traincurves/{args.model_name}.csv')
    df.plot()
    plt.xlabel("epoch")
    #loss_metrics = configs["train_loop"]["loss_fn"]+" + "\
    #    +"qloss "+"* "+str(configs["train_loop"]["qloss_weight"])
    #if configs["train_loop"]["ewc"] is not None:
    #    loss_metrics += "ewc"
    #plt.ylabel(loss_metrics)
    path = f'./experiments/tmp/{args.model_name}_train.png'
    plt.savefig(path)
    md.new_line(md.new_reference_image(text="finetuned", path=path[1:]))
    plt.clf()
    plt.close()
#############
## Results ##
#############
md.new_header(level=2, title="Results")
#########################
## Pretraining Results ##
#########################
if "pretrain" in args.show:
    md.new_header(level=3, title="Pretraining")
    num_result = 5
    infer = inference.SimulationInference(
        args.model_name,
        is_finetuning = False,
        is_vibrate    = configs["pretrain_loop"]["is_vibrate"],
        with_align    = False                   ,
        )
    results = infer.get_result(num_result)
    evals = infer.evaluate(results)
    md.new_line(f'Segmentation: mean MSE: {np.mean(evals["MSEx"])}, mean BCE: {np.mean(evals["BCEx"])}')
    md.new_line(f'Luminance Estimation: mean MSE: {np.mean(evals["MSEz"])}, mean BCE: {np.mean(evals["BCEz"])}')
    infer.visualize(results)
    slice_list = ["plane", "depth"]
    type_list  = ["original","novibrate","aligned", "outputx", "labelx", "outputz", "labelz"]
    for n in range(num_result):
        md.new_header(level=3, title=f"{n}")
        for slice in slice_list:
            im_list = []
            for tp in type_list:
                path = f'./{configs["visualization"]["path"]}/{infer.pre_model_name}_{n}_{tp}_{slice}.png'
                im_list.append(md.new_reference_image(text=f"{infer.pre_model_name}_{n}_{tp}_{slice}", path=path[1:]))
            md.new_table(columns=len(type_list), rows=2, text=[*type_list, *im_list],)
            md.new_line(f'MSEx: {evals["MSEx"][n]}, BCEx: {evals["BCEx"][n]}')
            md.new_line(f'MSEz: {evals["MSEz"][n]}, BCEz: {evals["BCEz"][n]}')
            md.new_line()
    #infer.psf_visualize()
    #psf_list = []
    #timing_list = ["pre", "post"]
    #for t in timing_list:
    #    path = f'./{configs["visualization"]["path"]}/{infer.model_name}_psf_{t}.png'
    #    psf_list.append(md.new_reference_image(text=f"{infer.model_name}_psf_{t}", path=path[1:]))
    #md.new_table(columns=len(timing_list), rows=2, text=[*timing_list, *psf_list])
    infer.del_model()
    torch.cuda.empty_cache()

########################################
## Finetuning Results with Simulation ##
########################################
if "simulation" in args.show:
    md.new_header(level=3, title="Finetuning Results with Simulation")
    type_list = ["original","novibrate","aligned", "reconst", "heatmap", "outputx", "labelx", "outputz", "labelz"]
    infer = inference.SimulationInference(args.model_name, 
                                        is_finetuning=True,
                                        is_vibrate = True,
                                        with_align=False,)
    results = infer.get_result(num_result)
    evals  = infer.evaluate(results)
    infer.visualize(results)
    for n in range(len(results)):
        md.new_header(level=3, title=f"image {n}")
        for slice in slice_list:
            im_list = []
            for tp in type_list:
                path = f'./{configs["visualization"]["path"]}/{infer.model_name}_{n}_{tp}_{slice}.png'
                im_list.append(md.new_reference_image(text=f"{infer.model_name}_{n}_{tp}_{slice}", path=path[1:]))
            md.new_table(columns=len(type_list), rows=2, text=[*type_list, *im_list],)
            md.new_line(f'MSEz: {evals["MSEz"][n]}, quantized loss: {evals["qloss"][n]}')
            md.new_line()
    md.new_line("If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.")
    infer.psf_visualize()
    psf_list = []
    timing_list = ["pre", "post"]
    for t in timing_list:
        path = f'./{configs["visualization"]["path"]}/{infer.model_name}_psf_{t}.png'
        psf_list.append(md.new_reference_image(text=f"{infer.model_name}_psf_{t}", path=path[1:]))
    md.new_table(columns=2, rows=2, text=[*timing_list, *psf_list])
#######################################
## Finetuning Results with Microglia ##
#######################################
if "microglia" in args.show:
    md.new_header(level=3, title="Finetuning Results with Microglia")
    btype_list = ["original", "aligned", "outputx", "outputz", "reconst", "heatmap"]
    slice_list = ["plane", "depth"]
    for is_finetuning in [False, True]:
        md.new_header(level=4, title=f"finetuning == {is_finetuning}")
        binfer = inference.MicrogliaInference(args.model_name,
                                            is_finetuning=is_finetuning,
                                            with_align=False)
        results = binfer.get_result(5)
        binfer.visualize(results)
        for n in range(5):
            md.new_header(level=3, title=f"image {n}")
            for slice in slice_list:
                im_list = []
                for tp in btype_list:
                    path = f'./{configs["visualization"]["path"]}/{binfer.model_name}_microglia_{n}_{tp}_{slice}.png'
                    im_list.append(md.new_reference_image(text=f"{binfer.model_name}_microglia_{n}_{tp}_{slice}", path=path[1:]))
                md.new_table(columns=len(btype_list), rows=len(slice_list), text=[*btype_list, *im_list],)
                md.new_line()
    md.new_line("If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.")

###################################
## Finetuning Results with Beads ##
###################################
if "beads" in args.show:
    btype_list = ["original", "output", "reconst", "heatmap"]
    pretrain = True
    md.new_header(level=3, title="pretrain")    
    binfer = inference.BeadsInference(args.model_name, pretrain=pretrain, threshold=0.5)
    results = binfer.get_result()
    bevals  = binfer.evaluate(results)
    binfer.visualize(results)
    print(f'pretrain: volume mean: {bevals["mean"]}, volume sd: {bevals["sd"]}')
    md.new_line(f'volume mean: {bevals["mean"]}, volume sd: {bevals["sd"]}')
    for n in range(len(results)):
        image_name = binfer.images[n][len(binfer.datapath)+1:-3]
        md.new_header(level=3, title=image_name)
        im_list = []
        for tp in btype_list:
            path = f'./{configs["visualization"]["path"]}/{binfer.model_name}_{image_name}_{tp}_depth.png'
            im_list.append(md.new_reference_image(text=f"{binfer.model_name}_{image_name}_{tp}_{slice}", path=path[1:]))
        md.new_table(columns=4, rows=2, text=[*btype_list, *im_list],)
        md.new_line(f'volume: {bevals["volume"][n]}, MSE: {bevals["MSE"][n]}, quantized loss: {bevals["qloss"][n]}')
        md.new_line()
    md.new_line("If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.")

    pretrain = False
    md.new_header(level=3, title="finetuning")
    volumes = np.zeros(30)
    for i in range(0, 10):
        btype_list = ["original", "output", "reconst", "heatmap"]
        binfer = inference.BeadsInference(
            args.model_name, cv=i, pretrain=False, threshold=0.5)
        results = binfer.get_result(
            datapath=f"_20231208_tsuji_beads_roi_stackreged_cv_wise/{i}")
        bevals  = binfer.evaluate(results)
        binfer.visualize(results)    
        for n in range(len(results)):
            print(f'{i * 3 + n} volume: {bevals["volume"][n]}')
            volumes[i * 3 + n] = bevals["volume"][n]
            image_name = binfer.images[n][len(binfer.datapath)+1:-3]
            md.new_header(level=3, title=image_name)
            im_list = []
            for tp in btype_list:
                path = f'./{configs["visualization"]["path"]}/{binfer.model_name}_{image_name}_{tp}_depth.png'
                im_list.append(md.new_reference_image(text=f"{binfer.model_name}_{image_name}_{tp}_{slice}", path=path[1:]))
            md.new_table(columns=4, rows=2, text=[*btype_list, *im_list],)
            md.new_line(f'volume: {bevals["volume"][n]}, MSE: {bevals["MSE"][n]}, quantized loss: {bevals["qloss"][n]}')
            md.new_line()
        md.new_line(f'volume mean: {volumes.mean()}, volume sd: {volumes.std()}')
    md.new_line("If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.")

    binfer.psf_visualize()
    psf_list = []
    timing_list = ["pre", "post"]
    for t in timing_list:
        path = f'./{configs["visualization"]["path"]}/{binfer.model_name}_psf_{t}.png'
        psf_list.append(md.new_reference_image(text=f"{binfer.model_name}_psf_{t}", path=path[1:]))
    md.new_table(columns=2, rows=2, text=[*timing_list, *psf_list])

##################
## Architecture ##
##################
print(model.JNet(configs["params"]), file = codecs.open("experiments/tmp/"+args.model_name+".txt", "w", "utf-8"))
md.new_header(level=2, title="Architecture")
md.new_line()
md.new_line("```")
with open("experiments/tmp/"+args.model_name+".txt") as f:
    lines = f.readlines()
    for line in lines:
        md.new_line(line.rstrip("\n"))
    f.close()
md.new_line("```")
md.new_line()

#########
## End ##
#########
md.new_line()
md.create_md_file()