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
import tifffile

parser = argparse.ArgumentParser(description='generates report')
parser.add_argument('model_name')
args   = parser.parse_args() 
configs = open(os.path.join("experiments/configs", f"{args.model_name}.json"))
configs = json.load(configs)

###################################
## Finetuning Results with Beads ##
###################################
volumes = np.zeros(30)
os.makedirs(f"_results_for_paper/fig4/tenet/{args.model_name}/", exist_ok=True)
for i in range(0, 10):
    btype_list = ["original", "output", "reconst", "heatmap"]
    binfer = inference.BeadsInference(
        args.model_name, cv=i, pretrain=False, threshold=0.5)
    results = binfer.get_result(
        datapath=f"_20231208_tsuji_beads_roi_stackreged_cv_wise/{i}")
    bevals  = binfer.evaluate(results)
    binfer.visualize(results)
    #print(f'finetuning: volume mean: {bevals["mean"]}, volume sd: {bevals["sd"]}')
    namelist = os.listdir(f"_20231208_tsuji_beads_roi_stackreged_cv_wise/{i}")
    for n in range(len(results)):
        tifffile.imwrite(f"_results_for_paper/fig4/tenet/{args.model_name}/"+namelist[n], results[n][1])
        print(f'{i * 3 + n} volume: {bevals["volume"][n]}')
        volumes[i * 3 + n] = bevals["volume"][n]
mu = volumes.mean()
s2 = (volumes.var(ddof=1))
print("mu: ",  mu)
print("s^2: ", s2)
print("95%" "ci:", f"[{mu - 2.045*((s2/30)**0.5)},{mu + 2.045*((s2/30)**0.5)}]")