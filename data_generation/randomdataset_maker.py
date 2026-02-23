import os
import argparse
import json

import numpy as np
import time
import torch
from makedata import make_realistic_data, make_thick_realistic_data
torch.manual_seed(415)
np.random.seed(415)

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

parser = argparse.ArgumentParser(description='Pretraining model.')
parser.add_argument('model_name')
args   = parser.parse_args()

configs  = open(os.path.join("experiments/configs",f"{args.model_name}.json"))
configs  = json.load(configs)
param = configs["simulation_data_generation"]
dataset_name = param["dataset_name"]
os.makedirs(dataset_name, exist_ok=True)
train_object_diff = int((param["train_object_num_max"] - param["train_object_num_min"]) / param["train_num"])
valid_object_diff = int((param["valid_object_num_max"] - param["valid_object_num_min"]) / param["valid_num"])

for i in range(0, param["train_num"]):
    t1 = time.time()
    num = param["train_object_num_max"] - i * train_object_diff
    data    = make_thick_realistic_data(num, param["image_size"], p=0.05)
    data_x = data["data_x"]
    data_z = data["data_z"]
    t2 = time.time()
    print(f'{t2 - t1} s')
    np.save(f'{dataset_name}/{str(i).zfill(4)}_labelx.npy',data_x)
    np.save(f'{dataset_name}/{str(i).zfill(4)}_labelz.npy',data_z)


# testdata
for i in range(0, param["valid_num"]):
    t1 = time.time()
    num = param["valid_object_num_max"] - i * valid_object_diff
    inp     = make_thick_realistic_data(num, param["image_size"], p=0.05)
    data_x = inp["data_x"]
    data_z = inp["data_z"]
    t2 = time.time()
    print(f'{t2 - t1} s')
    np.save(f'{dataset_name}/{str(i + param["train_num"]).zfill(4)}_labelx.npy',
            data_x)
    np.save(f'{dataset_name}/{str(i + param["train_num"]).zfill(4)}_labelz.npy', 
            data_z)