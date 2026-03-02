from pathlib import Path
import numpy as np
import torch
from utils import save_label

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Building data on device {device}.")
folderpath    = '_var_num_realisticdataset'
outfolderpath = '_var_num_realisticdata'
labelname     = '_label'
outlabelname  = '_label'
save_label(folderpath, outfolderpath, labelname, outlabelname)