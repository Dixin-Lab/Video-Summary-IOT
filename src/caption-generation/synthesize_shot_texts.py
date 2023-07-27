from operator import index
import numpy as np
import glob
import os
import shutil 
import time
import subprocess
import matplotlib.pyplot as plt
from torch import Tensor, tensor
from scipy.special import softmax
import torch
from torch import nn
from torch.nn import functional as F
import random
import itertools
from sklearn.metrics import average_precision_score
import json
import h5py
import cv2
import pickle
import math
import scipy.io
import csv
import skvideo.io
from torch.nn.functional import normalize
import sys
sys.path.append("..") 
from home import get_project_base

"""
Synthesize the different shot texts into a whole file for one video. 
"""

BASE = get_project_base()
bs = os.path.join(BASE, 'raw_data', 'OVP', 'generated_texts', 'shot10_new_all')
new_bs = os.path.join(BASE, 'raw_data', 'OVP', 'generated_texts', 'shot10_new')

if not os.path.exists(new_bs):
    os.makedirs(new_bs)

files = os.listdir(bs)
for file in files:
    fdir = os.path.join(bs, file)
    ret = list()

    subfiles = sorted(os.listdir(fdir))

    # for HiTeA texts, just select all texts. 

    for subfile in subfiles:
        file_path = os.path.join(fdir, subfile)
        with open(file_path, 'r') as f:
            data = json.load(f)
        item = dict()
        item['time'] = subfile
        item['sentence'] = data['caption']
        ret.append(item)
    
    ret_file = file + '.json'
    ret_path = os.path.join(new_bs, ret_file)
    with open(ret_path, 'w') as f:
        json.dump(ret, f)




