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
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import sys
sys.path.append("..") 
from home import get_project_base

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device="cuda"

model_id = 'damo/multi-modal_hitea_video-captioning_base_en'
pipeline_caption = pipeline('video-captioning', 'damo/multi-modal_hitea_video-captioning_base_en')


BASE = get_project_base()
# note here, for the pretrained HiTeA model, the input form must be .avi .
base_dir = os.path.join(BASE, 'raw_data', 'OVP', 'mp4video_shot10_avi')
tgt_dir = os.path.join(BASE, 'raw_data', 'OVP', 'generated_texts', 'shot10_new_all')
files = os.listdir(base_dir)

for file in files:
    fdir = os.path.join(base_dir, file)

    o_fdir = os.path.join(tgt_dir, file)
    if os.path.exists(o_fdir):
        continue
    os.makedirs(o_fdir)

    subfiles = os.listdir(fdir)

    for subfile in subfiles:
        video_path = os.path.join(fdir, subfile)
        video_name = subfile[:-4]

        input_caption = video_path
        result = pipeline_caption(input_caption)

        json_name = video_name + '.json'
        json_path = os.path.join(o_fdir, json_name)

        with open(json_path, 'w') as f:
            json.dump(result, f)

        print(subfile)
        print(result)
        


