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

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device="cuda"

model_id = 'damo/multi-modal_hitea_video-captioning_base_en'
input_caption = '/home/yutong/workspace/data/Youtube/new_database/v71.avi'
pipeline_caption = pipeline('video-captioning', 'damo/multi-modal_hitea_video-captioning_base_en')

result = pipeline_caption(input_caption)
print(result)


