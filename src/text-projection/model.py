import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import random 

import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Text_MLP_Encoder(nn.Module):
    """MLP-based Text Projector"""
    def __init__(self, x_dim=512, z_dim=512, device='cpu'):
        super(Text_MLP_Encoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        # [I, xdim]
        self.encoder = nn.Sequential(
            nn.Linear(self.x_dim, self.z_dim), 
            nn.ReLU(True),
            nn.Linear(self.z_dim, self.z_dim),
        ).to(device)

    def forward(self, x):
        return self.encoder(x)