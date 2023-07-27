import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import json
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import argparse
import pickle
import h5py
import time
import sys
sys.path.append("..") 
from home import get_project_base


def get_k_max(array, k):
    _k_sort = np.argpartition(array, -k)[-k:]  # 最大的k个数据的下标
    return _k_sort


def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C] distance matrix
    """
    x_col = pts_src.unsqueeze(1) # [J,1,z_dim] or [I,1,z_dim]
    y_row = pts_dst.unsqueeze(0) # [1,J,z_dim] or [1,I,z_dim]
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    # distance = -torch.sum((x_col * y_row), 2)

    return distance


def distance_matrix_sqrt(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C] distance matrix
    """
    x_col = pts_src.unsqueeze(1) # [J,1,z_dim] or [I,1,z_dim]
    y_row = pts_dst.unsqueeze(0) # [1,J,z_dim] or [1,I,z_dim]
    distance = torch.sqrt(torch.sum((torch.abs(x_col - y_row)) ** p, 2))

    return distance


def bregman_admm_iteration(C: torch.Tensor, p_v: torch.Tensor, p_w: torch.Tensor, trans0: torch.Tensor,
                            tau: float = 0.1, rho: float = 1.0, 
                            error_bound: float = 1e-3, max_iter: int = 50, device: str = 'cuda', 
                            ) -> torch.Tensor:
    '''
    Bregman-ADMM iteration algorithm

    Args:
        C: (I, J) array representing distance between nodes
        p_v: (I, 1) array representing the distribution of source nodes
        p_w: (J, 1) array representing the distribution of target nodes
        trans0: (I, J) initial array of optimal transport
        tau: the weight of entropic regularizer
        rho: a hyperpara controlling rate of convergence in badmm
        error_bound: the error bound to check convergence
        max_iter: the maximum number of iterations
        device: running on cuda or cpu
    Returns:
        T: final OT matrix
    
    '''
    I = C.size(0)
    J = C.size(1)

    if p_v is None:
        p_v = torch.ones(I, 1) / I
        p_v = p_v.to(device)
    
    if p_w is None:
        p_w = torch.ones(J, 1) / J
        p_w = p_w.to(device)

    if trans0 == None:
        trans0 = p_v @ torch.t(p_w)
        trans0 = trans0.to(device)
    
    trans1 = torch.zeros(trans0.shape).to(device)
    u = torch.ones(I,1).to(device) / I
    mu = torch.ones(J,1).to(device) / J
    Z = torch.zeros(trans0.shape).to(device)
    z1 = torch.zeros(u.shape).to(device)
    z2 = torch.zeros(mu.shape).to(device)

    relative_error = error_bound + 1.0
    i = 1

    while relative_error > error_bound and i <= max_iter:
        
        tmp = (-C  + rho * torch.log(trans0) - Z) / rho
        trans1 = torch.diag(u.reshape(-1)) @ F.softmax(tmp, dim=1)   # T^{k+1}

        tmp = (Z + rho * torch.log(trans1)) / rho
        trans0 = F.softmax(tmp, dim=0) @ torch.diag(mu.reshape(-1))   # S^{k+1}

        tmp = (rho * torch.log(torch.sum(trans1, dim=1, keepdim=True)) + tau * torch.log(torch.ones(I, 1).to(device) / I) - z1) / (rho + tau)
        u = F.softmax(tmp, dim=0)                              # u^{k+1}

        tmp = (rho * torch.log(torch.sum(torch.t(trans0), dim=1, keepdim=True)) + tau * torch.log(torch.ones(J, 1).to(device) / J) - z2) / (rho + tau)
        mu = F.softmax(tmp, dim=0)                             # mu^{k+1}
        

        Z = Z + rho * (trans1 - trans0)
        z1 = z1 + rho * (u - torch.sum(trans1, dim=1, keepdim=True))
        z2 = z2 + rho * (mu - torch.sum(torch.t(trans0), dim=1, keepdim=True))
        
        relative_error = torch.sum(torch.abs(trans1.detach().data - trans0.detach().data)) / torch.sum(torch.abs(trans1.detach().data)) + \
                            torch.sum(torch.abs(u.detach().data - torch.sum(trans1.detach().data, dim=1, keepdim=True))) / torch.sum(torch.abs(u.detach().data)) + \
                               torch.sum(torch.abs(mu.detach().data - torch.sum(torch.t(trans0.detach().data), dim=1, keepdim=True))) / torch.sum(torch.abs(mu.detach().data))
        
        if i%200==0:
            print('{} : badmm relative error: {}'.format(i, relative_error))
        # print('{} : badmm relative error: {}'.format(i, relative_error))

        i += 1
    
    return trans1


def unbalanced_wasserstein_distance(C: torch.Tensor, args):
    """
    compute the unbalanced OT with BADMM

    Args:
        C: cost matrix between two distributions
    Returns:
        d_uot: unbalanced wasserstein distance
        T: the optimal transport matrix between two distributions
    """
    I = C.size(0)
    J = C.size(1)
    T = torch.ones(I, J) / (I * J) # [I,J]
    T = T.to(args.device)
    T = bregman_admm_iteration(C, None, None, T, args.tau, args.badmm_rho, args.badmm_error_bound, args.badmm_loops, args.device)
    if torch.isnan(T).sum() > 0:
        T = (torch.ones(I, J) / (I * J)).to(args.device)
    
    d_uot = (C * T.detach().data).sum()

    return d_uot, T.detach().data


def plot_matrix(impact, output_name: str = None):
    '''
    plot headmap for provided matrix

    params:
        impact: matrix that need to be plotted
        outputname: heatmap figure's name
    '''
    plt.figure(figsize=(5, 5))
    plt.imshow(impact)
    plt.colorbar()
    if output_name is None:
        plt.savefig('figure.svg')
    else:
        plt.savefig(output_name)
    plt.close("all")


parser = argparse.ArgumentParser()
parser.add_argument('--badmm-loops', type=int, default=2000, help='the iteration number in badmm')
parser.add_argument('--badmm-error-bound', type=float, default=1e-2, help='the iteration error bound in badmm')
parser.add_argument('--badmm-rho', type=float, default=1e-2, help='a hyperpara controlling rate of convergence in badmm')
parser.add_argument('--tau', type=float, default=0.1, help='the weight of entropic regularizer')
parser.add_argument('--threshold', type=float, default=0.5, help='the threshold in the alignment score')
parser.add_argument('--recon-range', type=float, default=0.5, help='the threshold in the representation score')
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--score-way", type=str, default="score5", help="the method to compute the ot pseudo score")
parser.add_argument("--recon-way", type=str, default="1-2", help="the method to compute the ot pseudo score")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--dataset", type=str, default='OVP')
parser.add_argument("--exp", type=str, default='',help="save dir name")
args = parser.parse_args()

base_dir = '.'
dataset = args.dataset
ds = dataset.lower()

# the save dir name <exp>
exp = 'ovp-newshot-score5-thre05-recon1-2-recon_thre05-1kl-fps'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda'

BASE = get_project_base()
videos_path = os.path.join(BASE, 'raw_data', dataset, 'mp4video')
videos = os.listdir(videos_path)

clip_v_path = os.path.join(BASE, 'dataset', 'v_feat', '{}_clip_feats_norm.pkl'.format(ds))
clip_w_path = os.path.join(BASE, 'dataset', 'w_feat', '{}_projected_text_feats_norm.pkl'.format(ds))
clip_datasets = np.load(clip_v_path, allow_pickle=True)
clip_text_datasets = np.load(clip_w_path, allow_pickle=True)

dir1 = os.path.join(base_dir, exp)
if not os.path.exists(dir1):
    os.mkdir(dir1)

dir2 = os.path.join(dir1, 'matrix') # OT matrix visualization of each video is saved here. 
if not os.path.exists(dir2):
    os.mkdir(dir2)

T_set = dict()
score_set = dict()
score1_set = dict() # score1: representation score
score2_set = dict() # score2: alignment score

for video in videos:
    print(video)
    video = video[:-4] # xxx.mp4

    v_feat = torch.Tensor(np.array(clip_datasets[video])).to(device)
    w_feat = torch.Tensor(np.array(clip_text_datasets[video])).to(device)

    # compute OT matrix T between frames and texts.
    C = distance_matrix(v_feat, w_feat, p=2).to(device)
    ot_distance, T = unbalanced_wasserstein_distance(C, args)

    T = T.detach().cpu().numpy()
    T_set[video] = T
    score2 = np.zeros(T.shape[0])
    threshold = args.threshold

    if args.score_way == 'score3': # consider ot matrix T and cost matrix C 
        K = int(T.shape[1] * threshold)
        for idx in range(T.shape[0]):
            selected = get_k_max(T[idx], K)
            for idxx in selected:
                score2[idx] -= C[idx][idxx] * T[idx][idxx]

        score2 = (score2 - score2.min()) / (score2.max() - score2.min())

    elif args.score_way == 'score4': # consider cost matrix C only  
        K = int(T.shape[1] * threshold)
        for idx in range(T.shape[0]):
            selected = get_k_max(T[idx], K)
            for idxx in selected:
                score2[idx] -= C[idx][idxx]

        score2 = (score2 - score2.min()) / (score2.max() - score2.min())
    
    # score5: 
    elif args.score_way == 'score5': 
        C = C.detach().cpu().numpy()
        K = int(T.shape[0] * threshold)
        for idx in range(T.shape[1]):
            selected = get_k_max(T.T[idx], K)
            scorej = np.zeros(T.shape[0])
            dcost = C.T[idx][selected]
            dsum = np.sum(dcost)
            scorej[selected] = dsum / C.T[idx][selected]
            scorej = (scorej - scorej.min()) / (scorej.max() - scorej.min())
            for idxx in range(T.shape[0]):
                score2[idxx] = max(score2[idxx], scorej[idxx])

        score2 = (score2 - score2.min()) / (score2.max() - score2.min())
    

    if args.recon_way == '1-1':
        score1 = np.zeros(v_feat.size(0))
        C_v = distance_matrix_sqrt(v_feat, v_feat)
        for idx in range(v_feat.size(0)):
            score1[idx] = C_v[idx].sum() / (v_feat.size(0) - 1)
        
        score1 = (score1 - score1.min()) / (score1.max() - score1.min())

    elif args.recon_way == '1-2':
        score1 = np.zeros(v_feat.size(0))
        C_v = distance_matrix_sqrt(v_feat, v_feat)
        window = np.ceil(args.recon_range * v_feat.size(0))
        for idx in range(v_feat.size(0)):
            l = np.ceil(idx - window / 2.0)
            r = np.ceil(idx + window / 2.0 + 1)
            if l < 0:
                l = 0
                r = l + window + 1
            elif r > v_feat.size(0):
                r = v_feat.size(0)
                l = r - window - 1
            
            l = int(l)
            r = int(r)
            score1[idx] = C_v[idx][l:r].sum() / (window)
        
        score1 = (score1 - score1.min()) / (score1.max() - score1.min())
    

    if args.recon_way == '0':
        score_set[video] = -score2 # here, the score2 is negative 
    elif args.score_way == '0':
        score_set[video] = -1 + score1 # here, the score2 is negative 
    else:
        score_set[video] = (- score2 - 1 + score1) / 2.0 # here, the score2 is negative 
        score1_set[video] = -1 + score1 # here, the score2 is negative 
        score2_set[video] = - score2 # here, the score2 is negative 

    fig_path = video + '.pdf'
    fig_path = os.path.join(dir2, fig_path)
    plot_matrix(T, fig_path)

    print('{} complete'.format(video))


path1 = os.path.join(dir1, '{}_ot_scores1.pkl'.format(ds))
with open(path1, 'wb') as f:
    pickle.dump(score1_set, f)
path1 = os.path.join(dir1, '{}_ot_scores2.pkl'.format(ds))
with open(path1, 'wb') as f:
    pickle.dump(score2_set, f)

path1 = os.path.join(dir1, '{}_ot_scores.pkl'.format(ds))
with open(path1, 'wb') as f:
    pickle.dump(score_set, f)
path2 = os.path.join(dir1, '{}_ot_matrix.pkl'.format(ds))
with open(path2, 'wb') as f:
    pickle.dump(T_set, f)
