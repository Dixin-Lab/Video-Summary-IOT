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
from torch.nn.functional import normalize
from model import Text_MLP_Encoder
import math 
import time 
import sys
sys.path.append("..") 
from home import get_project_base


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


def get_C(v_feat, w_feat, device):
    C = distance_matrix(v_feat, w_feat, p=2).to(device)
    # order info ? 
    # A = C.size(0)
    # B = C.size(1)
    # order_info = np.zeros((A, B)).astype(np.float32)
    # Lambda = 1.0
    # for i in range(A):
    #     for j in range(B):
    #         order_info[i][j] = order_info[i][j] + Lambda * (abs(i / A - j / B))

    # C = C + torch.from_numpy(order_info).to(device)

    return C


parser = argparse.ArgumentParser()
parser.add_argument('--badmm-loops', type=int, default=2000, help='the iteration number in badmm')
parser.add_argument('--badmm-error-bound', type=float, default=1e-2, help='the iteration error bound in badmm')
parser.add_argument('--badmm-rho', type=float, default=1e-2, help='a hyperpara controlling rate of convergence in badmm')
parser.add_argument('--tau', type=float, default=0.5, help='the weight of entropic regularizer')
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument('--input_size', type=int, default=512)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--enable_gap', type=int, default=1)
parser.add_argument("--dataset", type=str, default='OVP')
parser.add_argument("--exp", type=str, default='',help="save dir name")
args = parser.parse_args()

base_dir = '.'
dataset = args.dataset
ds = dataset.lower()

exp = 'ovp_clip_shot10_mlp-100epoch-bd1e-2-rho1e-2-gap-only-1kl-fps-lr1e-3-vdetach' 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = args.device

BASE = get_project_base()
videos_path = os.path.join(BASE, 'raw_data', dataset, 'mp4video')
videos = os.listdir(videos_path)

clip_v_path = os.path.join(BASE, 'dataset', 'v_feat', '{}_clip_feats_norm.pkl'.format(ds))
clip_w_path = os.path.join(BASE, 'dataset', 'w_feat', '{}_clip_shot10_new_text_feats_norm.pkl.pkl'.format(ds))
clip_datasets = np.load(clip_v_path, allow_pickle=True)
clip_text_datasets = np.load(clip_w_path, allow_pickle=True)

fps_dir = 'fps-{}.json'.format(dataset)
with open(fps_dir, 'r') as f:
    fps_datasets = json.load(f)

dir1 = os.path.join(base_dir, exp)
if not os.path.exists(dir1):
    os.mkdir(dir1)

arg_dir = os.path.join(dir1, 'args.json')
with open(arg_dir, 'w') as f:
    if not isinstance(args, dict):
        args_ = vars(args)
    json.dump(args_, f, indent=True)

model_w = Text_MLP_Encoder(device=device)
optimizer = optim.Adam(list(model_w.parameters()), lr=args.lr, betas=(0.9, 0.999))

# plot the loss figure
x = np.arange(args.n_epochs)
loss_set = []
ot_set = []
nce_set = []
gap_set = []
kl_set = []

BASE = get_project_base()
videos_path = os.path.join(BASE, 'raw_data', dataset, 'mp4video')
videos = os.listdir(videos_path)

# note the different between TVSum and other datasets. 
shot_length = 20 if dataset=='TVSum' else 10

for epoch in range(args.n_epochs):
    loss_ = []
    ot_ = []
    nce_ = []
    gap_ = []
    kl_ = []
    print('Epoch: {}'.format(epoch))
    text_feat = dict()

    for video in videos:
        video_fps = fps_datasets[video]
        video = video[:-4]

        print(video)

        optimizer.zero_grad()

        v_feat = torch.Tensor(np.array(clip_datasets[video])).to(device)
        w_feat = torch.Tensor(np.array(clip_text_datasets[video])).to(device)

        w_feat = model_w(w_feat)
        w_feat = normalize(w_feat, p=2.0, dim = 1)

        text_feat[video] = w_feat.detach().cpu().numpy()

        v_cluster_feat = torch.zeros_like(w_feat.detach())
        n_w = w_feat.size(0)
        n_v = v_feat.size(0)

        for idx in range(n_w):
            if dataset != 'WikiHow':
                shot_range = shot_length * 2 * (video_fps / 30.0) # (5.0 / 6.0) for summe videos 
            else:
                shot_range = shot_length * video_fps / 8.0

            l = math.ceil(idx * shot_range)
            r = min(math.ceil((idx + 1) * shot_range), n_v)

            if l > r or l >= n_v: 
                continue

            v_cluster_feat[idx] = torch.mean(v_feat[l:r], dim=0)

        v_cluster_feat = v_cluster_feat.to(device).detach()

        C_gt = get_C(v_cluster_feat, w_feat, device)
        gap_loss = 0
        for idx in range(n_w):
            gap_loss += C_gt[idx][idx]
        gap_loss /= n_w

        # compute KL divergence between C_w and C_bc
        C_w = get_C(w_feat, w_feat, device)
        C_w = C_w / C_w.max()
        C_bc_np = np.ones((n_w, n_w))
        row, col = np.diag_indices_from(C_bc_np)
        C_bc_np[row,col] = 0
        C_bc = torch.from_numpy(C_bc_np).to(device).detach()
        d_kl = F.kl_div(C_bc.log(), C_w)


        if n_w > 1 and d_kl.item()!=0:
            loss = gap_loss + 1*d_kl
        else:
            loss = gap_loss

        loss.backward()
        optimizer.step()
        loss_.append(loss.item())

        gap_.append(gap_loss.item())
        kl_.append(1*d_kl.item())
        print('gap loss: {}'.format(gap_loss.item()))
        print('kl loss: {}'.format(1*d_kl.item()))
        print('loss: {}'.format(loss.item()))

    loss_set.append(np.mean(loss_))
    gap_set.append(np.mean(gap_))
    kl_set.append(np.mean(kl_))
    print('Epoch: {} : mean loss {}'.format(epoch, np.mean(loss_)))

    if epoch % 10 == 0:
        feat_dir = os.path.join(dir1, '{}_projected_text_feats_norm_{}.pkl'.format(dataset,epoch))
        with open(feat_dir, 'wb') as f:
            pickle.dump(text_feat, f)
            print('feats save successfully.')

# save model_w parameters
net_dir = os.path.join(dir1, 'network_w.net')
torch.save(model_w.state_dict(), net_dir)


# plot a loss figure
y = np.array(loss_set)
plt.plot(x,y,'g',label='loss')
y2 = np.array(gap_set)
plt.plot(x,y2,'b',label='gap')
y3 = np.array(kl_set)
plt.plot(x,y3,'r',label='kl')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
fig_dir = os.path.join(dir1, 'loss.png')
plt.savefig(fig_dir)
plt.close('all')

#model_w.load_state_dict(torch.load(net_dir))

# generate final text embeddings 
text_feat = dict()
with torch.no_grad():
    for video in videos:
        video = video[:-4]
        w_feat = torch.Tensor(np.array(clip_text_datasets[video])).to(device)
        w_feat = model_w(w_feat)
        w_feat = normalize(w_feat, p=2.0, dim = 1)
        text_feat[video] = w_feat.detach().cpu().numpy()

feat_dir = os.path.join(dir1, '{}_projected_text_feats_norm.pkl'.format(dataset))
with open(feat_dir, 'wb') as f:
    pickle.dump(text_feat, f)
    print('feats save successfully.')
