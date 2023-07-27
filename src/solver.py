# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import numpy as np
import json
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from summarizer import Summarizer_transformer
# from utils import TensorboardWriter
# from feature_extraction import ResNetFeature
import time
import numpy as np
#from knapsack import knapsack_dp
from knapsack import knapsack_ortools
import math
import os
import pickle

def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
    """Generate keyshot-based video summary i.e. a binary vector.
    Args:
    ---------------------------------------------
    - ypred: predicted importance scores.
    - cps: change points, 2D matrix, each row contains a segment.
    - n_frames: original number of frames.
    - nfps: number of frames per segment.
    - positions: positions of subsampled frames in the original video.
    - proportion: length of video summary (compared to original video length).
    - method: defines how shots are selected, ['knapsack', 'rank'].
    """
    n_segs = cps.shape[0]
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i >= len(ypred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred[i]

    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx,0]), int(cps[seg_idx,1]+1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    limits = int(math.floor(n_frames * proportion))

    if method == 'knapsack':
        #picks = knapsack_dp(seg_score, nfps, n_segs, limits)
        picks = knapsack_ortools(seg_score, nfps, n_segs, limits)
    elif method == 'rank':
        order = np.argsort(seg_score)[::-1].tolist()
        picks = []
        total_len = 0
        for i in order:
            if total_len + nfps[i] < limits:
                picks.append(i)
                total_len += nfps[i]
    else:
        raise KeyError("Unknown method {}".format(method))

    summary = np.zeros((1), dtype=np.float32) # this element should be deleted
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))

    summary = np.delete(summary, 0) # delete the first element
    return summary


def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    """Compare machine summary with user summary (keyshot-based).
    Args:
    --------------------------------
    machine_summary and user_summary should be binary vectors of ndarray type.
    eval_metric = {'avg', 'max'}
    'avg' averages results of comparing multiple human summaries.
    'max' takes the maximum (best) out of multiple comparisons.
    """
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)
    n_users,n_frames = user_summary.shape

    # binarization
    machine_summary[machine_summary > 0] = 1
    user_summary[user_summary > 0] = 1

    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])

    f_scores = []
    prec_arr = []
    rec_arr = []

    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx,:]
        overlap_duration = (machine_summary * gt_summary).sum()
        precision = overlap_duration / (machine_summary.sum() + 1e-8)
        recall = overlap_duration / (gt_summary.sum() + 1e-8)
        if precision == 0 and recall == 0:
            f_score = 0.
        else:
            f_score = (2 * precision * recall) / (precision + recall)
        f_scores.append(f_score)
        prec_arr.append(precision)
        rec_arr.append(recall)

    if eval_metric == 'avg':
        final_f_score = np.mean(f_scores)
        final_prec = np.mean(prec_arr)
        final_rec = np.mean(rec_arr)
    elif eval_metric == 'max':
        final_f_score = np.max(f_scores)
        max_idx = np.argmax(f_scores)
        final_prec = prec_arr[max_idx]
        final_rec = rec_arr[max_idx]
    
    return final_f_score, final_prec, final_rec


class Solver_transformer(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

    def build(self):
        self.summarizer = Summarizer_transformer(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size).to(self.config.device)
        
        self.model = nn.ModuleList([
            self.summarizer])

        if self.config.mode == 'train':
            self.transformer_optimizer = optim.Adam(
                list(self.summarizer.pos_enc.parameters())
                + list(self.summarizer.transformer_encoder.parameters())
                + list(self.summarizer.fc.parameters()),
                lr = self.config.lr,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
            )
            self.model.train()

    @staticmethod
    def freeze_model(module):
        for p in module.parameters():
            p.requires_grad = False

    def mse_loss(self, x, y):
        return F.mse_loss(x, y, reduction='mean')

    def train(self):
        step = 0
        loss_set = []
        ot_set = []

        max_fscore = 0
        ret_result = dict()

        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            loss_history = []
            ot_history = []

            for batch_i, (image_features, text_features, video_name) in enumerate(tqdm(
                    self.train_loader, desc='Batch', ncols=80, leave=False)):

                image_features = image_features.view(-1, self.config.input_size).to(self.config.device)
                # image_features_ = Variable(image_features).to(self.config.device)
                original_features = image_features.unsqueeze(1)
                
                # read in score2(pre-generated pseudo scores)
                if self.train_loader.dataset_type == '': # standard 
                    score2 = torch.Tensor(
                        self.train_loader.ot_score[self.train_loader.dataset_name][video_name]
                        ).to(self.config.device)
                else: # augment / transfer
                    dataset_name, video_name = video_name.split('/')
                    dataset_name = dataset_name.split('_')[2]
                    score2 = torch.Tensor(
                        self.train_loader.ot_score[dataset_name][video_name]
                        ).to(self.config.device)

                scores = self.summarizer(original_features)
                scores = scores.view(-1)

                if torch.isnan(score2).any():
                    continue

                ot_loss = self.mse_loss(scores.view(-1, 1), score2.view(-1, 1))

                loss = 1*ot_loss

                self.transformer_optimizer.zero_grad()
                loss.backward()  # retain_graph=True)
                # Gradient cliping
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
                self.transformer_optimizer.step()

                loss_history.append(loss.data)
                ot_history.append(1*ot_loss.data)
                print('loss: {}'.format(loss.data.detach().cpu()))
                print('ot loss: {}'.format(1*ot_loss.data.detach().cpu()))

                step += 1

            loss_st = torch.stack(loss_history).mean()
            ot_loss_st = torch.stack(ot_history).mean()

            loss_set.append(loss_st.detach().cpu())
            ot_set.append(ot_loss_st.detach().cpu())

            result = self.evaluate(epoch_i)

            if result["f-score"] > max_fscore:
                max_fscore = result["f-score"]
                ret_result = result
                # Save parameters at checkpoint
                ckpt_path = str(self.config.ckpt_path) + f'/split-{self.test_loader.split_id}-epoch-{epoch_i}.pkl'
                tqdm.write(f'Save parameters at {ckpt_path}')
                torch.save(self.model.state_dict(), ckpt_path)

            self.model.train()

        x = np.arange(1, len(loss_set) + 1, 1)
        fig, ax = plt.subplots()
        ax.plot(x, loss_set, label="loss")
        ax.plot(x, ot_set, label="ot")
        ax.set_title = f'split-{self.test_loader.split_id}'
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend()
        plt.savefig(os.path.join(self.config.loss_dir, f'split-{self.test_loader.split_id}.png'))

        return ret_result

    def evaluate(self, epoch_i):

        self.model.eval()
        out_dict = {}

        with torch.no_grad(): 
            for video_tensor, text_features, video_name in tqdm(
                    self.test_loader, desc='Evaluate', ncols=80, leave=False):


                video_tensor = video_tensor.view(-1, self.config.input_size).to(self.config.device)
                video_feature = Variable(video_tensor, volatile=True).to(self.config.device)
                video_feature = video_feature.unsqueeze(1)

                scores = self.summarizer(video_feature)
                scores = scores.squeeze(1)

                scores = np.array(scores.data.detach().cpu()).tolist()
                out_dict[video_name] = scores

                score_save_path = os.path.join(self.config.score_dir, f'score-split-{self.test_loader.split_id}-epoch-{epoch_i}.json')
                with open(score_save_path, 'w') as f:
                    tqdm.write(f'Saving score at {str(score_save_path)}.')
                    json.dump(out_dict, f)
                # score_save_path.chmod(0o777)
        
        fms = []
        precs = []
        recs = []
        
        for key_idx, key in enumerate(self.test_loader.test_keys):
            
            if self.test_loader.dataset_name == 'wikihow':
                
                # read in binary annotation of wikihow dataset
                annts = self.test_loader.wikihow_annt

                score = out_dict[key]
                score = torch.from_numpy(np.array(score)).view(-1)
                
                # becasue of the wikihow video frames are downsampled by 8 when extracting their visual embeddings,
                # so here, we downsample the binary annts to match the scores (in size). 
                orig_labels = torch.FloatTensor(annts[key])
                num_frames_in_video = len(orig_labels)
                sample_rate = 8
                n_segments_whole_video = math.ceil(num_frames_in_video / sample_rate)
                labels = torch.zeros(n_segments_whole_video, dtype=torch.long)  # downsampled binary labels
                label_scores = torch.zeros(n_segments_whole_video)  # downsampled frame-level annt scores
                for x in range(n_segments_whole_video):
                    left_bound = x * sample_rate
                    right_bound = min((x + 1) * sample_rate, num_frames_in_video)
                    count_1 = np.count_nonzero(
                        orig_labels[left_bound: right_bound]
                    )
                    count_0 = right_bound - left_bound - count_1
                    if count_1 > count_0:
                        labels[x] = 1
                    else:
                        labels[x] = 0
                    label_scores[x] = orig_labels[left_bound: right_bound].mean()

                if score.size(0) < labels.size(0):
                    zeros = torch.zeros(
                        (labels.size(0) - score.size(0)),
                    ).float()
                    score = torch.cat((score, zeros), axis=0)
                elif score.size(0) > labels.size(0):
                    score = score[:labels.size(0)]

                # select ids by the machine-generated scores. 
                summary_ids = (
                    score.detach().cpu().view(-1).topk(int(self.config.summary_rate * len(labels)))[1] # 0.7
                )
                summary = np.zeros((labels.size(0)))
                summary[summary_ids] = 1 

                fm, prec, rec = evaluate_summary(summary, labels.detach().numpy().reshape((1,-1)))
                fms.append(fm)
                precs.append(prec)
                recs.append(rec)
            
            else:
                if self.test_loader.dataset_type != '': # not standard
                    dataset_name, video_name = key.split('/')
                    dataset_name = dataset_name.split('_')[2]
                else:
                    dataset_name = self.test_loader.dataset_name
                    video_name = key

                eval_metric = 'avg' if self.test_loader.dataset_name == 'tvsum' else 'max'

                d = self.test_loader.datasets[dataset_name][video_name] # h5 files provided by the datasets. 
                probs = out_dict[key] # machine-generated scores

                if 'change_points' not in d:
                    print("ERROR: No change points in dataset/video ",key)

                cps = d['change_points'][...]
                num_frames = d['n_frames'][()]
                nfps = d['n_frame_per_seg'][...].tolist()
                positions = d['picks'][...]
                user_summary = d['user_summary'][...]
                gt_summary = d['gtsummary'][()]

                machine_summary = generate_summary(probs, cps, num_frames, nfps, positions)
                fm, prec, rec = evaluate_summary(machine_summary, user_summary, eval_metric)
                fms.append(fm)
                precs.append(prec)
                recs.append(rec)

        mean_fm = np.mean(fms)
        mean_prec = np.mean(precs)
        mean_rec = np.mean(recs)

        result = dict()
        result["f-score"] = mean_fm
        result["prec"] = mean_prec
        result["rec"] = mean_rec
        result_save_path = os.path.join(self.config.save_dir, f'split-{self.test_loader.split_id}-epoch-{epoch_i}.json')
        with open(result_save_path, 'w') as f:
            tqdm.write(f'Saving score at {str(result_save_path)}.')
            json.dump(result, f)

        return result


if __name__ == '__main__':
    pass
