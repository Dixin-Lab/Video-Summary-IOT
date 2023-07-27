# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from pathlib import Path

import h5py
import numpy as np
import os
import json
from home import get_project_base

def parse_splits_filename(splits_filename):
    # Parse split file and count number of k_folds
    spath, sfname = os.path.split(splits_filename)
    sfname, _ = os.path.splitext(sfname)
    dataset_name = sfname.split('_')[0]  # Get dataset name e.g. tvsum
    dataset_type = sfname.split('_')[1]  # augmentation type e.g. aug

    # The keyword 'splits' is used as the filename fields terminator from historical reasons.
    if dataset_type == 'splits':
        # Split type is not present
        dataset_type = ''

    # Get number of discrete splits within each split json file
    with open(splits_filename, 'r') as sf:
        splits = json.load(sf)

    return dataset_name, dataset_type, splits


class VideoData(Dataset):
    def __init__(self, root, splits, split, with_name=False, ot_score_dir=""):
        self.with_name = with_name  # true->test; false->train

        self.dataset_name, self.dataset_type, self.splitsfile = parse_splits_filename(splits)
        self.split_id = split
        self.train_keys = self.splitsfile[self.split_id]['train_keys']
        self.test_keys = self.splitsfile[self.split_id]['test_keys']

        BASE = get_project_base()
        dataset_path = os.path.join(BASE, 'dataset')
        h5_files_path = os.path.join(dataset_path, 'h5_files')

        # read in h5 files for 4 datasets. 
        # summe_path = os.path.join(h5_files_path, 'eccv16_dataset_summe_google_pool5.h5')
        tvsum_path = os.path.join(h5_files_path, 'eccv16_dataset_tvsum_google_pool5.h5')
        # youtube_path = os.path.join(h5_files_path, 'eccv16_dataset_youtube_google_pool5.h5')
        # ovp_path = os.path.join(h5_files_path, 'eccv16_dataset_ovp_google_pool5.h5')
        self.datasets = {}
        # self.datasets["summe"] = h5py.File(summe_path, 'r')
        self.datasets["tvsum"] = h5py.File(tvsum_path, 'r')
        # self.datasets["youtube"] = h5py.File(youtube_path, 'r')
        # self.datasets["ovp"] = h5py.File(ovp_path, 'r')

        # wikihow_annt_path = os.path.join(h5_files_path, 'wikihowto_annt.json')
        # with open(wikihow_annt_path) as f:
        #     self.wikihow_annt = json.load(f)

        # read in visual features for 5 datasets. 
        self.clip_datasets = {}
        v_feat_path = os.path.join(dataset_path, 'v_feat')
        # summe_path = os.path.join(v_feat_path, 'summe_clip_feats_norm.pkl')
        tvsum_path = os.path.join(v_feat_path, 'tvsum_clip_feats_norm.pkl')
        # youtube_path = os.path.join(v_feat_path, 'youtube_clip_feats_norm.pkl')
        # ovp_path = os.path.join(v_feat_path, 'ovp_clip_feats_norm.pkl')
        # wikihow_path = os.path.join(v_feat_path, 'wikihow_clip_feats_8_norm.pkl')

        self.clip_datasets["tvsum"] = np.load(tvsum_path, allow_pickle=True)
        # self.clip_datasets["summe"] = np.load(summe_path, allow_pickle=True)
        # self.clip_datasets["youtube"] = np.load(youtube_path, allow_pickle=True)
        # self.clip_datasets["ovp"] = np.load(ovp_path, allow_pickle=True)
        # self.clip_datasets["wikihow"] = np.load(wikihow_path, allow_pickle=True)

        # read in textual features for 5 datasets. 
        self.clip_text_datasets = {}
        w_feat_path = os.path.join(dataset_path, 'w_feat')
        # summe_path = os.path.join(w_feat_path, 'summe_clip_shot10_new_text_feats_norm.pkl')
        tvsum_path = os.path.join(w_feat_path, 'tvsum_clip_shot20_new_text_feats_norm.pkl')
        # youtube_path = os.path.join(w_feat_path, 'youtube_clip_shot10_new_text_feats_norm.pkl')
        # ovp_path = os.path.join(w_feat_path, 'ovp_clip_shot10_new_text_feats_norm.pkl')
        # wikihow_path = os.path.join(w_feat_path, 'wikihow_clip_shot10_new_text_feats_norm.pkl')

        self.clip_text_datasets["tvsum"] = np.load(tvsum_path, allow_pickle=True)
        # self.clip_text_datasets["summe"] = np.load(summe_path, allow_pickle=True)
        # self.clip_text_datasets["youtube"] = np.load(youtube_path, allow_pickle=True)
        # self.clip_text_datasets["ovp"] = np.load(ovp_path, allow_pickle=True)
        # self.clip_text_datasets["wikihow"] = np.load(wikihow_path, allow_pickle=True)

        # read in pseudo scores for 5 datasets. 
        self.ot_score = {}
        score_path = os.path.join(dataset_path, 'pseudo_scores')
        # summe_path = os.path.join(score_path, 'summe_ot_scores.pkl')
        tvsum_path = os.path.join(score_path, 'tvsum_ot_scores.pkl')
        # youtube_path = os.path.join(score_path, 'youtube_ot_scores.pkl')
        # ovp_path = os.path.join(score_path, 'ovp_ot_scores.pkl')
        # wikihow_path = os.path.join(score_path, 'wikihow_ot_scores.pkl')

        self.ot_score['tvsum'] = np.load(tvsum_path, allow_pickle=True)
        # self.ot_score['summe'] = np.load(summe_path, allow_pickle=True)
        # self.ot_score['youtube'] = np.load(youtube_path, allow_pickle=True)
        # self.ot_score['ovp'] = np.load(ovp_path, allow_pickle=True)
        # self.ot_score['wikihow'] = np.load(wikihow_path, allow_pickle=True)

        for ds in ['tvsum']:
            for key in self.ot_score[ds].keys():
                # the saved pseudo scores are negative, change them to positive.
                self.ot_score[ds][key] = 1 - self.ot_score[ds][key] 
                # normalize the pseudo scores to [0,1]. 
                self.ot_score[ds][key] = (self.ot_score[ds][key] - self.ot_score[ds][key].min()) / (self.ot_score[ds][key].max() - self.ot_score[ds][key].min())
        

    def __len__(self):
        if self.with_name:
            return len(self.test_keys)
        else:
            return len(self.train_keys)

    def __getitem__(self, index):
            
        if self.with_name:
            video_path = self.test_keys[index]
        else:
            video_path = self.train_keys[index]

        if self.dataset_type == '': #standard
            return torch.Tensor(np.array(self.clip_datasets[self.dataset_name][video_path])),  torch.Tensor(np.array(self.clip_text_datasets[self.dataset_name][video_path])), video_path

        else: # aug or transfer
            dataset_name, video_name = video_path.split('/')
            dataset_name = dataset_name.split('_')[2]

            return torch.Tensor(np.array(self.clip_datasets[dataset_name][video_name])),  torch.Tensor(np.array(self.clip_text_datasets[dataset_name][video_name])), video_path


def get_loader(root, mode, splits, split, ot_score_dir=""):
    if mode.lower() == 'train':
        return VideoData(root, splits, split, ot_score_dir=ot_score_dir)
    else:
        return VideoData(root, splits, split, with_name=True, ot_score_dir=ot_score_dir)


if __name__ == '__main__':
    pass
