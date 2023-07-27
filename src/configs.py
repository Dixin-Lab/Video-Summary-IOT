# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pprint
import os
from home import get_project_base

def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, exp="", **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.set_dataset_dir(exp=exp)

    def set_dataset_dir(self, exp=""):
        BASE = get_project_base()
        self.video_root_dir = os.path.join(BASE, 'dataset', 'splits')
        logbase = os.path.join(BASE, 'log')

        # the project name "exp" you use in the log file. 
        exp = 'tvsum-std'

        self.save_dir = os.path.join(logbase, exp)
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_dir = self.save_dir
        self.ckpt_path = os.path.join(self.save_dir, 'ckpt')
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.score_dir = os.path.join(self.save_dir, 'score')
        os.makedirs(self.score_dir, exist_ok=True)
        self.loss_dir = os.path.join(self.save_dir, 'loss')
        os.makedirs(self.loss_dir, exist_ok=True)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, exp="", **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=str, default='3')
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--summary_rate', type=float, default=0.3,help="pre-defined summary length, 0.3 for generic, 0.7 for wikihow")
    parser.add_argument('--n_epochs', type=int, default=100,help="training epochs")
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate of keyframe selector")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="weight decay of keyframe selector")
    parser.add_argument("--dataset_setting", type=str, default='tvsum_splits',help="tvsum_splits/tvsum_aug_splits/tvsum_transfer_splits/summe_splits/summe_aug_splits/summe_transfer_splits/wikihow_splits.")
    kwargs = parser.parse_args()

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(exp, **kwargs)


if __name__ == '__main__':
    config = get_config()
    import ipdb
    ipdb.set_trace()
