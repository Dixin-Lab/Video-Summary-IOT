from configs import get_config
from solver import Solver_transformer
from data_loader import get_loader
import json
import numpy as np
import os
from pprint import pprint
from home import get_project_base

if __name__ == '__main__':
    config = get_config(mode='train')
    test_config = get_config(mode='test')
    print(config)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    device = config.device

    BASE = get_project_base()
    data_setting = config.dataset_setting + ".json"
    split_path = os.path.join(BASE, 'dataset', 'splits', data_setting)
    
    with open(split_path, 'r') as sf:
        splits_file = json.load(sf)

    splits_num = len(splits_file)

    f_score = []
    prec = []
    rec = []  

    for split in range(splits_num):
        print('split {}:'.format(split))
        train_loader = get_loader(config.video_root_dir, config.mode, split_path, split)
        test_loader = get_loader(test_config.video_root_dir, test_config.mode, split_path, split)

        solver = Solver_transformer(config, train_loader, test_loader)
        solver.build()
        ret_score = solver.train()
        f_score.append(ret_score["f-score"])
        prec.append(ret_score["prec"])
        rec.append(ret_score["rec"])
        pprint(ret_score)
        print('---------------------------------------')

        score_save_path = os.path.join(solver.config.save_dir, f'result-split-{split}.json')
        with open(score_save_path, 'w') as f:
            json.dump(ret_score, f)

    
    print("Result f_score :{}".format(np.mean(f_score)))
    print("Result prec :{}".format(np.mean(prec)))
    print("Result rec :{}".format(np.mean(rec)))
