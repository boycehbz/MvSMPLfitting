# -*- coding: utf-8 -*-
import sys
import os

import os.path as osp

import time
import torch

from cmd_parser import parse_config
from init import init
from utils.init_guess import init_guess
from utils.non_linear_solver import non_linear_solver
from utils.utils import save_results
def main(**args):

    dataset_obj, setting = init(**args)

    start = time.time()

    # init = {}
    s_last = None # the name of last sequence
    seq_begin = False # indicate the first frame of the sequence
    for idx, data in enumerate(dataset_obj):
        serial = data['serial']
        # viewpoint should greater than 1 if not use 3D annotation
        keypoints = data['keypoints']
        views = 0
        for kep in keypoints:
            if kep is not None:
                views += 1
        if views < 2 and not setting["use_3d"]:
            s_last = None
            continue
        setting['views'] = views
        print('Processing: {}'.format(data['img_path']))

        # init guess
        init_guess(setting, data, use_torso=True, **args)
        # linear solve
        print("linear solve, to do...")
        # non-linear solve
        results = non_linear_solver(setting, data, **args)
        # save results
        save_results(setting, data, results, **args)

    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))

if __name__ == "__main__":

    sys.argv = ["", "--config=cfg_files\\fit_smpl.yaml"
    ] 
    args = parse_config()
    main(**args)





