# -*- coding: utf-8 -*-
'''
 @FileName    : main.py
 @EditTime    : 2021-09-19 21:46:57
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
from copy import deepcopy
import sys
import os

import os.path as osp

import time
import torch
import numpy as np
from cmd_parser import parse_config
from init import init
from utils.init_guess import init_guess, load_init, fix_params
from utils.non_linear_solver import non_linear_solver
from utils.utils import save_results, change
import cv2
keyps = []

def main(**args):
    global keyps

    dataset_obj, setting = init(**args)

    start = time.time()

    results = {}
    s_last = None  # the name of last sequence
    setting['seq_start'] = False  # indicate the first frame of the sequence
    for idx, data in enumerate(dataset_obj):

        serial = data['serial']
        if serial != s_last:
            setting['seq_start'] = True
            s_last = serial
        else:
            setting['seq_start'] = False
        # filter out the view without annotaion
        keypoints = data['keypoints']
        views = 0
        extrinsics = []
        intrinsics = []
        keyps = []
        img_paths = []
        imgs = []
        cameras = []
        for v in range(len(keypoints)):
            if keypoints[v] is not None:
                extrinsics.append(setting['extrinsics'][v])
                intrinsics.append(setting['intrinsics'][v])
                cameras.append(setting['cameras'][v])
                keyps.append(keypoints[v])
                img_paths.append(data['img_path'][v])
                imgs.append(data['img'][v])
                views += 1
        # viewpoint should greater than 1 if not use 3D annotation
        # if views < 2 and not setting["use_3d"] or len(keyps) < 1:
        #     s_last = None
        #     continue
        setting['views'] = views
        setting['extris'] = np.array(extrinsics)
        setting['intris'] = np.array(intrinsics)
        setting['camera'] = cameras
        data['img'] = imgs
        data['img_path'] = img_paths
        data['keypoints'] = keyps
        # print("keypoints: ", keypoints)
        print('Processing: {}'.format(data['img_path']))

        if setting['adjustment']:
            change(img_paths, keyps)

        # init guess
        if setting['seq_start'] or not args.get('is_seq'):
            init_guess(setting, data, use_torso=True, **args)  # 运行这个
        else:
            load_init(setting, data, results, use_torso=True, **args)

        fix_params(
            setting, scale=setting['fixed_scale'], shape=setting['fixed_shape'])

        print("linear solve, to do...")

        results = non_linear_solver(setting, data, **args)

        # save results
        save_results(setting, data, results, **args)

    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))

if __name__ == "__main__":

    sys.argv = ["", "--config=cfg_files/fit_smpl.yaml"]
    args = parse_config()
    main(**args)