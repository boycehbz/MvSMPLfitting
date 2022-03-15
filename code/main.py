# -*- coding: utf-8 -*-
'''
 @FileName    : main.py
 @EditTime    : 2021-09-19 21:46:57
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import sys
import os

import os.path as osp

import time
import torch
import numpy as np
from cmd_parser import parse_config
from init import init
from utils.init_guess import init_guess, load_init, fix_params, guess_init
from utils.non_linear_solver import non_linear_solver
from utils.utils import save_results
def main(**args):

    dataset_obj, setting = init(**args)

    start = time.time()

    results = {}
    s_last = None # the name of last sequence
    setting['seq_start'] = False # indicate the first frame of the sequence
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
        print('Processing: {}'.format(data['img_path']))

        # init guess
        # if setting['seq_start'] or not args.get('is_seq'):
        #     init_guess(setting, data, use_torso=True, **args) ## 根据2djoints->3djoints 估计初始旋转和平移
        # else:
        #     load_init(setting, data, results, use_torso=True, **args)
        gt_joints = torch.tensor(data['keypoints'][0][:,:,:2]).to(device=setting['device'],dtype=setting['dtype'])
        body_model = setting['model']
        init_t = guess_init(
            body_model,gt_joints,[[5,12],[2,9]],
            use_vposer=args['use_vposer'], vposer=setting['vposer'],
            pose_embedding=setting['pose_embedding'],
            model_type=args['model_type'],
            focal_length=setting['camera'][0].focal_length_x, dtype=setting['dtype'])

        from utils import fitting
        ## 补充非线性 全局变换优化
        camera_loss = fitting.create_loss('camera_init',
                                        trans_estimation=init_t,
                                        init_joints_idxs=torch.tensor([9,12,2,5]).to(device=setting['device']),
                                        depth_loss_weight=0.0,
                                        camera_mode='fixed',
                                        dtype=setting['dtype']).to(device=setting['device'])
        camera_loss.trans_estimation[:] = init_t
        monitor = fitting.FittingMonitor(batch_size=1, **args)
        body_mean_pose = torch.zeros([1, 32],
                                     dtype=setting['dtype'])
        body_model.reset_params(body_pose=body_mean_pose, transl=init_t)
        camera_opt_params = [body_model.transl, body_model.global_orient]
        from optimizers import optim_factory
        camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
            camera_opt_params,
            **args)
        fit_camera = monitor.create_fitting_closure(
            camera_optimizer, body_model, setting['camera'][0], gt_joints,
            camera_loss, create_graph=camera_create_graph,
            use_vposer=args['use_vposer'], vposer=setting['vposer'],
            pose_embedding=setting['pose_embedding'],
            scan_tensor=None,
            return_full_pose=False, return_verts=False)
        cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                fit_camera,
                                                camera_opt_params, body_model,
                                                use_vposer=args['use_vposer'],
                                                pose_embedding=setting['pose_embedding'],
                                                vposer=setting['vposer'])
        orient = body_model.global_orient.detach().cpu().numpy()
        body_transl = body_model.transl.clone().detach()
        from collections import defaultdict
        new_params = defaultdict(transl=body_transl,
                                     global_orient=orient,
                                     body_pose=body_mean_pose)
        body_model.reset_params(**new_params)
        if args.get('use_vposer'):
            with torch.no_grad():   
                setting['pose_embedding'].fill_(0)

        fix_params(setting, scale=setting['fixed_scale'], shape=setting['fixed_shape']) ## 设置第一步初始化的全局旋转平移，选择是否优化scale和shape
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

def transProx2coco(js):
    idx = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]
    return js[idx]

if __name__ == "__main__":

    sys.argv = ["", "--config=cfg_files/fit_smpl_test.yaml"]
    args = parse_config()
    main(**args)