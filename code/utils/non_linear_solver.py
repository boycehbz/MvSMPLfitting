'''
 @FileName    : non_linear_solver.py
 @EditTime    : 2021-09-19 21:48:01
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

import cv2
import PIL.Image as pil_img

from optimizers import optim_factory

from utils import fitting

def non_linear_solver(
                    setting,
                    data,
                    batch_size=1,
                    data_weights=None,
                    body_pose_prior_weights=None,
                    shape_weights=None,
                    coll_loss_weights=None,
                    use_joints_conf=False,
                    use_3d=False,
                    rho=100,
                    interpenetration=False,
                    loss_type='smplify',
                    visualize=False,
                    use_vposer=True,
                    interactive=True,
                    use_cuda=True,
                    is_seq=False,
                    **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    views = setting['views']
    device = setting['device']
    dtype = setting['dtype']
    vposer = setting['vposer']
    keypoints = data['keypoints']
    joint_weights = setting['joints_weight']
    model = setting['model']
    camera = setting['camera']
    pose_embedding = setting['pose_embedding']
    seq_start = setting['seq_start']
    if data['3d_joint'] is None:
        use_3d = False

    assert (len(data_weights) ==
            len(body_pose_prior_weights) and len(shape_weights) ==
            len(body_pose_prior_weights) and len(coll_loss_weights) ==
            len(body_pose_prior_weights)), "Number of weight must match"
    
    # process keypoints
    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :, :2]
    if use_joints_conf:
        joints_conf = []
        for v in keypoint_data:
            conf = v[:, :, 2].reshape(1, -1)
            conf = conf.to(device=device, dtype=dtype)
            joints_conf.append(conf)

    if use_3d: #:
        joints3d = data['3d_joint'][0]
        joints_data = torch.tensor(joints3d, dtype=dtype)
        gt_joints3d = joints_data[:, :3]
        if use_joints_conf:
            joints3d_conf = joints_data[:, 3].reshape(1, -1).to(device=device, dtype=dtype)
            if not kwargs.get('use_hip'):
                joints3d_conf[0][11] = 0
                joints3d_conf[0][12] = 0

        gt_joints3d = gt_joints3d.to(device=device, dtype=dtype)
    else:
        gt_joints3d = None
        joints3d_conf = None
    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights

    # get weights for each stage
    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # create fitting loss
    loss = fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=setting['body_pose_prior'],
                               shape_prior=setting['shape_prior'],
                               angle_prior=setting['angle_prior'],
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               dtype=dtype,
                               use_3d=use_3d,
                               **kwargs)
    loss = loss.to(device=device)

    monitor = fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs)

    H, W, _ = data['img'][0].shape

    data_weight = 500 / H

    # Step 1: Optimize the full model
    final_loss_val = 0
    opt_start = time.time()

    for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
        # pass stage1 and stage2 if it is a sequence
        if not seq_start and is_seq:
            if opt_idx < 2:
                continue
            elif opt_idx == 2:
                curr_weights['body_pose_weight'] *= 0.15

        body_params = list(model.parameters())

        final_params = list(
            filter(lambda x: x.requires_grad, body_params))

        if vposer is not None:
            final_params.append(pose_embedding)

        body_optimizer, body_create_graph = optim_factory.create_optimizer(
            final_params,
            **kwargs)
        body_optimizer.zero_grad()

        curr_weights['data_weight'] = data_weight
        curr_weights['bending_prior_weight'] = (
            3.17 * curr_weights['body_pose_weight'])
        loss.reset_loss_weights(curr_weights)

        closure = monitor.create_fitting_closure(
            body_optimizer, model,
            camera=camera, gt_joints=gt_joints,
            joints_conf=joints_conf,
            gt_joints3d=gt_joints3d,
            joints3d_conf=joints3d_conf,
            joint_weights=joint_weights,
            loss=loss, create_graph=body_create_graph,
            use_vposer=use_vposer, vposer=vposer,
            pose_embedding=pose_embedding,
            return_verts=True, return_full_pose=True, use_3d=use_3d)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            stage_start = time.time()
        final_loss_val = monitor.run_fitting(
            body_optimizer,
            closure, final_params,
            model,
            pose_embedding=pose_embedding, vposer=vposer, camera=camera, img_path=data['img_path'],
            use_vposer=use_vposer)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - stage_start
            if interactive:
                tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                    opt_idx, elapsed))

    if setting['adjustment']:
        ## 1. 每个视角的 3D投影 和 2D关键点 可视化，调整2D关键点
        ## 2. 根据新的2D关键点 再次优化一轮
        ## 3. 调整 3D优化结果，增加reset选项
        result = {key: val.detach().cpu().numpy()
                for key, val in model.named_parameters()}
        result['pose_embedding'] = pose_embedding
        from utils.utils import changeNew
        changeNew(data['img_path'], data['keypoints'], result, setting)
        keypoints = data['keypoints']
        keypoint_data = torch.tensor(keypoints, dtype=dtype)
        gt_joints = keypoint_data[:, :, :, :2]
        gt_joints = gt_joints.to(device=device, dtype=dtype)

        opt_idx += 1
        curr_weights = opt_weights[-1]
        body_params = list(model.parameters())
        final_params = list(
            filter(lambda x: x.requires_grad, body_params))
        if vposer is not None:
            final_params.append(pose_embedding)
        body_optimizer, body_create_graph = optim_factory.create_optimizer(
            final_params,
            **kwargs)
        body_optimizer.zero_grad()
        curr_weights['data_weight'] = data_weight
        curr_weights['bending_prior_weight'] = (
            3.17 * curr_weights['body_pose_weight'])
        loss.reset_loss_weights(curr_weights)

        closure = monitor.create_fitting_closure(
            body_optimizer, model,
            camera=camera, gt_joints=gt_joints,
            joints_conf=joints_conf,
            gt_joints3d=gt_joints3d,
            joints3d_conf=joints3d_conf,
            joint_weights=joint_weights,
            loss=loss, create_graph=body_create_graph,
            use_vposer=use_vposer, vposer=vposer,
            pose_embedding=pose_embedding,
            return_verts=True, return_full_pose=True, use_3d=use_3d)
        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            stage_start = time.time()
        final_loss_val = monitor.run_fitting(
            body_optimizer,
            closure, final_params,
            model,
            pose_embedding=pose_embedding, vposer=vposer, camera=camera, img_path=data['img_path'],
            use_vposer=use_vposer)
        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - stage_start
            if interactive:
                tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                    opt_idx, elapsed))



    if interactive:
        if use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - opt_start
        tqdm.write(
            'Body fitting done after {:.4f} seconds'.format(elapsed))
        tqdm.write('Body final loss val = {:.5f}'.format(
            final_loss_val))

        # Get the result of the fitting process
        result = {key: val.detach().cpu().numpy()
                        for key, val in model.named_parameters()}
        result['loss'] = final_loss_val
        result['pose_embedding'] = pose_embedding
    return result
