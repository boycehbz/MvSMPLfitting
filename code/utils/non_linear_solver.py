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
                    hand_pose_prior_weights=None,
                    jaw_pose_prior_weights=None,
                    face_joints_weights=None,
                    hand_joints_weights=None,
                    expr_weights=None,
                    use_face=True,
                    use_hands=True,
                    use_contact=True,
                    use_foot_contact=True,
                    shape_weights=None,
                    contact_loss_weights=None,
                    foot_contact_loss_weights=None,
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

    if use_3d: ## 估计的3djoint只用于估计初始旋转和平移
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

    # Weights used for the pose prior and the shape prior ## 四轮约束吗？
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    if use_face:
        jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')), jaw_pose_prior_weights)
        jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if use_contact:
        opt_weights_dict['contact_loss_weight'] = contact_loss_weights   
    if use_foot_contact:
        opt_weights_dict['foot_contact_loss_weights'] = foot_contact_loss_weights
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
                               left_hand_prior=setting['left_hand_prior'],
                               right_hand_prior=setting['left_hand_prior'],
                               expr_prior=setting['expr_prior'],
                               jaw_prior=setting['jaw_prior'],
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               dtype=dtype,
                               use_3d=use_3d,
                               body_model=model,
                               use_cuda=use_cuda,
                               use_contact=use_contact,
                               use_foot_contact=use_foot_contact,
                               **kwargs)
    loss = loss.to(device=device)

    monitor = fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs)
    # with fitting.FittingMonitor(
    #         batch_size=batch_size, visualize=visualize, **kwargs) as monitor:

    H, W, _ = data['img'][0].shape

    data_weight = 500 / H

    # Reset the parameters to estimate the initial translation of the
    # body model
    # body_model.reset_params(body_pose=body_mean_pose, transl=init['init_t'], global_orient=init['init_r'], scale=init['init_s'], betas=init['init_betas'])

    # we do not change rotation in multi-view task
    orientations = [model.global_orient]

    # store here the final error for both orientations,
    # and pick the orientation resulting in the lowest error
    results = []

    # Step 1: Optimize the full model
    final_loss_val = 0
    opt_start = time.time()

    vis=None
    visFlag=False
    if visFlag:
        import open3d as o3d
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        body_pose = vposer.forward(
                    pose_embedding).view(1,-1) if use_vposer else None
        model_output = model(
                    return_verts=True, body_pose=body_pose)
                # vertices = model_output.vertices.detach()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
                    model_output.vertices.detach().cpu().numpy()[0]
                )
        mesh.triangles = o3d.utility.Vector3iVector(
                    model.faces
                )
        mesh.compute_vertex_normals()
        sceneVis = o3d.io.read_triangle_mesh(kwargs.get('scene'))
        sceneVis.compute_vertex_normals()
        vis.add_geometry(mesh)
        vis.add_geometry(sceneVis)

    for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
        # pass stage1 and stage2 if it is a sequence
        if not seq_start and is_seq:
            if opt_idx < 2:
                continue
            elif opt_idx == 2:
                curr_weights['body_pose_weight'] *= 0.15

        body_params = list(model.parameters()) # shape + t + r + s

        # print("--------------------------------")
        # for name,param in model.named_parameters():
        #     print(name)
        #     print(param.shape)
        #     print(param.requires_grad)
        #     print("---------------------------------")

        final_params = list(
            filter(lambda x: x.requires_grad, body_params))

        if vposer is not None:
            final_params.append(pose_embedding) # shape + t + r + s + vp

        body_optimizer, body_create_graph = optim_factory.create_optimizer(
            final_params,
            **kwargs)
        body_optimizer.zero_grad()

        curr_weights['data_weight'] = data_weight
        curr_weights['bending_prior_weight'] = (
            3.17 * curr_weights['body_pose_weight'])
        if use_hands:
            joint_weights[:, 25:76] = curr_weights['hand_weight']
        if use_face:
            joint_weights[:, 76:] = curr_weights['face_weight']

        
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
        # if curr_weights['contact_loss_weight'] < 1.0:
        if opt_idx < 4:
            final_loss_val = monitor.run_fitting(
                body_optimizer,
                closure, final_params,
                model,
                pose_embedding=pose_embedding, vposer=vposer, camera=camera, img_path=data['img_path'],
                use_vposer=use_vposer,
                visflag=False,scene=kwargs.get('scene'),viser=vis)
        else:
            final_loss_val = monitor.run_fitting(
                body_optimizer,
                closure, final_params,
                model,
                pose_embedding=pose_embedding, vposer=vposer, camera=camera, img_path=data['img_path'],
                use_vposer=use_vposer,
                visflag=visFlag,scene=kwargs.get('scene'),viser=vis)

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
