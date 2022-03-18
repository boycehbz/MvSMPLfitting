'''
 @FileName    : init_guess.py
 @EditTime    : 2021-09-19 21:47:56
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

from utils.recompute3D import recompute3D
import torch
import numpy as np
from utils.umeyama import umeyama
import cv2
from collections import defaultdict
from utils.utils import cal_trans

def guess_init(model,
               joints_2d,
               edge_idxs,
               focal_length=5000,
               pose_embedding=None,
               vposer=None,
               use_vposer=True,
               dtype=torch.float32,
               model_type='smpl',
               **kwargs):
    ''' Initializes the camera translation vector

        Parameters
        ----------
        model: nn.Module
            The PyTorch module of the body
        joints_2d: torch.tensor 1xJx2
            The 2D tensor of the joints
        edge_idxs: list of lists
            A list of pairs, each of which represents a limb used to estimate
            the camera translation
        focal_length: float, optional (default = 5000)
            The focal length of the camera
        pose_embedding: torch.tensor 1x32
            The tensor that contains the embedding of V-Poser that is used to
            generate the pose of the model
        dtype: torch.dtype, optional (torch.float32)
            The floating point type used
        vposer: nn.Module, optional (None)
            The PyTorch module that implements the V-Poser decoder
        Returns
        -------
        init_t: torch.tensor 1x3, dtype = torch.float32
            The vector with the estimated camera location

    '''
    
    # body_pose = vposer.decode(
    #     pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
    body_pose = vposer.forward(
        pose_embedding).view(1,-1) if use_vposer else None
    if use_vposer and model_type == 'smpl':
        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                 dtype=body_pose.dtype,
                                 device=body_pose.device)
        body_pose = torch.cat([body_pose, wrist_pose], dim=1)
    model.reset_params()
    output = model(body_pose=body_pose, return_verts=False,
                   return_full_pose=False)
    joints_3d = output.joints
    joints_2d = joints_2d.to(device=joints_3d.device)

    diff3d = []
    diff2d = []
    for edge in edge_idxs:
        diff3d.append(joints_3d[:, edge[0]] - joints_3d[:, edge[1]])
        diff2d.append(joints_2d[:, edge[0]] - joints_2d[:, edge[1]])

    diff3d = torch.stack(diff3d, dim=1)
    diff2d = torch.stack(diff2d, dim=1)

    length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
    length_3d = diff3d.pow(2).sum(dim=-1).sqrt()

    height2d = length_2d.mean(dim=1)
    height3d = length_3d.mean(dim=1)

    est_d = focal_length * (height3d / height2d)

    # just set the z value
    batch_size = joints_3d.shape[0]
    x_coord = torch.zeros([batch_size], device=joints_3d.device,
                          dtype=dtype)
    y_coord = x_coord.clone()
    init_t = torch.stack([x_coord, y_coord, est_d], dim=1)
    return init_t

def init_guess(setting, data, use_torso=False, **kwargs):
    model = setting['model']
    dtype = setting['dtype']
    keypoints = data['keypoints']
    batch_size = setting['batch_size']
    device = setting['device']
    est_scale = not setting['fix_scale']
    fixed_scale = 1. if setting['fixed_scale'] is None else setting['fixed_scale']

    # reset model
    init_t = torch.zeros((1,3), dtype=dtype)
    init_r = torch.zeros((1,3), dtype=dtype)
    init_s = torch.tensor(fixed_scale, dtype=dtype)
    init_shape = torch.zeros((1,10), dtype=dtype)
    # model.reset_params(transl=init_t, global_orient=init_r, scale=init_s, betas=init_shape)
    model.reset_params(transl=init_t, global_orient=init_r, betas=init_shape)

    init_pose = torch.zeros((1,63), dtype=dtype).cuda()
    model_output = model(return_verts=True, return_full_pose=True, body_pose=init_pose) ## joints有lsp+face组成
    verts = model_output.vertices[0]
    joints = model_output.joints[0].detach().cpu().numpy()

    if len(keypoints) == 1:
        # guess depth for single-view input
        ## 序号是lsp那里对应的
        # 5 is L shoulder, 11 is L hip
        # 6 is R shoulder, 12 is R hip
        ## 此处序号是openpose25对应的
        # 5 is L shoulder, 12 is L hip
        # 2 is R shoulder, 9 is R hip
        torso3d = joints[[5,2,12,9]]
        torso2d = keypoints[0][0][[5,2,12,9]]
        torso3d = np.insert(torso3d, 3, 1, axis=1).T
        torso3d = (np.dot(setting['extris'][0], torso3d).T)[:,:3]

        diff3d = np.array([torso3d[0] - torso3d[2], torso3d[1] - torso3d[3]])
        mean_height3d = np.mean(np.sqrt(np.sum(diff3d**2, axis=1)))

        diff2d = np.array([torso2d[0] - torso2d[2], torso2d[0] - torso2d[2]])
        mean_height2d = np.mean(np.sqrt(np.sum(diff2d**2, axis=1)))

        est_d = setting['intris'][0][0][0] * (mean_height3d / mean_height2d)
        # just set the z value
        cam_joints = np.dot(setting['extris'][0], np.insert(joints.copy(), 3, 1, axis=1).T)
        cam_joints[2,:] += est_d
        joints3d = (np.dot(np.linalg.inv(setting['extris'][0]), cam_joints).T)[:,:3]

        # trans = cal_trans(camcoord, keypoints[0][0][[5,6,11,12]], setting['intris'][0])
        # trans = np.dot(np.linalg.inv(setting['extris'][0]), np.insert(trans.reshape(3,1), 3, 1, axis=0)).reshape(1,-1)[:,:3]
        # joints3d = joints + trans
    else:
        joints3d = recompute3D(setting['extris'], setting['intris'], keypoints)
    if kwargs.get('use_3d') and data['3d_joint'] is not None:
        joints3d = data['3d_joint'][0][:,:3]

    if use_torso:
        joints3d = joints3d[[5,2,12,9]]
        joints = joints[[5,2,12,9]]
    # get transformation
    rot, trans, scale = umeyama(joints, joints3d, est_scale) ## 根据2d-3d估计的joints 估计初始旋转和平移
    rot = cv2.Rodrigues(rot)[0]
    # apply to model
    if est_scale:
        init_s = torch.tensor(scale, dtype=dtype)
    else:
        init_s = torch.tensor(fixed_scale, dtype=dtype)
    init_t = torch.tensor(trans, dtype=dtype)
    init_r = torch.tensor(rot, dtype=dtype).reshape(1,3)
    # model.reset_params(transl=init_t, global_orient=init_r, scale=init_s)
    model.reset_params(transl=init_t, global_orient=init_r)
    if kwargs.get('use_vposer'):
        with torch.no_grad():   
            setting['pose_embedding'].fill_(0)

    # # load fixed parameters
    # init_s = torch.tensor(7., dtype=dtype)
    # init_shape = torch.tensor([2.39806, 0.678491, -1.38193, -0.966748, -1.29383,-0.795755, -0.303195, -1.1032, -0.197056, -0.102728 ], dtype=dtype)
    # model.reset_params(transl=init_t, global_orient=init_r, scale=init_s)
    # model.betas.requires_grad = False
    # model.scale.requires_grad = False

    # visualize
    if False:
        if kwargs.get('use_vposer'):
            vposer = setting['vposer']
            init_pose = vposer.decode(
                setting['pose_embedding'], output_type='aa').view(
                    1, -1)
        else:
            init_pose = torch.zeros((1,69), dtype=dtype).cuda()
        model_output = model(return_verts=True, return_full_pose=True, body_pose=init_pose)
        joints = model_output.joints.detach().cpu().numpy()[0]
        verts = model_output.vertices.detach().cpu().numpy()[0]

        from utils.utils import joint_projection, surface_projection
        for i in range(1):
            joint_projection(joints3d, setting['extris'][i], setting['intris'][i], data['img'][i][:,:,::-1], True)
            surface_projection(verts, model.faces, joints, setting['extris'][i], setting['intris'][i], data['img'][i][:,:,::-1], 5)

def init_guess_nonlinear(setting, data, use_torso=False, **kwargs):
    gt_joints = torch.tensor(data['keypoints'][0][:,:,:2]).to(device=setting['device'],dtype=setting['dtype'])
    body_model = setting['model']
    init_t = guess_init(
        body_model,gt_joints,[[5,12],[2,9]],
        use_vposer=kwargs['use_vposer'], vposer=setting['vposer'],
        pose_embedding=setting['pose_embedding'],
        model_type=kwargs['model_type'],
        focal_length=setting['camera'][0].focal_length_x, dtype=setting['dtype'])
    from utils import fitting
    ## 补充非线性 全局变换优化
    camera_loss = fitting.create_loss(
        'camera_init',
        trans_estimation=init_t,
        init_joints_idxs=torch.tensor([9,12,2,5]).to(device=setting['device']),
        depth_loss_weight=0.0,
        camera_mode='fixed',
        dtype=setting['dtype']).to(device=setting['device'])
    camera_loss.trans_estimation[:] = init_t
    monitor = fitting.FittingMonitor(batch_size=1, **kwargs)
    body_mean_pose = torch.zeros([1, 32],
    dtype=setting['dtype'])
    body_model.reset_params(body_pose=body_mean_pose, transl=init_t)
    camera_opt_params = [body_model.transl, body_model.global_orient]
    from optimizers import optim_factory
    camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
        camera_opt_params,
        **kwargs)
    fit_camera = monitor.create_fitting_closure(
        camera_optimizer, body_model, setting['camera'][0], gt_joints,
        camera_loss, create_graph=camera_create_graph,
        use_vposer=kwargs['use_vposer'], vposer=setting['vposer'],
        pose_embedding=setting['pose_embedding'],
        scan_tensor=None,
        return_full_pose=False, return_verts=False)
    cam_init_loss_val = monitor.run_fitting(
        camera_optimizer,
        fit_camera,
        camera_opt_params, body_model,
        use_vposer=kwargs['use_vposer'],
        pose_embedding=setting['pose_embedding'],
        vposer=setting['vposer'])
    orient = body_model.global_orient.detach().cpu().numpy()
    body_transl = body_model.transl.clone().detach()
    from collections import defaultdict
    new_params = defaultdict(
        transl=body_transl,
        global_orient=orient,
        body_pose=body_mean_pose)
    body_model.reset_params(**new_params)
    if kwargs.get('use_vposer'):
        with torch.no_grad():   
            setting['pose_embedding'].fill_(0)

def load_init(setting, data, results, use_torso=False, **kwargs):
    model = setting['model']
    dtype = setting['dtype']
    device = setting['device']
    # if the loss of last frame is too large, we use init_guess to get initial value
    if results['loss'] > 5000:
        init_guess(setting, data, use_torso=use_torso, **kwargs)
        setting['seq_start'] = True
        return

    init_t = torch.tensor(results['transl'], dtype=dtype)
    init_r = torch.tensor(results['global_orient'], dtype=dtype)
    init_s = torch.tensor(results['scale'], dtype=dtype)
    init_shape = torch.tensor(results['betas'], dtype=dtype)
    if kwargs.get('use_vposer'):
        setting['pose_embedding'] = torch.tensor(results['pose_embedding'], dtype=dtype, device=device, requires_grad=True)
    else:
        # gmm prior, to do...
        pass
        #init_pose = torch.tensor(results['body_pose'], dtype=dtype)

    # initial value
    new_params = defaultdict(global_orient=init_r,
                                # body_pose=body_mean_pose,
                                transl=init_t,
                                scale=init_s,
                                betas=init_shape,
                                )
    model.reset_params(**new_params)

    # visualize
    if False:
        if kwargs.get('use_vposer'):
            vposer = setting['vposer']
            init_pose = vposer.decode(
                setting['pose_embedding'], output_type='aa').view(
                    1, -1)
        else:
            init_pose = torch.zeros((1,69), dtype=dtype).cuda()
        model_output = model(return_verts=True, return_full_pose=True, body_pose=init_pose)
        joints = model_output.joints.detach().cpu().numpy()[0]
        verts = model_output.vertices.detach().cpu().numpy()[0]

        from utils.utils import joint_projection, surface_projection
        for i in range(1):
            # joint_projection(joints3d, setting['extris'][i], setting['intris'][i], data['img'][i][:,:,::-1], True)
            surface_projection(verts, model.faces, joints, setting['extris'][i], setting['intris'][i], data['img'][i][:,:,::-1], 5)

def fix_params(setting, scale=None, shape=None):
    dtype = setting['dtype']
    model = setting['model']
    init_t = model.transl
    init_r = model.global_orient
    # init_s = model.scale
    init_shape = model.betas
    # if scale is not None:
    #     init_s = torch.tensor(scale, dtype=dtype)
    #     model.scale.requires_grad = False
    if shape is not None:
        init_shape = torch.tensor(shape, dtype=dtype)
        model.betas.requires_grad = False
    model.reset_params(transl=init_t, global_orient=init_r, betas=init_shape)    
    # model.reset_params(transl=init_t, global_orient=init_r, scale=init_s, betas=init_shape)
        