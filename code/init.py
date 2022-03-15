'''
 @FileName    : init.py
 @EditTime    : 2021-09-19 21:48:33
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import os
import os.path as osp
import yaml
import torch
import sys
import numpy as np
from utils.data_parser import create_dataset
from utils.utils import JointMapper, load_camera_para, get_rot_trans
import smplx
from camera import create_camera
from prior import create_prior
from utils.prior import load_vposer
from vposer.vposer import VPoserDecoder
def init(**kwarg):

    setting = {}
    # create folder
    output_folder = kwarg.pop('output_folder')
    output_folder = osp.expandvars(output_folder)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    # Store the arguments for the current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(kwarg, conf_file)

    result_folder = kwarg.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    mesh_folder = kwarg.pop('mesh_folder', 'meshes')
    mesh_folder = osp.join(output_folder, mesh_folder)
    if not osp.exists(mesh_folder):
        os.makedirs(mesh_folder)

    out_img_folder = osp.join(output_folder, 'images')
    if not osp.exists(out_img_folder):
        os.makedirs(out_img_folder)

    # assert cuda is available
    use_cuda = kwarg.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    #read gender
    input_gender = kwarg.pop('gender', 'neutral')
    model_type = kwarg.get('model_type')
    if model_type == 'smpllsp':
        assert(input_gender=='neutral'), 'smpl-lsp model support neutral only'
    gender_lbl_type = kwarg.pop('gender_lbl_type', 'none')

    if model_type == 'smpllsp':
        # the hip joint of smpl is different with 2D annotation predicted by openpose/alphapose, so we use smpl-lsp model to replace
        pose_format = 'lsp14' 
    elif model_type == 'smpl':
        pose_format = 'coco17'
    elif model_type == 'smplx':
        pose_format = 'coco25'

    dataset_obj = create_dataset(pose_format=pose_format, **kwarg)

    float_dtype = kwarg.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    # map smpl joints to 2D keypoints
    joint_mapper = JointMapper(dataset_obj.get_model2data())

    model_params = dict(model_path=kwarg.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not kwarg.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=True, #set transl in multi-view task  --Buzhen Huang 07/31/2019
                        create_scale=False,
                        dtype=dtype,
                        **kwarg)

    model = smplx.create(gender=input_gender,**model_params) ## JointMapper 由 smpl 类自行映射
    # model = smplx.create_scale(gender=input_gender, **model_params) ## 多了 lsp 以及 scale 参数

    # load camera parameters
    cam_params = kwarg.pop('cam_param')
    extris, intris = load_camera_para(cam_params)
    trans, rot = get_rot_trans(extris, photoscan=False)


    # Create the camera object
    # create camera
    views = len(extris)
    camera = []
    for v in range(views):
        focal_length = float(intris[v][0][0])
        rotate = torch.tensor(rot[v],dtype=dtype).unsqueeze(0)
        translation = torch.tensor(trans[v],dtype=dtype).unsqueeze(0)
        center = torch.tensor(intris[v][:2,2],dtype=dtype).unsqueeze(0)
        camera_t = create_camera(focal_length_x=focal_length,
                            focal_length_y=focal_length,
                            translation=translation,
                            rotation=rotate,
                            center=center,
                            dtype=dtype,
                            **kwarg) ## 相机外参和内参投影
        camera.append(camera_t) # n_c

    # fix rotation and translation of camera
    for cam in camera: ## 不优化外参
        if hasattr(cam, 'rotation'):
            cam.rotation.requires_grad = False
        if hasattr(cam, 'translation'):
            cam.translation.requires_grad = False

    # create prior ## 定义损失函数
    body_pose_prior = create_prior(
        prior_type=kwarg.get('body_prior_type'),
        dtype=dtype,
        **kwarg)
    shape_prior = create_prior(
        prior_type=kwarg.get('shape_prior_type', 'l2'),
        dtype=dtype, **kwarg)
    angle_prior = create_prior(prior_type='angle', dtype=dtype) ## 轴角表示中每个维度和xyz轴转动角度有啥关系？

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')

        for cam in camera:
            cam = cam.to(device=device)
        model = model.to(device=device)
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
    else:
        device = torch.device('cpu')
    
    # A weight for every joint of the model ## ignore neck lhip rhip
    joint_weights = dataset_obj.get_joint_weights().to(device=device,
                                                       dtype=dtype)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)

    # load vposer
    vposer = None
    pose_embedding = None
    batch_size = 1
    # if kwarg.get('use_vposer'):
    #     vposer_ckpt = osp.expandvars(kwarg.get('prior_folder'))
    #     vposer = load_vposer(vposer_ckpt, vp_model='snapshot')
    #     vposer = vposer.to(device=device)
    #     vposer.eval()
    #     pose_embedding = torch.zeros([batch_size, 32],
    #                                  dtype=dtype, device=device,
    #                                  requires_grad=True)
    if kwarg.get('use_vposer'):
        vposer_ckpt = osp.expandvars(kwarg.get('prior_folder'))
        vposer = VPoserDecoder(vposer_ckpt=vposer_ckpt,latent_dim=32,dtype=dtype,**kwarg)
        vposer = vposer.to(device=device)
        vposer.eval()
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)

    # process fixed parameters
    setting['fix_scale'] = kwarg.get('fix_scale')
    if kwarg.get('fix_scale'):
        setting['fixed_scale'] = np.array(kwarg.get('scale'))
    else:
        setting['fixed_scale'] = None
    if kwarg.get('fix_shape'):
        setting['fixed_shape'] = np.array(kwarg.get('shape'))
    else:
        setting['fixed_shape'] = None

    model.reset_params()
    # return setting
    setting['use_3d'] = kwarg.pop("use_3d")
    setting['extrinsics'] = extris
    setting['intrinsics'] = intris
    setting['model'] = model
    setting['dtype'] = dtype
    setting['device'] = device
    setting['vposer'] = vposer
    setting['joints_weight'] = joint_weights
    setting['body_pose_prior'] = body_pose_prior
    setting['shape_prior'] = shape_prior
    setting['angle_prior'] = angle_prior
    setting['cameras'] = camera
    setting['img_folder'] = out_img_folder
    setting['result_folder'] = result_folder
    setting['mesh_folder'] = mesh_folder
    setting['pose_embedding'] = pose_embedding
    setting['batch_size'] = batch_size
    return dataset_obj, setting

