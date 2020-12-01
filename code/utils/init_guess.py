from utils.recompute3D import recompute3D
import torch
from utils.umeyama import umeyama
import cv2
from collections import defaultdict

def init_guess(setting, data, use_torso=False, **kwargs):
    model = setting['model']
    dtype = setting['dtype']
    keypoints = data['keypoints']
    batch_size = setting['batch_size']
    device = setting['device']
    est_scale = not setting['fix_scale']
    fixed_scale = 1. if setting['fixed_scale'] is None else setting['fixed_scale']
    joints3d = recompute3D(setting['extris'], setting['intris'], keypoints)
    if kwargs.get('use_3d'):
        joints3d = data['3d_joint'][0][:,:3]


    # reset model
    init_t = torch.zeros((1,3), dtype=dtype)
    init_r = torch.zeros((1,3), dtype=dtype)
    init_s = torch.tensor(fixed_scale, dtype=dtype)
    init_shape = torch.zeros((1,10), dtype=dtype)
    model.reset_params(transl=init_t, global_orient=init_r, scale=init_s, betas=init_shape)

    init_pose = torch.zeros((1,69), dtype=dtype).cuda()
    model_output = model(return_verts=True, return_full_pose=True, body_pose=init_pose)
    verts = model_output.vertices[0]
    if kwargs.get('model_type') == 'smpllsp':
        J = torch.matmul(model.joint_regressor, verts)
    else:
        J = torch.matmul(model.J_regressor, verts)

    verts = verts.unsqueeze(0)
    J = J.unsqueeze(0)
    joints = model.vertex_joint_selector(verts, J)
    # Map the joints to the current dataset
    if model.joint_mapper is not None:
        joints = model.joint_mapper(joints).detach().cpu().numpy()[0]
    if use_torso:
        joints3d = joints3d[[5,6,11,12]]
        joints = joints[[5,6,11,12]]
    # get transformation
    rot, trans, scale = umeyama(joints, joints3d, est_scale)
    rot = cv2.Rodrigues(rot)[0]
    # apply to model
    if est_scale:
        init_s = torch.tensor(scale, dtype=dtype)
    else:
        init_s = torch.tensor(fixed_scale, dtype=dtype)
    init_t = torch.tensor(trans, dtype=dtype)
    init_r = torch.tensor(rot, dtype=dtype).reshape(1,3)
    model.reset_params(transl=init_t, global_orient=init_r, scale=init_s)

    if kwargs.get('use_vposer'):
        with torch.no_grad():   
            setting['pose_embedding'].fill_(0)

    # # load fixed parameters
    # init_s = torch.tensor(7., dtype=dtype)
    # init_shape = torch.tensor([2.39806, 0.678491, -1.38193, -0.966748, -1.29383,-0.795755, -0.303195, -1.1032, -0.197056, -0.102728 ], dtype=dtype)
    # model.reset_params(transl=init_t, global_orient=init_r, scale=init_s)
    # model.betas.requires_grad = False
    # model.scale.requires_grad = False

    # # visualize
    # init_pose = torch.zeros((1,69), dtype=dtype).cuda()
    # model_output = model(return_verts=True, return_full_pose=True, body_pose=init_pose)
    # joints = model_output.joints.detach().cpu().numpy()[0]
    # verts = model_output.vertices.detach().cpu().numpy()[0]
    # import numpy as np
    # from test_code.projection import joint_projection, surface_projection
    # for i in range(6):
    #     joint_projection(joints3d, setting['extris'][i], setting['intris'][i], data['img'][i][:,:,::-1], True, op=0)
    #     surface_projection(verts, model.faces, joints, setting['extris'][i], setting['intris'][i], data['img'][i][:,:,::-1], 5)


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

    # # visualize
    # init_pose = torch.tensor(results['body_pose'], dtype=dtype).cuda()
    # model_output = model(return_verts=True, return_full_pose=True, body_pose=init_pose)
    # joints = model_output.joints.detach().cpu().numpy()[0]
    # verts = model_output.vertices.detach().cpu().numpy()[0]
    # import numpy as np
    # from test_code.projection import joint_projection, surface_projection
    # for i in range(6):
    #     # joint_projection(joints3d, setting['extris'][i], setting['intris'][i], data['img'][i][:,:,::-1], True)
    #     surface_projection(verts, model.faces, joints, setting['extris'][i], setting['intris'][i], data['img'][i][:,:,::-1], 0)

def fix_params(setting, scale=None, shape=None):
    dtype = setting['dtype']
    model = setting['model']
    init_t = model.transl
    init_r = model.global_orient
    init_s = model.scale
    init_shape = model.betas
    if scale is not None:
        init_s = torch.tensor(scale, dtype=dtype)
        model.scale.requires_grad = False
    if shape is not None:
        init_shape = torch.tensor(shape, dtype=dtype)
        model.betas.requires_grad = False
    model.reset_params(transl=init_t, global_orient=init_r, scale=init_s, betas=init_shape)
        