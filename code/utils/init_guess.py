from utils.recompute3D import recompute3D
import torch
from utils.umeyama import umeyama
import cv2

def init_guess(setting, data, use_torso=False, **kwargs):
    model = setting['model']
    dtype = setting['dtype']
    keypoints = data['keypoints']
    joints3d = recompute3D(setting['extris'], setting['intris'], keypoints)

    if kwargs.get('model_type') == 'smpllsp':
        J = torch.matmul(model.joint_regressor, model.v_template)
    else:
        J = torch.matmul(model.J_regressor, model.v_template)
    verts = model.v_template.unsqueeze(0)
    J = J.unsqueeze(0)
    joints = model.vertex_joint_selector(verts, J)
    # Map the joints to the current dataset
    if model.joint_mapper is not None:
        joints = model.joint_mapper(joints).detach().cpu().numpy()[0]
    if use_torso:
        joints3d = joints3d[[5,6,11,12]]
        joints = joints[[5,6,11,12]]
    # get transformation
    rot, trans, scale = umeyama(joints, joints3d, True)
    rot = cv2.Rodrigues(rot)[0]
    # apply to model
    init_t = torch.tensor(trans, dtype=dtype)
    init_s = torch.tensor(scale, dtype=dtype)
    init_r = torch.tensor(rot, dtype=dtype).reshape(1,3)
    model.reset_params(transl=init_t, global_orient=init_r, scale=init_s)

    # # visualize
    # init_pose = torch.zeros((1,69), dtype=dtype).cuda()
    # model_output = model(return_verts=True, return_full_pose=True, body_pose=init_pose)
    # joints = model_output.joints.detach().cpu().numpy()[0]
    # verts = model_output.vertices.detach().cpu().numpy()[0]
    # import numpy as np
    # from test_code.projection import joint_projection, surface_projection
    # for i in range(6):
    #     joint_projection(joints3d, setting['extris'][i], setting['intris'][i], data['img'][i][:,:,::-1], True)
    #     surface_projection(verts, model.faces, joints, setting['extris'][i], setting['intris'][i], data['img'][i][:,:,::-1], 0)