# -*- coding: utf-8 -*-
'''
 @FileName    : utils.py
 @EditTime    : 2021-09-19 21:47:05
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description :
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from operator import index

import numpy as np

import torch
import torch.nn as nn
import os.path as osp
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import cv2
from copy import deepcopy
import sys
import pyrender
from pyrender.constants import RenderFlags
import trimesh

vis_count = 0
r = 10
a_index = -1
points_index = -1
joints_index = -1
draw_color = True
points_to = []
ix = 0
iy = 0
draw_begin = False
draw_move = False
coordinate_1 = []
points_index_list = []
mouse_list = []
flag_change_value = False
num = 0
status = -1
key = -1
keypoints = []
keypoint = []
d2j_points_index = -1
imgs = []
img_dir = []
keyAxis = 0
shapeDim = 0
gloablOrientAxis = 0
gloablTranAxis = 0
gimg = {}

def change(image_path, keyps):
    global imgs, img_dir, mouse_list
    for v in range(len(image_path)):
        img_dir = image_path[v]
        if sys.platform == 'linux':
            view = img_dir.split('/')[-2]
        else:
            view = img_dir.split('\\')[-2]
        global keypoints,keypoint
        while True:
            imgs = cv2.imread(img_dir)

            ratiox = 800/int(imgs.shape[0])
            ratioy = 800/int(imgs.shape[1])
            if ratiox < ratioy:
                ratio = ratiox
            else:
                ratio = ratioy

            keypoints = keyps
            keypoint = keypoints[v]
            for p in keypoints[v][0]:
                cv2.circle(imgs, (int(p[0]), int(p[1])), 3, (0, 255, 0), 10)
            if draw_move == True:
                cv2.circle(imgs, (int(ix), int(iy)), 3, (0, 0, 255), 10)
            cv2.namedWindow(img_dir, 0)
            cv2.resizeWindow(img_dir, int(
                imgs.shape[1]*ratio), int(imgs.shape[0]*ratio))
            cv2.imshow(img_dir, imgs)
            cv2.setMouseCallback(img_dir, points_move)
            for number in range(len(mouse_list)):
                keypoints[v][0][points_index_list[number]
                                ][0:2] = mouse_list[number]  # 保存移动后的点
            key = cv2.waitKey(1)
            if key == 27:
                break
        mouse_list = []
        cv2.destroyAllWindows()

def result2mesh(result,setting,use_vposer=True):
    vposer = setting['vposer']
    model = setting['model']
    if use_vposer:
        pose_embedding = result['pose_embedding']
        body_pose = vposer.decode(
            pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
        if True:
            body_pose[:, 18:24] = 0.
            body_pose[:, 27:33] = 0.
            body_pose[:, 57:] = 0.
        orient = np.array(model.global_orient.detach().cpu().numpy())
        temp_pose = body_pose.detach().cpu().numpy()
        pose = np.hstack((orient, temp_pose))
    else:
        if True:
            result['body_pose'][:, 18:24] = 0.
            result['body_pose'][:, 27:33] = 0.
            result['body_pose'][:, 57:] = 0.
        pose = np.hstack((result['global_orient'], result['body_pose']))
    model_output = model(
        global_orient=torch.tensor(pose[:,:3],device=setting['device']), 
        transl=torch.tensor(result['transl'],device=setting['device']),
        return_verts=True, 
        body_pose=torch.tensor(pose[:,3:],device=setting['device']), 
        betas=torch.tensor(result['betas'],device=setting['device']))
    body_joints = model_output.joints
    verts = model_output.vertices
    return verts, body_joints, model.faces

def changeNew(image_path, keyps, results, setting):
    global imgs, img_dir, mouse_list
    verts, joints, faces = result2mesh(results,setting,setting['use_vposer'])
    verts = verts.squeeze().detach().cpu().numpy()
    for v in range(len(image_path)):
        img_dir = image_path[v]
        if sys.platform == 'linux':
            view = img_dir.split('/')[-2]
        else:
            view = img_dir.split('\\')[-2]
        global keypoints,keypoint

        imgs = cv2.imread(img_dir)
        ratiox = 800/int(imgs.shape[0])
        ratioy = 800/int(imgs.shape[1])
        if ratiox < ratioy:
            ratio = ratiox
        else:
            ratio = ratioy
        cv2.namedWindow(img_dir, 0)
        cv2.resizeWindow(img_dir, int(
            imgs.shape[1]*ratio), int(imgs.shape[0]*ratio))
        cv2.setMouseCallback(img_dir, points_move)

        render = Renderer((imgs.shape[1],imgs.shape[0]))
        cam = setting['cameras'][v]
        camIn = [
                [cam.focal_length_x.detach().squeeze().cpu().numpy(),0,cam.center[0][0].detach().squeeze().cpu().numpy()],
                [0,cam.focal_length_y.detach().squeeze().cpu().numpy(),cam.center[0][1].detach().squeeze().cpu().numpy()],
                [0,0,1]]
        imgs = render(verts,faces,cam.rotation.detach().cpu().squeeze().numpy(),cam.translation.detach().cpu().squeeze().numpy(),camIn,imgs,viz=False)
        joints2D = cam(joints).detach().cpu().numpy().astype(np.int32)
        for p in joints2D[0]:
            imgs = cv2.circle(
                imgs, (int(p[0]), int(p[1])), 3, (0, 0, 255), 10)
        imgs = imgs.astype(np.uint8)

        while True:
            imgscopy = imgs.copy()
            keypoints = keyps
            keypoint = keypoints[v]
            for p in keypoints[v][0]:
                imgscopy = cv2.circle(imgscopy, (int(p[0]), int(p[1])), 3, (0, 255, 0), 10)
            if draw_move == True:
                imgscopy = cv2.circle(imgscopy, (int(ix), int(iy)), 3, (255, 0, 0), 10)

            cv2.imshow(img_dir, imgscopy)
            
            for number in range(len(mouse_list)):
                keypoints[v][0][points_index_list[number]
                                ][0:2] = mouse_list[number]  # 保存移动后的点
            key = cv2.waitKey(1)
            if key == 27:
                break
        mouse_list = []
        cv2.destroyAllWindows()

def points_move(event, x, y, flags, param):
    global draw_begin, ix, iy, draw_move, points_index_list, mouse_list, keypoints, points_index, imgs, keypoint
    ix = x
    iy = y
    if event == cv2.EVENT_LBUTTONDOWN:
        points_index = WhichSelected(keypoint[0,:,:2], x, y)  # 确定选择的点坐标
        if points_index != -1:
            draw_begin = True
    if draw_begin == True:
        if event == cv2.EVENT_MOUSEMOVE:
            draw_move = True
    if draw_move == True:
        if event == cv2.EVENT_LBUTTONUP:  # 左键弹起，点已经移动好
            draw_begin = False
            draw_move = False
            print("after move the points is:", [ix, iy])
            points_index_list.append(points_index)
            mouse_list.append([ix, iy])


def click_event(event, x, y, flags, param):
    global status, flag_change_value, num
    global d2j_points_index, ix, iy
    ix = x
    iy = y

    if event == cv2.EVENT_LBUTTONDOWN:
        status = -1
        num = num+1
        d2j_points_index = WhichSelected(param, x, y)
        if num % 2 == 1:
            flag_change_value = True
        else:
            flag_change_value = False

def estimate_translation_from_intri(S, joints_2d, joints_conf, fx=5000., fy=5000., cx=128., cy=128.):
    num_joints = S.shape[0]
    # focal length
    f = np.array([fx, fy])
    # optical center
   # center = np.array([img_size/2., img_size/2.])
    center = np.array([cx, cy])
    # transformations
    Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
    XY = np.reshape(S[:, 0:2], -1)
    O = np.tile(center, num_joints)
    F = np.tile(f, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array([F*np.tile(np.array([1, 0]), num_joints), F *
                  np.tile(np.array([0, 1]), num_joints), O-np.reshape(joints_2d, -1)]).T
    c = (np.reshape(joints_2d, -1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # test
    A += np.eye(A.shape[0]) * 1e-6

    # solution
    trans = np.linalg.solve(A, b)
    return trans


def cal_trans(J3d, J2d, intri):
    fx = intri[0][0]
    fy = intri[1][1]
    cx = intri[0][2]
    cy = intri[1][2]
    j_conf = J2d[:, 2]
    gt_cam_t = estimate_translation_from_intri(
        J3d, J2d[:, :2], j_conf, cx=cx, cy=cy, fx=fx, fy=fy)
    return gt_cam_t


def surface_projection(vertices, faces, joint, exter, intri, image, op):
    im = deepcopy(image)

    intri_ = np.insert(intri, 3, values=0., axis=1)
    temp_v = np.insert(vertices, 3, values=1., axis=1).transpose((1, 0))

    out_point = np.dot(exter, temp_v)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1]
    out_point = (out_point.astype(np.int32)).transpose(1, 0)
    max = dis.max()
    min = dis.min()
    t = 255./(max-min)

    img_faces = []
    color = (255, 255, 255)
    for f in faces:
        point = out_point[f]
        im = cv2.polylines(im, [point], True, color, 1)

    temp_joint = np.insert(joint, 3, values=1., axis=1).transpose((1, 0))
    out_point = np.dot(exter, temp_joint)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1].astype(np.int32)
    out_point = out_point.transpose(1, 0)
    for i in range(len(out_point)):
        if i == op:
            im = cv2.circle(im, tuple(out_point[i]), 2, (0, 0, 255), -1)
        else:
            im = cv2.circle(im, tuple(out_point[i]), 2, (255, 0, 0), -1)

    ratiox = 800/int(im.shape[0])
    ratioy = 800/int(im.shape[1])
    if ratiox < ratioy:
        ratio = ratiox
    else:
        ratio = ratioy

    cv2.namedWindow("mesh", 0)
    cv2.resizeWindow("mesh", int(im.shape[1]*ratio), int(im.shape[0]*ratio))
    cv2.moveWindow("mesh", 0, 0)
    cv2.imshow('mesh', im)
    cv2.waitKey()

    return out_point, im


def joint_projection(joint, extri, intri, image, viz=False):

    im = deepcopy(image)

    intri_ = np.insert(intri, 3, values=0., axis=1)
    temp_joint = np.insert(joint, 3, values=1., axis=1).transpose((1, 0))
    out_point = np.dot(extri, temp_joint)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1].astype(np.int32)
    out_point = out_point.transpose(1, 0)

    if viz:
        for i in range(len(out_point)):
            im = cv2.circle(im, tuple(out_point[i]), 10, (0, 0, 255), -1)

        ratiox = 800/int(im.shape[0])
        ratioy = 800/int(im.shape[1])
        if ratiox < ratioy:
            ratio = ratiox
        else:
            ratio = ratioy

        cv2.namedWindow("mesh", 0)
        cv2.resizeWindow("mesh", int(
            im.shape[1]*ratio), int(im.shape[0]*ratio))
        cv2.moveWindow("mesh", 0, 0)
        cv2.imshow('mesh', im)
        cv2.waitKey()

    return out_point, im


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


def load_camera_para(file):
    """"
    load camera parameters
    """
    campose = []
    intra = []
    campose_ = []
    intra_ = []
    f = open(file, 'r')
    for line in f:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        if len(words) == 3:
            intra_.append([float(words[0]), float(words[1]), float(words[2])])
        elif len(words) == 4:
            campose_.append([float(words[0]), float(words[1]),
                            float(words[2]), float(words[3])])
        else:
            pass

    index = 0
    intra_t = []
    for i in intra_:
        index += 1
        intra_t.append(i)
        if index == 3:
            index = 0
            intra.append(intra_t)
            intra_t = []

    index = 0
    campose_t = []
    for i in campose_:
        index += 1
        campose_t.append(i)
        if index == 3:
            index = 0
            campose_t.append([0., 0., 0., 1.])
            campose.append(campose_t)
            campose_t = []

    return np.array(campose), np.array(intra)


def get_rot_trans(campose, photoscan=False):
    trans = []
    rot = []
    for cam in campose:
        # for photoscan parameters
        if photoscan:
            cam = np.linalg.inv(cam)
        trans.append(cam[:3, 3])
        rot.append(cam[:3, :3])
        # rot.append(cv2.Rodrigues(cam[:3,:3])[0])

    return trans, rot


class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)


class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist


def smpl_to_annotation(model_type='smplx', use_hands=False, use_face=False,
                       use_face_contour=False, pose_format='coco17'):

    if pose_format == 'coco17':
        if model_type == 'smpl':
            # coco17 order: Nose Leye Reye Lear Rear LS RS LE RE LW RW LH RH LK RK LA RA
            return np.array([24, 25, 26, 27, 28, 16, 17, 18, 19, 20, 21, 1,
                             2, 4, 5, 7, 8],
                            dtype=np.int32)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif pose_format == 'lsp14':
        if model_type == 'smpllsp':
            # lsp order: Nose Leye Reye Lear Rear LS RS LE RE LW RW LH RH LK RK LA RA
            return np.array([14, 15, 16, 17, 18, 9, 8, 10, 7, 11, 6, 3,
                             2, 4, 1, 5, 0],
                            dtype=np.int32)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif pose_format == "coco25":
        if model_type == 'smplx':
            return np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                             8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                             63, 64, 65], dtype=np.int32)
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))


def WhichSelected(keypoints, x, y):
    dis = []
    # for i in range(len(keypoints)):
    #     dis.append(pow(x-keypoints[i][0], 2)+pow(y-keypoints[i][1], 2))
    dis = np.linalg.norm(np.array([x,y]) - keypoints,axis=1)
    points_index = np.argmin(dis)
    # mindis = min(dis)
    # points_index = -1
    # if(mindis < pow(r, 2)):
    #     for i in range(len(dis)):
    #         if(dis[i] == mindis):
    #             points_index = i
    #             print("the index of the selected point is: ", points_index)
    #             break
    
    if(points_index != -1):
        selected_point = keypoints[points_index]
        print("the selected points is: ",
              (selected_point[0], selected_point[1]))

    return points_index


def index_to(points_index):
    global draw_color
    if points_index != -1:
        joints_index = points_to[points_index]-1
        if joints_index >= 22:
            joints_index = -1
        if joints_index == -1:
            draw_color = False
    else:
        joints_index = -1
    return joints_index

def keyboardCall(key, init_param=None):
    global status, keyAxis, shapeDim, gloablOrientAxis, gloablTranAxis, flag_change_value, num
    global my_body_pose, my_transl, my_global_orient, my_betas
    betas_max_value = 10
    betas_min_value = -10
    pose_min_value = -0.5+original_pose
    pose_max_value = 0.5+original_pose
    tranl_max_value = 1 + original_transl
    tranl_min_value = -1 + original_transl
    orient_min_value = -1 + original_orient
    orient_max_value = 1 + original_orient
    # key = cv2.waitKey(1)
    if key == 27:    # 退出
        cv2.destroyAllWindows()
    elif key != -1:
        if flag_change_value:
            if (key == ord("w")):
                keyAxis += 1
                keyAxis %= 3
                print("poseAxis:",keyAxis)
        else:
            if (key == ord("s")):
                shapeDim += 1
                shapeDim %= 10
                print("shapeDim:",shapeDim)
            if (key == ord("x")):
                gloablOrientAxis += 1
                gloablOrientAxis %= 3
                print("globalOrientAxis:",gloablOrientAxis)
            if (key == ord("b")):
                gloablTranAxis += 1
                gloablTranAxis %= 3
                print("gloablTranAxis:",gloablTranAxis)
    
    if key == ord("r"):
        my_betas = init_param['betas'].clone()
        my_global_orient = init_param['global_orient'].clone()
        my_transl = init_param['transl'].clone()
        my_body_pose = init_param['body_pose'].clone()

    if key == ord("a"):
        if(my_betas[0, shapeDim] >= betas_min_value):
            my_betas[0, shapeDim] = my_betas[0, shapeDim]-1
    if key == ord("d"):
        if(my_betas[0, shapeDim] <= betas_max_value):
            my_betas[0, shapeDim] = my_betas[0, shapeDim]+1

    if key == ord("z"):
        if(my_global_orient[0, gloablOrientAxis] >= orient_min_value[0, gloablOrientAxis]):
            my_global_orient[0, gloablOrientAxis] = my_global_orient[0, gloablOrientAxis]-0.05
    if key == ord("c"):
        if(my_global_orient[0, gloablOrientAxis] <= orient_max_value[0, gloablOrientAxis]):
            my_global_orient[0, gloablOrientAxis] = my_global_orient[0, gloablOrientAxis]+0.05

    if key == ord("v"):
        if(my_transl[0, gloablTranAxis] >= tranl_min_value[0, gloablTranAxis]):
            my_transl[0, gloablTranAxis] = my_transl[0, gloablTranAxis]-0.05
    if key == ord("n"):
        if(my_transl[0, gloablTranAxis] <= tranl_max_value[0, gloablTranAxis]):
            my_transl[0, gloablTranAxis] = my_transl[0, gloablTranAxis]+0.05


    if flag_change_value == True:
        if key == ord("q"):
            if(my_body_pose[0, 3*joints_index+keyAxis] >= pose_min_value[0, 3*joints_index+keyAxis]):
                my_body_pose[0, 3*joints_index+keyAxis] = my_body_pose[0, 3*joints_index+keyAxis]-0.02
        if key == ord("e"):
            if(my_body_pose[0, 3*joints_index+keyAxis] <= pose_max_value[0, 3*joints_index+keyAxis]):
                my_body_pose[0, 3*joints_index+keyAxis] = my_body_pose[0, 3*joints_index+keyAxis]+0.02

def project_to_img(joints, verts, faces, gt_joints, camera, image_path, renderList, keyMain,viz=False, inter=False, path=None, points_to=None, init_param=None, test=False):

    global d2j, img, imagepath, vertices, keypoints
    imagepath = path

    d2j = []
    # vertices = []
    for cam in camera:
        d2j_ = cam(joints).detach().cpu().numpy().astype(np.int32)
        d2j.append(d2j_)

    if inter:
        for v in range(len(image_path)):
            visualize_results(d2j[v], verts,
                                faces, gt_joints[v], image_path[v], camera[v], renderList[v], keyMain, save=False, path=None)
            cv2.setMouseCallback(name, click_event, np.array(d2j[v])[0])
        keyboardCall(keyMain, init_param=init_param)
    else:
        if not test:
            for v in range(len(image_path)):
                visualize_results(d2j[v], verts,faces, gt_joints[v], image_path[v], camera[v], renderList[v], keyMain, save=True, path=path)
        else:
            for v in range(len(image_path)):
                visualize_results(d2j[v], verts,faces, gt_joints[v], image_path[v], camera[v], renderList[v], keyMain, save=False, path=None)

def visualize_fitting(joints, verts, faces, camera, image_path, save=False, path='./temp_vis'):
    global vis_count, a_index
    d2j = []
    vertices = []
    for cam in camera:
        d2j_ = cam(joints).detach().cpu().numpy().astype(np.int32)
        vert = cam(verts).detach().cpu().numpy().astype(np.int32)
        d2j.append(d2j_)
        vertices.append(vert)

    for v in range(len(image_path)):
        img_dir = image_path[v]
        if sys.platform == 'linux':
            view = img_dir.split('/')[-2]
        else:
            view = img_dir.split('\\')[-2]
        img = cv2.imread(img_dir)
        for f in faces:
            color = 255
            point = vertices[v][0][f]
            img = cv2.polylines(img, [point], True, (color, color, color), 1)
        count = 0
        if d2j_points_index != -1:
            a_index = d2j_points_index
        for p in d2j[v][0]:
            if flag_change_value == True:
                print("11111111111")
                if count == a_index:  # 选中点画成红色
                    joints_index = index_to(d2j_points_index)
                    # print("joints_index: ", joints_index)
                    cv2.circle(
                        img, (int(p[0]), int(p[1])), 3, (0, 0, 255), 10)  # 红色
                else:  # 未选中的点正常画
                    cv2.circle(
                        img, (int(p[0]), int(p[1])), 3, (0, 255, 0), 10)
            else:  # 未选中的点正常画
                print("22222222")
                cv2.circle(
                    img, (int(p[0]), int(p[1])), 3, (0, 255, 0), 10)
            # print("draw_move:", draw_move)
            # if draw_move == True:
            #     cv2.circle(
            #         img, (int(ix), int(iy)), 3, (0, 255, 0), 10)

            count = count+1
        global name
        name = "%s/%s.jpg" % (path, view)

        vis_img(name, img)
        if save:
            cv2.imwrite("%s/%s.jpg" % (path, view), img)
        # if save:
        #     cv2.imwrite("%s/%s_%05d.jpg" % (path, view, vis_count), img)
    vis_count += 1

def renderMultiview(keyMain, save=False):
    if (keyMain != -1) or (save):
        pass
    pass

def visualize_results(d2jD, vertices, faces, gt_joints, image_path, camera, renderList, keyMain, save=False, path=None):
    global view, img, a_index, joints_index, gimg

    mesh = trimesh.Trimesh(vertices=vertices.detach().squeeze().cpu().numpy(),faces=faces)

    img_dir = image_path
    cam = camera
    if sys.platform == 'linux':
        view = img_dir.split('/')[-2]
    else:
        view = img_dir.split('\\')[-2]
    img = cv2.imread(img_dir)
    render = renderList
    camIn = [
            [cam.focal_length_x.detach().squeeze().cpu().numpy(),0,cam.center[0][0].detach().squeeze().cpu().numpy()],
            [0,cam.focal_length_y.detach().squeeze().cpu().numpy(),cam.center[0][1].detach().squeeze().cpu().numpy()],
            [0,0,1]
    ]
    global name
    name = "%s/%s.jpg" % (path, view)
    
    if (keyMain != -1) or (save):
        #img_v = cam(vertices).detach().cpu().numpy()
        img = render(mesh.vertices,mesh.faces,cam.rotation.detach().cpu().squeeze().numpy(),cam.translation.detach().cpu().squeeze().numpy(),camIn,img,viz=False)
        # for f in faces:
        #     color = 255
        #     #point = vertices.detach().cpu().numpy()[0][f]
        #     point = img_v[0][f]
        #     img = cv2.polylines(
        #         img, [point.astype(np.int32)], True, (color, color, color), 1)
        gimg[name] = img.copy()

    count = 0
    if d2j_points_index != -1:
        a_index = d2j_points_index
    for p in d2jD[0]:
        if flag_change_value == True:
            if count == a_index:
                joints_index = index_to(d2j_points_index)
                gimg[name] = cv2.circle(
                    gimg[name], (int(p[0]), int(p[1])), 3, (255, 0, 0), 10)
            else:
                gimg[name] = cv2.circle(
                    gimg[name], (int(p[0]), int(p[1])), 3, (0, 0, 255), 10)
        else:
            gimg[name] = cv2.circle(
                gimg[name], (int(p[0]), int(p[1])), 3, (0, 0, 255), 10)
        count = count+1

    if save:
        cv2.imwrite("%s/%s.jpg" % (path, view), gimg[name])
    else:
        vis_img(name, gimg[name])


def vis_img(name, im):
    ratiox = 800/int(im.shape[0])
    ratioy = 800/int(im.shape[1])
    if ratiox < ratioy:
        ratio = ratiox
    else:
        ratio = ratioy

    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, int(im.shape[1]*ratio), int(im.shape[0]*ratio))
    if im.max() > 1:
        im = im/255.
    cv2.imshow(name, im)


def save_results(setting, data, result,
                 use_vposer=True,
                 save_meshes=False, save_images=False,
                 **kwargs):
    global keypoints
    model_type = kwargs.get('model_type', 'smpl')
    vposer = setting['vposer']
    model = setting['model']
    camera = setting['camera']
    serial = data['serial']
    fn = data['fn']
    img_path = data['img_path']
    keypoints = data["keypoints"]
    person_id = 0

    if use_vposer:
        pose_embedding = result['pose_embedding']
        body_pose = vposer.decode(
            pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
        # the parameters of foot and hand are from vposer
        # we do not use this inaccurate results
        if True:
            body_pose[:, 18:24] = 0.
            body_pose[:, 27:33] = 0.
            body_pose[:, 57:] = 0.
        result['body_pose'] = body_pose.detach().cpu().numpy()
        # print("body_pose:", body_pose)
        # print("body_shape", body_pose.shape)
        orient = np.array(model.global_orient.detach().cpu().numpy())
        temp_pose = body_pose.detach().cpu().numpy()
        pose = np.hstack((orient, temp_pose))
        result['pose'] = pose
        result['pose_embedding'] = pose_embedding.detach().cpu().numpy()
    else:
        if True:
            result['body_pose'][:, 18:24] = 0.
            result['body_pose'][:, 27:33] = 0.
            result['body_pose'][:, 57:] = 0.
        pose = np.hstack((result['global_orient'], result['body_pose']))
        result['pose'] = pose

    renderList = []
    for imgPath in img_path:
        img = cv2.imread(imgPath)
        renderList.append(Renderer((img.shape[1],img.shape[0])))

    if setting['adjustment']:
        global my_betas, my_transl, my_global_orient, origianl_betas, original_transl, original_orient, original_pose, my_changed_body_pose, my_body_pose, key
        model.betas.requires_grad = False
        model.global_orient.requires_grad = False
        model.transl.requires_grad = False
        origianl_betas = model.betas
        original_transl = model.transl
        original_orient = model.global_orient
        original_pose = body_pose
        my_betas = model.betas
        my_global_orient = model.global_orient
        my_transl = model.transl
        my_body_pose = body_pose
        distinat = smpl_to_annotation(model_type=model_type)
        global points_to
        points_to = dict(
            zip(np.arange(0, distinat.shape[0]), distinat))
        if not use_vposer:
            body_pose = model.body_pose.detach()
            body_pose[:, 18:24] = 0.
            body_pose[:, 27:33] = 0.
            body_pose[:, 57:] = 0.

        model_output = model(
            global_orient=my_global_orient, transl=my_transl,
            return_verts=True, body_pose=my_body_pose, betas=my_betas)  # ! 主要
        body_joints = model_output.joints
        verts = model_output.vertices

        curr_image_fn = osp.join(setting['img_folder'], serial, fn)
        project_to_img(
            body_joints, verts, model.faces, keypoints,
            camera, img_path, renderList, 1,viz=False, inter=setting['adjustment'], path=curr_image_fn, points_to=points_to)

        init_param = {}
        init_param = {
            'global_orient':my_global_orient.clone(),
            'transl':my_transl.clone(),
            'body_pose':my_body_pose.clone(),
            'betas':my_betas.clone(),
        }

        while setting['adjustment']:  # 没按下退出键
            keyMain = cv2.waitKey(1)
            # if keyMain != -1:
            #     model_output = model(global_orient=my_global_orient, transl=my_transl,
            #                         return_verts=True, body_pose=my_body_pose, betas=my_betas)  # ! 主要
            #     body_joints = model_output.joints
            #     verts = model_output.vertices
            project_to_img(
                body_joints, verts, model.faces, keypoints,
                camera, img_path, renderList, keyMain,viz=False, inter=setting['adjustment'], path=curr_image_fn, points_to=points_to,init_param=init_param)
            if keyMain != -1:
                model_output = model(global_orient=my_global_orient, transl=my_transl,
                                    return_verts=True, body_pose=my_body_pose, betas=my_betas)  # ! 主要
                body_joints = model_output.joints
                verts = model_output.vertices
                project_to_img(
                    body_joints, verts, model.faces, keypoints,
                    camera, img_path, renderList, keyMain,viz=False, inter=False, path=curr_image_fn, points_to=points_to,init_param=init_param, test=True)
            if keyMain == 27:    # 退出
                break
        cv2.destroyAllWindows()

        if use_vposer:
            result['body_pose'] = my_body_pose.detach().cpu().numpy()
            # print("body_pose:", body_pose)
            # print("body_shape", body_pose.shape)
            orient = my_global_orient.detach().cpu().numpy()
            temp_pose = my_body_pose.detach().cpu().numpy()
            pose = np.hstack((orient, temp_pose))
            result['pose'] = pose
            result['pose_embedding'] = pose_embedding.detach().cpu().numpy()
        else:
            if True:
                result['body_pose'][:, 18:24] = 0.
                result['body_pose'][:, 27:33] = 0.
                result['body_pose'][:, 57:] = 0.
            pose = np.hstack((my_global_orient.detach().cpu().numpy(), my_body_pose.detach().cpu().numpy()))
            result['pose'] = pose
        result['betas'] = my_betas.detach().cpu().numpy()
        result['transl'] = my_transl.detach().cpu().numpy()

    # save results
    curr_result_fn = osp.join(setting['result_folder'], serial, fn)
    if not osp.exists(curr_result_fn):
        os.makedirs(curr_result_fn)
    result_fn = osp.join(curr_result_fn, '{:03d}.pkl'.format(person_id))
    with open(result_fn, 'wb') as result_file:
        pickle.dump(result, result_file, protocol=2)

    if save_meshes or save_images:
        model_output = model(
            global_orient=torch.tensor(result['pose'][:,:3],device=setting['device']), 
            transl=torch.tensor(result['transl'],device=setting['device']),
            return_verts=True, 
            body_pose=torch.tensor(result['pose'][:,3:],device=setting['device']), 
            betas=torch.tensor(result['betas'],device=setting['device']))  # ! 主要
        body_joints = model_output.joints
        verts = model_output.vertices
        # save image
        if save_images:
            curr_image_fn = osp.join(setting['img_folder'], serial, fn)
            if not osp.exists(curr_image_fn):
                os.makedirs(curr_image_fn)
            project_to_img(
                body_joints, verts, model.faces, keypoints,
                camera, img_path, renderList, -1,viz=False, inter=False, path=curr_image_fn, points_to=points_to)

        if save_meshes:
            curr_mesh_fn = osp.join(setting['mesh_folder'], serial, fn)
            if not osp.exists(curr_mesh_fn):
                os.makedirs(curr_mesh_fn)
            mesh_fn = osp.join(curr_mesh_fn, '{:03d}.obj'.format(person_id))
            out_mesh = trimesh.Trimesh(verts.detach().squeeze().cpu().numpy(), model.faces, process=False)
            out_mesh.export(mesh_fn)

class Renderer:
    def __init__(self, resolution=(256, 256, 3), wireframe=False):

        self.resolution = resolution
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0)

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        self.colors = {
            'red': [.8, .1, .1],
            'bule': [.1, .1, .8], #[.7, .7, .6],#
            'green': [.1, .8, .1],
            'pink': [.7, .7, .9],
            'neutral': [.9, .9, .8], #[.7, .7, .6],#
            'capsule': [.7, .75, .5],
            'yellow': [.5, .7, .75],
        }
        
    def Extrinsic_to_ModelViewMatrix(self, extri):
        extri[1] = -extri[1]
        extri[2] = -extri[2]
        return extri

    def vis_img(self, name, im):
        ratiox = 800/int(im.shape[0])
        ratioy = 800/int(im.shape[1])
        if ratiox < ratioy:
            ratio = ratiox
        else:
            ratio = ratioy

        cv2.namedWindow(name,0)
        cv2.resizeWindow(name,int(im.shape[1]*ratio),int(im.shape[0]*ratio))
        #cv2.moveWindow(name,0,0)
        if im.max() > 1:
            im = im/255.
        cv2.imshow(name,im)
        cv2.waitKey(0)
        if name != 'mask':
            cv2.waitKey(1)

    def add_pointLight(self, intensity,center,size):
        light_pose = np.eye(4)
        thetas = np.pi * np.array([1.0 / 6.0, 3.0 / 6.0, 5.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        r = np.linalg.norm(size)
        for theta in thetas:
            for phi in phis:
                xp = np.sin(theta) * np.cos(phi)
                yp = np.sin(theta) * np.sin(phi)
                zp = np.cos(theta)
                light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity*r*r)
                light_pose[:3, 3] = center + np.array([xp,yp,zp]) * r
                self.scene.add(light, pose=light_pose)

    def add_points_light(self, intensity=1.0, bbox=None):
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        # Use 3 directional lights
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -2, 2]) + bbox[0]
        self.scene.add(light, pose=light_pose)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        light_pose[:3, 3] = np.array([0, 2, 2]) + bbox[0]
        self.scene.add(light, pose=light_pose)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        light_pose[:3, 3] = np.array([2, 2, 2]) + bbox[0]
        self.scene.add(light, pose=light_pose)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)

        # Use 3 directional lights
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -2, 2]) + bbox[1]
        self.scene.add(light, pose=light_pose)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        light_pose[:3, 3] = np.array([0, 2, 2]) + bbox[1]
        self.scene.add(light, pose=light_pose)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        light_pose[:3, 3] = np.array([2, 2, 2]) + bbox[1]
        self.scene.add(light, pose=light_pose)

    def __call__(self, verts, faces, rotation, trans, intri, img=None, color=[0.5,0.5,0.5], viz=False):
        
        # Add mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )
        
        rot = np.eye(4)
        rot[:3,:3] = rotation
        mesh.apply_transform(rot)

        center = (np.max(np.array(mesh.vertices),axis=0) + np.min(np.array(mesh.vertices),axis=0))/2.0
        size = (np.max(np.array(mesh.vertices),axis=0) - np.min(np.array(mesh.vertices),axis=0))/2.0

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        mesh_node = self.scene.add(mesh, 'mesh')

        # Add cameras
        camera = pyrender.IntrinsicsCamera(fx=intri[0][0], fy=intri[1][1], cx=intri[0][2], cy=intri[1][2], zfar=8000)
        camera_pose = np.eye(4)
        trans = trans.reshape(-1,)
        trans[0] = -trans[0]
        camera_pose[:3,3] = trans
        camera_pose = self.Extrinsic_to_ModelViewMatrix(camera_pose)
        cam_node = self.scene.add(camera, pose=camera_pose)

        self.add_pointLight(1.0, center,size)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA 

        image, _ = self.renderer.render(self.scene, flags=render_flags)

        visible_weight = 1
        if img is not None:
            valid_mask = (image[:, :, -1] > 0)[:, :,np.newaxis]
            if image.shape[-1] == 4:
                image = image[:,:,:-1]
            
            image = image * valid_mask * visible_weight + img * valid_mask * (1-visible_weight) + (1 - valid_mask) * img

        if viz:
            self.vis_img('img', image)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)
        return image

    def render_multiperson(self, verts, faces, rotation, trans, intri, img=None, viz=False):
        # assert len(verts) < 5
        # Add mesh
        mesh_nodes = []
        mesh_bounds = []
        for i, (vert, color) in enumerate(zip(verts, self.colors)):
            if vert is None:
                continue
            else:
                vert = vert.detach().cpu().numpy()
            color = self.colors[color]
            mesh = trimesh.Trimesh(vertices=vert, faces=faces, process=False)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0,
                alphaMode='OPAQUE',
                baseColorFactor=(color[0], color[1], color[2], 1.0)
            )

            rot = np.eye(4)
            rot[:3,:3] = rotation
            mesh.apply_transform(rot)

            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            mesh_node = self.scene.add(mesh, 'mesh')
            mesh_nodes.append(mesh_node)
            mesh_bounds.append(mesh.bounds)

        if len(mesh_bounds) < 1:
            return img

        mesh_bounds = np.array(mesh_bounds)
        top = np.mean(mesh_bounds[:,0,:], axis=0)
        bottom = np.mean(mesh_bounds[:,1,:], axis=0)
        pos = (top + bottom) / 2
        # Add light
        light_nodes = self.use_raymond_lighting(15, trans=pos-np.array([0,0,3]))

        # Add cameras
        camera = pyrender.IntrinsicsCamera(fx=intri[0][0], fy=intri[1][1], cx=intri[0][2], cy=intri[1][2], zfar=8000)
        camera_pose = np.eye(4)
        trans = trans.reshape(-1,)
        trans[0] = -trans[0]
        camera_pose[:3,3] = trans
        camera_pose = self.Extrinsic_to_ModelViewMatrix(camera_pose)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA 

        image, _ = self.renderer.render(self.scene, flags=render_flags)

        visible_weight = 1
        if img is not None:
            valid_mask = (image[:, :, -1] > 0)[:, :,np.newaxis]
            if image.shape[-1] == 4:
                image = image[:,:,:-1]
            
            image = image * valid_mask * visible_weight + img * valid_mask * (1-visible_weight) + (1 - valid_mask) * img

        if viz:
            self.vis_img('img', image)
        
        for n in mesh_nodes:
            self.scene.remove_node(n)
        self.scene.remove_node(cam_node)
        for n in light_nodes:
            self.scene.remove_node(n)
        return image

    def _add_raymond_light(self, trans):
        from pyrender.light import DirectionalLight
        from pyrender.light import PointLight
        from pyrender.node import Node

        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            nodes.append(Node(
                light=PointLight(color=np.ones(3), intensity=1.0),
                translation=trans+3.0*np.array([xp, yp, zp])
            ))
        return nodes

    def use_raymond_lighting(self, intensity=1.0, trans=np.array([0,0,0])):
        nodes = []
        for n in self._add_raymond_light(trans):
            n.light.intensity = intensity / 3.0
            if not self.scene.has_node(n):
                self.scene.add_node(n)#, parent_node=pc)
            nodes.append(n)
        return nodes