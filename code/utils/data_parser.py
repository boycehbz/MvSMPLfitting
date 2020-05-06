# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

import json

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from utils.utils import smpl_to_annotation

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(dataset='offline', data_folder='data', pose_format='coco17', **kwargs):
    if dataset.lower() == 'offline':
        return FittingData(data_folder, pose_format=pose_format, **kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)

    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt)

def read_joints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    """
    load 3D annotation
    """
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_3d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 4])
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_3d'],
                dtype=np.float32).reshape([-1, 4])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_3d'],
                dtype=np.float32).reshape([-1, 4])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_3d'],
                dtype=np.float32).reshape([-1, 4])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 4)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_3d'],
                    dtype=np.float32).reshape([-1, 4])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)

    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt)

# class OpenPose(Dataset):

#     NUM_BODY_JOINTS = 17
#     NUM_HAND_JOINTS = 20

#     def __init__(self, data_folder, img_folder='images',
#                  keyp_folder='keypoints',
#                  use_hands=False,
#                  use_face=False,
#                  dtype=torch.float32,
#                  model_type='smplx',
#                  joints_to_ign=None,
#                  use_face_contour=False,
#                  openpose_format='coco19',  #alphapose格式为coco
#                  **kwargs):
#         super(OpenPose, self).__init__()

#         self.use_hands = use_hands
#         self.use_face = use_face
#         self.model_type = model_type
#         self.dtype = dtype
#         self.joints_to_ign = joints_to_ign
#         self.use_face_contour = use_face_contour

#         self.openpose_format = openpose_format

#         self.num_joints = (self.NUM_BODY_JOINTS +
#                            2 * self.NUM_HAND_JOINTS * use_hands)

#         self.img_folder = osp.join(data_folder, img_folder)
#         self.keyp_folder = osp.join(data_folder, keyp_folder)

#         self.img_paths = [osp.join(self.img_folder, img_fn)
#                           for img_fn in os.listdir(self.img_folder)
#                           if img_fn.endswith('.png') or
#                           img_fn.endswith('.jpg') and
#                           not img_fn.startswith('.')]
#         self.img_paths = sorted(self.img_paths)
#         self.cnt = 0

#     def get_model2data(self):
#         return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
#                                 use_face=self.use_face,
#                                 use_face_contour=self.use_face_contour,
#                                 openpose_format=self.openpose_format)

#     def get_left_shoulder(self):
#         return 2

#     def get_right_shoulder(self):
#         return 5

#     def get_joint_weights(self):
#         # The weights for the joint terms in the optimization
#         optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
#                                 self.use_face * 51 +
#                                 17 * self.use_face_contour,
#                                 dtype=np.float32)

#         # Neck, Left and right hip
#         # These joints are ignored because SMPL has no neck joint and the
#         # annotation of the hips is ambiguous.
        
#         # if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
#         #     optim_weights[self.joints_to_ign] = 0.
#         # return torch.tensor(optim_weights, dtype=self.dtype)
#         # alphapose 忽略hip就行
#         optim_weights[11] = 0.
#         optim_weights[12] = 0.
#         return torch.tensor(optim_weights, dtype=self.dtype)


#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, idx):
#         img_path = self.img_paths[idx]
#         return self.read_item(img_path)

#     def read_item(self, img_path):
#         img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
#         img_fn = osp.split(img_path)[1]
#         img_fn, _ = osp.splitext(osp.split(img_path)[1])

#         keypoint_fn = osp.join(self.keyp_folder,
#                                img_fn + '_keypoints.json')
        
#         if not os.path.exists(keypoint_fn):
#             keypoints = np.stack([0])
#         else:
#             keyp_tuple = read_keypoints(keypoint_fn, use_hands=self.use_hands,
#                                         use_face=self.use_face,
#                                         use_face_contour=self.use_face_contour)

#             if len(keyp_tuple.keypoints) < 1:
#                 return {}
#             keypoints = np.stack(keyp_tuple.keypoints)

#         output_dict = {'fn': img_fn,
#                        'img_path': img_path,
#                        'keypoints': keypoints, 'img': img}
#         # if keyp_tuple.gender_gt is not None:
#         #     if len(keyp_tuple.gender_gt) > 0:
#         #         output_dict['gender_gt'] = keyp_tuple.gender_gt
#         # if keyp_tuple.gender_pd is not None:
#         #     if len(keyp_tuple.gender_pd) > 0:
#         #         output_dict['gender_pd'] = keyp_tuple.gender_pd
#         return output_dict

#     def __iter__(self):
#         return self

#     def __next__(self):
#         return self.next()

#     def next(self):
#         if self.cnt >= len(self.img_paths):
#             raise StopIteration

#         img_path = self.img_paths[self.cnt]
#         self.cnt += 1

#         return self.read_item(img_path)

class FittingData(Dataset):

    NUM_BODY_JOINTS = 17
    NUM_HAND_JOINTS = 20

    def __init__(self, data_folder, img_folder='images',
                    keyp_folder='keypoints',
                    use_hands=False,
                    use_face=False,
                    dtype=torch.float32,
                    model_type='smplx',
                    joints_to_ign=None,
                    use_face_contour=False,
                    pose_format='coco17',
                    use_3d=False,
                    use_hip=True,
                    **kwargs):
        super(FittingData, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.use_3d = use_3d
        self.use_hip = use_hip
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.pose_format = pose_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                            2 * self.NUM_HAND_JOINTS * use_hands)

        self.img_folder = osp.join(data_folder, img_folder)
        self.keyp_folder = osp.join(data_folder, keyp_folder)

        img_serials = sorted(os.listdir(self.img_folder))
        self.img_paths = []
        for i_s in img_serials:
            i_s_dir = osp.join(self.img_folder, i_s)
            img_cameras = sorted(os.listdir(i_s_dir))
            this_serials = []
            for i_cam in img_cameras:
                i_c_dir = osp.join(i_s_dir, i_cam)
                cam_imgs = [osp.join(i_c_dir, img_fn)
                            for img_fn in os.listdir(i_c_dir)
                            if img_fn.endswith('.png') or
                            img_fn.endswith('.jpg') and
                            not img_fn.startswith('.')]
                cam_imgs = sorted(cam_imgs)
                this_serials.append(cam_imgs)
            self.img_paths.append(this_serials)

        self.cnt = 0
        self.serial_cnt = 0

    def get_model2data(self):
        return smpl_to_annotation(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                pose_format=self.pose_format)

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        
        # if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
        #     optim_weights[self.joints_to_ign] = 0.
        # return torch.tensor(optim_weights, dtype=self.dtype)
        if self.pose_format != 'lsp14' or not self.use_hip:
            optim_weights[11] = 0.
            optim_weights[12] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return self.read_item(img_path)

    def read_item(self, img_paths):
        img = []
        keypoints = []
        joints3d = None
        for img_path in img_paths:
            img_ = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
            img_fn = osp.split(img_path)[1]
            areas = osp.split(img_path)[0].split("\\")
            serial = areas[-2]
            cam = areas[-1]
            img_fn, _ = osp.splitext(osp.split(img_path)[1])

            keypoint_fn = osp.join(self.keyp_folder,
                                    serial + '\\' + cam + '\\' + img_fn + '_keypoints.json')

            if not os.path.exists(keypoint_fn):
                keypoints_ = None # keypoints may not exist
            else:
                keyp_tuple = read_keypoints(keypoint_fn, use_hands=self.use_hands,
                                            use_face=self.use_face,
                                            use_face_contour=self.use_face_contour)
    
                if len(keyp_tuple.keypoints) < 1:
                    return {}
                keypoints_ = np.stack(keyp_tuple.keypoints)

            img.append(img_)
            keypoints.append(keypoints_)

            if self.use_3d and joints3d is None and os.path.exists(keypoint_fn):
                joints3d = read_joints(keypoint_fn, use_hands=self.use_hands,
                                            use_face=self.use_face,
                                            use_face_contour=self.use_face_contour)
                joints3d = joints3d.keypoints

        output_dict = {'fn': img_fn,
                        'serial': serial,
                        'img_path': img_paths,
                        'keypoints': keypoints,
                        'img': img,
                        '3d_joint': joints3d}

        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.serial_cnt >= len(self.img_paths):
            raise StopIteration

        img_path = self.img_paths[self.serial_cnt]

        img_paths = []
        for cam in img_path:
            img_paths.append(cam[self.cnt])
        self.cnt += 1
        if self.cnt >= len(cam):
            self.cnt = 0
            self.serial_cnt += 1


        return self.read_item(img_paths)
