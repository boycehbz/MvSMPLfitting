'''
 @FileName    : alphapose.py
 @EditTime    : 2023-02-03 18:51:59
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import cv2
import numpy as np
import torch
from alphapose_core.alphapose.models import builder
from alphapose_core.alphapose.utils.config import update_config
from alphapose_core.alphapose.utils.presets import SimpleTransform
from alphapose_core.alphapose.utils.transforms import flip, flip_heatmap
from alphapose_core.alphapose.utils.pPose_nms import pose_nms, write_json
from alphapose_core.alphapose.utils.transforms import get_func_heatmap_to_coord
from utils.module_utils import draw_keyp, vis_img

class AlphaPose_Predictor(object):
    def __init__(
        self,
        pose_config,
        pose_checkpoint,
        thres,
        decoder=None,
        device=torch.device('cuda'),
        fp16=True
    ):
        self.device = device
        self.pose_checkpoint = pose_checkpoint
        # Load pose model
        self.load_model(pose_config)

        self.posebatch = 80
        self.flip = False
        self.pose_track = False
        self.posenms = False
        self.heatmap_to_coord = get_func_heatmap_to_coord(self.cfg)
        self.min_box_area = 0
        self.use_heatmap_loss = True

        self._input_size = self.cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = self.cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = self.cfg.DATA_PRESET.SIGMA

        if self.cfg.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                self, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False)


    def load_model(self, cfg):
        
        self.cfg = update_config(cfg)
        self.pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)

        print(f'Loading pose model from {self.pose_checkpoint}...')
        self.pose_model.load_state_dict(torch.load(self.pose_checkpoint, map_location=self.device))

        self.pose_model.to(self.device)
        self.pose_model.eval()

    def inference(self, data):
        inps = data['inps']
        assert inps.shape[0] == 1

        inps = inps[0]
        orig_img_k = data['orig_img_k'][0]
        cropped_boxes = data['cropped_boxes'][0]

        # Pose Estimation
        inps = inps.to(self.device)
        datalen = inps.size(0)
        leftover = 0
        if (datalen) % self.posebatch:
            leftover = 1
        num_batches = datalen // self.posebatch + leftover

        hm = []
        for j in range(num_batches):
            inps_j = inps[j * self.posebatch:min((j + 1) * self.posebatch, datalen)]
            if self.flip:
                inps_j = torch.cat((inps_j, flip(inps_j)))
            hm_j = self.pose_model(inps_j)
            if self.flip:
                hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], self.joint_pairs, shift=True)
                hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
            hm.append(hm_j)
        hm = torch.cat(hm)
        hm = hm.cpu()

        # save result
        orig_img = np.array(orig_img_k, dtype=np.uint8)[:, :, ::-1]
        hm_data = hm
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)

        # location prediction (n, kp, 2) | score prediction (n, kp, 1)
        assert hm_data.dim() == 4

        face_hand_num = 110
        if hm_data.size()[1] == 136:
            self.eval_joints = [*range(0,136)]
        elif hm_data.size()[1] == 26:
            self.eval_joints = [*range(0,26)]
        elif hm_data.size()[1] == 133:
            self.eval_joints = [*range(0,133)]
        elif hm_data.size()[1] == 68:
            face_hand_num = 42
            self.eval_joints = [*range(0,68)]
        elif hm_data.size()[1] == 21:
            self.eval_joints = [*range(0,21)]
        pose_coords = []
        pose_scores = []
        for i in range(hm_data.shape[0]):
            bbox = cropped_boxes[i].tolist()
            if isinstance(self.heatmap_to_coord, list):
                pose_coords_body_foot, pose_scores_body_foot = self.heatmap_to_coord[0](
                    hm_data[i][self.eval_joints[:-face_hand_num]], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords_face_hand, pose_scores_face_hand = self.heatmap_to_coord[1](
                    hm_data[i][self.eval_joints[-face_hand_num:]], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
            else:
                pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
        preds_img = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)

        if not self.pose_track and self.posenms:
            assert len(scores) == len(preds_img)
            boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                pose_nms(boxes, scores, ids, preds_img, preds_scores, self.min_box_area, use_heatmap_loss=self.use_heatmap_loss)

        pred_poses = torch.cat([preds_img, preds_scores], dim=2)
        pred_poses = pred_poses.detach().cpu().numpy()

        return pred_poses

    def predict(self, img, boxes):

        boxes = torch.from_numpy(np.array(boxes))
        scores = torch.from_numpy(np.ones((boxes.shape[0])))
        ids = torch.from_numpy(np.arange((boxes.shape[0]))).to(torch.int32)
        orig_img_k = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        inps = torch.zeros(boxes.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes.size(0), 4)
        for i, box in enumerate(boxes):
            inps[i], cropped_box = self.transformation.test_transform(orig_img_k, box)
            cropped_boxes[i] = torch.FloatTensor(cropped_box)

        # Pose Estimation
        inps = inps.to(self.device)
        datalen = inps.size(0)
        leftover = 0
        if (datalen) % self.posebatch:
            leftover = 1
        num_batches = datalen // self.posebatch + leftover

        hm = []
        for j in range(num_batches):
            inps_j = inps[j * self.posebatch:min((j + 1) * self.posebatch, datalen)]
            if self.flip:
                inps_j = torch.cat((inps_j, flip(inps_j)))
            hm_j = self.pose_model(inps_j)
            if self.flip:
                hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], self.joint_pairs, shift=True)
                hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
            hm.append(hm_j)
        hm = torch.cat(hm)
        hm = hm.cpu()

        # save result
        orig_img = np.array(orig_img_k, dtype=np.uint8)[:, :, ::-1]
        hm_data = hm
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)

        # location prediction (n, kp, 2) | score prediction (n, kp, 1)
        assert hm_data.dim() == 4

        face_hand_num = 110
        if hm_data.size()[1] == 136:
            self.eval_joints = [*range(0,136)]
        elif hm_data.size()[1] == 26:
            self.eval_joints = [*range(0,26)]
        elif hm_data.size()[1] == 133:
            self.eval_joints = [*range(0,133)]
        elif hm_data.size()[1] == 68:
            face_hand_num = 42
            self.eval_joints = [*range(0,68)]
        elif hm_data.size()[1] == 21:
            self.eval_joints = [*range(0,21)]
        pose_coords = []
        pose_scores = []
        for i in range(hm_data.shape[0]):
            bbox = cropped_boxes[i].tolist()
            if isinstance(self.heatmap_to_coord, list):
                pose_coords_body_foot, pose_scores_body_foot = self.heatmap_to_coord[0](
                    hm_data[i][self.eval_joints[:-face_hand_num]], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords_face_hand, pose_scores_face_hand = self.heatmap_to_coord[1](
                    hm_data[i][self.eval_joints[-face_hand_num:]], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
            else:
                pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
        preds_img = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)

        if not self.pose_track and self.posenms:
            assert len(scores) == len(preds_img)
            boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                pose_nms(boxes, scores, ids, preds_img, preds_scores, self.min_box_area, use_heatmap_loss=self.use_heatmap_loss)

        pred_poses = torch.cat([preds_img, preds_scores], dim=2)
        pred_poses = pred_poses.detach().cpu().numpy()

        return pred_poses

    def visualize(self, img, poses, format='halpe', viz=False):
        # colors = [(96 , 153, 246),(215, 160, 110), ]

        for i, person in enumerate(poses):
            img = draw_keyp(img, person, color=None, format=format)
        if viz:
            vis_img('image', img)

        return img

    @property
    def stopped(self):
        if self.opt.sp:
            return self._stopped
        else:
            return self._stopped.value
    @property
    def length(self):
        return len(self.all_imgs)

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]
