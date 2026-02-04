'''
 @FileName    : Demo_AlphaPose.py
 @EditTime    : 2024-04-05 16:04:00
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description : 
'''
import sys
sys.path.append('./')
from yolox.yolox import Predictor
from alphapose_core.alphapose_core import AlphaPose_Predictor
import cv2
import os
from utils.FileLoaders import save_keypoints

folder = 'data/images'
viz = False

# human detection
yolox_model_dir = R'pretrained/yolox_data/bytetrack_x_mot17.pth.tar'
yolox_thres = 0.23
yolox_predictor = Predictor(yolox_model_dir, yolox_thres)

# alphapose
alpha_config = R'alphapose_core/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'
alpha_checkpoint = R'pretrained/alphapose_data/halpe26_fast_res50_256x192.pth'
alpha_thres = 0.1
alpha_predictor = AlphaPose_Predictor(alpha_config, alpha_checkpoint, alpha_thres)

seqs = os.listdir(folder)
for seq in seqs:
    seq_dir = os.path.join(folder, seq)
    cameras = sorted(os.listdir(seq_dir))
    for camera in cameras:
        cam_dir = os.path.join(seq_dir, camera)
        imgs = sorted(os.listdir(cam_dir))

        for name in imgs:
            img_path = os.path.join(cam_dir, name)

            img = cv2.imread(img_path)

            results, result_img = yolox_predictor.predict(img, viz=False)

            pose = alpha_predictor.predict(img, results['bbox'])[:1]
            
            # format: coco17, halpe
            result_img = alpha_predictor.visualize(img, pose, format='coco17', viz=viz)
            output_name = os.path.join('data/keypoints', seq, camera, name.split('.')[0] + '_keypoints.json')
            os.makedirs(os.path.dirname(output_name), exist_ok=True)
            save_keypoints(pose, output_name)
            # cv2.imwrite(os.path.join('output/alphapose', name), result_img)
