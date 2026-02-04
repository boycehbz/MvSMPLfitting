'''
 @FileName    : alphapose_module.py
 @EditTime    : 2023-02-04 20:14:13
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import torch
from collections import namedtuple
from alphapose_core.alphapose.utils.presets import SimpleTransform
import cv2
import numpy as np

dataset = namedtuple('dataset', [
    # Position
    'joint_pairs',
])
dataset.joint_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]

transformation = SimpleTransform(
    dataset, scale_factor=0,
    input_size=[256, 192],
    output_size=[64, 48],
    rot=0, sigma=2,
    train=False, add_dpg=False)

def prepare(img, boxes):
    boxes = torch.from_numpy(np.array(boxes))
    scores = torch.from_numpy(np.ones((boxes.shape[0])))
    ids = torch.from_numpy(np.arange((boxes.shape[0]))).to(torch.int32)
    orig_img_k = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inps = torch.zeros(boxes.size(0), 3, *[256, 192])
    cropped_boxes = torch.zeros(boxes.size(0), 4)
    for i, box in enumerate(boxes):
        inps[i], cropped_box = transformation.test_transform(orig_img_k, box)
        cropped_boxes[i] = torch.FloatTensor(cropped_box)

    data = {'scores':scores, 'ids':ids, 'inps':inps, 'orig_img_k':orig_img_k, 'cropped_boxes':cropped_boxes}
    return data