import sys
sys.path.append('./')
import glob
import os
import shutil
import json
import numpy as np

## lsp
# def transProx2coco(js):
#     idx = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]
#     return js[idx]
# if __name__ == '__main__':
#     path = R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\keypoints'
#     imgPath = R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\recordings'
#     desPath = R'H:\YangYuan\Code\phy_program\MvSMPLfitting\dataprox'
#     for seq in glob.glob(os.path.join(path,'*')):
#         seqname = os.path.basename(seq)
#         for frame in glob.glob(os.path.join(seq,'*')):
#             framename = os.path.basename(frame)[:4]
#             if not os.path.exists(os.path.join(imgPath,seqname,'img',os.path.basename(frame)[:-15]+'.jpg')):
#                 continue
#             os.makedirs(
#                 os.path.join(desPath,'images',seqname+'_'+framename,'Camera00'),exist_ok=True
#             )
#             shutil.copyfile(
#                 os.path.join(imgPath,seqname,'img',os.path.basename(frame)[:-15]+'.jpg'),
#                 os.path.join(desPath,'images',seqname+'_'+framename,'Camera00','00001.jpg')
#             )
#             os.makedirs(
#                 os.path.join(desPath,'keypoints',seqname+'_'+framename,'Camera00'),exist_ok=True
#             )
#             with open(frame, 'rb') as file:
#                 data = json.load(file)
#             js = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1,3)
#             js = transProx2coco(js)
#             temData = {}
#             temData['people'] = []
#             temData['people'].append(
#                 {'pose_keypoints_2d':list(js.reshape(-1))}
#             )
#             with open(os.path.join(desPath,'keypoints',seqname+'_'+framename,'Camera00','00001_keypoints.json'),'w',encoding="utf8") as file:
#                 json.dump(temData,file)

# coc25
def transProx2coco(js):
    idx = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]
    return js[idx]
if __name__ == '__main__':
    path = R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\keypoints'
    imgPath = R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\recordings'
    desPath = R'H:\YangYuan\Code\phy_program\MvSMPLfitting\dataproxTest'
    for seq in glob.glob(os.path.join(path,'*')):
        seqname = os.path.basename(seq)
        for frame in glob.glob(os.path.join(seq,'*')):
            framename = os.path.basename(frame)[:4]
            if not os.path.exists(os.path.join(imgPath,seqname,'img',os.path.basename(frame)[:-15]+'.jpg')):
                continue
            os.makedirs(
                os.path.join(desPath,'images',seqname+'_'+framename,'Camera00'),exist_ok=True
            )
            shutil.copyfile(
                os.path.join(imgPath,seqname,'img',os.path.basename(frame)[:-15]+'.jpg'),
                os.path.join(desPath,'images',seqname+'_'+framename,'Camera00','00001.jpg')
            )
            os.makedirs(
                os.path.join(desPath,'keypoints',seqname+'_'+framename,'Camera00'),exist_ok=True
            )
            shutil.copyfile(
                os.path.join(frame),
                os.path.join(desPath,'keypoints',seqname+'_'+framename,'Camera00','00001_keypoints.json')
            )