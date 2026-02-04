'''
 @FileName    : yolox.py
 @EditTime    : 2023-02-03 13:42:18
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import torch
import torch.nn as nn
import cv2
import numpy as np
import os.path as osp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking, vis_bboxes
from yolox.utils.timer import Timer
from yolox.data.data_augment import preproc
from utils.module_utils import vis_img

class Predictor(object):
    def __init__(
        self,
        pretrained_model,
        thres,
        decoder=None,
        device=torch.device("cuda"),
        fp16=True
    ):
        self.fp16 = fp16
        self.thresthold = [thres]
        self.model = self.get_model(pretrained_model)
        self.decoder = decoder
        self.num_classes = 1
        self.confthre = 0.001
        self.nmsthre = 0.7
        self.test_size = (800, 1440)
        self.device = device
        
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def predict(self, img_path, viz=False):
        timer = Timer()
        outputs, img_info = self.inference(img_path, timer)

        frame = {}
        if isinstance(img_path, str):
            frame['img_path'] = img_path
            frame['h_w'] = (img_info['height'], img_info['width'])

        if outputs[0] is not None:
            ratio = img_info["ratio"]
            output = outputs[0].cpu().numpy()

            # delete bbox which is not a person 
            cls = output[:,6]
            not_person_idx = np.where(cls != 0)
            output = np.delete(output, not_person_idx, axis=0)

            # delete low confidence
            scores = output[:, 4] * output[:, 5]
            ## set thresthold
            low_conf_1 = np.where(scores < 0.46)[0]
            output_1 = np.delete(output, low_conf_1, axis=0)
            low_conf_2 = np.union1d(np.where(scores > 0.46)[0], np.where(scores < self.thresthold[0])[0])
            output_2 = np.delete(output, low_conf_2, axis=0)
            output = np.vstack((output_1, output_2))

            result_image = self.visual(torch.tensor(output), img_info, self.thresthold)

            bboxes = output[:, 0:4]
            bboxes /= ratio
            bboxes_conf = output[:, 4] * output[:, 5]

        else:
            timer.toc()
            result_image = img_info['raw_img']
            bboxes = None
            bboxes_conf = None

        if viz:
            vis_img('detection', result_image)

        frame['bbox'] = bboxes
        frame['bboxes_conf'] = bboxes_conf

        return frame, result_image

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img = img.copy()
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def model_inference(self, img, img_info):

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )

        if outputs[0] is not None:
            ratio = img_info["ratio"]
            output = outputs[0].cpu().numpy()

            # delete bbox which is not a person 
            cls = output[:,6]
            not_person_idx = np.where(cls != 0)
            output = np.delete(output, not_person_idx, axis=0)

            # delete low confidence
            scores = output[:, 4] * output[:, 5]
            ## set thresthold
            low_conf_1 = np.where(scores < 0.46)[0]
            output_1 = np.delete(output, low_conf_1, axis=0)
            low_conf_2 = np.union1d(np.where(scores > 0.46)[0], np.where(scores < self.thresthold[0])[0])
            output_2 = np.delete(output, low_conf_2, axis=0)
            output = np.vstack((output_1, output_2))

            # result_image = self.visual(torch.tensor(output), img_info, self.thresthold)

            bboxes = output[:, 0:4]
            bboxes /= ratio.cpu().numpy()
            bboxes_conf = output[:, 4] * output[:, 5]

        else:
            result_image = img_info['raw_img']
            bboxes = None
            bboxes_conf = None

        return bboxes, bboxes_conf

    def get_model(self, pretrained_model):
        from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

        # ---------------- model config ---------------- #
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        self.model.to(torch.device('cuda'))

        print("Model Summary: {}".format(get_model_info(self.model, (800, 1440))))
        self.model.eval()

        trt = False
        if not trt:
            ckpt_file = pretrained_model

            print("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            self.model.load_state_dict(ckpt["model"])
            print("loaded checkpoint done.")

        fuse = True
        if fuse:
            print("\tFusing model...")
            self.model = fuse_model(self.model)

        if self.fp16:
            self.model = self.model.half()  # to FP16

        return self.model

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis_bboxes(img, bboxes, scores, cls, cls_conf)
        return vis_res


