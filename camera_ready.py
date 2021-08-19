import argparse
import glob
import os
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox

from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
     xyxy2xywh, strip_optimizer, set_logging, increment_path, clip_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

class LoadImages:  # for inference
    def __init__(self, data, img_size=640, stride=32):
        self.data = data
        self.img_size = img_size
        self.stride = stride
        self.nf = 1
        self.mode = 'image'
        self.cap = None
    def load_info(self):
        img0 = self.data
        assert img0 is not None, 'Image Not Found '
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img, img0, self.cap

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def detect(source = "data/images/zidane.jpg",half=None,model=None,device=None,imgsz=640,stride=32):

    dataset = LoadImages(source, img_size=imgsz, stride=stride).load_info()
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    # for path, img, im0s, vid_cap in dataset:
    img, im0s, vid_cap = dataset[:]
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=False)[0]
    # Apply NMS
    pred = non_max_suppression(pred,0.25, 0.45, classes=0+1+2, agnostic=False)
    t2 = time_synchronized()
    s, im0, frame =  str(''), im0s, getattr(dataset, 'frame', 0)
    for i, det in enumerate(pred):
        # Rescale boxes from img_size to im0 size
        det[:, 0:4] = scale_coords([img.shape[2], img.shape[3]], det[:, :4], im0.shape).round()
        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        # Write results
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        cv2.putText(im0,s,(100,1200),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,0),3)#目标放到图片上
    print(f'{s}Done. ({t2 - t1:.3f}s)')
    return  im0

