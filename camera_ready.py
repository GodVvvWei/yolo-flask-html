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
# img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
# vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
class LoadImages:  # for inference
    def __init__(self, data, img_size=640, stride=32):

        # videos = [x for x in files]
        # ni, nv = len(images), len(videos)
        self.data = data
        self.img_size = img_size
        self.stride = stride
        self.nf = 1
        self.mode = 'image'
        # if any(videos):
        #     self.new_video(videos[0])  # new video
        # else:
        #     self.cap = None
        self.cap = None
        # assert self.nf > 0, f'No images or videos found in {p}. ' \
        #                     f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'


    def load_info(self):

        # img0 = cv2.imread(path)  # BGR

        img0 = self.data
        h0, w0 = img0.shape[:2]
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


# class LoadImages:  # for inference
#     def __init__(self, path, img_size=640, stride=32):
#         p = str(Path(path).absolute())  # os-agnostic absolute path
#         if '*' in p:
#             files = sorted(glob.glob(p, recursive=True))  # glob
#         elif os.path.isdir(p):
#             files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
#         elif os.path.isfile(p):
#             files = [p]  # files
#         else:
#             raise Exception(f'ERROR: {p} does not exist')
#
#         images = [x for x in files]
#         # videos = [x for x in files]
#         # ni, nv = len(images), len(videos)
#         ni = len(images)
#         self.img_size = img_size
#         self.stride = stride
#         self.files = images
#         self.nf = ni   # number of files
#         self.video_flag = [False] * ni
#         self.mode = 'image'
#         # if any(videos):
#         #     self.new_video(videos[0])  # new video
#         # else:
#         #     self.cap = None
#         self.cap = None
#         # assert self.nf > 0, f'No images or videos found in {p}. ' \
#         #                     f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'
#
#     def __iter__(self):
#         self.count = 0
#         return self
#
#     def __next__(self):
#         if self.count == self.nf:
#             raise StopIteration
#         path = self.files[self.count]
#
#         if self.video_flag[self.count]:
#             # Read video
#             self.mode = 'video'
#             ret_val, img0 = self.cap.read()
#             if not ret_val:
#                 self.count += 1
#                 self.cap.release()
#                 if self.count == self.nf:  # last video
#                     raise StopIteration
#                 else:
#                     path = self.files[self.count]
#                     self.new_video(path)
#                     ret_val, img0 = self.cap.read()
#
#             self.frame += 1
#             print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')
#
#         else:
#             # Read image
#             self.count += 1
#             img0 = cv2.imread(path)  # BGR
#             assert img0 is not None, 'Image Not Found ' + path
#             print(f'image {self.count}/{self.nf} {path}: ', end='')
#
#         # Padded resize
#         img = letterbox(img0, self.img_size, stride=self.stride)[0]
#
#         # Convert
#         img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#         img = np.ascontiguousarray(img)
#
#         return path, img, img0, self.cap
#
#     def new_video(self, path):
#         self.frame = 0
#         self.cap = cv2.VideoCapture(path)
#         self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     def __len__(self):
#         return self.nf  # number of files

def detect(source = "data/images/zidane.jpg"):
    conf_thres = 0.25
    iou_thres = 0.45
    device = ''
    weights, view_img, save_txt, imgsz = 'yolov5s.pt', False, False, 640

    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16


    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

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
    pred = non_max_suppression(pred,0.25, 0.45, classes=None, agnostic=False)
    t2 = time_synchronized()

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    # Process detections
    # for i, det in enumerate(pred):  # detections per image

    s, im0, frame =  str(''), im0s, getattr(dataset, 'frame', 0)
    # im0, frame = im0s, getattr(dataset, 'frame', 0)
    # save_path = str(save_dir / p.name)  # img.jpg
    # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt

    s += '%gx%g    ' % img.shape[2:]  # print string
    # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

    # if len(det):
    # if 0 != det:

    for i, det in enumerate(pred):
        # Rescale boxes from img_size to im0 size

        det[:, 0:4] = scale_coords([img.shape[2], img.shape[3]], det[:, :4], im0.shape).round()


        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        for *xyxy, conf, cls in reversed(det):
            # if save_txt:  # Write to file
            #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #     line = (cls, *xywh, conf) if False else (cls, *xywh)  # label format
                # with open(txt_path + '.txt', 'a') as f:
                #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

    # Print time (inference + NMS)
    print(f'{s}Done. ({t2 - t1:.3f}s)')

    # Stream results
    return  im0
    # while True:
    #     cv2.imshow(str(p), im0)
    #     cv2.waitKey(1)

if __name__ == '__main__':
    a = cv2.imread('data/images/bus.jpg')
    b = detect(source= a)
    while True:
        cv2.imshow('b',b)
        cv2.waitKey(10)
