import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import torchvision
import json

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadTileImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized


submit_result = []

# clw modify
def non_max_suppression_big(prediction, iou_thres=0.6):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    ###output = [None] * prediction.shape[0]
    boxes, scores = prediction[:, :4] , prediction[:, 4]  # boxes (offset by class), scores
    i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
    return prediction[i]


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    save_img = True
    #dataset = LoadImages(source, img_size=imgsz)
    dataset = LoadTileImages(source)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    preds = torch.Tensor([]).cuda()
    step_h = int(0.8 * imgsz)
    step_w = int(0.8 * imgsz)
    for path, img, im0s in dataset: # im0s: (h, w, c)  img: (c, h, w)
        ######################################################
        img_h, img_w = im0s.shape[:2]
        img_name = path.split('/')[-1]  # clw added

        for start_h in range(0, img_h, step_h):  # imgsz is crop step here,
            if start_h + imgsz > img_h:                  # 如果最后剩下的不到imgsz,则step少一些,保证切的图尺寸不变
                start_h = img_h - imgsz

            for start_w in range(0, img_h, step_w):
                if start_w + imgsz > img_w:  # 如果最后剩下的不到imgsz,则step少一些,保证切的图尺寸不变
                    start_w = img_w - imgsz
                # crop
                img_crop = img[:, start_h:start_h+imgsz, start_w:start_w+imgsz ]
                img_crop = torch.from_numpy(img_crop).to(device)
                img_crop = img_crop.half() if half else img_crop.float()  # uint8 to fp16/32
                img_crop /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img_crop.ndimension() == 3:
                    img_crop = img_crop.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img_crop, augment=opt.augment)[0]  # (1, n, 4+1+class_nums),  4+1+class_nums = xyxy + obj_conf + cls_conf

                # first nms, because there are too many results
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

                # box coord map to big img
                if pred[0] is None:
                    continue
                else:
                    pred[0][ :, 0:4:2] += start_w
                    pred[0][ :, 1:5:2] += start_h
                    preds = torch.cat((preds, pred[0]), 0)  # (n, 6)

        # 和原代码统一,因为前面都是对img_crop做的操作
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        ######################################################

        # Apply NMS
        pred = non_max_suppression_big(preds, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred = pred.unsqueeze(0)  # 和原代码统一

        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        # Process detections
        for i, det in enumerate(pred):  # detections per image

            submit_result.append(
                {'name': img_name, 'category': det[4], 'bbox': det[:4], 'score': det[5]})





            p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()  # clw delete

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        ##########################################
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        # clw modify
                        pass

                        ##########################################

                    if save_img or view_img:  # Add bbox to image
                        label = '%s' % (names[int(cls)])
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)


    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    ###
    print(submit_result)
    with open('result.json', 'w') as fp:
        json.dump(submit_result, fp, indent=4, ensure_ascii=False)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4-p5.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
            #detect(save_img=True)  # clw modify
