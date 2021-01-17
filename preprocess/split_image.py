#!/user/bin/python
# -*- coding: utf-8 -*-
"""
@File: division_image.py
@Description: 描述
@Author: dongyanqiang
@Email: 181228331@163.com
@Date: 2021/1/10
"""
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import time
from multiprocessing import Pool

from utils import get_json_data, json_data_to_images_data, check_or_make_dir, generate_val_dataset


def division_according_sliding(image, size, stride, padding=False):
    """
    通过滑动窗口的方式分割图像
    :param image: 要分割的图像，array数组
    :param size: 分割后的图像大小
    :param stride: 移动步长
    :param padding: 是否填充
    :return: 分割后的图像列表
    """
    ori_size = image.shape[:2]
    size = np.array(size)
    stride = np.array(stride)
    # 计算分割后的图像数量
    num = (ori_size - size) // stride + 1
    if padding:
        num += 1
        pad = (num - 1) * stride + size - ori_size
        image = np.pad(image, ((0, pad[0]), (0, pad[1]), (0, 0)), 'constant', constant_values=0)
    else:
        crop = ori_size - (num - 1) * stride - size
        image = image[:-crop[0], :-crop[1], :]
    # print(ori_size, size, stride, num)
    division_images = []
    original_position = []
    for i in range(num[0]):
        for j in range(num[1]):
            start = np.array([i, j]) * stride
            end = start + size
            crop_img = image[start[0]:end[0], start[1]:end[1], :]
            # print(start, end, crop_img.shape)
            division_images.append(crop_img)
            original_position.append(list(start[::-1]))
    return division_images, original_position


def division_according_point(image, size, boxes):
    """
    通过指定点的方式分割图片
    :param image: 要分割的图像
    :param size: 分割后的图像大小
    :param boxes: 瑕疵box list
    :return: 分割后的图像列表
    """
    ori_size = image.shape[:2]
    size = np.array(size)
    num = len(boxes)
    division_images = []
    for i in range(num):
        box = np.array(boxes[i], dtype=np.int)
        point = np.array(box[:2], dtype=np.int)
        box_size = box[2:] - box[:2]
        high = np.where(size-box_size <= 10, size, size-box_size)
        crop_y = np.random.randint(0, high[0])
        crop_x = np.random.randint(0, high[1])
        crop_lt = np.array([crop_x, crop_y])
        crop_lt = point - crop_lt
        crop_lt = np.where(crop_lt < 0, 0, crop_lt)
        crop_tl = crop_lt[::-1]
        crop_br = crop_tl + size
        crop_br = np.where(crop_br > ori_size, ori_size, crop_br)
        crop_tl = crop_br - size
        crop = np.append(crop_tl, crop_br)
        print(ori_size, crop)
        crop_img = image[crop[0]:crop[2], crop[1]:crop[3], :]
        division_images.append(crop_img)
    return division_images


def show(img, win_name):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


def deal_bbox(original_position, ori_bbox):
    """
    将给定的bbox坐标转换为分割后的坐标
    :param original_position: 分割后的图像原点在原图的坐标
    :param ori_bbox: 原始的bbox坐标
    :return: bbox在分割后的图片中的坐标
    """
    bboxes = []
    for x, y in original_position:
        bbox = ori_bbox
        bbox = np.array(bbox) - np.array([x, y, x, y])
        bboxes.append(bbox)
    return bboxes


def filter_bbox_and_images(bboxes, category, image_size, min_size=5):
    """
    过滤不在图像内的bbox和没有bbox的图像
    :param bboxes: 转换后的bbox的坐标
    :param category: 瑕疵类别
    :param image_size: 分割的图像大小
    :param min_size: bbox在图像边缘的最小值
    :return: 有效的bbox信息
    """
    valid_images_info = {}
    for i, bbox in enumerate(bboxes):
        if np.all(bbox < 0):
            continue
        valid_bbox = []
        valid_category = []
        for j in range(len(bbox)):
            x_min = bbox[j][0]
            x_max = bbox[j][2]
            y_min = bbox[j][1]
            y_max = bbox[j][3]
            if x_max <= min_size or y_max <= min_size:
                continue
            if x_min >= image_size[0] - min_size or y_min >= image_size[1] - min_size:
                continue
            box = np.where(bbox[j] < 0, 0, bbox[j])
            box[2:] = np.where(box[2:] > image_size, image_size, box[2:])
            valid_bbox.append(box)
            valid_category.append(category[j])
        if valid_bbox:
            valid_images_info[i] = {
                "bbox": valid_bbox,
                "category": valid_category,
            }
    return valid_images_info


def save_images(save_path, save_name, save_img):
    """
    保存图片
    :param save_path:
    :param save_name:
    :param save_img:
    :return:
    """
    path = os.path.join(save_path, save_name)
    if not os.path.exists(path):
        cv2.imwrite(path, save_img)


def save_yolo_labels(save_path, save_name, save_label):
    """
    保存yolo格式的label
    :param save_path:
    :param save_name:
    :param save_label:
    :return:
    """
    path = os.path.join(save_path, save_name)
    if os.path.exists(path):
        return
    category = save_label["category"]
    height = save_label["height"]
    width = save_label["width"]
    bbox = save_label["bbox"]
    with open(path, "w") as fp:
        for i, box in enumerate(bbox):
            center_x = (box[0] + box[2]) / 2 / width
            center_y = (box[1] + box[3]) / 2 / height
            bw = (box[2] - box[0]) / width
            bh = (box[3] - box[1]) / height
            #print(center_x, center_y, bw, bh, path)  # clw delete
            data = "%d %.6f %.6f %.6f %.6f\n" % (category[i] - 1, center_x, center_y, bw, bh)
            fp.write(data)


def save_yolo_dataset(save_path, save_name, save_img, save_label):
    """
    将分割后的图像和label保存成yolo格式的
    :param save_path:
    :param save_name:
    :param save_img:
    :param save_label:
    :return:
    """
    save_images_dir = "images"
    save_labels_dir = "labels"
    save_images_path = os.path.join(save_path, save_images_dir)
    if not os.path.exists(save_images_path):
        os.mkdir(save_images_path)
    train_images_path = os.path.join(save_images_path, "train")
    if not os.path.exists(train_images_path):
        os.mkdir(train_images_path)

    save_labels_path = os.path.join(save_path, save_labels_dir)
    if not os.path.exists(save_labels_path):
        os.mkdir(save_labels_path)
    train_labels_path = os.path.join(save_labels_path, "train")
    if not os.path.exists(train_labels_path):
        os.mkdir(train_labels_path)

    save_images(train_images_path, save_name, save_img)
    save_name = save_name.split(".")[0] + ".txt"
    save_yolo_labels(train_labels_path, save_name, save_label)

count = 0
def deal_one_image(img_name, images_data, images_dir_path, divide_size, stride, save_path):
    global count
    print(count)
    count+=1
    """
    分割一张图像
    :param img_name: 图像名
    :param images_data: 图像label信息
    :param images_dir_path: 图像所在目录
    :param divide_size: 分割大小
    :param stride: 移动步长
    :param save_path: 保存路径
    :return:
    """
    #print(img_name)  # clw delete
    image_path = os.path.join(images_dir_path, img_name)
    image_data = images_data.get(img_name, None)
    if image_data is None:
        return
    #print(image_path, image_data) # clw delete
    if not os.path.exists(image_path):
        return
    img = cv2.imread(image_path)
    divide_images, original_position = division_according_sliding(img, divide_size, stride, True)
    bboxes = deal_bbox(original_position, image_data["bbox"])
    valid_images_info = filter_bbox_and_images(bboxes, image_data["category"], divide_size[::-1])
    if not valid_images_info:
        return
    for i, (index, item) in enumerate(valid_images_info.items()):
        save_name = img_name[:-4] + "_{:02}".format(i) + img_name[-4:]
        save_img = divide_images[index]
        item["height"] = divide_size[0]
        item["width"] = divide_size[1]
        save_yolo_dataset(save_path, save_name, save_img, item)
        #print(save_name, index, item)  # clw delete


def main():
    #root = r"D:\python\competition\dataset\tile_round1_train_20201231"
    root = "/home/user/dataset/tile_round1_train_20201231"
    json_file = "train_annos.json"
    images_dir = "train_imgs"
    save_path = "/home/user/dataset/tile_round1_split"
    # divide_size = [128, 128]  # [h, w]
    # stride = [120, 120]       # [h, w]
    # max_num = 1000

    #divide_size = [640, 640]  # [h, w]
    divide_size = [416, 416]
    overlap_ratio = 0.2
    stride = [int( divide_size[0]*(1-overlap_ratio) ), int( divide_size[1]*(1-overlap_ratio)) ]       # [h, w]
    max_num = 1000000

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = check_or_make_dir(save_path, "size{}X{}".format(divide_size[0], divide_size[1]), mkdir=True)

    images_dir_path = os.path.join(root, images_dir)
    json_file_path = os.path.join(root, json_file)

    images_name = os.listdir(images_dir_path)
    json_data = get_json_data(json_file_path)
    images_data = json_data_to_images_data(json_data)
    random.shuffle(images_name)
    p = Pool(12)  # clw modify: origin is 4
    for i, img_name in enumerate(tqdm(images_name)):
        if max_num != -1:
            if i >= max_num:
                continue
        try:
            # deal_one_image(img_name, images_data, images_dir_path, divide_size, stride, save_path)
            p.apply_async(deal_one_image, args=(img_name, images_data, images_dir_path, divide_size, stride, save_path, ))
        except Exception as e:
            print(e, img_name)
            continue
    p.close()
    p.join()
    generate_val_dataset(save_path, p=0.2)  #


if __name__ == "__main__":
    print("split_image.py")
    start = time.time()
    main()
    print("time use: %.3fs" % (time.time() - start))

'''
    Pool设置不同大小耗时对比    max_num=100, 切图100张测试
    1: time use: 43.348s
    2: time use: 22.464s
    4: time use: 12.209s
    6: time use: 8.993s
    8: time use: 7.855s
    12: time use: 7.619s  7.118s  7.202s
    24: time use: 7.525s
    48: time use: 7.619s
    
    既然python解释器是单线程的，还有进行多线程编程的必要吗？
    有必要，至少能解决很多IO阻塞问题。能产生IO阻塞的情况很多，比如网络、磁盘，等等。当发生阻塞时，Python是不耗CPU的，此时如果就一个线程就没法处理其他事情了。
    所以对于含有IO阻塞的环境。多线程至少有机会让你把一个CPU核心跑到100%。另一个用处来自于Python的C扩展模块。在扩展模块里是可以释放GIL的。
    但释放GIL期间不应该调用任何Python API。所以，对于一些非常繁重的计算，可以写成C模块，计算前释放GIL，计算后重新申请GIL，并将结果返回给Python。
    这样就可以让Python这个进程利用更多的CPU资源。每个Python的线程都是OS级别pthread的线程。利用Python来管理这些线程比在C层级操作pthread更方便。

'''