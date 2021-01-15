# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/1/5
"""
import os
import shutil
import random
import json
import cv2
import torch
import numpy as np


def get_json_data(file_path):
    if not os.path.exists(file_path):
        print("Error ", file_path)

    with open(file_path, "r") as fp:
        return json.loads(fp.read())


def json_data_to_images_data(json_data):
    """
    将给定的label转换成以图片名为key的字典
    :param json_data:
    :return:
    """
    images_data = {}
    for item in json_data:
        if item["name"] not in images_data:
            images_data[item["name"]] = {
                'height': item["image_height"],
                'width': item["image_width"],
                'category': [item["category"]],
                'bbox': [item["bbox"]],
            }
        else:
            images_data[item["name"]]["category"].append(item["category"])
            images_data[item["name"]]["bbox"].append(item["bbox"])
    return images_data


def read_txt_label(label_path):
    data = []
    with open(label_path, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            if line.strip():
                data.append(line.replace("\n", ""))
    return data


def show(img, win_name="show"):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


def check_or_make_dir(root, dir_name, mkdir=False):
    dir_path = os.path.join(root, dir_name)
    if not os.path.exists(dir_path):
        if mkdir:
            os.mkdir(dir_path)
        else:
            raise ValueError("Error path: " + dir_path)
    return dir_path


def generate_val_dataset(root, p=0.3):
    train_images_path = os.path.join(root, "images", "train")
    val_images_path = os.path.join(root, "images", "val")
    if not os.path.exists(val_images_path):
        os.mkdir(val_images_path)

    train_labels_path = os.path.join(root, "labels", "train")
    val_labels_path = os.path.join(root, "labels", "val")
    if not os.path.exists(val_labels_path):
        os.mkdir(val_labels_path)
    images_files = os.listdir(train_images_path)
    labels_files = os.listdir(train_labels_path)

    assert len(images_files) == len(labels_files)
    print(images_files)
    print(labels_files)

    random.shuffle(images_files)

    train_len = int(len(images_files) * p)
    for i in range(train_len):
        img_name = images_files[i]
        label_name = img_name.replace(".jpg", ".txt")
        ori_img_file = os.path.join(train_images_path, img_name)
        ori_lab_file = os.path.join(train_labels_path, label_name)

        new_img_file = os.path.join(val_images_path, img_name)
        new_lab_file = os.path.join(val_labels_path, label_name)

        shutil.move(ori_img_file, new_img_file)
        shutil.move(ori_lab_file, new_lab_file)


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.mat = None

    def update(self, a, b):
        if self.mat is None:
            self.mat = torch.zeros((3, ), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            inds = a.to(torch.int64) + b
            count = torch.bincount(inds)
            for i in range(count.shape[0]):
                # print(i, count[i])
                self.mat[i] += count[i]
            # print(self.mat)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # acc_global = torch.diag(h).sum() / h.sum()
        # acc = torch.diag(h) / h.sum(1)
        iu = h[2] / (h[1] + h[2])
        return iu

    def __str__(self):
        iu = self.compute()
        return 'mean IoU: {:.3f}'.format(
                iu.mean().item() * 100)


def main():
    root = r"D:\learnspace\dataset\project_001\divide_tile_round1"
    images_dir = "images"
    labels_dir = "labels"
    # labels_mask_dir = "labels_mask"

    images_dir_path = check_or_make_dir(root, images_dir)
    labels_dir_path = check_or_make_dir(root, labels_dir)

    # save_mask_path = check_or_make_dir(root, labels_mask_dir, mkdir=True)
    # generate_mask_dataset(images_dir_path, labels_dir_path, save_mask_path)


if __name__ == '__main__':
    print("utils.py")
    # main()