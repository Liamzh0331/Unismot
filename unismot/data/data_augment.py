#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Copyright (c) Lian Zhang and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import cv2
import math
import numpy as np
import random
import torch

from unismot.utils import xyxy2cxcywh


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates


def random_perspective(
    img,
    imgl,
    imgv,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            imgl = cv2.warpPerspective(imgl, M, dsize=(width, height), borderValue=(114, 114, 114))
            imgv = cv2.warpPerspective(imgv, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            imgl = cv2.warpAffine(imgl, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            imgv = cv2.warpAffine(imgv, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # filter candidates
        i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, :4] = xy[i]
        
        targets = targets[targets[:, 0] < width]
        targets = targets[targets[:, 2] > 0]
        targets = targets[targets[:, 1] < height]
        targets = targets[targets[:, 3] > 0]
        
    return img, imgl, imgv, targets


def _distort(image, imagel, imagev):
    def _convert(image, imagel, imagev, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmpl = imagel.astype(float) * alpha + beta
        tmpv = imagev.astype(float) * alpha + beta

        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        tmpl[tmpl < 0] = 0
        tmpl[tmpl > 255] = 255
        tmpv[tmpv < 0] = 0
        tmpv[tmpv > 255] = 255
        image[:] = tmp
        imagel[:] = tmpl
        imagev[:] = tmpv

    image = image.copy()
    imagel = imagel.copy()
    imagev = imagev.copy()

    if random.randrange(2):
        _convert(image, imagel, imagev, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, imagel, imagev, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imagel = cv2.cvtColor(imagel, cv2.COLOR_BGR2HSV)
    imagev = cv2.cvtColor(imagev, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmpl = imagel[:, :, 0].astype(int) + random.randint(-18, 18)
        tmpv = imagev[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp
        imagel[:, :, 0] = tmpl
        imagev[:, :, 0] = tmpv

    if random.randrange(2):
        _convert(image[:, :, 1], imagel[:, :, 1], imagev[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    imagel = cv2.cvtColor(imagel, cv2.COLOR_HSV2BGR)
    imagev = cv2.cvtColor(imagev, cv2.COLOR_HSV2BGR)

    return image, imagel, imagev


def _mirror(image, imagel, imagev, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        imagel = imagel[:, ::-1]
        imagev = imagev[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, imagel, imagev, boxes

def hist(padded_img, mean):
    padded_img_unit = padded_img.astype(np.uint8)
    b,g,r = cv2.split(padded_img_unit)
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    padded_img_merge = cv2.merge([b_eq,g_eq,r_eq])
    padded_img_merge = padded_img_merge.astype(np.float32)

    return padded_img_merge

def preproc(image, imagel, imagev, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        padded_imgl = np.ones((input_size[0], input_size[1], 3)) * 114.0
        padded_imgv = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
        padded_imgl = np.ones(input_size) * 114.0
        padded_imgv = np.ones(input_size) * 114.0
    img = np.array(image)
    imgl = np.array(imagel)
    imgv = np.array(imagev)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    rl = min(input_size[0] / imgl.shape[0], input_size[1] / imgl.shape[1])
    rv = min(input_size[0] / imgv.shape[0], input_size[1] / imgv.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    resized_imgl = cv2.resize(
        imgl,
        (int(imgl.shape[1] * rl), int(imgl.shape[0] * rl)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_imgl[: int(imgl.shape[0] * rl), : int(imgl.shape[1] * rl)] = resized_imgl
    resized_imgv = cv2.resize(
        imgv,
        (int(imgv.shape[1] * rv), int(imgv.shape[0] * rv)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_imgv[: int(imgv.shape[0] * rv), : int(imgv.shape[1] * rv)] = resized_imgv

    padded_img = hist(padded_img, mean)

    padded_img = padded_img[:, :, ::-1]
    padded_imgl = padded_imgl[:, :, ::-1]
    padded_imgv = padded_imgv[:, :, ::-1]
    padded_img /= 255.0
    padded_imgl /= 255.0
    padded_imgv /= 255.0
    if mean is not None:
        padded_imgl -= mean
        padded_imgv -= mean
    if std is not None:
        padded_imgv /= std

    padded_img = padded_img.transpose(swap)
    padded_imgl = padded_imgl.transpose(swap)
    padded_imgv = padded_imgv.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    padded_imgl = np.ascontiguousarray(padded_imgl, dtype=np.float32)
    padded_imgv = np.ascontiguousarray(padded_imgv, dtype=np.float32)
    return padded_img, padded_imgl, padded_imgv, r


class TrainTransform:
    def __init__(self, p=0.5, rgb_means=None, std=None, max_labels=100):
        self.means = rgb_means
        self.std = std
        self.p = p
        self.max_labels = max_labels

    def __call__(self, image, imagel, imagev, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        ids = targets[:, 5].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 6), dtype=np.float32)
            image, imagel, imagev, r_o = preproc(image, imagel, imagev, input_dim, self.means, self.std)
            # image = np.ascontiguousarray(image, dtype=np.float32)
            return image, imagel, imagev, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        ids_o = targets_o[:, 5]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        image_t, imagel_t, imagev_t = _distort(image, imagel, imagev)
        image_t, imagel_t, imagev_t, boxes = _mirror(image_t, imagel_t, imagev_t, boxes)
        height, width, _ = image_t.shape
        image_t, imagel_t, imagev_t, r_ = preproc(image_t, imagel_t, imagev_t, input_dim, self.means, self.std)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        ids_t = ids[mask_b]

        if len(boxes_t) == 0:
            imagel_o = imagel.copy()
            imagev_o = imagev.copy()
            image_t, imagel_t, imagev_t, r_o = preproc(image_o, imagel_o, imagev_o, input_dim, self.means, self.std)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            ids_t = ids_o

        labels_t = np.expand_dims(labels_t, 1)
        ids_t = np.expand_dims(ids_t, 1)

        targets_t = np.hstack((labels_t, boxes_t, ids_t))
        padded_labels = np.zeros((self.max_labels, 6))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        image_t = np.ascontiguousarray(image_t, dtype=np.float32)
        imagel_t = np.ascontiguousarray(imagel_t, dtype=np.float32)
        imagev_t = np.ascontiguousarray(imagev_t, dtype=np.float32)
        return image_t, imagel_t, imagev_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, rgb_means=None, std=None, swap=(2, 0, 1)):
        self.means = rgb_means
        self.swap = swap
        self.std = std

    # assume input is cv2 img for now
    def __call__(self, imgi, imgl, imgv, res, input_size):
        imgi, imgl, imgv, _ = preproc(imgi, imgl, imgv, input_size, self.means, self.std, self.swap)
        return imgi, imgl, imgv, np.zeros((1, 5))
