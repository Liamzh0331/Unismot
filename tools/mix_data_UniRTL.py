# encoding: utf-8
# Copyright (c) Lian Zhang and its affiliates.

import json
import os

"""
cd datasets/UniRTL
mkdir annotations
cd ..
"""

def mix(datasets_ROOT):
    sot_json = json.load(open(datasets_ROOT+'/datasets/UniRTL/SOT/annotations/train.json', 'r'))

    train_img_list = sot_json['images']
    train_ann_list = sot_json['annotations']

    train_video_list = sot_json['videos']
    train_category_list = sot_json['categories']

    max_img = 132690
    max_ann = 132690
    max_video = 626

    mot_json = json.load(open(datasets_ROOT+'/datasets/UniRTL/MOT/annotations/train.json', 'r'))
    img_id_count = 0
    for img in mot_json['images']:
        img_id_count += 1
        img['file_name'] = img['file_name']
        img['frame_id'] = img_id_count
        img['prev_image_id'] = img['id'] + max_img
        img['next_image_id'] = img['id'] + max_img
        img['id'] = img['id'] + max_img
        img['video_id'] = img['video_id'] + max_video
        train_img_list.append(img)

    for ann in mot_json['annotations']:
        ann['id'] = ann['id'] + max_ann
        ann['image_id'] = ann['image_id'] + max_img
        train_ann_list.append(ann)

    train_video_list = train_video_list + mot_json['videos']

    print('SOT MOT train.json complete')

    sot_half_json = json.load(open(datasets_ROOT + '/datasets/UniRTL/SOT/annotations/train_half.json', 'r'))

    train_half_img_list = sot_half_json['images']
    train_half_ann_list = sot_half_json['annotations']

    train_half_video_list = sot_half_json['videos']
    train_half_category_list = sot_half_json['categories']

    max_half_img = 66836
    max_half_ann = 66836
    max_half_video = 626

    mot_half_json = json.load(open(datasets_ROOT + '/datasets/UniRTL/MOT/annotations/train_half.json', 'r'))
    img_id_count = 0
    for img in mot_half_json['images']:
        img_id_count += 1
        img['file_name'] = img['file_name']
        img['frame_id'] = img_id_count
        img['prev_image_id'] = img['id'] + max_half_img
        img['next_image_id'] = img['id'] + max_half_img
        img['id'] = img['id'] + max_half_img
        img['video_id'] = img['video_id'] + max_half_video
        train_half_img_list.append(img)

    for ann in mot_half_json['annotations']:
        ann['id'] = ann['id'] + max_half_ann
        ann['image_id'] = ann['image_id'] + max_half_img
        train_half_ann_list.append(ann)

    train_half_video_list = train_half_video_list + mot_half_json['videos']


    print('SOT MOT train_half.json complete')



    sot_val_json = json.load(open(datasets_ROOT + '/datasets/UniRTL/SOT/annotations/val_half.json', 'r'))

    val_img_list = sot_val_json['images']
    val_ann_list = sot_val_json['annotations']

    val_video_list = sot_val_json['videos']
    val_category_list = sot_val_json['categories']



    max_img_val = 65854
    max_ann_val = 65854
    max_video_val = 626

    mot_val_json = json.load(open(datasets_ROOT + '/datasets/UniRTL/MOT/annotations/val_half.json', 'r'))
    val_id_count = 0
    for img in mot_val_json['images']:
        val_id_count += 1
        img['file_name'] = img['file_name']
        img['frame_id'] = val_id_count
        img['prev_image_id'] = img['id'] + max_img_val
        img['next_image_id'] = img['id'] + max_img_val
        img['id'] = img['id'] + max_img_val
        img['video_id'] = img['video_id'] + max_video_val
        val_img_list.append(img)

    for ann in mot_val_json['annotations']:
        ann['id'] = ann['id'] + max_ann_val
        ann['image_id'] = ann['image_id'] + max_img_val
        val_ann_list.append(ann)

    val_video_list = val_video_list + mot_val_json['videos']

    print('SOT MOT val_half.json complete')

    mix_train_json = dict()
    mix_train_json['images'] = train_img_list
    mix_train_json['annotations'] = train_ann_list
    mix_train_json['videos'] = train_video_list
    mix_train_json['categories'] = train_category_list
    json.dump(mix_train_json, open(datasets_ROOT+'/datasets/UniRTL/annotations/train.json', 'w'))
    mix_train_half_json = dict()
    mix_train_half_json['images'] = train_half_img_list
    mix_train_half_json['annotations'] = train_half_ann_list
    mix_train_half_json['videos'] = train_half_video_list
    mix_train_half_json['categories'] = train_half_category_list
    json.dump(mix_train_half_json, open(datasets_ROOT + '/datasets/UniRTL/annotations/train_half.json', 'w'))
    mix_val_json = dict()
    mix_val_json['images'] = val_img_list
    mix_val_json['annotations'] = val_ann_list
    mix_val_json['videos'] = val_video_list
    mix_val_json['categories'] = val_category_list
    json.dump(mix_val_json, open(datasets_ROOT + '/datasets/UniRTL/annotations/val_half.json', 'w'))

if __name__ == '__main__':
    datasets_ROOT = '/home/liam/Unismot'  #  your path
    mix(datasets_ROOT)