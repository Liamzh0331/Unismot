# encoding: utf-8
# Copyright (c) Lian Zhang and its affiliates.

import json
import os

"""
cd datasets
mkdir -p annotations
cd ..
"""

def mix(datasets_ROOT):
    GTOT_train_json = json.load(open(datasets_ROOT+'/datasets/annotations_GTOT/train.json', 'r'))

    train_img_list = GTOT_train_json['images']
    train_ann_list = GTOT_train_json['annotations']

    train_video_list = GTOT_train_json['videos']
    train_category_list = GTOT_train_json['categories']

    print('GTOT complete')

    max_img = len(train_img_list)
    max_ann = len(train_img_list)
    max_video = len(train_category_list)

    RGBT234_train_json = json.load(open(datasets_ROOT+'/datasets/annotations_RGBT234/train.json', 'r'))
    img_id_count = 0
    for img in RGBT234_train_json['images']:
        img_id_count += 1
        img['file_name'] = img['file_name']
        img['file_name_rgb'] = img['file_name_rgb']
        img['frame_id'] = img_id_count
        img['prev_image_id'] = img['id'] + max_img
        img['next_image_id'] = img['id'] + max_img
        img['id'] = img['id'] + max_img
        img['video_id'] = img['video_id'] + max_video
        train_img_list.append(img)

    for ann in RGBT234_train_json['annotations']:
        ann['id'] = ann['id'] + max_ann
        ann['image_id'] = ann['image_id'] + max_img
        train_ann_list.append(ann)

    for cate in RGBT234_train_json['categories']:
        cate['id'] = cate['id'] + max_video
        cate['name'] = cate['name']
        train_category_list.append(cate)

    train_video_list = train_video_list + RGBT234_train_json['videos']

    print('RGB234 complete')





    GTOT_val_json = json.load(open(datasets_ROOT + '/datasets/annotations_GTOT/val_half.json', 'r'))

    val_img_list = GTOT_val_json['images']
    val_ann_list = GTOT_val_json['annotations']

    val_video_list = GTOT_val_json['videos']
    val_category_list = GTOT_val_json['categories']

    print('GTOT val complete')

    max_img_val = len(val_img_list)
    max_ann_val = len(val_img_list)
    max_video_val = len(val_video_list)

    RGBT234_val_json = json.load(open(datasets_ROOT + '/datasets/annotations_RGBT234/val_half.json', 'r'))
    val_id_count = 0
    for img in RGBT234_val_json['images']:
        val_id_count += 1
        img['file_name'] = img['file_name']
        img['file_name_rgb'] = img['file_name_rgb']
        img['frame_id'] = val_id_count
        img['prev_image_id'] = img['id'] + max_img_val
        img['next_image_id'] = img['id'] + max_img_val
        img['id'] = img['id'] + max_img_val
        img['video_id'] = img['video_id'] + max_video_val
        val_img_list.append(img)

    for ann in RGBT234_val_json['annotations']:
        ann['id'] = ann['id'] + max_ann_val
        ann['image_id'] = ann['image_id'] + max_img_val
        val_ann_list.append(ann)

    for cate in RGBT234_val_json['categories']:
        cate['id'] = cate['id'] + max_video
        cate['name'] = cate['name']
        val_category_list.append(cate)

    val_video_list = val_video_list + RGBT234_val_json['videos']

    print('RGBT234 val complete')

    mix_train_json = dict()
    mix_train_json['images'] = train_img_list
    mix_train_json['annotations'] = train_ann_list
    mix_train_json['videos'] = train_video_list
    mix_train_json['categories'] = train_category_list
    json.dump(mix_train_json, open(datasets_ROOT+'/datasets/annotations/train.json', 'w'))
    mix_val_json = dict()
    mix_val_json['images'] = val_img_list
    mix_val_json['annotations'] = val_ann_list
    mix_val_json['videos'] = val_video_list
    mix_val_json['categories'] = val_category_list
    json.dump(mix_val_json, open(datasets_ROOT + '/datasets/annotations/val_half.json', 'w'))

if __name__ == '__main__':
    datasets_ROOT = '/home/liam/Unismot'
    mix(datasets_ROOT)