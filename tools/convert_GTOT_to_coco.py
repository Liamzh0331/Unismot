# encoding: utf-8
# Copyright (c) Lian Zhang and its affiliates.

import os
import numpy as np
import json
import cv2
from unismot.data import GTOT_CLASSES

# Use the same script for MOT16
DATA_PATH = 'datasets/RGBT_mix/GTOT'
DATA_ROOT = 'datasets/RGBT_mix'
OUT_PATH = os.path.join(DATA_ROOT, 'annotations_GTOT')
SPLITS = ['train_half', 'val_half', 'train']  # --> split training data to train_half and val_half.
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True
classGTOT = GTOT_CLASSES

if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        if split == "test":
            data_path = os.path.join(DATA_PATH, 'SOT', 'test')
        else:
            data_path = os.path.join(DATA_PATH)
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': []}
        seqs = sorted(os.listdir(data_path))
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1
        cat_id = 0
        print(seqs)
        for i, clss in enumerate(classGTOT):
            classinfo = {'id': i+1, 'name': clss}
            out['categories'].append(classinfo)
        for seq in sorted(seqs):

            cat_id += 1
            if '.DS_Store' in seq:
                continue
            if 'MOT' in DATA_PATH and split == 'test':
                continue
            video_cnt += 1  # video sequence number.
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(data_path, seq)
            # img_path = os.path.join(seq_path, 'i')
            RGB_img_list = sorted([p for p in os.listdir(os.path.abspath(DATA_PATH) + '/' + seq + '/v') if
                                   os.path.splitext(p)[1] == '.png'])
            T_img_list = sorted([p for p in os.listdir(os.path.abspath(DATA_PATH) + '/' + seq + '/i') if
                                 os.path.splitext(p)[1] == '.png'])
            ann_path = os.path.join(seq_path, 'groundTruth_i.txt')
            if not os.path.exists(ann_path):
                continue
            # images = os.listdir(img_path)
            # num_images = len([image for image in images if 'png' in image])  # half and half
            num_images = len(T_img_list)

            if HALF_VIDEO and ('half' in split):
                image_range = [0, num_images // 2] if 'train' in split else \
                              [num_images // 2 + 1, num_images - 1]
            elif 'random' in split:
                image_range = [0, num_images // 4]
            else:
                image_range = [0, num_images - 1]

            for i, imgzip in enumerate(zip(T_img_list, RGB_img_list)):
                IR_img = imgzip[0]
                rgb_img = imgzip[1]
                if i < image_range[0] or i > image_range[1]:
                    continue
                img = cv2.imread(os.path.join(data_path, '{}/i/{}'.format(seq, IR_img)))
                height, width = img.shape[:2]
                image_info = {'file_name': 'GTOT/{}/i/{}'.format(seq, IR_img),  # image name.
                              'file_name_rgb': 'GTOT/{}/v/{}'.format(seq, rgb_img),
                              'id': image_cnt + i + 1,  # image number in the entire training set.
                              'frame_id': i + 1 - image_range[0],  # image number in the video sequence, starting from 1.
                              'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height, 'width': width}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))
            if split != 'test':
                #det_path = os.path.join(seq_path, 'det/det.txt')
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=None)
                x_min = np.min(anns[:, [0, 2]], axis=1)[:, None]
                y_min = np.min(anns[:, [1, 3]], axis=1)[:, None]
                x_max = np.max(anns[:, [0, 2]], axis=1)[:, None]
                y_max = np.max(anns[:, [1, 3]], axis=1)[:, None]
                anns = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
                #dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')
                if CREATE_SPLITTED_ANN and ('half' in split):
                    anns_out = np.array([anns[i] for i in range(anns.shape[0])
                                         if i - 1 >= image_range[0] and
                                         i - 1 <= image_range[1]], np.float32)
                    anns_out[:, 0] -= image_range[0]
                    gt_out = os.path.join(seq_path, 'gt_{}.txt'.format(split))
                    fout = open(gt_out, 'w')
                    for o in anns_out:
                        fout.write('{:d},{:d},{:d},{:d}\n'.format(
                                    int(o[0]), int(o[1]), int(o[2]), int(o[3])))
                    fout.close()
                if CREATE_SPLITTED_ANN and ('random' in split):
                    anns_out = np.array([anns[i] for i in range(anns.shape[0])
                                         if i - 1 >= image_range[0] and
                                         i - 1 <= image_range[1]], np.float32)
                    anns_out[:, 0] -= image_range[0]
                    gt_out = os.path.join(seq_path, 'gt_{}.txt'.format(split))
                    fout = open(gt_out, 'w')
                    for o in anns_out:
                        fout.write('{:d},{:d},{:d},{:d}\n'.format(
                            int(o[0]), int(o[1]), int(o[2]), int(o[3])))
                    fout.close()
                '''
                if CREATE_SPLITTED_DET and ('half' in split):
                    dets_out = np.array([dets[i] for i in range(dets.shape[0])
                                         if int(dets[i][0]) - 1 >= image_range[0] and
                                         int(dets[i][0]) - 1 <= image_range[1]], np.float32)
                    dets_out[:, 0] -= image_range[0]
                    det_out = os.path.join(seq_path, 'det/det_{}.txt'.format(split))
                    dout = open(det_out, 'w')
                    for o in dets_out:
                        dout.write('{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n'.format(
                                    int(o[0]), int(o[1]), float(o[2]), float(o[3]), float(o[4]), float(o[5]),
                                    float(o[6])))
                    dout.close()
                '''

                print('{} ann images'.format(int(anns[:, 0].max())))
                for i in range(anns.shape[0]):
                    frame_id = i + 1
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    tid_curr = 0

                    ann_cnt += 1

                    category_id = cat_id
                    ann = {'id': ann_cnt,
                           'category_id': category_id,
                           'image_id': image_cnt + frame_id,
                           'track_id': tid_curr,
                           'bbox': anns[i][0:4].tolist(),
                           'conf': float(1),
                           'iscrowd': 0,
                           'area': float(anns[i][2] * anns[i][3])}
                    out['annotations'].append(ann)
            image_cnt += num_images
            print(tid_curr, tid_last)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))