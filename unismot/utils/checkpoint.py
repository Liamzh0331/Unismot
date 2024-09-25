#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Copyright (c) Lian Zhang and its affiliates.

import torch
import os
import shutil

from loguru import logger


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        key_model_r = key_model
        if key_model_r in ckpt:
            v_ckpt = ckpt[key_model]
        else:
            # if 'backbone_i' in key_model_r:
            #     key_model_r = key_model_r.replace('backbone_i', 'backbone')
            #     if key_model_r in ckpt:
            #         v_ckpt = ckpt[key_model_r]
            #     else:
            #         continue
            # else:
            #     logger.warning(
            #         "{} is not in the ckpt. Please double check and see if this is desired.".format(
            #             key_model
            #         )
            #     )
            #     continue
            ##################### for yolo_m.pth ##################
            # if 'stem_v' in key_model_r:
            #     key_model_r = key_model_r.replace('stem_v', 'stem')
            #     if key_model_r in ckpt:
            #         v_ckpt = ckpt[key_model_r]
            #     else:
            #         continue
            # else:
            #     logger.warning(
            #         "{} is not in the ckpt. Please double check and see if this is desired.".format(
            #             key_model
            #         )
            #     )
            #     continue
        #########################################################
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth.tar")
        shutil.copyfile(filename, best_filename)
