#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Copyright (c) Lian Zhang and its affiliates.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, needlow=False):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(35)

        self.backbone = backbone
        self.head = head
        self.nlow = needlow

    def forward(self, xi, xv, xl=None, targets=None, FTP=None):
        # fpn output content features of [dark3, dark4, dark5]
        if self.nlow:
            fpn_outs = self.backbone(xi, xv, xl)
        else:
            fpn_outs = self.backbone(xi, xv, input_l=None)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, xi
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            if FTP is not None:
                outputs = self.head(fpn_outs, labels=None, imgs=None, FTP=FTP)
            else:
                outputs = self.head(fpn_outs)


        return outputs
