#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Copyright (c) Lian Zhang and its affiliates.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        nlow=False,
    ):
        super().__init__()

        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act, nlow=nlow)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        #################################################################
        # self.backbone_i = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        # self.feature_fuse0 = BaseConv(
        #     int(in_channels[2] * width * 2), int(in_channels[2] * width), 1, 1, act=act
        # )
        # self.feature_fuse1 = BaseConv(
        #     int(in_channels[1] * width * 2), int(in_channels[1] * width), 1, 1, act=act
        # )
        # self.feature_fuse2 = BaseConv(
        #     int(in_channels[0] * width * 2), int(in_channels[0] * width), 1, 1, act=act
        # )
        ##################################################
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )

        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input_i, input_v, input_l = None):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone

        if input_l is not None:
            out_features = self.backbone(input_i, input_v, input_l)
        else:
            out_features = self.backbone(input_i, input_v)
        features = [out_features[f] for f in self.in_features]

        ######################################tou#############################
        # out_featuresi = self.backbone_i(input_i)
        # out_featuresv = self.backbone(input_v)
        # features_i = [out_featuresi[f] for f in self.in_features]
        # features_v = [out_featuresv[f] for f in self.in_features]
        # if input_l is not None:
        #     out_featuresl = self.backbone(input_l)
        #     features_l = [out_featuresl[f] for f in self.in_features]
        #     #features_v = [a + b for a, b in zip(features_v, features_l)]
        #     features_v = [torch.cat([a, b], 1) for a, b in zip(features_v, features_l)]
        #     [xv2, xv1, xv0] = features_v
        #     xv0 = self.feature_fuse0(xv0)
        #     xv1 = self.feature_fuse1(xv1)
        #     xv2 = self.feature_fuse2(xv2)
        #     features_v = [xv2, xv1, xv0]
        # # features = [a + b for a, b in zip(features_v, features_i)]
        # features = [torch.cat([a, b], 1) for a, b in zip(features_v, features_i)]
        #######################################wei#####################################

        [x2, x1, x0] = features
        ##################tou###################3
        # x0 = self.feature_fuse0(x0)
        # x1 = self.feature_fuse1(x1)
        # x2 = self.feature_fuse2(x2)
        ##################wei############################

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
