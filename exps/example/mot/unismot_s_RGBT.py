# encoding: utf-8
# Copyright (c) Lian Zhang and its affiliates.

import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from unismot.exp import Exp as MyExp
from unismot.data import get_yolox_datadir

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 35
        self.depth = 0.33 # tiny:0.33 #s:0.33 # l:1.0 # m:0.67  # x:1.33
        self.width = 0.50 # tiny:0.375 #s:0.50 # l:1.0 # m:0.75  # x:1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = "train_half.json"
        self.val_ann = "val_half.json"
        self.input_size = (480, 640)   #(800, 1440)
        self.test_size = (480, 640)    #(800, 1440)
        self.random_size = (18, 32)
        self.max_epoch = 15  # 15
        self.print_interval = 20 # 20
        self.eval_interval = 1 # 5
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.no_aug_epochs = 8  # 8
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        self.nlow = False

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from unismot.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "UniRTL"),
            json_file=self.train_ann,
            name='train',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.403, 0.409, 0.375),
                std=(0.164, 0.161, 0.162),
                max_labels=120,
            ),
        ) # std=(0.229, 0.224, 0.225), # rgb_means=(0.485, 0.456, 0.406),

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.403, 0.409, 0.375),
                std=(0.164, 0.161, 0.162),
                max_labels=120,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from unismot.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "UniRTL"),
            json_file=self.val_ann,
            img_size=self.test_size,
            name='train',
            preproc=ValTransform(
                rgb_means=(0.403, 0.409, 0.375),
                std=(0.164, 0.161, 0.162),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from unismot.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
