#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.distributed as dist
import random

from unismot.utils import synchronize


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_inputi, self.next_inputl, self.next_inputv, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_inputi = None
            self.next_inputl = None
            self.next_inputv = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputi = self.next_inputi
        inputl = self.next_inputl
        inputv = self.next_inputv
        target = self.next_target
        if inputi is not None:
            self.record_stream(inputi)
            self.record_stream(inputl)
            self.record_stream(inputv)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return inputi, inputl, inputv, target

    def _input_cuda_for_image(self):
        self.next_inputi = self.next_inputi.cuda(non_blocking=True)
        self.next_inputl = self.next_inputl.cuda(non_blocking=True)
        self.next_inputv = self.next_inputv.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


def random_resize(data_loader, exp, epoch, rank, is_distributed):
    tensor = torch.LongTensor(1).cuda()
    if is_distributed:
        synchronize()

    if rank == 0:
        if epoch > exp.max_epoch - 10:
            size = exp.input_size
        else:
            size = random.randint(*exp.random_size)
            size = int(32 * size)
        tensor.fill_(size)

    if is_distributed:
        synchronize()
        dist.broadcast(tensor, 0)

    input_size = data_loader.change_input_dim(multiple=tensor.item(), random_range=None)
    return input_size
