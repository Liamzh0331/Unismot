# encoding: utf-8
# Copyright (c) Lian Zhang and its affiliates.

import argparse
import random
import warnings
import torch
import torch.backends.cudnn as cudnn

from loguru import logger
from unismot.core import Trainer, launch
from unismot.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser(description="YOLOX training parser")
    
    # Experiment and model configuration
    parser.add_argument("-expn", "--experiment-name", type=str, default=None, help="Name of the experiment")
    parser.add_argument("-n", "--name", type=str, default=None, help="Model name")
    
    # Distributed training configuration
    parser.add_argument("--dist-backend", type=str, default="nccl", help="Distributed backend")
    parser.add_argument("--dist-url", type=str, default=None, help="URL used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("-d", "--devices", type=int, default=2, help="Number of devices for training")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    
    # Experiment file and resuming training
    parser.add_argument("-f", "--exp_file", type=str, default='exps/example/mot/unismot_l_RGBTL2.py', help="Path to the experiment description file")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint")
    parser.add_argument("-c", "--ckpt", type=str, default='pretrained/unismot_l_RGBT.pth.tar', help="Path to the checkpoint file")
    parser.add_argument("-e", "--start_epoch", type=int, default=None, help="Epoch to start training from when resuming")
    
    # Multi-node training configuration
    parser.add_argument("--num_machines", type=int, default=1, help="Number of nodes for training")
    parser.add_argument("--machine_rank", type=int, default=0, help="Rank of the current node for multi-node training")
    
    # Training options
    parser.add_argument("--fp16", action="store_true", default=True, help="Use mixed precision training")
    parser.add_argument("--nlow", action="store_true", default=False, help="Add low light channels for training")
    parser.add_argument("--occupy", action="store_true", default=True, help="Occupy GPU memory for training to avoid memory fragmentation")
    
    # Additional options
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None, help="Modify config options using the command line")
    
    return parser


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name, args.nlow)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args),
    )
