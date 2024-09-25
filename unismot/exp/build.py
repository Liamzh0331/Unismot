#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Copyright (c) Lian Zhang and its affiliates.

import importlib
import os
import sys


def get_exp_by_file(exp_file, needlow):
    try:
        # Resolve the absolute path of the experiment file
        exp_file = os.path.realpath(exp_file)
        # Add the directory containing the experiment file to the system path
        sys.path.append(os.path.dirname(exp_file))
        # Dynamically import the module corresponding to the experiment file
        module_name = os.path.basename(exp_file).split(".")[0]
        current_exp = importlib.import_module(module_name)
        # Instantiate the 'Exp' class from the imported module
        exp = current_exp.Exp()
        # Set the 'nlow' attribute of the experiment
        exp.nlow = needlow
    except Exception as e:
        raise ImportError(f"{exp_file} doesn't contain a class named 'Exp'. Error: {e}")
    return exp


def get_exp_by_name(exp_name):
    import unismot

    yolox_path = os.path.dirname(os.path.dirname(unismot.__file__))
    filedict = {
        "yolox-s": "yolox_s.py",
        "yolox-m": "yolox_m.py",
        "yolox-l": "yolox_l.py",
        "yolox-x": "yolox_x.py",
        "yolox-tiny": "yolox_tiny.py",
        "yolox-nano": "nano.py",
        "yolov3": "yolov3.py",
    }
    filename = filedict[exp_name]
    exp_path = os.path.join(yolox_path, "exps", "default", filename)
    return get_exp_by_file(exp_path)


def get_exp(exp_file, exp_name, needlow):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file, needlow)
    else:
        return get_exp_by_name(exp_name)
