# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import torch
from os.path import isfile

__version__ = '0.1.1'


def set_all(seed, n_gpu=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset


if is_tf_available():
    import tensorflow as tf


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list.
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def load(model, ckpt_path):
    assert isfile(ckpt_path), 'No model checkpoint found!'
    try:
        model.load_state_dict(torch.load(ckpt_path))
    except:
        state_dict = torch.load(ckpt_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return model