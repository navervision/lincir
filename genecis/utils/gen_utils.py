# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Sagar Vaze from https://github.com/sgvaze/generalized-category-discovery/blob/main/project_utils/general_utils.py

import argparse
import torch
import os
import random
from datetime import date
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def _to_rgb(image: Image):
    return image.convert('RGB')

def bool_flag(s):

    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def none_flag(s):

    """
    Parse None arguments from the command line.
    """
    NONEY_STRINGS = {None, 'None', 'none'}
    if s in NONEY_STRINGS:
        return None
    else:
        return s

def strip_state_dict(state_dict: torch.nn.Module.state_dict, strip_key: str = 'module.'):

    """
    Strip strip_key from start of state_dict keys
    Useful if model has been trained as DDP model
    """

    for k in list(state_dict.keys()):
        if k.startswith(strip_key):
            state_dict[k[len(strip_key):]] = state_dict[k]
            del state_dict[k]

    return state_dict

def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:
            p.grad.data = p.grad.data.float() 

def step_scheduler(scheduler, metric, args):

    if args.scheduler == 'reduce_on_plateau':
        scheduler.step(metric)
    else:
        scheduler.step()

def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]

def create_uq_log_dir(root_dir):

    # Create random log file
    log_str = '_'.join([date.today().strftime("%d.%m.%Y"), f'{random.getrandbits(16):04x}'])
    log_dir = os.path.join(root_dir, log_str)
    
    return log_dir

def set_requires_grad_clip(finetune_mode, clip_model):


    if finetune_mode == 'text_only':

        for p in clip_model.transformer.parameters():
            p.requires_grad = True

    elif finetune_mode == 'image_only':
        
        for p in clip_model.visual.attnpool.parameters():
            p.requires_grad = True

    elif finetune_mode == 'image_plus_text':

        for p in clip_model.visual.attnpool.parameters():
            p.requires_grad = True

        for p in clip_model.transformer.resblocks[11].parameters():
            p.requires_grad = True

    elif finetune_mode == 'text_only_whole':

        for p in clip_model.transformer.parameters():
            p.requires_grad = True
        
    elif finetune_mode == 'image_only_whole':

        for p in clip_model.visual.parameters():
            p.requires_grad = True

    elif finetune_mode == 'image_plus_text_whole':

        for p in clip_model.parameters():
            p.requires_grad = True

    elif finetune_mode == None:

        pass

    else:

        raise ValueError

