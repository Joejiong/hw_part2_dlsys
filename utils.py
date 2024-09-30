from __future__ import print_function

import os
import sys 

import torch
import torch.nn as nn

def print_layer_info(model):
    index = 0
    print()
    for m in model.modules():
        if hasattr(m, 'alpha'):
            print('MaskLayer', index, ':',
                    m.alpha.data.nelement()-int(m.alpha.data.eq(1.0).sum()), 'of',
                    m.alpha.data.nelement(), 'is blocked')
            index += 1
    print()
    return

def print_args(args):
    print('\n==> Setting params:')
    for key in vars(args):
        print(key, ':', getattr(args, key))
    print('====================\n')
    return

def load_state(model, state_dict):
    param_dict = dict(model.named_parameters())
    state_dict_keys = state_dict.keys()
    cur_state_dict = model.state_dict()
    for key in cur_state_dict:
        if key in state_dict_keys:
            cur_state_dict[key].copy_(state_dict[key])
        elif key.replace('module.','') in state_dict_keys:
            cur_state_dict[key].copy_(state_dict[key.replace('module.','')])
        elif 'module.'+key in state_dict_keys:
            cur_state_dict[key].copy_(state_dict['module.'+key])
    return

class dropout_update():
    def __init__(self, dropout_list, mask_list):
        self.dropout_list = dropout_list
        self.mask_list = mask_list
        self.dropout_param_list = []
        for i in range(len(self.dropout_list)):
            self.dropout_param_list.append(self.dropout_list[i].p)
        return

    def update(self):
        for i in range(len(self.dropout_list)):
            mask = self.mask_list[i].alpha.data
            dropout_tmp_value = float(mask.eq(1.0).sum()) / float(mask.nelement())
            dropout_tmp_value = dropout_tmp_value * self.dropout_param_list[i]
            self.dropout_list[i].p = dropout_tmp_value
        return