# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os.path as osp

def make_dir(dataset):
    if dataset == 'NTU':
        output_dir = os.path.join('./results/NTU/')
    elif dataset == 'NTU120':
        output_dir = os.path.join('./results/NTU120/')
    elif dataset == 'NTU_SRID':
        output_dir = os.path.join('./results/NTU_SRID/')
    elif dataset == 'SITC_SAR':
        output_dir = os.path.join('./results/SITC_SAR/')
    elif dataset == 'SITC_SRID':
        output_dir = os.path.join('./results/SITC_SRID/')
    elif dataset == 'PP-SITC_SAR':
        output_dir = os.path.join('./results/PP-SITC_SAR/')
    elif dataset == 'PP-SITC_SRID':
        output_dir = os.path.join('./results/PP-SITC_SRID/')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir

def get_num_classes(dataset):
    if dataset == 'NTU':
        return 60
    elif dataset == 'NTU120':
        return 120
    elif dataset == 'NTU_SRID':
        return 40
    elif dataset == 'SITC_SAR' or dataset == ('PP-SITC_SAR'):
        return 2
    elif dataset == 'SITC_SRID' or dataset == ('PP-SITC_SRID'):
        return 8
    
