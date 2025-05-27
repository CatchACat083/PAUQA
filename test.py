import argparse
import random
from argparse import ArgumentParser

import os
import numpy as np
import random

import logging
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch import optim
from torchvision.transforms.functional import to_tensor
from torchsummary import summary
import csv

import matplotlib.pyplot as plt

from utiles import *
from dataloader import *

from models.uiqa.pauqa_model import PAUQA 



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def logistic_function(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def map_quality_score(q, theta1, theta2, theta3, theta4, theta5):
    exp_component = np.exp(theta2 * (q - theta3))
    logistic_term = theta1 * (0.5 - 1 / (1 + exp_component))
    linear_term = theta4 * q
    constant_term = theta5
    return logistic_term + linear_term + constant_term

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Underwater IQA')

    parser.add_argument('--database', default='LUIQD_TEST', type=str, help='database name')
    parser.add_argument('--model', default='PAUQA', type=str, help='model name')

    # parser.add_argument('--pretrained', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    parser.add_argument('--multi_gpu', action='store_true', help='flag whether to use multiple GPUs')
    
    parser.add_argument('--save_csv', default=True, type=bool, help='flag whether to save result CSV files')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    # >>> [model] set model info.
    # >>>> <1> set models. >>>>> 
    if args.model == 'PAUQA':
        model = PAUQA(img_size=384, num_classes=1, num_stages=4,  
                        num_paths=[2, 3, 3, 3], patch_size=[[3,3],[3,3,3],[3,3,3],[3,3,3]], dilation_size=[[1,2],[1,2,3],[1,2,3],[1,2,3]],
                        blk_depths=[1, 2, 4, 1], embed_dims=[64, 128, 192, 256], #res_dims=[3, 4, 6, 3],
                        mlp_ratios=[4, 4, 4, 4], num_heads=[8, 8, 8, 8])
        args.pretrained = "./checkpoints/PAUQA_ckpt.pth"

    args.output_csv_path = './outputs_data/output_' + args.database + '_' + args.model + '.csv'

    model = model.to(memory_format=torch.channels_last)
    ckpt = torch.load(args.pretrained)
    model.load_state_dict(ckpt["model"])
    model.to(device=device)
    model.eval()

    # >>> [dataloader] set train data loader. >>>>>
    _, val_loader = data_loader(dataset_name = args.database, batch_size = 1)

    # Disable gradient computation for evaluation
    with torch.no_grad():
        outputs = []
        mos_values = []

        results = []

        metric_source = UWIQAPerformance(status='train')
        metric_source.reset()

        for batch in tqdm(val_loader, total=len(val_loader), desc='Validation round', unit='batch', leave=True, ncols=150):
            if args.save_csv:
                images, scores, names = batch
            else:
                images, scores = batch

            # move images and labels to correct device and type
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            scores = scores.to(device=device, dtype=torch.float32)

            output = model(images)
            outputs.extend(output.flatten().tolist())
            mos_values.extend(scores.tolist())


            metric_source.update((output, scores))

            if args.save_csv:
                predicted_scores = output.cpu().detach().numpy()
                for path, true_score, pred_score in zip(names, scores, predicted_scores):
                    parent_folder, filename = os.path.split(path)
                    parent_folder_name = os.path.basename(parent_folder)
                    short_path = os.path.join(parent_folder_name, filename)

                    results.append([short_path, true_score.item(), pred_score.item()])
    ### print validation metric info.
    performance = metric_source.compute()
    val_srocc, val_krocc, val_plcc, val_rmse, val_mae = \
                    performance['SROCC'], performance['KROCC'], performance['PLCC'], \
                    performance['RMSE'], performance['MAE']    
    print('Validation SROCC: {:.4f}, KROCC: {:.4f}, PLCC: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}'.format(val_srocc, val_krocc, val_plcc, val_rmse, val_mae))

