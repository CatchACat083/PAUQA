'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-10-30 13:47:02
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-11-25 15:12:58
FilePath: /workspace/Single_image_regression/my_NR-IQM/utiles.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
from scipy import stats
import torch
from torch import nn
import copy

from ignite.metrics.metric import Metric


class UWIQAPerformance(Metric):
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE.

    `update` must receive output of the form (y_pred, y).
    """
    def __init__(self, status='train'):
        super(UWIQAPerformance, self).__init__()
        self.status = status

    def reset(self):
        self._y_pred = []
        self._y      = []

    def update(self, output):
        y_pred, y = output

        # y = torch.unsqueeze(y, 1)
        # self._y = y.tolist().copy()
        # #self._y.append(y[0].item())
        # self._y_pred = y_pred.tolist().copy()
        # #self._y_pred.append(torch.mean(y_pred).item())

        self._y.extend([t.item() for t in y])
        self._y_pred.extend([t.item() for t in y_pred])


    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
        #sq_std = np.reshape(np.asarray(self._y_std), (-1,))
        q = np.reshape(np.asarray(self._y_pred), (-1,))

        # sq = np.asarray(self._y)
        # q = np.asarray(self._y_pred)
        
        srocc = stats.spearmanr(sq, q)[0]
        krocc = stats.stats.kendalltau(sq, q)[0]
        plcc = stats.pearsonr(sq, q)[0]
        rmse = np.sqrt(((sq - q) ** 2).mean())
        mae = np.abs((sq - q)).mean()

        # return srocc, krocc, plcc, rmse, mae
        return  {'SROCC': srocc,
                 'KROCC': krocc,
                 'PLCC': plcc,
                 'RMSE': rmse,
                 'MAE': mae
                }

    '''
    description: 
                        recall=0.1 
                        positive label should be 1, negative label should be -1
    param {*} label_gt  ground truth label, (n, 1) matrix, n is the number of images
    param {*} prob_es   confidence map of being positive, matrix (n, 1),
    param {*} ratio     recall ratio, e.g. 0.1 means for the precision when
    return {*}
    '''    
    def mapcompt(self, label_gt_list, prob_es_list, ratio = 1):
        positive_label = 1
        positive_indexs = [i for i, x in enumerate(label_gt_list) if x == positive_label ]

        sorted_prob_positive = []
        for i in positive_indexs:
            sorted_prob_positive.append(prob_es_list[i])
        sorted_prob_positive.sort(reverse=True)

        ap_cc = 0
        for i in range(1, len(positive_indexs) * ratio):
            index_retrived = [j for j, x in enumerate(prob_es_list) if x >= sorted_prob_positive[i]]

            label_retrived = []
            for j in index_retrived:
                label_retrived.append(label_gt_list[j])

            a = 0
            for j in label_retrived:
                if j ==positive_label:
                    a = a + 1
            ap_cc = ap_cc + a / len(label_retrived)
            
        ap = ap_cc / (len(positive_indexs) * ratio)
        return ap

    