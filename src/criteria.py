# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F


def calculate_acc(output, target):
    pred = output.data.max(1)[1]
    correct = pred.eq(target.data).cpu().sum().numpy()
    return correct * 100.0 / target.size()[0]


def calculate_correct(output, target):
    pred = output.data.max(1)[1]
    correct = pred.eq(target.data).cpu().sum().numpy()
    return correct


def contrast_loss(output, target):
    gt_value = output
    noise_value = torch.zeros_like(gt_value)
    G = gt_value - noise_value
    zeros = torch.zeros_like(gt_value)
    zeros.scatter_(1, target.view(-1, 1), 1.0)
    return F.binary_cross_entropy_with_logits(G, zeros)
    