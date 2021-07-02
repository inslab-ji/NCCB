#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import numpy as np
from .Update import DatasetSplit
from .Dataset import NLPDataset


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def test_nlp(model, dataset, args):
    model.eval()
    next_words = 1
    test_loss = 0
    perplex = []
    data_loader = DataLoader(DatasetSplit(NLPDataset(dataset), "0", length=1), batch_size=args.local_bs,
                             shuffle=True)
    loss_func = torch.nn.CrossEntropyLoss()
    state_h, state_c = model.init_state(10)
    state_h, state_c = state_h.to(args.device), state_c.to(args.device)
    for idx, (data, target, mask) in enumerate(data_loader):
        data = data.to(args.device)
        target = data.to(args.device)
        mask = mask.to(args.device)
        for i in range(0, next_words):
            y_pred, (state_h, state_c) = model(data, (state_h, state_c))
            test_loss += loss_func(y_pred.transpose(1, 2), target)
        perplex.append(np.exp(test_loss.item()).item())

    test_loss /= len(dataset)

    return sum(perplex) / len(perplex), test_loss
