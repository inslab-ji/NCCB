#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, length):
        self.dataset = dataset
        self.idxs = idxs
        self.length = length
        #  self.l = min([self.dataset[i]['x'].__len__() for i in self.dataset.data.keys()])
        self.l = self.dataset[idxs]['x'].__len__()

    def __len__(self):
        return self.l

    def __getitem__(self, item):
        xy = self.dataset[self.idxs]
        item = item % self.l
        return torch.tensor(xy['x'][item]), torch.tensor(xy['y'][item]), torch.ByteTensor(xy['mask'][item])


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdate_nlp(object):
    def __init__(self, args, dataset=None, idxs=None, len=10, batch_size=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if batch_size is None:
            batch_size = self.args.local_bs
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, length=len), batch_size=batch_size,
                                    shuffle=True)

    def train(self, net, lr):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        epoch_loss = []
        epoch_correct = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_correct = []
          #  perplex = []
            state_h, state_c = net.init_state()
            state_h, state_c = state_h.to(self.args.device), state_c.to(self.args.device)

            for batch, (x, y, mask) in enumerate(self.ldr_train):
                acc = 0
                x, y = x.to(self.args.device), y.to(self.args.device)
                optimizer.zero_grad()

                y_pred, (state_h, state_c) = net(x, (state_h, state_c))
                loss = self.loss_func(y_pred.transpose(1, 2), y)

                state_h = state_h.detach()
                state_c = state_c.detach()

                loss.to(self.args.device)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=5, norm_type=2)
                optimizer.step()
                for i, pred in enumerate(y_pred):
                    last_word_logits = pred[-1]
                    p = nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
                    indices = np.argsort(p)[-5:]
                    acc += y.cpu()[i][-1].item() in indices
                batch_correct.append(acc)

                if self.args.verbose and batch % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch * len(x), len(self.ldr_train.dataset),
                              100.0 * batch / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
        #        perplex.append(np.exp(loss.item()).item())
            epoch_loss.append((sum(batch_loss) * len(batch_loss)) ** 0.5)
            epoch_correct.append(sum(batch_correct) / len(batch_correct))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_correct) / len(epoch_correct)

    def test(self, model):
        model.eval()
        test_loss = 0
    #    perplex = 0
        state_h, state_c = model.init_state(10)
        state_h, state_c = state_h.to(self.args.device), state_c.to(self.args.device)
        acc = 0
        for idx, (data, target, mask) in enumerate(self.ldr_train):
            data = data.to(self.args.device)
            target = target.to(self.args.device)
            y_pred, (state_h, state_c) = model(data, (state_h, state_c))
            last_word_logits = y_pred[0][-1]
            p = nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
            indices = np.argsort(p)[-5:]
            acc += target.cpu()[0][-1].item() in indices
            test_loss += self.loss_func(y_pred.transpose(1, 2), target).item()
            state_h = state_h.detach()
            state_c = state_c.detach()
    #        perplex += np.exp(self.loss_func(y_pred.transpose(1, 2), target).item()).item()

        test_loss /= len(self.ldr_train.dataset)

        return test_loss, acc / len(self.ldr_train.dataset)
