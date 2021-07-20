#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import pickle
import os
import json
import collections
from torch.utils.tensorboard import SummaryWriter
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, superuser_noniid
from utils.options import args_parser
from models.Update import LocalUpdate, LocalUpdate_nlp
from models.Nets import MLP, CNNMnist, CNNCifar, NLPModel
from models.Dataset import NLPDataset
from models.Fed import FedAvg
from models.test import test_img, test_nlp
from models.Client import degree, sort_degree
from bandit.main import Bandit


import logging

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='new1.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )
                

superuser_train_path = "/home/sjtu/hangrui/federated-learning-master/data/superuser/superuser_trainnew.json"
superuser_test_path = "/home/sjtu/hangrui/federated-learning-master/data/superuser/superuser_test.json"
superuser_vocab_path = "/home/sjtu/hangrui/federated-learning-master/data/vocab/superuser_vocab.pck"

yelp_train_path = "/home/sjtu/hangrui/federated-learning-master/data/yelp_leaf/train/"

yelp_test_path = "/home/sjtu/hangrui/federated-learning-master/data/yelp_leaf/test/"
yelp_vocab_path = "/home/sjtu/hangrui/federated-learning-master/data/vocab/yelp_vocab.pck"

yelp_feature_path = "/home/sjtu/yelp_hash_floc_64.pck"

FIXED_TEST = False
if __name__ == '__main__':
    # parse args
    args = args_parser()
    print("CLIENT SELECTION Method:"+args.selection_method)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print("the device used for training is : ", args.device)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'superuser':
        logging.info("Use superuser dataset")
        dataset_train = {"users": [], "user_data": {}}
        with open(superuser_train_path, "rb") as file:
            jf = json.load(file)
            dataset_train["users"].extend(jf["users"])
            dataset_train["user_data"].update(jf["user_data"])
            all_idx = dataset_train['users']

        dataset_test = {"users": [], "user_data": {}}
        with open(superuser_test_path, "rb") as file:
            jf = json.load(file)
            dataset_test["users"].extend(jf["users"])
            dataset_test["user_data"].update(jf["user_data"])

        if FIXED_TEST:
            test_id = ["567231", "201818", "38001", "219655", "213663", "37440", "170233", "9556", "114058", "542839"]
            dataset_test_new = dict()
            for u in test_id:
                dataset_test_new[u] = dataset_test["user_data"][u]
                all_idx.remove(u)
        else:
            
            pass
        vocab_file = pickle.load(open(superuser_vocab_path, "rb"))
        vocab = collections.defaultdict(lambda: vocab_file['unk_symbol'])
        vocab.update(vocab_file['vocab'])
        dict_users = superuser_noniid(dataset_train['user_data'], vocab)
        if FIXED_TEST:
            test_users = superuser_noniid(dataset_test_new, vocab)
        else:
            test_users = superuser_noniid(dataset_test["user_data"], vocab)
        args.num_users = len(all_idx)
        if args.selection_method == "BANDIT":
            """f = np.load("./data/val.npy")
            new_f = np.zeros((CLIENTNUMBER * args.epochs, args.num_users, 16), dtype=np.float32)
            with open("./data/superuser/rev.json") as file:
                rev = json.load(file)
            for i, u in enumerate(all_idx):
                for j in range(CLIENTNUMBER * args.epochs):
                    new_f[j, i] = f[int(rev[u])]
            f = new_f"""
            new_f = np.zeros((args.client_number * args.epochs, args.num_users, args.fsize), dtype=np.float32)
            with open("./data/supergfuser_hash_floc.pck", "rb") as file:
                f = pickle.load(file)
            for i, u in enumerate(all_idx):
                for j in range(args.client_number * args.epochs):
                    new_f[j, i] = f[u]
            f = new_f
    elif args.dataset == 'yelp':
        data_files = [f for f in os.listdir(yelp_train_path) if f.endswith('.json')]
        dataset_train = {"users": [], "user_data": {}}

        for f in data_files:
            with open(yelp_train_path + f, "rb") as file:
                jf = json.load(file)
                dataset_train["users"].extend(jf["users"])
                dataset_train["user_data"].update(jf["user_data"])
        data_files = [f for f in os.listdir(yelp_test_path) if f.endswith('.json')]
        dataset_test = {"users": [], "user_data": {}}
        for f in data_files:
            with open(yelp_test_path +f, "rb") as file:
                jf = json.load(file)
                dataset_test["users"].extend(jf["users"])
                dataset_test["user_data"].update(jf["user_data"])
        vocab_file = pickle.load(open(yelp_vocab_path, "rb"))
        vocab = collections.defaultdict(lambda: vocab_file['unk_symbol'])
        vocab.update(vocab_file['vocab'])
        dict_users = superuser_noniid(dataset_train['user_data'], vocab)
        test_users = superuser_noniid(dataset_test['user_data'], vocab)
        all_idx = dataset_train['users']
        test_id = dataset_test['users']
        args.num_users = len(all_idx)
        if args.selection_method == "BANDIT":
            # f = np.load("./data/val.npy")
            # new_f = np.zeros((args.client_number * args.epochs, args.num_users, 16), dtype=np.float32)
            # with open("./data/superuser/rev.json") as file:
            #     rev = json.load(file)
            # for i, u in enumerate(all_idx):
            #     for j in range(args.client_number * args.epochs):
            #         new_f[j, i] = f[int(rev[u])]
            # f = new_f
            print('load 64 floc')
            new_f = np.zeros((args.client_number * args.epochs, args.num_users, args.fsize), dtype=np.float32)
            with open(yelp_feature_path, "rb") as file:
                f = pickle.load(file)
            for i, u in enumerate(all_idx):
                for j in range(args.client_number*args.epochs):
                    new_f[j,i] = f[u]
            f = new_f
    else:
        exit('Error: unrecognized dataset')

    if args.selection_method == "CLUSTER":
        with open("./data/superuser/" +  args.cluster_filepath + ".json", "rb") as file:
            rev_d = json.load(file)

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        img_size = dataset_train[0][0].shape
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'nlp':
        net_glob = NLPModel(vocab=vocab)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    min_loss = 0
    # training
    loss_train = []
    test_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    write = SummaryWriter('./tensorboard/' + args.dataset + args.selection_method + args.save_filename + '/' + str(args.client_number))
    if args.selection_method == "BANDIT":
        bandit = Bandit(args.client_number * args.epochs, args.num_users, args.fsize, f, True)

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    if args.selection_method == "CLUSTER":
        cluster_ref = set(rev_d.keys())
    lr = 1e-3
    chosen_clients_per_round = []
    for iter in range(args.epochs):
        loss_locals = []
        acc_locals = []

        if not args.all_clients:
            w_locals = []
        m = args.client_number
        if args.selection_method == "RANDOM":
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        elif args.selection_method == "CLUSTER":
            idxs_users, cluster_ref = degree(rev_d, m, cluster_ref)
            if cluster_ref.__len__() < m:
                cluster_ref = set(rev_d.keys())
        elif args.selection_method == "BANDIT":
            idxs_users = [bandit.get_arm(iter * args.client_number)]

        chosen_clients_per_round.append(idxs_users)
        # idxs_users = sort_degree(rev_d, m)
        # ma = max(map(int, rev_d.keys()))
        # rev_d[str(min(map(int, rev_d.keys())) - 1)] = rev_d[str(ma)]
        # del rev_d[str(ma)]

        lr = lr * 0.993
        train_losses = []
        train_acc = []
        for it, idx in enumerate(idxs_users):
            if args.selection_method == "RANDOM" or args.selection_method == "BANDIT":
                id = all_idx[idx]
            else:
                id = idx
            local = LocalUpdate_nlp(args=args, dataset=NLPDataset(dict_users), idxs=id, len=len)
            w, loss, acc = local.train(net=copy.deepcopy(net_glob).to(args.device), lr=lr, topk=args.topk)
            train_losses.append(loss)
            train_acc.append(acc)
            if args.selection_method == "BANDIT":
                bandit.set_reward(iter * args.client_number + it, loss)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            if args.selection_method == "BANDIT" and it + 1 < args.client_number:
                idxs_users.append(bandit.get_arm(iter * args.client_number + it + 1))

        # update global weights
        w_glob = FedAvg(w_locals)
        logging.info('The training loss in round ', (iter + 1), ' is ', np.average(train_losses), 'Acc is ', np.average(train_acc))

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        write.add_scalar("Train Acc", np.average(train_acc))
        write.add_scalar("Train loss", np.average(train_losses))

        # print loss
        if (iter + 1) % args.test_round == 0:
            # net = copy.deepcopy(net_glob).to(args.device)
            for it, idx in enumerate(idxs_users):
                if args.selection_method == "RANDOM" or args.selection_method == "BANDIT":
                    id = all_idx[idx]
                else:
                    id = idx
                local = LocalUpdate_nlp(args=args, dataset=NLPDataset(test_users), idxs=id, len=len, batch_size=1)
                loss, acc = local.test(net_glob.to(args.device), args.topk)
                
                loss_locals.append(copy.deepcopy(loss))
                acc_locals.append(copy.deepcopy(acc))
            loss_avg = sum(loss_locals) / loss_locals.__len__()
            acc_avg = sum(acc_locals) / acc_locals.__len__()
            loss_train.append(acc_avg)
            write.add_scalar("Test Acc", acc_avg, iter)
            write.add_scalar("Test Loss", loss_avg, iter)
            print()
            logging.info('Round ' + str(iter + 1) + 'Test Acc ' + str(acc_avg) + '; Test Loss' + str(loss_avg))
            if acc_avg > min_loss:
                min_loss = acc_avg
                print("save model")
                torch.save(net_glob.state_dict(), 'weights/' + args.dataset + args.selection_method + args.save_filename +
                           str(args.client_number) + '.pth')
    write.close()
    torch.save(net_glob.state_dict(), 'weights/' + args.dataset + args.selection_method + args.save_filename + 'clientnumber'+
               str(args.client_number) + 'fsize' + str(args.fsize)+'end.pth')
    if args.selection_method == "RANDOM":
        with open("save/random" + args.save_filename + ".json", "w") as file:
            file.write(json.dumps(loss_train))
    elif args.selection_method == "CLUSTER":
        with open("save/partition" + args.save_filename + ".json", "w") as file:
            file.write(json.dumps(loss_train))
    elif args.selection_method == "BANDIT":
        with open("save/bandit" + args.save_filename + 'clientnumber'+
               str(args.client_number) +  'fsize' + str(args.fsize) + ".json", "w") as file:
            file.write(json.dumps(loss_train))
    with open("save/" + args.selection_method + args.save_filename + 'clientnumber'+
               str(args.client_number) +  'fsize' + str(args.fsize) + "test.json", "w") as file:
        file.write(json.dumps(test_train))

    with open("save/" + args.selection_method + args.save_filename  + 'clientnumber'+
               str(args.client_number) +  'fsize' + str(args.fsize) + "chosen_client.pck", "wb") as file:
        pickle.dump(chosen_clients_per_round, file)
