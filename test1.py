import torch
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

args = args_parser()

dataset_train = {"users": [], "user_data": {}}
with open("./data/superuser/superuser_trainnew.json", "rb") as file:
    jf = json.load(file)
    dataset_train["users"].extend(jf["users"])
    dataset_train["user_data"].update(jf["user_data"])
dataset_test = {"users": [], "user_data": {}}
with open("./data/superuser/superuser_test.json", "rb") as file:
    jf = json.load(file)
    dataset_test["users"].extend(jf["users"])
    dataset_test["user_data"].update(jf["user_data"])
test_id = ["567231", "201818", "38001", "219655", "213663", "37440", "170233", "9556", "114058", "542839"]
dataset_test_new = dict()
all_idx = dataset_train['users']
for u in test_id:
    dataset_test_new[u] = dataset_test["user_data"][u]
    # all_idx.remove(u)
vocab_file = pickle.load(open("./data/vocab/superuser_vocab.pck", "rb"))
vocab = collections.defaultdict(lambda: vocab_file['unk_symbol'])
vocab.update(vocab_file['vocab'])
dict_users = superuser_noniid(dataset_train['user_data'], vocab)
test_users = superuser_noniid(dataset_test_new, vocab)

model = NLPModel(vocab=vocab)
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
model.load_state_dict(torch.load('./model/superuserBANDIT56.pth'))

loss_locals = []
acc_locals = []
# net = copy.deepcopy(net_glob).to(args.device)
for it, idx in enumerate(test_id):
    local = LocalUpdate_nlp(args=args, dataset=NLPDataset(test_users), idxs=idx, len=len)
    ppl, loss = local.test(model)
    loss_locals.append(copy.deepcopy(loss))
    acc_locals.append(copy.deepcopy(ppl))