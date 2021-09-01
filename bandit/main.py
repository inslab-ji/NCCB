import numpy as np
from .bandit import ContextualBandit
from .neuralucb import NeuralUCB
from .linucb import LinUCB
import itertools
import matplotlib.pyplot as plt


# Bandit settings
class Bandit:
    def __init__(self, T, n_arms, n_features, features, use_cuda):
        self.bandit = ContextualBandit(T, n_arms, n_features, noise_std=0.1, seed=42, features=features)
        self.model = NeuralUCB(self.bandit,
                               hidden_size=16,
                               reg_factor=1.0,
                               delta=0.1,
                               confidence_scaling_factor=1,
                               training_window=10,
                               p=0.2,
                               learning_rate=0.01,
                               epochs=100,
                               train_every=1,
                               use_cuda=use_cuda
                               )

    def get_arm(self, iter):
        arm = self.model.per_run_1(1)
        self.model.action = arm[0].item()
        return arm[0].item()

    def set_reward(self, iter, r):
        self.model.per_run_2(iter, r)


class LinBandit:
    def __init__(self, T, n_arms, n_features, features):
        noise_std = 0.1
        self.bandit = ContextualBandit(T, n_arms, n_features, noise_std=0.1, seed=42, features=features)
        self.model = LinUCB(self.bandit,
                            reg_factor=0.1,
                            delta=0.1,
                            confidence_scaling_factor=noise_std
                            )

    def get_arm(self, iter):
        arm = self.model.per_run_1(1)
        self.model.action = arm[0]
        return arm

    def set_reward(self, iter, r):
        self.model.per_run_2(iter, r)


import math


class OortBandit:
    def __init__(self, args):
        self.num_clients = args.num_users

        self.clients = np.arange(self.num_clients, dtype='int')
        self.uninitialized = [i for i in range(self.num_clients)]
        self.lastinit = 0  # 0: initializing, 1: the last round for initialization, 2: working
        self.participation = [0] * self.num_clients
        self.available_clients = set([i for i in range(self.num_clients)])

        self.u = [0] * self.num_clients
        self.lastround = [1] * self.num_clients
        self.round = 0

        self.lamb = 0.2
        self.clientpoolsize = int(self.num_clients * self.lamb)
        self.maxparticipation = 100

    def requireArms(self, num_picked):
        self.round += 1

        if num_picked > self.num_clients:
            print('Too much clients picked')
            exit(0)

        # All required arms is uninitialized
        if len(self.uninitialized) >= num_picked:
            if len(self.uninitialized) == num_picked and self.lastinit == 0:
                self.lastinit = 1
            result = np.random.choice(self.uninitialized, num_picked, replace=False)
            for i in result:
                self.uninitialized.remove(i)
            return result

        if self.lastinit == 0:
            self.lastinit = 1

        # Part of arms is uninitialized
        if len(self.uninitialized) > 0:
            reserved = np.array(self.uninitialized, dtype='int')
            num_left = num_picked - len(self.uninitialized)
            self.uninitialized.clear()
            temp = self.clients.copy()
            for i in reserved:
                temp = np.delete(temp, np.argwhere(temp == i))
            newpicked = np.random.choice(temp, num_left, replace=False)
            result = np.concatenate([reserved, newpicked])
            return result

        # All arms initialized
        clientpoolsize = max(self.clientpoolsize, num_picked)
        util = self.__util()
        sortarms = sorted(util.items(), key=lambda x: x[1], reverse=True)
        clientpool = np.zeros(clientpoolsize, dtype='int')
        clientutil = np.zeros(clientpoolsize, dtype='float')
        for i in range(clientpoolsize):
            clientpool[i] = sortarms[i][0]
            clientutil[i] = sortarms[i][1]
        clientutil = clientutil / clientutil.sum()
        draw = np.random.choice(clientpool, num_picked, p=clientutil, replace=False)
        return draw

    def updateWithRewards(self, loss):
        arm, reward = loss
        self.lastround[arm] = self.round
        self.u[arm] = reward
        self.participation[arm] += 1
        if self.participation[arm] >= self.maxparticipation and arm in self.available_clients and len(
                self.available_clients) > 10:
            self.available_clients.remove(arm)

    def __util(self):
        util = {}
        for i in self.available_clients:
            util[i] = self.u[i] + math.sqrt(0.1 * math.log(self.round) / self.lastround[i])
        return util


import math
import logging


class CCMAB:
    def __init__(self, args, cluster_dict, T, n_arms, n_features, features, use_cuda):
        self.bandit = ContextualBandit(T, n_arms, n_features, noise_std=0.1, seed=42, features=features)
        self.model = NeuralUCB(self.bandit,
                               hidden_size=16,
                               reg_factor=1.0,
                               delta=0.1,
                               confidence_scaling_factor=1,
                               training_window=10,
                               p=0.2,
                               learning_rate=0.01,
                               epochs=100,
                               train_every=1,
                               use_cuda=use_cuda
                               )
        #    self.num_cluster = args.num_clusters
        #   self.num_clients = args.num_users
        #    self.rewards = {}
        self.cluster_dict = cluster_dict
        self.reverse_dict = {}
        self.counter = {}
        for cl, users in cluster_dict.items():
            for u in users:
                self.reverse_dict[u] = cl
            self.counter[cl] = 0
        #      self.uninitialized = [i for i in range(self.num_cluster)]
        self.T = 1

    def kfunc(self, t):

        return 10

    def getrandFromcluster(self, chosenids):
        arms = []
        for cid in chosenids:
            arms.append(np.random.choice(self.cluster_dict[cid], 1)[0])
        return arms

    def getUnexplored(self):
        Unexplored_cluster = []
        for clusterid, Cp in self.counter.items():

            if Cp <= self.kfunc(self.T):
                Unexplored_cluster.append(clusterid)
        return Unexplored_cluster

    def getArms(self, B):
        # Check for the uninitialized
        c = self.getUnexplored()
        idx_users = []
        if len(c) >= B:
            chosen_cluster = np.random.choice(c, B, replace=False)
            idx_users = self.getrandFromcluster(chosen_cluster)
        else:
            if len(c) > 0:
                idx_users = self.getrandFromcluster(c)
            ids = self.model.per_run_1(B)
            for i in ids:
                if len(idx_users) < B and i not in idx_users:
                    idx_users.append(i)
        return idx_users

    def updateReward(self, idx, reward):
        """
            First Update the Reward then Update the Counter:
        """
        self.model.per_run_2(self.T - 1, reward, idx)
        self.counter[self.reverse_dict[idx]] += 1
        self.T += 1
    #  for id, loss in idrewarddict:
    #     self.model.per_run_2(self.T, loss, id)
    #    #    self.rewards[cid] = (self.counter(cid) * self.rewards[cid] + loss) / (self.counter[cid] + 1)
    #   self.counter[self.reverse_dict[id]] += 1
    #  self.T += 1
