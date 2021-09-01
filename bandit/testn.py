import numpy as np
from bandit import ContextualBandit
from main import LinBandit
from neuralucb import NeuralUCB
from linucb import LinUCB
import itertools
import matplotlib.pyplot as plt

### Bandit settings

T = int(200)
n_arms = 1000
n_features = 256
noise_std = 0.1

confidence_scaling_factor = noise_std

n_sim = 2

SEED = 42
np.random.seed(SEED)
x = np.random.randn(T, n_arms, n_features)
x /= np.repeat(np.linalg.norm(x, axis=-1, ord=2), n_features).reshape(T, n_arms, n_features)
features = x

### mean reward function
a = np.random.randn(n_features)
a /= np.linalg.norm(a, ord=2)
h =  lambda x: np.cos(10*np.pi*np.dot(x, a))

rewards = np.array(
            [
                h(features[t, k]) + noise_std*np.random.randn()
                for t, k in itertools.product(range(T), range(n_arms))
            ]
        ).reshape(T, n_arms)

p = 0.2
hidden_size = 16
epochs = 100
train_every = 10
confidence_scaling_factor = 1.0
use_cuda = True


linbandit = LinBandit(T,n_arms,n_features,features)
model_lin=linbandit.model

for i in range(T):
    a=model_lin.per_run_1(i)
    model_lin.per_run_2(i,rewards[i, a])


