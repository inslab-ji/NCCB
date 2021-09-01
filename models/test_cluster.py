import pickle
import json
import os
import numpy as np
from sklearn.cluster import KMeans

data_files = [f for f in os.listdir("../data/yelp_leaf/train") if f.endswith('.json')]
users = []

for f in data_files:
    with open("../data/yelp_leaf/train/" + f, "rb") as file:
        jf = json.load(file)
        users.extend(jf["users"])
users = users[:-50]
with open("../data/yelp_leaf/user.json", "w") as file:
    json.dump(users, file)
with open('../data/yelp_leaf/yelp_hash_floc_64.pck', 'rb') as file:
    feature = pickle.load(file)
f = np.zeros((len(users), 64))
for i, u in enumerate(users):
    f[i] = feature[u]
estimator = KMeans(n_clusters=5)
estimator.fit(f)
pred_classes = estimator.predict(f)
cluster = {}
for i, pred in enumerate(pred_classes):
    if pred not in cluster.keys():
        cluster[pred] = []
    cluster[pred].append(i)
with open("../data/yelp_leaf/yelp_cluster_5.pck", "wb") as file:
    pickle.dump(cluster, file)
