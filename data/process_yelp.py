import json

import pickle
import networkx

user_name_id_map = {}

with open("D:\fl\data\fl\sx-yelp-new.txt") as fp:
    lines = fp.readlines()
    for line in lines:
        u1, u2 = line.split()
