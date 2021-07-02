import numpy as np


def degree(rev_d, m, client_ref):
    degrees = np.random.choice(list(rev_d.keys()), m, replace=False)
    clients = []
    client_ref = client_ref - set(degrees)
    for d in degrees:
        clients.append(np.random.choice(rev_d[d], 1)[0])
    return clients, client_ref


def sort_degree(rev_d,m):
    degrees = [max(map(int,rev_d.keys()))]
    clients = []
    for d in degrees:
        clients.append(np.random.choice(rev_d[str(d)], 1)[0])
    return clients
