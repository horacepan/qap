import torch
import numpy as np

def get_graph(num_feats, prob=0.5):
    '''
    Make an erdos reyni graph with some random number of vertices
    Features are constant
    '''
    size = np.random.randint(5, 30)
    feats = torch.ones(size, num_feats)
    feats = torch.rand((size, num_feats))
    adj = erdos_reyni(size, prob)
    return feats, adj

def get_pair_graphs(num_feats, noise, prob=0.5):
    f1, adj1 = get_graph(num_feats, prob)
    f2, adj2 = f1.clone(), adj1.clone()
    return f1, adj1, f2, adj2

def erdos_reyni(n, prob):
    adj = torch.zeros((n, n))
    for i in range(n):
        for j in range(i):
            adj[i, j] = adj[j, i] = (1 if np.random.random() > prob else 0)
    return adj

def get_pair_erdos_reyni(n, prob, noise):
    pass
