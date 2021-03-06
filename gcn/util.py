import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_size, hid_sizes, out_size, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.in_size = in_size
        self.hid_sizes = hid_sizes
        self.out_size = out_size

        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        self.num_layers = len(dims) - 1
        for idx, (_in, out) in enumerate(zip(dims[:-1], dims[1:])):
            weight = nn.Parameter(torch.rand((_in, out)))
            bias = nn.Parameter(torch.rand(out))
            setattr(self, 'weight_{}'.format(idx), weight)
            setattr(self, 'bias_{}'.format(idx), bias)

    def weight(self, lvl):
        return getattr(self, 'weight_{}'.format(lvl))

    def bias(self, lvl):
        return getattr(self, 'bias_{}'.format(lvl))

    def forward(self, input):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        act = input
        for lvl in range(self.num_layers):
            h = torch.mm(act, self.weight(lvl)) + self.bias(lvl)
            act = nn.ReLU(h)

        return h

def count_triangles(adj):
    n = len(adj)
    num_tri = 0
    for i in range(len(adj)):
        for j in range(i):
            for k in range(j):
                if adj[i, j] == 1 and adj[i, k] == 1 and adj[j, k] == 1:
                    print('triangle at ({}, {}, {})'.format(i, j, k))
                num_tri += (1 if (adj[i, j] == 1 and adj[i, k] == 1 and adj[j, k] == 1) else 0)

    return num_tri

def add_triangle(adj, i, j, k):
    adj[i, j] = adj[j, i] = 1
    adj[i, k] = adj[k, i] = 1
    adj[j, k] = adj[k, j] = 1

def test_triangle():

    adj = np.zeros((10, 10))
    add_triangle(adj, 0, 1, 2)
    add_triangle(adj, 3, 1, 2)
    add_triangle(adj, 3, 5, 2)
    add_triangle(adj, 3, 5, 1)
    '''
    triangle at (2, 1, 0)
    triangle at (3, 2, 1)
    triangle at (5, 2, 1)
    triangle at (5, 3, 1)
    triangle at (5, 3, 2)
    '''
    print(count_triangles(adj))

if __name__ == '__main__':
    test_triangle()
