import pdb
import numpy as np
from layers import GraphConvLayer
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.gc1 = GraphConvLayer(in_feats, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, out_feats)
        self.out = nn.Linear(out_feats, 1)
        self.dropout_rate = dropout_rate
        self.nonlinearity = F.relu
        # output is a linear layer?


    def forward(self, x, adj):
        x = self.nonlinearity(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout_rate, training=self.training)
        #x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.gc2(x, adj)
        # x is now what? n x out_feats
        x = self.out(x.sum(dim=0))
        return x

def erdos_reyni(n, prob):
    adj = torch.zeros((n, n))
    for i in range(n):
        for j in range(i):
            adj[i, j] = adj[j, i] = (1 if np.random.random() > prob else 0)
    return adj

def get_graph(num_feats, prob=0.5):
    '''
    Make an erdos reyni graph with some random number of vertices
    '''
    size = np.random.randint(5, 30)
    feats = torch.rand(size, num_feats)
    adj = erdos_reyni(size, prob)
    return feats, adj

def test_eval(model, test_size):
    preds = np.zeros(test_size)
    truth = np.zeros(test_size)
    for i in range(test_size):
        feats, adj = get_graph(num_feats)
        y = len(adj)
        truth[i] = y
        v_feats, v_adj = Variable(feats), Variable(adj)

        output = gcn(Variable(feats), Variable(adj))
        preds[i] = output.data[0]

    loss = np.mean(np.sum(np.square(preds - truth)))
    return loss

if __name__ == '__main__':
    in_feats = num_feats = 3
    hidden_size = 3
    out_feats = 4
    dropout_rate = 0.5
    num_iters = 50000
    test_size = 100


    gcn = GCN(in_feats, hidden_size, out_feats, dropout_rate)
    loss_fn = F.mse_loss
    optimizer = optim.Adam(gcn.parameters(), lr=0.001, weight_decay=1e-4)
    i = 0

    while True:
        optimizer.zero_grad()

        feats, adj = get_graph(num_feats)
        y = len(adj)
        output = gcn(Variable(feats), Variable(adj))
        loss = loss_fn(output, Variable(torch.Tensor([y]), requires_grad=False))
        loss.backward()
        optimizer.step()

        if i % 100 == 0 and i > 0:
            test_error = test_eval(gcn, test_size)
            print('Iter {} | MSE on {} test graphs: {}'.format(i, test_size, test_error))
            if test_error < 1e-2:
                break

        i += 1
    print('Done training after {} iterations.'.format(i))
