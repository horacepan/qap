import pdb
import argparse
import numpy as np
import graph_generators
from layers import GraphConvLayer
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import util

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, dropout=0.5, layers=3, output_mode='graph_repr'):
        super(GCN, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats

        self.layers = layers
        self.dropout = dropout
        self.nonlinearity = F.relu
        self._output_mode = output_mode
        self.final_repr_size = self._final_repr_size(in_feats, hidden_size, out_feats, layers, output_mode)
        self._init_layers(in_feats, hidden_size, self.final_repr_size, out_feats)

        print(self.__repr__)

    def _init_layers(self, in_feats, hidden_size, repr_size, out_feats):
        setattr(self, 'gc1', GraphConvLayer(in_feats, hidden_size))
        for i in range(2, self.layers+1):
            setattr(self, 'gc{}'.format(i), GraphConvLayer(hidden_size, hidden_size))
        self.fc = nn.Linear(repr_size, out_feats)

    def _final_repr_size(self, in_feats, hidden_size, out_feats, layers, output_mode):
        if output_mode is 'vtx_repr' or output_mode is 'graph_repr':
            return hidden_size
        elif output_mode is 'vtx_repr_cat' or output_mode is 'graph_repr_cat':
            return layers * hidden_size + in_feats
        else:
            raise Exception("Invalid output mode: {}".format(output_mode))

    def forward(self, x, adj):
        # x = self.nonlinearity(self.gc1(x, adj))
        lvl_feats = [x]
        for i in range(1, self.layers+1):
            gc_layer = getattr(self, 'gc{}'.format(i))
            x = gc_layer(x, adj)

            if i is not self.layers:
                x = self.nonlinearity(x)
            lvl_feats.append(x)

        if self._output_mode is 'vtx_repr':
            final_repr = x
        elif self._output_mode is 'vtx_repr_cat':
            final_repr = torch.cat(lvl_feats, 1)
        elif self._output_mode is 'graph_repr':
            final_repr = x.sum(dim=0)
        elif self._output_mode is 'graph_repr_cat':
            final_repr = torch.cat(lvl_feats, 1).sum(dim=0)
        else:
            raise Exception("Invalid output mode: {}".format(self._output_mode))
        return self.fc(final_repr)

def test_eval(model, test_size, num_feats, gcn):
    preds = np.zeros(test_size)
    truth = np.zeros(test_size)
    for i in range(test_size):
        feats, adj = graph_generators.get_graph(num_feats)
        y = len(adj)
        truth[i] = y
        v_feats, v_adj = Variable(feats), Variable(adj)

        output = gcn(Variable(feats), Variable(adj))
        preds[i] = output.data[0]

    loss = np.mean(np.sum(np.square(preds - truth)))
    return loss

def train(args):
    in_feats = num_feats = args.in_feats
    hidden_size = args.hidden_size
    out_feats = args.out_feats
    dropout_rate = args.dropout
    layers = args.layers
    test_size = 100
    output_mode = 'graph_repr'

    gcn = GCN(in_feats, hidden_size, out_feats, dropout_rate, layers, output_mode)

    loss_fn = F.mse_loss
    optimizer = optim.Adam(gcn.parameters(), lr=0.001, weight_decay=1e-4)
    i = 0
    pdb.set_trace()
    while True:
        optimizer.zero_grad()

        feats, adj = graph_generators.get_graph(num_feats)
        y = len(adj)
        output = gcn(Variable(feats), Variable(adj))
        loss = loss_fn(output, Variable(torch.Tensor([y]), requires_grad=False))
        loss.backward()
        optimizer.step()

        if i % 100 == 0 and i > 0:
            test_error = test_eval(gcn, test_size, num_feats, gcn)
            print('Iter {} | MSE on {} test graphs: {}'.format(i, test_size, test_error))
            if test_error < 1e-2:
                break

        i += 1
    print('Done training after {} iterations.'.format(i))
    pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action='store_true', help="cuda flag(if true use CUDA)")
    parser.add_argument("--in_feats",dest="in_feats", type=int, help="num input features", default=2)
    parser.add_argument("--out_feats",dest="out_feats", type=int, help="num output features", default=1)
    parser.add_argument("--hidden", dest="hidden_size", type=int, help="size of hidden layer", default=3)
    parser.add_argument("--samples", dest="samples", type=int, help="number of samples to train on", default=10)
    parser.add_argument("--layers", dest="layers", type=int, help="number of layers", default=2)
    parser.add_argument("--dropout", action="store_true", help="dropout flag(if true use dropout)")
    parser.add_argument("--lr",dest="learning_rate", type=float, help="ADAM learning rate",
                        default=0.001)

    args = parser.parse_args()
    train(args)
