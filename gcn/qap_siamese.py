import pdb
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from gcn import GCN
import graph_generators

class QAP_Siamese(nn.Module):
    def __init__(self, gcn_params):
        '''
        gcn_params: dictionary containing GCN parameters
        '''
        super(QAP_Siamese, self).__init__()
        self.gcn = GCN(**gcn_params)

    def forward(self, x_1, adj_1, x_2, adj_2):
        e_1 = self.forward_single(x_1, adj_1)
        e_2 = self.forward_single(x_2, adj_2)
        return e_1, e_2

    def forward_single(self, x, adj):
        return self.gcn(x, adj)

def make_stochastic(e_1, e_2):
    k = torch.mm(e_1, e_2.t())
    row_norm_k = nn.Softmax()(k)
    col_norm_k = nn.Softmax()(row_norm_k.t())
    return col_norm_k

def train(gcn_args):
    qnet = QAP_Siamese(gcn_params)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(qnet.gcn.parameters(), lr=0.01, weight_decay=1e-4)
    losses = torch.zeros(100)
    for i in range(10000):
        optimizer.zero_grad()

        f1, adj1, f2, adj2 = graph_generators.get_pair_graphs(gcn_args['in_feats'], 0)
        true_perm = Variable(torch.eye(len(adj1)))
        e_1, e_2 = qnet(Variable(f1), Variable(adj1), Variable(f2), Variable(adj2))
        perm = make_stochastic(e_1, e_2)
        loss = loss_fn(perm, true_perm)
        loss.backward()
        optimizer.step()
        losses[i%100] = loss.data[0]

        if i % 100 == 0 and i > 0:
            print("Epoch {} | Last 100 loss: {:.2f}".format(i, torch.mean(losses)))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action='store_true', help="cuda flag(if true use CUDA)")
    parser.add_argument("--in_feats",dest="in_feats", type=int, help="num input features", default=2)
    parser.add_argument("--out_feats",dest="out_feats", type=int, help="dims the network should return", default=1)
    parser.add_argument("--hidden", dest="hidden_size", type=int, help="size of hidden layer", default=3)
    parser.add_argument("--samples", dest="samples", type=int, help="number of samples to train on", default=10)
    parser.add_argument("--layers", dest="layers", type=int, help="number of layers", default=2)
    parser.add_argument("--dropout", dest="dropout", type=float, help="dropout rate(if 0, no dropout)")
    parser.add_argument("--lr",dest="learning_rate", type=float, help="ADAM learning rate",
                        default=0.001)

    parser.add_argument("--wd",dest="weight_decay", type=float, help="ADAM weight decay",
                        default=1e-4)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    gcn_params = {
        'in_feats': args.in_feats,
        'out_feats': args.out_feats,
        'hidden_size': args.hidden_size,
        'layers': args.layers,
        'dropout': args.dropout,
        'output_mode': 'vtx_repr'
    }
    train(gcn_params)
