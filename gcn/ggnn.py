import pdb
import torch
import graph_generators
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import util
from layers import GRU 
from torch.autograd import Variable

class GGNN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, num_steps):
        super(GGNN, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = hidden_size 
        self.in_feats = in_feats
        self.num_steps = num_steps

        # assume input feature is zero padded to length of hidden size
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.weight = nn.Parameter(torch.rand((hidden_size, hidden_size)))
        self.gru = GRU(hidden_size, hidden_size)
        self.reg_gate = util.MLP(in_size=self.hidden_size+self.in_feats, hid_sizes=[], out_size=1)
        self.reg_transform = util.MLP(in_size=self.hidden_size+self.in_feats, hid_sizes=[], out_size=1)
        #self.init_params()

    def compute_node_reprs(self, node_repr, adj):
        h = node_repr
        for i in range(self.num_steps):
            m = torch.mm(h, self.weight) + self.bias
            acts = torch.mm(adj, m)
            h = self.gru(acts, h)
        last_h = h
        return last_h

    def gated_regression(self, last_h, init_node_feats, regression_gate, regression_transform):
        gate_input = torch.cat([last_h, init_node_feats], 1)
        gate_output = F.sigmoid(regression_gate(gate_input) * \
                                regression_transform(gate_input))
        output = F.tanh(torch.sum(gate_output)) 
        return output

    def forward(self, node_repr, adj):
        last_reprs = self.compute_node_reprs(node_repr, adj)
        return self.gated_regression(last_reprs, node_repr, self.reg_gate, self.reg_transform)

def train():
    in_feats = 6
    hid_feats = 6
    out_feats = 5
    num_steps = 4
    net = GGNN(in_feats, hid_feats, out_feats, num_steps)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = F.mse_loss
    i = 0

    while True:
        optimizer.zero_grad()

        feats, adj = graph_generators.get_graph(in_feats)
        y = len(adj)
        output = net(Variable(feats), Variable(adj))
        loss = loss_fn(output, Variable(torch.Tensor([y]), requires_grad=False))
        loss.backward()
        optimizer.step()

        if i % 100 == 0 and i > 0:
            print("Epoch {}".format(i))

        i += 1
    print('Done training after {} iterations.'.format(i))

if __name__ == '__main__':
    train()
