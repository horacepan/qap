import pdb
import torch
import torch.nn as nn
import math
from torch.autograd import Variable

# Refs:
# https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
# https://discuss.pytorch.org/t/does-pytorch-support-autograd-on-sparse-matrix/6156/7
class SparseMM(torch.autograd.Function):
    def forward(self, m1, m2):
        self.save_for_backward(m1, m2)
        return torch.mm(m1, m2)

    def backward(self, grad_output):
        grad_m1 = grad_m2 = None
        m1, m2 = self.saved_tensors

        if self.needs_input_grad[0]:
            grad_m1 = torch.mm(grad_output, m2)
        if self.needs_input_grad[1]:
            grad_m2 = torch.mm(m1.t(), grad_output)
        return grad_m1, grad_m2

class GraphConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # initialize weight matrix to uniform in [+/- 1/sqrt[input size]]
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        '''
        input(Tensor):
        adj(Tensor): adjacency matrix of a graph
        '''
        x = torch.mm(input, self.weight)
        x = SparseMM()(adj, x)
        if self.bias is not None:
            return x + self.bias
        else:
            return x

    def __repr__(self):
        return '{} ({} -> {})'.format(self.__class__.name__, self.in_feats, self.out_feats)


if __name__ == '__main__':
    in_feats = 3
    out_feats = 4
    graph_size = 3
    x = Variable(torch.rand(3, in_feats))
    adj = Variable(torch.Tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
    gc_layer = GraphConvLayer(in_feats, out_feats)
    gc_layer(x, adj)
