'''
Implementation of the Struc2Vec architecture used in:
Learning Combinatorial Optimization Algorithms over Graphs
Hanjun Dai, Elias B. Khalil, Yuyu Zhang, Bistra Dilkina, Le Song
https://arxiv.org/abs/1704.01665
'''
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class GraphGenerator(object):
    def __init__(self):
        pass

class GraphBatcher(object):
    def __init__(self, graph_generator, batch_size, max_graph_size, node_label_size):
        self.graph_generator = graph_generator
        self.batch_size = bath_size
        self.max_graph_size = max_graph_size
        self.node_label_size = node_label_size

    def next_batch(self):
        batch_node_labels = torch.zeros((b, self.node_label_size))
        batch_adjs = torch.zeros((b, self.max_graph_size, self.max_graph_size))
        batch_edge_weights = torch.zeros((b, self.max_graph_size, self.max_graph_size))

        for i in range(batch_size):
            # generator here should give node labels, edge weights, adjacency matrix
            node_labels, adj, weights = graph_generator.next()
            n = len(adj)
            batch_node_labels[i, :n] = node_labels
            batch_adjs[i, :n, :n] = adj
            batch_edge_weights[i, :n, :n] = weights
        return batch_node_labels, batch_adjs, batch_edge_weights

class Struc2Vec(nn.Module):
    def __init__(self, embed_dim, iters):
        super(Struc2Vec, self).__init__()
        self.embed_dim = embed_dim
        self.iters = iters

        # param names in accordance with the paper. Here embed_dim = p
        self.theta_1 = nn.Parameter(torch.Tensor(embed_dim))
        self.theta_2 = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.theta_3 = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.theta_4 = nn.Parameter(torch.Tensor(embed_dim))
        self._reset_weights()
    def _reset_weights(self):
        '''
        Initializes the weights of the nn
        '''
        # TODO: better weight initialization
        self.theta_1.data.normal_(0, 1)
        self.theta_2.data.normal_(0, 1)
        self.theta_3.data.normal_(0, 1)
        self.theta_4.data.normal_(0, 1)

    def forward(self, node_labels, embeddings, edge_weights, adj):
        '''
        node_labels: torch Tensor of the node labels
        embeddings: torch Tensor
        edge_weights: torch Tensor
        adj: torch Tensor of 0/1s
        '''
        vtx_contrib = self.theta_1 * node_labels # n x p
        nbr_edge_weights = F.relu(self.theta_4 * torch.sum(edge_weights, dim=1, keepdim=True)) # n x p
        mixed_edge_weights = torch.mm(nbr_edge_weights, self.theta_3) # n x p
        for t in range(self.iters):
            nbr_update = torch.mm(torch.mm(adj, embeddings), self.theta_2)
            embeddings = F.relu(vtx_contrib + nbr_update + mixed_edge_weights)
        print(embeddings.size())
        return embeddings

def test():
    embed_dim = 10
    iters = 4
    graph_size = 20

    node_labels = Variable(torch.rand((graph_size, 1)) > 0.5).float()
    embeddings = Variable(torch.rand((graph_size, embed_dim)))
    edge_weights = Variable(torch.rand((graph_size, graph_size)))
    adj = Variable(torch.rand((graph_size, graph_size)))
    net = Struc2Vec(embed_dim, iters)
    embeddings = net(node_labels, embeddings, edge_weights, adj)

if __name__ == '__main__':
    test()
