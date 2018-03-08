import torch

class GGNN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, dropout, layers):
        super(GGNN, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.in_feats = in_feats
        self.num_steps = num_steps

        # assume input feature is zero padded to length of hidden size
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.weight = nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.reg_gate = MLP(layers=1, in_size =None, out_size=None, hidden_size=None, weights=None)
        self.reg_transform = MLP(layers=1, in_size =None, out_size=None, hidden_size=None, weights=None)
        self.init_params()

    def init_params(self):
        pass

    def compute_node_reprs(self, node_repr, adj):
        h = node_repr:
        for i in range(self.num_steps):
            m = torch.mm(h, self.weight) + self.bias
            acts = torch.mm(adj, m)
            h = self.gru(acts, h)
        last_h = h
        return last_h

    def gated_regression(self, last_h, init_node_feats, regression_gate, regression_transform):
        gate_input = torch.concat([last_h, init_node_feats], 1)
        gate_output = torch.nn.sigmoid(regression_gate(gate_input) * \
                                       regression_transform(gate_input)

        return gate_output

    def forward(self, node_repr, adj):
        last_reprs = self.compute_node_reprs(node_repr, adj)
        return self.gated_regression(last_reprs, node_repr, self.reg_gate, self.reg_transform)
