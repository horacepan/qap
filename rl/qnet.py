import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# Do I want this as an abstract class that then gets subclassed by other classes
# that have specific forward passes for TSP/QAP/etc?
class QNet(nn.Module):
    def __init__(self, embed_dim):
        super(QNet, self).__init__()
        self.theta_5 = nn.Parameter(torch.rand(1, 2 * embed_dim))
        self.theta_6 = nn.Parameter(torch.rand((embed_dim, embed_dim)))
        self.theta_7 = nn.Parameter(torch.rand((embed_dim, embed_dim)))

    def forward(self, state, action):
        '''
        state: torch Tensor of size batch * p(or just p)
        action: torch Tensor of size batch * p(or just p)
        '''
        mixed_summed_tour = self.theta_6.matmul(state.unsqueeze(-1)) # batch x p
        mixed_action = self.theta_7.matmul(action.unsqueeze(-1))
        state_action_vec = torch.cat([mixed_summed_tour, mixed_action],
                                     dim=action.dim() - 1) # batch x 2p x 1
        return self.theta_5.matmul(F.relu(state_action_vec)).squeeze(-1) # batch x 1

def test():
    embed_dim = 7
    batch_size = 11
    batch_state = Variable(torch.rand((batch_size, embed_dim)))
    batch_action = Variable(torch.rand((batch_size, embed_dim)))

    qnet = QNet(embed_dim)
    result = qnet(batch_state, batch_action)
    assert result.size() == (batch_size, 1)

if __name__ == '__main__':
    test()
