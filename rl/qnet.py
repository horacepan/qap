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
        state: list or iterable of 1-dim torch Tensors        
        '''
        summed_tour = sum(state).unsqueeze(1) # sum gives 1 dim tensor. Unsqueeze to get px1
        mixed_summed_tour = self.theta_6.mm(summed_tour)
        mixed_action = self.theta_7.mm(action.unsqueeze(1))
        state_action_vec = torch.cat([mixed_summed_tour, mixed_action], dim=0) # 2p x 1
        return self.theta_5.mm(F.relu(state_action_vec))

def test():
    embed_dim = 7
    state = (Variable(torch.rand(7)), Variable(torch.rand(7)), Variable(torch.rand(7)))
    action = Variable(torch.rand(7))

    qnet = QNet(embed_dim)
    result = qnet(state, action)
    assert result.size() == (1,1)

if __name__ == '__main__':
    test()
