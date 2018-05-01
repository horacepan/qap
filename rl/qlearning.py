import sys
sys.path.append('../')
import pdb
#from gcn.gcn import GCN
from struc2vec import Struc2Vec, GraphBatcher, GraphGenerator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from replaybuffer import ReplayBuffer

def binary_state(n, indices):
    '''
    n: int, length of vector
    indices: list of ints
    '''
    state = torch.zeros(n)
    for i in indices:
        state[i] = 1
    return Variable(state)

class QNetwork(nn.Module):
    def __init__(self, s2v_params):
        super(QNetwork, self).__init__()
        self.struc2vec = Struc2Vec(**s2v_params)
        self.embed_dim = self.struc2vec.embed_dim
        self.state_weight = nn.Parameter(torch.zeros(self.embed_dim, self.embed_dim)) # p x p
        self.action_weight = nn.Parameter(torch.zeros(self.embed_dim, self.embed_dim)) # p x p
        self.fc = nn.Parameter(torch.zeros(2 * self.embed_dim, 1)) #2p x 1
        self.reset_weights()

    def reset_weights(self):
        self.state_weight.data.normal_(0, 1)
        self.action_weight.data.normal_(0, 1)
        self.fc.data.normal_(0, 1)

    def qvalue(self, state, vtx):
        '''
        State = list of ints(vertices)
        vertex = int
        '''
        # TODO: should the state already be the torch tensor of 1s?
        n = len(self.embedding)
        state_rep = self.embedding[state].sum(dim=0).unsqueeze(1)
        vtx_rep = self.embedding[vtx].unsqueeze(1)
        qval = self._qvalue(state_rep, vtx_rep)
        return qval

    def _qvalue(self, state_vec, action_vec):
        '''
        state_vec:  p x 1 representation of the state
        action_vec: p x 1 representation of the action(vertex) chosen
        '''
        state_mixed = self.state_weight.mm(state_vec) # p x 1
        action_mixed = self.action_weight.mm(action_vec) # p x 1
        sa_vec = F.relu(torch.cat([state_mixed, action_mixed], dim=0)) # 2p x 1
        return self.fc.t().mm((sa_vec))

    def best_action(self, state, allowed_actions=None):
        n = len(self.embedding)
        if allowed_actions is None:
            allowed_actions = range(n)

        qvalues = [float(self.qvalue(state, v)) for v in allowed_actions]
        max_vtx = allowed_actions[np.argmax(qvalues)]
        return max_vtx

    def take_action(self, state, v_t):
        return self.qvalue(state, v_t)

    def backprop_batch(self, batch, optimizer):
        '''
        batch: list of tuples of state, vertex chosen, list of rewards, new state
        batch: list of tuples of state (sum of embeddings)
                                 embedding of vertex chosen
                                 rewards during the n-step period
                                 new state (this is after the nth step
        # Can also store the max action from the new state
        # or also just store the qvalue
        '''
        optimizer.zero_grad()

        expected = Variable(torch.zeros(len(batch)))
        computed = Variable(torch.zeros(len(batch)))
        for ind, (n_prev_state, v_t_n, reward, curr_state) in enumerate(batch):
            '''
            y = reward + discount * Q(current state, current action)
            loss = y - Q(state_t,
            '''
            curr_v = self.best_action(curr_state)
            expected[ind] = reward + discount * self.qvalue(curr_state, curr_v)
            computed[ind] = self.qvalue(n_prev_state, v_t_n)

        loss = F.mse_loss(expected, computed)
        loss.backward(retain_graph=True)

        for param in self.parameters():
            param.grad.data.clamp_(-10, 10)
        optimizer.step()
        return loss

    def embed_graph(self, node_labels, edge_weights, adj):
        '''
        Pass the graph through the gcn. Return the embedding of all nodes
        '''
        self.embedding = self.struc2vec(node_labels, edge_weights, adj)
        return self.embedding

def get_optimizer(opt_params):
    return optim.Adam(**opt_params)

def update_exploration(eps):
    return 0.01 * eps

def train(graph_distr, epochs, batch_size, eps, n_step, discount, capacity, gcn_params, opt_params):
    '''
    graph_distr: object that wraps the graph generating distr
    epochs: int
    batch_size: int
    eps: float for exploration probability
    n: int, num steps for n-step Q-learning
    discount: float, how much to discount future state/action value
    capacity: int, number of episodes to keep in memory
    gcn_params: dictionary of graph conv net parameters
    opt_params: dictionary of params for optimizer
    '''
    qnet = QNetwork(gcn_params)
    memory = ReplayBuffer(capacity)
    opt_params['params'] = qnet.parameters()
    optimizer = get_optimizer(opt_params)

    for e in range(epochs):
        node_labels, edge_weights, adj = graph_distr.next()
        qnet.embed_graph(node_labels, edge_weights, adj)

        # should just have an arbitrary vertex here?
        state = [0]
        rewards = []
        s_complement = set(range(len(adj)))
        s_complement.remove(0)
        losses = []
        for t in range(len(adj)-1):
            if random.random() < eps:
                v_t = random.choice(tuple(s_complement))
            else:
                v_t = qnet.best_action(state, tuple(s_complement))

            vprev = state[-1]
            r_t = -edge_weights[vprev, v_t]
            s_complement.remove(v_t)

            if t > n_step:
                #state_vec = get_sum(embedding, state[:t-n])
                #action_vec = embedding[state[t-n]]
                #new_state_vec = get_sum(embedding, state)
                #episode = (state_vec, action_vec, sum(rewards[t-n:]), new_state_vec)
                new_state = state[:]
                episode = (state[:t-n_step], state[t-n_step], sum(rewards[t-n_step:]), new_state)
                memory.push(*episode)
                if len(memory) > batch_size:
                    batch = memory.sample(batch_size)
                    batch_loss = qnet.backprop_batch(batch, optimizer)
                    losses.append(batch_loss)
            state.append(v_t)
            rewards.append(r_t)
        epoch_loss = torch.mean(torch.cat(losses))
        print('Epoch {} | avg loss: {:.3f}'.format(e, float(epoch_loss)))
        eps = update_exploration(eps)

if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    epochs = 1000
    batch_size = 10
    eps = 0.9
    n_step = 3
    discount = 0.99
    capacity = 100
    gcn_params = {'embed_dim': 11, 'iters': 4}
    adam_params = {'lr': 0.01, 'weight_decay': 1e-4}
    graph_distr = GraphGenerator(16, 16)
    train(graph_distr, epochs, batch_size, eps, n_step, discount, capacity, gcn_params, adam_params)
