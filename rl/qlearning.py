from gcn import GCN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from util import Memory

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
    def __init__(self, gcn_params):
        super(QNetwork, self).__init__()
        self.gcn = GCN(**gcn_params)

        self.state_weight = nn.Linear()
        self.new_vtx_weight = nn.Linear()
        self.fc_weight = nn.Linear()

    def qvalue(self, state, vtx):
        '''
        State = list of ints(vertices)
        vertex = int
        '''
        # TODO: should the state already be the torch tensor of 1s?
        n = len(self.embedding)
        state_rep = self.embedding.t().mm(binary_state(n, state)))
        vtx_rep = self.embedding[vtx]
        t = torch.cat([self.state_wt(state_rep), state.new_vtx_wt(vtx_rep)], 0)
        qval = self.fc_wt(F.relu(t))
        return qval

    def argmax(self, state):
        n = len(self.embedding)
        max_vtx = None
        max_qval = 0

        for v in range(n): 
            qval = self.qvalue(state, v)
            if qval > max_qval:
                max_qval = qval
                max_vtx = v

        return max_vtx
 
    def take_action(self, state, v_t):
        pass

    def backprop_batch(self, batch, optimizer):
        '''
        batch: list of tuples of state, vertex chosen, list of rewards, new state
        '''
        optimizer.zero_grad()

        expected = Variable(torch.zeros(len(batch)))
        computed = Variable(torch.zeros(len(batch)))
        for ind, (n_prev_state, v_t, reward, curr_state) in enumerate(batch):
            '''
            y = reward + discount * Q(current state, current action)
            loss = y - Q(state_t, 
            '''
            best_v = self.argmax(curr_state)
            expected[ind] = reward + discount * self.qvalue(curr_state, best_v)
            computed[ind] = self.qvalue(n_prev_state, best_v)

        loss = F.mse_loss(expected, computed)
        loss.backward()
        optimizer.step()

    def embed_graph(self, feats, adj):
        '''
        Pass the graph through the gcn. Return the embedding of all nodes
        '''
        self.embedding = self.gcn(feats, adj)

def get_optimizer(opt_params):
    return optim.Adam(**opt_params)

def update_exploration(eps):
    pass

def train(graph_distr, epochs, batch_size, eps, n, discount, capacity, gcn_params, opt_params):
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
    memory = Memory(capacity)
    optimizer = get_optimizer(opt_params)

    for e in range(epochs):
        adj, feats = graph_distr.draw_graph()
        qnet.embed_graph(adj, feats)

        state = []
        rewards = []
        s_complement = set(range(len(g)))

        for t in range(len(g)):
            if torch.random.random() < eps:
                v_t = random.choice(tuple(s_complement))
            else:
                v_t = qnet.argmax(state, v_t)

            r_t = qnet.take_action(state, v_t)
            rewards.append(r_t)
            state.append(v_t)
            s_complement.remove(v_t)

            if t > n:
                episode = (state[:t-n], state[t-n], sum(rewards[t-n:]), state)
                memory.add_episode(episode)
                batch = memory.sample(batch_size)
                qnet.backprop_batch(batch)

        eps = update_exploration(eps)
