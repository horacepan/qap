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
from replaybuffer import ReplayBuffer, Transition

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

    def _qvalue(self, state_vec, action_vec):
        '''
        state_vec:  p x 1 representation of the state
        action_vec: p x k representation of the action(vertex) chosen. k = number of actions
        to compute
        '''
        actions = action_vec.size(-1)
        state_mixed = self.state_weight.mm(state_vec.unsqueeze(-1)) # p x 1
        action_mixed = self.action_weight.mm(action_vec.unsqueeze(-1)) # p x 1
        sa_vec = F.relu(torch.cat([state_mixed, action_mixed], dim=0)) # 2p x 1
        return self.fc.t().mm((sa_vec))

    def batch_qvalues(self, state_vecs, action_vecs):
        if state_vecs.size(-1) != action_vecs.size(-1):
            assert state_vecs.size(-1) == 1

        state_mixed = state_vecs.mm(self.state_weight) # 1 x p
        action_mixed = action_vecs.mm(self.action_weight)# (k x p) x (p x p)
        if state_mixed.size(0) != action_vecs.size(0):
            assert state_mixed.size(0) == 1 # 1 x p
            state_mixed = state_mixed.repeat(action_mixed.size(0), 1) # k x p

        sa_vecs = F.relu(torch.cat([state_mixed, action_mixed], dim=1)) # k x 2p
        return sa_vecs.mm(self.fc) # try to return a 2k x 1 thing

    def best_action(self, state, allowed_actions=None, embedding=None):
        n = len(embedding)
        if allowed_actions is None:
            allowed_actions = range(n)

        actions = embedding[allowed_actions]
        state_vec = embedding[state].sum(dim=0, keepdim=True)
        qvalues = self.batch_qvalues(state_vec, actions)
        max_vtx = allowed_actions[np.argmax(qvalues.data)]
        return max_vtx

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
        batch_t = Transition(*zip(*batch))

        batch_state = torch.cat(batch_t.state, dim=0)
        batch_action = torch.cat(batch_t.action, dim=0)
        batch_new_state = torch.cat(batch_t.new_state, dim=0)
        batch_best_action = torch.cat(batch_t.best_action, dim=0)
        batch_reward = Variable(torch.FloatTensor(batch_t.reward).unsqueeze(1))

        expected = batch_reward + discount * self.batch_qvalues(batch_new_state, batch_best_action)
        computed = self.batch_qvalues(batch_state, batch_action)

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
        return self.struc2vec(node_labels, edge_weights, adj)

def get_optimizer(opt_params):
    return optim.Adam(**opt_params)

def update_exploration(eps):
    return 0.99 * eps

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
        embedding = qnet.embed_graph(node_labels, edge_weights, adj)

        state = [] # s_0
        state_vec = Variable(torch.zeros((1, qnet.embed_dim)))
        state_vec_prev = None
        actions = []
        rewards = []
        s_complement = set(range(len(adj)))
        losses = []
        best_actions = []

        for t in range(len(adj)):
            if t > 0:
                v_best_t = qnet.best_action(state, list(s_complement), embedding)
            if random.random() < eps or t == 0:
                v_t = random.choice(tuple(s_complement))
            else:
                v_t = v_best_t

            action_vec = embedding[v_t].unsqueeze(0)
            vprev = None if t == 0 else state[-1]
            r_t = 0 if t == 0 else -edge_weights.data[vprev, v_t]
            s_complement.remove(v_t)

            # ideally store: s_0 , a_0, r_0, s_1, v_best_1
            # ideally store: s_1 , a_1, r_1, s_2, v_best_2
            if t >= n_step:
                new_state = state[:]
                # the action prev is what action got taken.
                # v_best_t must be the argmax action of the current state
                v_best_embedding = embedding[v_best_t].unsqueeze(0)
                episode = (state_vec_prev, action_vec_prev, rewards[-1], state_vec, v_best_embedding)
                # should try to add v_best_t so we dont recompute later

                memory.push(*episode)
                if len(memory) > batch_size:
                    batch = memory.sample(batch_size)
                    batch_loss = qnet.backprop_batch(batch, optimizer)
                    losses.append(batch_loss)


            state_vec_prev = state_vec
            action_vec_prev = action_vec
            state.append(v_t)
            state_vec = state_vec +  action_vec
            rewards.append(r_t)

        epoch_loss = torch.mean(torch.cat(losses))
        print('Epoch {} | avg loss: {:.3f} | Exploration rate: {:.3f}'.format(e, float(epoch_loss), eps))
        eps = update_exploration(eps)

if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    epochs = 1000
    batch_size = 10
    eps = 0.9
    n_step = 1
    discount = 0.99
    capacity = 100
    gcn_params = {'embed_dim': 11, 'iters': 4}
    adam_params = {'lr': 0.01, 'weight_decay': 1e-4}
    graph_distr = GraphGenerator(16, 20)
    train(graph_distr, epochs, batch_size, eps, n_step, discount, capacity, gcn_params, adam_params)
