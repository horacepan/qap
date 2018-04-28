import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from qnet import QNet
from struc2vec import Struc2Vec
import util

def get_reward(curr_tour, new_vtx, edge_weights):
    '''
    # reward is just the neg distance from the last vertex of the current state
    # to the new vertex
    edge_weights: pairwise distance matrices
    '''
    if len(curr_tour) == 0:
        return 0
    last_vtx = curr_tour[-1]
    return -edge_weights[last_vtx, new_vtx]

def arg_max_action(qnet, vtx_features, remaining_vertices):
        best_vtx = None
        best_reward = 0

        for v in vtx_features:
            action = vtx_features[v]
            reward = qnet(state, action)
            if reward > best_reward
                best_reward = reward
                best_vtx = v

        return best_vtx, best_reward

def train(eps_start, eps_end, eps_decy, n_step, mem_capacity, num_episodes, embed_dim, iters):
    graph_generator = util.GraphGenerator()
    memory = ReplayMemory(mem_capacity)
    steps_done = 0
    gnn = Struc2Vec(embed_dim, iters)
    qnet = QNet(embed_dim)
    optimizer = optim.Adam(gnn.parameters() + qnet.parameters(), lr=0.0001, weight_decay=1e-4)
    for e in range(num_episodes):
        node_labels, adj, edge_weights = graph_generator.next()
        vtx_feats = gnn(node_labels, adj, edge_weights)
        remaining_vertices = set([i for i in range(len(adj))])
        state = Variable(torch.zeros(embed_dim))
        curr_tour = []
        T = len(adj)
        rewards = []
        states = [state]

        for t in range(T):
            eps_threshold = util.get_eps_threshold(eps_start, eps_end, eps_decay, steps_done)
            if random.random() > eps_threshold:
                # arg max action
                curr_vtx = arg_max_action(qnet, vtx_features, remaining_vertices)
            else:
                # random action
                curr_vtx = random.sample(remaining_vertices, 1)[0]

            action = vtx_feats[curr_vtx]
            # reward maintenance
            est_reward = qnet(state, curr_vtx)
            reward = get_reward(curr_tour, curr_vtx, edge_weights)
            rewards.append(reward)

            # update states
            curr_tour.append(curr_vtx)
            remaining_vertices.remove(curr_vtx)
            states.append(state + action)
            # wait till after doing the memory stuff to add the state

            # we only do these updates after n steps
            if t >= n_step:
                _, next_reward = arg_max_action(qnet, vtx_features, remaining_vertices)
                state_tminusn = states[-n_step] # this is a torch tensor
                action_tminusn = vtx_feats[curr_tour[-nstep]] # this gives the vertex id
                reward_tminusn = sum(reward[-n:])
                memory.push(state_minusn, action_tminusn, reward_tminusn, state, action)

                transitions = memory.sample(batch_size)
                # batch.state, batch.action, batch.reward, etc are now tuples
                # TODO: this looks a bit gross....
                batch = Transition(*zip(*batch))
                state_batch = torch.cat([s.unsqueeze(0) for s in  batch.state], dim=0)
                action_batch = torch.cat([a.unsqueeze(0) for a in  batch.action], dim=0)
                reward_batch = torch.cat(batch.reward)
                newstate_batch = torch.cat([ns.unsqueeze(0) for ns in batch.new_state], dim=0)
                max_action_batch = torch.cat([ma.unsqueeze(0) for ma in batch.max_action], dim=0)

                # TODO: make qnet allow batch
                # does the experience replay memory contain state/action/reward/next_state
                # from only the current episode's graph? Or can any graph seen before be
                # in the memory?
                # The argmax action is the thing taken at time t-n_step right?
                oldstate_action_value = qnet(state_batch, action_batch)
                newstate_action_value = qnet(new_state_batch, max_action_batch)
                expected_sa_values = reward_batch + gamma * newstate_action_value
                loss = F.mse_loss(oldstate_action_value, expected_sa_values)

                optimizer.zero_grad()
                loss.backward()
                # clamp grads?

            state += action
            steps_done += 1
