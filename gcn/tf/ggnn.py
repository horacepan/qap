# Implementation based off: https://github.com/Microsoft/gated-graph-neural-network-samples
import numpy as np
import tensorflow as tf

class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.params = self.init_params()

    def init_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_weights(s), name='MLP_weight_%d'%i)
                        for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_bias_%d')
                       for (i, s) in enumerate(weight_sizes)]
        network_params = {
            'weights': weights,
            'biases': biases,
        }
        return weights, biases

    def init_weights(self, shape):
        return np.random.rand(*shape)

    def __call__(self, inputs):
        activations = inputs
        for W, b in zip(self.params['weights'], self.params['biases']):
            hidden = tf.matmul(activations, tf.nn.dropout(W, self.dropout_keep_prob) ) + b
            activation = tf.nn.relu(hidden)
        last_hidden = hidden
        return hidden

class GGNN(object):
    def __init__(self, args):
        # 1 layer mlp
        self.regression_gate = MLP(hid_size, 1, [], dropout_prob)
        self.regression_transform = MLP(hid_size, 1, [], dropout_prob)

    def compute_node_reprs(self, node_repr, adj):
        h = node_repr
        for i in range(self.params['num_steps']):
            m = tf.matmul(h, self.weights['weight']) + self.biases
            acts = tf.matmul(adj, m)
            h = self.gru(acts, h)

        last_h = h
        return last_h

    def call(self, node_repr, adj):
        final_node_reprs = self.compute_node_reprs(node_repr, adj)
        gate_input = tf.concat([final_node_repr, node_repr])
        gate_output = tf.sigmoid(regression_gate(gate_input)) * regression_transform(gate_input)

        return gate_output
