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

def get_eps_threshold(eps_start, eps_end, eps_decay, steps_done):
    decay_factor = math.exp(-1.0 * steps_done / eps_decay)
    eps_threshold = eps_end + (eps_start - eps_end) * decay_factor
    return eps_threshold
