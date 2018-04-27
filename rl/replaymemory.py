from collections import NamedTuple

Transition = namedtuple('Transition', ['state', 'action', 'new_state', 'reward'])

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, *args):
        if len(self.memory) < capacity:
            self.memory.append(None)

        self.memory[position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
