import torch as T
import random
from collections import namedtuple, deque
import numpy as np

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, action_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        states = T.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = T.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = T.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = T.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = T.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)