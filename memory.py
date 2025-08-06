import numpy as np
import torch

from config import Config
from help import propagate_nb, retrieve_nb, batch_retrieve_nb, batch_retrieve_par_nb


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.pointer = 0
        self.n_entries = 0
        self.max_priority = (1.0 + Config.PRIO_EPS) ** Config.PRIO_ALPHA

    def _propagate(self, idx, change):
        propagate_nb(self.tree, idx, change)

    def _retrieve(self, s):
        return retrieve_nb(self.tree, self.capacity, s)
    
    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.pointer + self.capacity - 1
        
        old_priority = self.tree[idx]
        change = priority - old_priority
        self.tree[idx] = priority
        self._propagate(idx, change)
            
        self.data[self.pointer] = data
        self.pointer = (self.pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        
        if priority > self.max_priority:
            self.max_priority = priority

    def update(self, idx, priority):
        old_priority = self.tree[idx]
        change = priority - old_priority
        self.tree[idx] = priority
        self._propagate(idx, change)
        
        if priority > self.max_priority:
            self.max_priority = priority

    def get_batch(self, s_values):
        # 使用并行版本检索
        indices = batch_retrieve_nb(self.tree, self.capacity, s_values)
        data_idx = indices - (self.capacity - 1)
        priorities = self.tree[indices]
        data_list = [self.data[i] for i in data_idx]
        return indices, priorities, data_list

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=Config.PRIO_ALPHA, 
                 beta_start=Config.PRIO_BETA_START, 
                 beta_frames=Config.PRIO_BETA_FRAMES):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.tree = SumTree(capacity)

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        initial_priority = (1.0 + Config.PRIO_EPS) ** self.alpha
        self.tree.add(initial_priority, data)

    def sample(self, batch_size):
        total_priority = self.tree.total()
        segment = total_priority / batch_size
        
        s_values = np.random.uniform(
            segment * np.arange(batch_size),
            segment * (np.arange(batch_size) + 1)
        )
        
        indices, priorities, batch = self.tree.get_batch(s_values)
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        probs = priorities / total_priority
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        
        max_weight = weights.max() if weights.max() > 0 else 1.0
        weights /= max_weight
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.as_tensor(np.array(states), 
                             dtype=torch.float32,
                             device=Config.device).contiguous()
                             
        actions = torch.as_tensor(np.array(actions), 
                              dtype=torch.long,
                              device=Config.device).contiguous()
                              
        rewards = torch.as_tensor(np.array(rewards), 
                              dtype=torch.float32,
                              device=Config.device).contiguous()
                              
        next_states = torch.as_tensor(np.array(next_states), 
                                  dtype=torch.float32,
                                  device=Config.device).contiguous()
                                  
        dones = torch.as_tensor(np.array(dones, dtype=np.float32), 
                             dtype=torch.float32,
                             device=Config.device).contiguous()
                             
        weights = torch.as_tensor(weights, 
                              dtype=torch.float32,
                              device=Config.device).contiguous()
        
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, new_priorities):
        new_priorities = np.abs(new_priorities) + Config.PRIO_EPS
        new_priorities = np.power(new_priorities, self.alpha)
            
        for idx, p in zip(indices, new_priorities):
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries