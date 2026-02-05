import numpy as np
from typing import Any, Optional
import random


class ReplayBuffer:
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool
    ) -> None:
        
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> dict[str, Any]:
        
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for experience in batch:
            states.append(experience['state'])
            actions.append(experience['action'])
            rewards.append(experience['reward'])
            next_states.append(experience['next_state'])
            dones.append(float(experience['done']))
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones)
        }
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self) -> None:
        self.buffer = []
        self.position = 0