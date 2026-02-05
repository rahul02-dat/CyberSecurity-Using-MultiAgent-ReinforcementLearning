import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any
import copy


class DQN:
    
    def __init__(
        self,
        q_network: nn.Module,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        target_update_freq: int = 100,
        double_dqn: bool = True
    ):
        self.q_network = q_network
        self.target_network = copy.deepcopy(q_network)
        
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        
        self.optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
        
        self.device = next(q_network.parameters()).device
        
        self.update_counter = 0
    
    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states)
                next_q = next_q_values.gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_network(next_states)
                next_q = next_q_values.max(dim=1)[0]
            
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._update_target_network()
        
        return {
            'loss': loss.item(),
            'mean_q_value': current_q.mean().item()
        }
    
    def _update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())