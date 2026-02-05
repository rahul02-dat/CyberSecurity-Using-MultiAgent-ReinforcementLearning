import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any, Optional


class PPO:
    
    def __init__(
        self,
        policy_network: nn.Module,
        value_network: Optional[nn.Module] = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64
    ):
        self.policy_network = policy_network
        self.value_network = value_network
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        params = list(policy_network.parameters())
        if value_network is not None:
            params += list(value_network.parameters())
        
        self.optimizer = optim.Adam(params, lr=learning_rate)
        
        self.device = next(policy_network.parameters()).device
    
    def update(self, memory: list[dict[str, Any]]) -> dict[str, float]:
        
        states, actions, old_log_probs, returns, advantages = self._prepare_batch(memory)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        dataset_size = len(states)
        
        for epoch in range(self.epochs):
            
            indices = np.random.permutation(dataset_size)
            
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                action_logits = self.policy_network(batch_states)
                action_log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
                
                if len(batch_actions.shape) == 1:
                    batch_actions = batch_actions.unsqueeze(1)
                
                new_log_probs = action_log_probs.gather(1, batch_actions).squeeze(1)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                if self.value_network is not None:
                    values = self.value_network(batch_states).squeeze()
                    value_loss = nn.MSELoss()(values, batch_returns)
                else:
                    value_loss = torch.tensor(0.0, device=self.device)
                
                entropy = -(torch.exp(action_log_probs) * action_log_probs).sum(dim=-1).mean()
                
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
                if self.value_network is not None:
                    torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        num_updates = self.epochs * (dataset_size // self.batch_size + 1)
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
    
    def _prepare_batch(self, memory: list[dict[str, Any]]) -> tuple:
        
        states = torch.FloatTensor(np.array([m['state'] for m in memory])).to(self.device)
        actions = torch.LongTensor(np.array([m['action'] for m in memory])).to(self.device)
        rewards = np.array([m['reward'] for m in memory])
        dones = np.array([m['done'] for m in memory])
        
        with torch.no_grad():
            action_logits = self.policy_network(states)
            action_log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
            
            if len(actions.shape) == 1:
                actions_for_gather = actions.unsqueeze(1)
            else:
                actions_for_gather = actions
            
            old_log_probs = action_log_probs.gather(1, actions_for_gather).squeeze(1)
        
        returns = self._compute_returns(rewards, dones)
        returns = torch.FloatTensor(returns).to(self.device)
        
        if self.value_network is not None:
            with torch.no_grad():
                values = self.value_network(states).squeeze()
            advantages = returns - values
        else:
            advantages = returns
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, old_log_probs, returns, advantages
    
    def _compute_returns(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        
        returns = np.zeros_like(rewards)
        running_return = 0.0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0.0
            
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns