import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Optional
from agents.base.base_agent import BaseAgent
from agents.monitoring.model import MonitoringPolicyNetwork
from agents.monitoring.reward import MonitoringReward


class MonitoringAgent(BaseAgent):
    
    def __init__(
        self, 
        agent_id: str, 
        observation_space: dict[str, Any], 
        action_space: dict[str, Any],
        num_features_to_select: int = 20,
        learning_rate: float = 3e-4
    ):
        super().__init__(agent_id, observation_space, action_space)
        
        self.input_dim = observation_space['feature_vector_dim']
        self.num_features_to_select = num_features_to_select
        
        self.policy_network = MonitoringPolicyNetwork(
            input_dim=self.input_dim,
            max_features=self.input_dim
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        self.reward_function = MonitoringReward()
        
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.exploration_min = 0.1
    
    def select_action(self, observation: dict[str, Any], training: bool = True) -> Any:
        
        feature_vector = torch.FloatTensor(observation['feature_vector']).to(self.device)
        
        if training and torch.rand(1).item() < self.exploration_rate:
            k = self.num_features_to_select
            selected_indices = torch.randperm(self.input_dim)[:k]
        else:
            with torch.no_grad():
                selected_indices = self.policy_network.select_top_k(
                    feature_vector.unsqueeze(0), 
                    self.num_features_to_select
                ).squeeze(0)
        
        if training:
            self.exploration_rate = max(
                self.exploration_min, 
                self.exploration_rate * self.exploration_decay
            )
        
        return {
            'selected_indices': selected_indices.cpu().numpy(),
            'num_features': self.num_features_to_select
        }
    
    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        logits, values = self.policy_network(states)
        
        action_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        selected_log_probs = action_log_probs.gather(1, actions)
        
        _, next_values = self.policy_network(next_states)
        advantages = rewards.unsqueeze(1) + 0.99 * next_values * (1 - dones.unsqueeze(1)) - values
        
        policy_loss = -(selected_log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        
        loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': loss.item()
        }
    
    def save(self, path: str) -> None:
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate
        }, path)
    
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.exploration_rate = checkpoint['exploration_rate']
    
    def get_state_dict(self) -> dict[str, Any]:
        return {
            'policy_network': self.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'num_features_to_select': self.num_features_to_select
        }