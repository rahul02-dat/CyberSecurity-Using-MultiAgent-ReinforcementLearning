import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Optional
from agents.base.base_agent import BaseAgent
from agents.response.model import ResponsePolicyNetwork
from agents.response.reward import ResponseReward


class ResponseAgent(BaseAgent):
    
    def __init__(
        self, 
        agent_id: str, 
        observation_space: dict[str, Any], 
        action_space: dict[str, Any],
        learning_rate: float = 3e-4
    ):
        super().__init__(agent_id, observation_space, action_space)
        
        self.input_dim = observation_space['state_dim']
        self.num_actions = 6
        
        self.policy_network = ResponsePolicyNetwork(
            input_dim=self.input_dim,
            num_actions=self.num_actions
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        self.reward_function = ResponseReward()
        
        self.temperature = 1.0
        self.exploration_bonus = 0.05
        self.cost_sensitivity = 1.0
        
        self.action_names = {
            0: 'DoNothing',
            1: 'Log',
            2: 'BlockIP',
            3: 'RateLimit',
            4: 'Honeypot',
            5: 'ShutDownSubnet'
        }
    
    def select_action(self, observation: dict[str, Any], training: bool = True) -> Any:
        
        state_vector = torch.FloatTensor(observation['state']).to(self.device)
        
        with torch.no_grad():
            action_logits, _, risk_score = self.policy_network(state_vector.unsqueeze(0))
            
            action_logits = action_logits / self.temperature
            
            action_probs = torch.softmax(action_logits, dim=-1)
        
        if training:
            action = torch.multinomial(action_probs, 1).item()
        else:
            action = torch.argmax(action_probs, dim=-1).item()
        
        return {
            'action': action,
            'action_name': self.action_names[action],
            'action_probabilities': action_probs.squeeze(0).cpu().numpy(),
            'risk_score': risk_score.item()
        }
    
    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        action_logits, values, risk_scores = self.policy_network(states)
        
        action_log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
        selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            _, next_values, _ = self.policy_network(next_states)
            td_target = rewards + 0.99 * next_values.squeeze(1) * (1 - dones)
        
        advantages = td_target - values.squeeze(1)
        
        policy_loss = -(selected_log_probs * advantages.detach()).mean()
        
        value_loss = advantages.pow(2).mean()
        
        entropy = -(torch.exp(action_log_probs) * action_log_probs).sum(dim=-1).mean()
        entropy_bonus = -0.01 * entropy
        
        total_loss = policy_loss + 0.5 * value_loss + entropy_bonus
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
    
    def save(self, path: str) -> None:
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'temperature': self.temperature,
            'cost_sensitivity': self.cost_sensitivity
        }, path)
    
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.temperature = checkpoint['temperature']
        self.cost_sensitivity = checkpoint['cost_sensitivity']
    
    def get_state_dict(self) -> dict[str, Any]:
        return {
            'policy_network': self.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'temperature': self.temperature,
            'cost_sensitivity': self.cost_sensitivity,
            'exploration_bonus': self.exploration_bonus
        }