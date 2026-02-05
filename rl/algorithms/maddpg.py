import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any, List
import copy


class MADDPG:
    
    def __init__(
        self,
        actor_networks: List[nn.Module],
        critic_networks: List[nn.Module],
        num_agents: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01
    ):
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        
        self.actors = actor_networks
        self.critics = critic_networks
        
        self.target_actors = [copy.deepcopy(actor) for actor in actor_networks]
        self.target_critics = [copy.deepcopy(critic) for critic in critic_networks]
        
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=learning_rate) 
            for actor in self.actors
        ]
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=learning_rate) 
            for critic in self.critics
        ]
        
        self.device = next(actor_networks[0].parameters()).device
    
    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        
        for agent_id in range(self.num_agents):
            
            critic_loss = self._update_critic(
                agent_id, states, actions, rewards[:, agent_id], next_states, dones
            )
            total_critic_loss += critic_loss
            
            actor_loss = self._update_actor(agent_id, states)
            total_actor_loss += actor_loss
            
            self._soft_update(self.target_actors[agent_id], self.actors[agent_id])
            self._soft_update(self.target_critics[agent_id], self.critics[agent_id])
        
        return {
            'critic_loss': total_critic_loss / self.num_agents,
            'actor_loss': total_actor_loss / self.num_agents
        }
    
    def _update_critic(
        self, 
        agent_id: int, 
        states: torch.Tensor, 
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        
        with torch.no_grad():
            next_actions = torch.cat([
                self.target_actors[i](next_states) for i in range(self.num_agents)
            ], dim=-1)
            
            target_q = self.target_critics[agent_id](next_states, next_actions)
            target_value = rewards + (1 - dones) * self.gamma * target_q.squeeze()
        
        current_q = self.critics[agent_id](states, actions)
        
        critic_loss = nn.MSELoss()(current_q.squeeze(), target_value)
        
        self.critic_optimizers[agent_id].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), 1.0)
        self.critic_optimizers[agent_id].step()
        
        return critic_loss.item()
    
    def _update_actor(self, agent_id: int, states: torch.Tensor) -> float:
        
        all_actions = []
        for i in range(self.num_agents):
            if i == agent_id:
                all_actions.append(self.actors[i](states))
            else:
                all_actions.append(self.actors[i](states).detach())
        
        all_actions = torch.cat(all_actions, dim=-1)
        
        actor_loss = -self.critics[agent_id](states, all_actions).mean()
        
        self.actor_optimizers[agent_id].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), 1.0)
        self.actor_optimizers[agent_id].step()
        
        return actor_loss.item()
    
    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )