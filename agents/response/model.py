import torch
import torch.nn as nn
from typing import Any


class ResponsePolicyNetwork(nn.Module):
    
    def __init__(self, input_dim: int, num_actions: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.action_head = nn.Linear(hidden_dim, num_actions)
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.risk_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        encoded = self.state_encoder(state)
        
        action_logits = self.action_head(encoded)
        
        value = self.value_head(encoded)
        
        risk_score = self.risk_estimator(encoded)
        
        return action_logits, value, risk_score


class ResponseQNetwork(nn.Module):
    
    def __init__(self, input_dim: int, num_actions: int = 6, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class CostPredictor(nn.Module):
    
    def __init__(self, input_dim: int, num_actions: int = 6, hidden_dim: int = 128):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor, action_one_hot: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([state, action_one_hot], dim=-1)
        return self.predictor(combined)