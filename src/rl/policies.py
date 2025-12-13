import torch
import torch.nn as nn
from typing import Dict, Tuple


class DetectionPolicyNetwork(nn.Module):
    
    def __init__(self, obs_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.flag_head = nn.Linear(hidden_dim, 2)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        flag_logits = self.flag_head(features)
        confidence_logits = self.confidence_head(features)
        confidence = torch.sigmoid(confidence_logits)
        
        return flag_logits, confidence


class ResponsePolicyNetwork(nn.Module):
    
    def __init__(self, obs_dim: int = 4, detection_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        
        input_dim = obs_dim + detection_dim
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.action_head = nn.Linear(hidden_dim, 3)
        self.duration_head = nn.Linear(hidden_dim, 3)
        
    def forward(self, obs: torch.Tensor, detection_info: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([obs, detection_info], dim=-1)
        features = self.shared(combined)
        action_logits = self.action_head(features)
        duration_logits = self.duration_head(features)
        
        return action_logits, duration_logits


class DetectionValueNetwork(nn.Module):
    
    def __init__(self, obs_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class ResponseValueNetwork(nn.Module):
    
    def __init__(self, obs_dim: int = 4, detection_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        
        input_dim = obs_dim + detection_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs: torch.Tensor, detection_info: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([obs, detection_info], dim=-1)
        return self.network(combined)