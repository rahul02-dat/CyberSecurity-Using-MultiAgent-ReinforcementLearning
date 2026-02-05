import torch
import torch.nn as nn
from typing import Any


class MonitoringPolicyNetwork(nn.Module):
    
    def __init__(self, input_dim: int, max_features: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.max_features = max_features
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.feature_scores = nn.Linear(hidden_dim, input_dim)
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.feature_encoder(state)
        
        feature_logits = self.feature_scores(encoded)
        
        value = self.value_head(encoded)
        
        return feature_logits, value
    
    def select_top_k(self, state: torch.Tensor, k: int) -> torch.Tensor:
        feature_logits, _ = self.forward(state)
        
        probs = torch.softmax(feature_logits, dim=-1)
        
        top_k_indices = torch.topk(probs, k, dim=-1).indices
        
        return top_k_indices


class MonitoringValueNetwork(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)