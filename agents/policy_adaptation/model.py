import torch
import torch.nn as nn
from typing import Any


class MetaControllerNetwork(nn.Module):
    
    def __init__(self, performance_metrics_dim: int, num_hyperparams: int, hidden_dim: int = 256):
        super().__init__()
        self.performance_metrics_dim = performance_metrics_dim
        self.num_hyperparams = num_hyperparams
        
        self.encoder = nn.Sequential(
            nn.Linear(performance_metrics_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.hyperparam_head = nn.Linear(hidden_dim, num_hyperparams)
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, performance_metrics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        encoded = self.encoder(performance_metrics)
        
        hyperparam_adjustments = self.hyperparam_head(encoded)
        
        confidence = self.confidence_head(encoded)
        
        return hyperparam_adjustments, confidence


class PerformancePredictor(nn.Module):
    
    def __init__(self, state_dim: int, hyperparams_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(state_dim + hyperparams_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, hyperparams: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([state, hyperparams], dim=-1)
        return self.predictor(combined)


class AdaptationMemory(nn.Module):
    
    def __init__(self, memory_size: int = 100, feature_dim: int = 256):
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        
        self.memory_bank = nn.Parameter(torch.randn(memory_size, feature_dim))
        self.memory_keys = nn.Parameter(torch.randn(memory_size, feature_dim))
        
        self.query_projection = nn.Linear(feature_dim, feature_dim)
        self.key_projection = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        
        projected_query = self.query_projection(query)
        projected_keys = self.key_projection(self.memory_keys)
        
        attention_scores = torch.matmul(
            projected_query.unsqueeze(1), 
            projected_keys.transpose(0, 1)
        ).squeeze(1)
        
        attention_weights = torch.softmax(attention_scores / (self.feature_dim ** 0.5), dim=-1)
        
        retrieved = torch.matmul(attention_weights.unsqueeze(1), self.memory_bank).squeeze(1)
        
        return retrieved