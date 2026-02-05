import torch
import torch.nn as nn
from typing import Any


class DetectionPolicyNetwork(nn.Module):
    
    def __init__(self, input_dim: int, num_classes: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        features = self.feature_extractor(state)
        
        class_logits = self.classifier(features)
        
        value = self.value_head(features)
        
        confidence = self.confidence_head(features)
        
        return class_logits, value, confidence


class DetectionCritic(nn.Module):
    
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


class AttackSignatureEncoder(nn.Module):
    
    def __init__(self, input_dim: int, embedding_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh()
        )
    
    def forward(self, signature: torch.Tensor) -> torch.Tensor:
        return self.encoder(signature)