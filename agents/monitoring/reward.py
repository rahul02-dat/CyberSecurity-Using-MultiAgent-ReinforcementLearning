import torch
from typing import Any


class MonitoringReward:
    
    def __init__(
        self, 
        information_gain_weight: float = 1.0,
        bandwidth_penalty_weight: float = 0.5,
        redundancy_penalty_weight: float = 0.3
    ):
        self.information_gain_weight = information_gain_weight
        self.bandwidth_penalty_weight = bandwidth_penalty_weight
        self.redundancy_penalty_weight = redundancy_penalty_weight
    
    def compute(
        self, 
        selected_features: torch.Tensor,
        attack_signatures: torch.Tensor,
        num_features: int,
        max_features: int
    ) -> float:
        
        information_gain = self._compute_information_gain(selected_features, attack_signatures)
        
        bandwidth_cost = (num_features / max_features)
        
        redundancy_penalty = self._compute_redundancy(selected_features)
        
        reward = (
            self.information_gain_weight * information_gain
            - self.bandwidth_penalty_weight * bandwidth_cost
            - self.redundancy_penalty_weight * redundancy_penalty
        )
        
        return reward
    
    def _compute_information_gain(
        self, 
        selected_features: torch.Tensor, 
        attack_signatures: torch.Tensor
    ) -> float:
        
        if attack_signatures.sum() == 0:
            return 0.0
        
        overlap = torch.isin(selected_features, torch.nonzero(attack_signatures).squeeze())
        
        gain = overlap.float().mean().item()
        
        return gain
    
    def _compute_redundancy(self, selected_features: torch.Tensor) -> float:
        
        unique_features = torch.unique(selected_features)
        
        redundancy = 1.0 - (len(unique_features) / len(selected_features))
        
        return redundancy