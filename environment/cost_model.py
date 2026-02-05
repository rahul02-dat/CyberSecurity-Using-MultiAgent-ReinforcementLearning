import numpy as np
from typing import Any


class CostModel:
    
    def __init__(self):
        
        self.bandwidth_cost_per_feature = 0.01
        
        self.compute_cost_base = 0.5
        self.compute_cost_per_class = 0.1
        
        self.action_costs = {
            0: 0.0,
            1: 1.0,
            2: 10.0,
            3: 5.0,
            4: 15.0,
            5: 50.0
        }
        
        self.resource_usage_weights = {
            'bandwidth': 0.3,
            'compute': 0.4,
            'mitigation': 0.3
        }
    
    def compute_monitoring_cost(self, num_features_selected: int, total_features: int) -> float:
        
        bandwidth_ratio = num_features_selected / total_features
        
        cost = bandwidth_ratio * self.bandwidth_cost_per_feature * total_features
        
        return cost
    
    def compute_detection_cost(self, num_classes: int, use_llm: bool = False) -> float:
        
        base_cost = self.compute_cost_base
        
        class_cost = self.compute_cost_per_class * num_classes
        
        llm_cost = 5.0 if use_llm else 0.0
        
        total_cost = base_cost + class_cost + llm_cost
        
        return total_cost
    
    def compute_response_cost(self, action: int) -> float:
        
        return self.action_costs.get(action, 0.0)
    
    def compute_total_resource_usage(
        self,
        monitoring_cost: float,
        detection_cost: float,
        response_cost: float
    ) -> float:
        
        weighted_sum = (
            self.resource_usage_weights['bandwidth'] * monitoring_cost +
            self.resource_usage_weights['compute'] * detection_cost +
            self.resource_usage_weights['mitigation'] * response_cost
        )
        
        resource_usage = min(1.0, weighted_sum / 50.0)
        
        return resource_usage
    
    def compute_damage(
        self,
        attack_type: int,
        infection_level: float,
        response_effectiveness: float
    ) -> float:
        
        base_damages = {
            0: 0.0,
            1: 30.0,
            2: 50.0,
            3: 80.0,
            4: 20.0
        }
        
        base_damage = base_damages.get(attack_type, 0.0)
        
        infection_multiplier = 1.0 + infection_level
        
        mitigated_damage = base_damage * infection_multiplier * (1.0 - response_effectiveness)
        
        return mitigated_damage