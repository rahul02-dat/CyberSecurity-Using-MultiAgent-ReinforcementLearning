import torch
from typing import Any


class PolicyAdaptationReward:
    
    def __init__(
        self,
        improvement_weight: float = 1.0,
        stability_weight: float = 0.5,
        adaptation_cost_weight: float = 0.2
    ):
        self.improvement_weight = improvement_weight
        self.stability_weight = stability_weight
        self.adaptation_cost_weight = adaptation_cost_weight
    
    def compute(
        self,
        previous_performance: dict[str, float],
        current_performance: dict[str, float],
        adaptation_magnitude: float
    ) -> float:
        
        improvement_reward = self._compute_improvement(
            previous_performance,
            current_performance
        )
        
        stability_penalty = self._compute_stability_penalty(
            previous_performance,
            current_performance
        )
        
        adaptation_cost = adaptation_magnitude
        
        reward = (
            self.improvement_weight * improvement_reward
            - self.stability_weight * stability_penalty
            - self.adaptation_cost_weight * adaptation_cost
        )
        
        return reward
    
    def _compute_improvement(
        self,
        previous: dict[str, float],
        current: dict[str, float]
    ) -> float:
        
        key_metrics = [
            'monitoring_avg_reward',
            'detection_f1_score',
            'response_efficiency_ratio'
        ]
        
        improvements = []
        for metric in key_metrics:
            prev_value = previous.get(metric, 0.0)
            curr_value = current.get(metric, 0.0)
            
            if prev_value != 0:
                improvement = (curr_value - prev_value) / abs(prev_value)
            else:
                improvement = curr_value
            
            improvements.append(improvement)
        
        avg_improvement = sum(improvements) / len(improvements)
        
        return avg_improvement
    
    def _compute_stability_penalty(
        self,
        previous: dict[str, float],
        current: dict[str, float]
    ) -> float:
        
        variance_metrics = [
            'monitoring_reward_std',
            'detection_accuracy_std',
            'response_reward_stability'
        ]
        
        variances = []
        for metric in variance_metrics:
            prev_var = previous.get(metric, 0.0)
            curr_var = current.get(metric, 0.0)
            
            variance_increase = max(0, curr_var - prev_var)
            variances.append(variance_increase)
        
        avg_variance_increase = sum(variances) / len(variances) if variances else 0.0
        
        return avg_variance_increase