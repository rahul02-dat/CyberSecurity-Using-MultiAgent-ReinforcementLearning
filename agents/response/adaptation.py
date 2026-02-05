from agents.base.base_adaptation import BaseAdaptation
from typing import Any


class ResponseAdaptation(BaseAdaptation):
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.min_lr = 1e-5
        self.max_lr = 1e-2
    
    def propose_adaptations(self, performance_data: dict[str, Any]) -> dict[str, Any]:
        
        mean_reward = performance_data.get('mean_reward', 0.0)
        efficiency_ratio = performance_data.get('mean_efficiency_ratio', 1.0)
        action_diversity = performance_data.get('mean_action_diversity', 0.0)
        
        adaptations = {}
        
        if mean_reward < 0:
            adaptations['learning_rate_multiplier'] = 1.5
        elif mean_reward > 50:
            adaptations['learning_rate_multiplier'] = 0.9
        else:
            adaptations['learning_rate_multiplier'] = 1.0
        
        if efficiency_ratio < 1.5:
            adaptations['cost_sensitivity_multiplier'] = 1.2
        elif efficiency_ratio > 5.0:
            adaptations['cost_sensitivity_multiplier'] = 0.8
        else:
            adaptations['cost_sensitivity_multiplier'] = 1.0
        
        if action_diversity < 1.5:
            adaptations['exploration_bonus'] = 0.1
            adaptations['temperature'] = 1.2
        elif action_diversity > 2.0:
            adaptations['exploration_bonus'] = 0.0
            adaptations['temperature'] = 0.8
        else:
            adaptations['exploration_bonus'] = 0.05
            adaptations['temperature'] = 1.0
        
        self.record_adaptation(adaptations)
        
        return adaptations
    
    def apply_adaptations(self, adaptations: dict[str, Any], agent: Any) -> None:
        
        if 'learning_rate_multiplier' in adaptations:
            for param_group in agent.optimizer.param_groups:
                new_lr = param_group['lr'] * adaptations['learning_rate_multiplier']
                param_group['lr'] = max(self.min_lr, min(self.max_lr, new_lr))
        
        if 'cost_sensitivity_multiplier' in adaptations and hasattr(agent, 'cost_sensitivity'):
            agent.cost_sensitivity *= adaptations['cost_sensitivity_multiplier']
        
        if 'temperature' in adaptations and hasattr(agent, 'temperature'):
            agent.temperature = adaptations['temperature']
        
        if 'exploration_bonus' in adaptations and hasattr(agent, 'exploration_bonus'):
            agent.exploration_bonus = adaptations['exploration_bonus']