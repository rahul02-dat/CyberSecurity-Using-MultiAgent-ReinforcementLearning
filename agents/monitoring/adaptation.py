from agents.base.base_adaptation import BaseAdaptation
from typing import Any


class MonitoringAdaptation(BaseAdaptation):
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.min_lr = 1e-5
        self.max_lr = 1e-2
        self.min_features = 5
        self.max_features = 50
    
    def propose_adaptations(self, performance_data: dict[str, Any]) -> dict[str, Any]:
        
        avg_reward = performance_data.get('mean_avg_reward', 0.0)
        information_gain = performance_data.get('mean_information_gain', 0.0)
        reward_stability = performance_data.get('reward_stability', 1.0)
        
        adaptations = {}
        
        if avg_reward < 0.3:
            adaptations['learning_rate_multiplier'] = 1.2
        elif avg_reward > 0.7:
            adaptations['learning_rate_multiplier'] = 0.9
        else:
            adaptations['learning_rate_multiplier'] = 1.0
        
        if information_gain < 0.5:
            adaptations['num_features_delta'] = 2
        elif information_gain > 0.85:
            adaptations['num_features_delta'] = -1
        else:
            adaptations['num_features_delta'] = 0
        
        if reward_stability > 0.3:
            adaptations['exploration_decay_rate'] = 0.995
        else:
            adaptations['exploration_decay_rate'] = 0.999
        
        self.record_adaptation(adaptations)
        
        return adaptations
    
    def apply_adaptations(self, adaptations: dict[str, Any], agent: Any) -> None:
        
        if 'learning_rate_multiplier' in adaptations:
            for param_group in agent.optimizer.param_groups:
                new_lr = param_group['lr'] * adaptations['learning_rate_multiplier']
                param_group['lr'] = max(self.min_lr, min(self.max_lr, new_lr))
        
        if 'num_features_delta' in adaptations:
            current_k = agent.num_features_to_select
            new_k = current_k + adaptations['num_features_delta']
            agent.num_features_to_select = max(self.min_features, min(self.max_features, new_k))
        
        if 'exploration_decay_rate' in adaptations and hasattr(agent, 'exploration_rate'):
            agent.exploration_decay = adaptations['exploration_decay_rate']