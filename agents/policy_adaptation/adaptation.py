from agents.base.base_adaptation import BaseAdaptation
from typing import Any


class PolicyAdaptationAdaptation(BaseAdaptation):
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.min_lr = 1e-5
        self.max_lr = 1e-2
    
    def propose_adaptations(self, performance_data: dict[str, Any]) -> dict[str, Any]:
        
        success_rate = performance_data.get('overall_success_rate', 0.0)
        mean_improvement = performance_data.get('mean_performance_improvement', 0.0)
        adaptation_magnitude = performance_data.get('mean_adaptation_magnitude', 0.0)
        
        adaptations = {}
        
        if success_rate < 0.4:
            adaptations['learning_rate_multiplier'] = 1.5
            adaptations['exploration_increase'] = 0.1
        elif success_rate > 0.8:
            adaptations['learning_rate_multiplier'] = 0.9
            adaptations['exploration_increase'] = -0.05
        else:
            adaptations['learning_rate_multiplier'] = 1.0
            adaptations['exploration_increase'] = 0.0
        
        if mean_improvement < 0:
            adaptations['adaptation_conservativeness'] = 1.5
        elif mean_improvement > 0.2:
            adaptations['adaptation_conservativeness'] = 0.8
        else:
            adaptations['adaptation_conservativeness'] = 1.0
        
        if adaptation_magnitude > 5.0:
            adaptations['magnitude_scaling'] = 0.7
        elif adaptation_magnitude < 1.0:
            adaptations['magnitude_scaling'] = 1.3
        else:
            adaptations['magnitude_scaling'] = 1.0
        
        self.record_adaptation(adaptations)
        
        return adaptations
    
    def apply_adaptations(self, adaptations: dict[str, Any], agent: Any) -> None:
        
        if 'learning_rate_multiplier' in adaptations:
            for param_group in agent.optimizer.param_groups:
                new_lr = param_group['lr'] * adaptations['learning_rate_multiplier']
                param_group['lr'] = max(self.min_lr, min(self.max_lr, new_lr))
        
        if 'adaptation_conservativeness' in adaptations and hasattr(agent, 'conservativeness'):
            agent.conservativeness *= adaptations['adaptation_conservativeness']
        
        if 'magnitude_scaling' in adaptations and hasattr(agent, 'magnitude_scaling'):
            agent.magnitude_scaling = adaptations['magnitude_scaling']