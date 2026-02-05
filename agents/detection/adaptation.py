from agents.base.base_adaptation import BaseAdaptation
from typing import Any


class DetectionAdaptation(BaseAdaptation):
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.min_lr = 1e-5
        self.max_lr = 1e-2
        self.min_confidence_threshold = 0.5
        self.max_confidence_threshold = 0.95
    
    def propose_adaptations(self, performance_data: dict[str, Any]) -> dict[str, Any]:
        
        accuracy = performance_data.get('mean_accuracy', 0.0)
        precision = performance_data.get('mean_precision', 0.0)
        recall = performance_data.get('mean_recall', 0.0)
        f1_score = performance_data.get('mean_f1_score', 0.0)
        
        adaptations = {}
        
        if f1_score < 0.6:
            adaptations['learning_rate_multiplier'] = 1.3
        elif f1_score > 0.85:
            adaptations['learning_rate_multiplier'] = 0.95
        else:
            adaptations['learning_rate_multiplier'] = 1.0
        
        if precision < 0.7:
            adaptations['confidence_threshold_delta'] = 0.05
        elif precision > 0.9 and recall < 0.7:
            adaptations['confidence_threshold_delta'] = -0.05
        else:
            adaptations['confidence_threshold_delta'] = 0.0
        
        if accuracy < 0.7:
            adaptations['increase_model_capacity'] = True
            adaptations['dropout_rate_delta'] = -0.05
        else:
            adaptations['increase_model_capacity'] = False
            adaptations['dropout_rate_delta'] = 0.0
        
        if recall < 0.65:
            adaptations['class_weight_adjustment'] = 'increase_attack_weights'
        elif recall > 0.9 and precision < 0.7:
            adaptations['class_weight_adjustment'] = 'decrease_attack_weights'
        else:
            adaptations['class_weight_adjustment'] = 'balanced'
        
        self.record_adaptation(adaptations)
        
        return adaptations
    
    def apply_adaptations(self, adaptations: dict[str, Any], agent: Any) -> None:
        
        if 'learning_rate_multiplier' in adaptations:
            for param_group in agent.optimizer.param_groups:
                new_lr = param_group['lr'] * adaptations['learning_rate_multiplier']
                param_group['lr'] = max(self.min_lr, min(self.max_lr, new_lr))
        
        if 'confidence_threshold_delta' in adaptations and hasattr(agent, 'confidence_threshold'):
            current_threshold = agent.confidence_threshold
            new_threshold = current_threshold + adaptations['confidence_threshold_delta']
            agent.confidence_threshold = max(
                self.min_confidence_threshold, 
                min(self.max_confidence_threshold, new_threshold)
            )
        
        if 'class_weight_adjustment' in adaptations and hasattr(agent, 'class_weights'):
            adjustment_type = adaptations['class_weight_adjustment']
            if adjustment_type == 'increase_attack_weights':
                agent.class_weights[1:] *= 1.2
            elif adjustment_type == 'decrease_attack_weights':
                agent.class_weights[1:] *= 0.9