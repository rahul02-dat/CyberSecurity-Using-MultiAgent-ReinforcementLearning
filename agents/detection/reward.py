import torch
from typing import Any


class DetectionReward:
    
    def __init__(
        self,
        true_positive_reward: float = 1.0,
        false_positive_penalty: float = 0.5,
        false_negative_penalty: float = 1.5,
        true_negative_reward: float = 0.2,
        confidence_bonus_weight: float = 0.3
    ):
        self.true_positive_reward = true_positive_reward
        self.false_positive_penalty = false_positive_penalty
        self.false_negative_penalty = false_negative_penalty
        self.true_negative_reward = true_negative_reward
        self.confidence_bonus_weight = confidence_bonus_weight
        
        self.class_mapping = {
            0: 'Normal',
            1: 'DDoS',
            2: 'Injection',
            3: 'Malware',
            4: 'Probe'
        }
    
    def compute(
        self,
        predicted_class: int,
        true_class: int,
        confidence: float
    ) -> float:
        
        reward = 0.0
        
        if predicted_class == true_class:
            if true_class == 0:
                reward = self.true_negative_reward
            else:
                reward = self.true_positive_reward
        else:
            if predicted_class == 0 and true_class != 0:
                reward = -self.false_negative_penalty
            elif predicted_class != 0 and true_class == 0:
                reward = -self.false_positive_penalty
            else:
                reward = -self.false_positive_penalty * 0.7
        
        if predicted_class == true_class:
            reward += self.confidence_bonus_weight * confidence
        else:
            reward -= self.confidence_bonus_weight * confidence
        
        return reward
    
    def compute_batch(
        self,
        predicted_classes: torch.Tensor,
        true_classes: torch.Tensor,
        confidences: torch.Tensor
    ) -> torch.Tensor:
        
        batch_size = predicted_classes.shape[0]
        rewards = torch.zeros(batch_size)
        
        for i in range(batch_size):
            rewards[i] = self.compute(
                predicted_classes[i].item(),
                true_classes[i].item(),
                confidences[i].item()
            )
        
        return rewards