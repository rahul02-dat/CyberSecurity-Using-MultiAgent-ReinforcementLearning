from typing import Dict, Any, List
import numpy as np


class MetricsTracker:
    
    def __init__(self):
        self.steps: List[int] = []
        self.attack_types: List[str] = []
        self.attack_active: List[bool] = []
        self.detection_flags: List[bool] = []
        self.response_actions: List[str] = []
        self.detection_rewards: List[float] = []
        self.response_rewards: List[float] = []
        self.attacker_rewards: List[float] = []
    
    def update(
        self,
        step: int,
        attack_type: str,
        attack_active: bool,
        detection_flag: bool,
        response_action: str,
        rewards: Dict[str, float]
    ):
        self.steps.append(step)
        self.attack_types.append(attack_type)
        self.attack_active.append(attack_active)
        self.detection_flags.append(detection_flag)
        self.response_actions.append(response_action)
        self.detection_rewards.append(rewards.get("detection", 0.0))
        self.response_rewards.append(rewards.get("response", 0.0))
        self.attacker_rewards.append(rewards.get("attacker", 0.0))
    
    def get_summary(self) -> Dict[str, Any]:
        total_steps = len(self.steps)
        
        true_positives = sum(
            1 for active, flag in zip(self.attack_active, self.detection_flags)
            if active and flag
        )
        false_positives = sum(
            1 for active, flag in zip(self.attack_active, self.detection_flags)
            if not active and flag
        )
        true_negatives = sum(
            1 for active, flag in zip(self.attack_active, self.detection_flags)
            if not active and not flag
        )
        false_negatives = sum(
            1 for active, flag in zip(self.attack_active, self.detection_flags)
            if active and not flag
        )
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "total_steps": total_steps,
            "total_attacks": sum(self.attack_active),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3),
            "avg_detection_reward": round(np.mean(self.detection_rewards), 3),
            "avg_response_reward": round(np.mean(self.response_rewards), 3),
            "total_detection_reward": round(sum(self.detection_rewards), 3),
            "total_response_reward": round(sum(self.response_rewards), 3)
        }
    
    def get_attack_breakdown(self) -> Dict[str, int]:
        breakdown = {}
        for attack_type in self.attack_types:
            breakdown[attack_type] = breakdown.get(attack_type, 0) + 1
        return breakdown