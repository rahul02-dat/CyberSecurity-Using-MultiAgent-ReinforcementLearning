from agents.base.base_evaluator import BaseEvaluator
from typing import Any
import numpy as np


class ResponseEvaluator(BaseEvaluator):
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.action_names = ['DoNothing', 'Log', 'BlockIP', 'RateLimit', 'Honeypot', 'ShutDownSubnet']
    
    def compute_metrics(self, episode_data: dict[str, Any]) -> dict[str, float]:
        
        actions = episode_data.get('actions', [])
        rewards = episode_data.get('rewards', [])
        damages_prevented = episode_data.get('damages_prevented', [])
        costs_incurred = episode_data.get('costs_incurred', [])
        
        if not actions or not rewards:
            return {}
        
        action_counts = np.bincount(actions, minlength=6)
        
        metrics = {
            'avg_reward': np.mean(rewards),
            'total_reward': sum(rewards),
            'total_damage_prevented': sum(damages_prevented) if damages_prevented else 0.0,
            'total_cost_incurred': sum(costs_incurred) if costs_incurred else 0.0,
            'action_diversity': self._compute_entropy(action_counts),
            'do_nothing_ratio': action_counts[0] / len(actions),
            'aggressive_action_ratio': (action_counts[4] + action_counts[5]) / len(actions)
        }
        
        for i, name in enumerate(self.action_names):
            metrics[f'{name}_count'] = int(action_counts[i])
        
        if damages_prevented and costs_incurred:
            total_prevented = sum(damages_prevented)
            total_cost = sum(costs_incurred)
            if total_cost > 0:
                metrics['efficiency_ratio'] = total_prevented / total_cost
            else:
                metrics['efficiency_ratio'] = float('inf') if total_prevented > 0 else 0.0
        
        self.update_history(metrics)
        
        return metrics
    
    def aggregate_metrics(self, num_episodes: int) -> dict[str, float]:
        
        if not self.metrics_history:
            return {}
        
        recent = self.metrics_history[-num_episodes:]
        
        aggregated = {
            'mean_reward': np.mean([m['avg_reward'] for m in recent]),
            'mean_damage_prevented': np.mean([m['total_damage_prevented'] for m in recent]),
            'mean_cost_incurred': np.mean([m['total_cost_incurred'] for m in recent]),
            'mean_efficiency_ratio': np.mean([m.get('efficiency_ratio', 0.0) for m in recent if m.get('efficiency_ratio', 0.0) != float('inf')]),
            'mean_action_diversity': np.mean([m['action_diversity'] for m in recent]),
            'reward_stability': np.std([m['avg_reward'] for m in recent])
        }
        
        return aggregated
    
    def _compute_entropy(self, counts: np.ndarray) -> float:
        
        total = counts.sum()
        if total == 0:
            return 0.0
        
        probabilities = counts / total
        probabilities = probabilities[probabilities > 0]
        
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy