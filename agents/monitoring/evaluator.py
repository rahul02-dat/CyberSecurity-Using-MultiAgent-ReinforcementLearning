from agents.base.base_evaluator import BaseEvaluator
from typing import Any
import numpy as np


class MonitoringEvaluator(BaseEvaluator):
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
    
    def compute_metrics(self, episode_data: dict[str, Any]) -> dict[str, float]:
        
        rewards = episode_data.get('rewards', [])
        
        selected_features_counts = episode_data.get('selected_features_counts', [])
        
        information_gains = episode_data.get('information_gains', [])
        
        metrics = {
            'avg_reward': np.mean(rewards) if rewards else 0.0,
            'total_reward': sum(rewards),
            'avg_features_selected': np.mean(selected_features_counts) if selected_features_counts else 0.0,
            'avg_information_gain': np.mean(information_gains) if information_gains else 0.0,
            'reward_std': np.std(rewards) if rewards else 0.0
        }
        
        self.update_history(metrics)
        
        return metrics
    
    def aggregate_metrics(self, num_episodes: int) -> dict[str, float]:
        
        if not self.metrics_history:
            return {}
        
        recent = self.metrics_history[-num_episodes:]
        
        aggregated = {
            'mean_avg_reward': np.mean([m['avg_reward'] for m in recent]),
            'mean_total_reward': np.mean([m['total_reward'] for m in recent]),
            'mean_features_selected': np.mean([m['avg_features_selected'] for m in recent]),
            'mean_information_gain': np.mean([m['avg_information_gain'] for m in recent]),
            'reward_stability': np.std([m['avg_reward'] for m in recent])
        }
        
        return aggregated