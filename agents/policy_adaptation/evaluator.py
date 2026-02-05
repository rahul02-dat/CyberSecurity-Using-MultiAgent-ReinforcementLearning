from agents.base.base_evaluator import BaseEvaluator
from typing import Any
import numpy as np


class PolicyAdaptationEvaluator(BaseEvaluator):
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
    
    def compute_metrics(self, episode_data: dict[str, Any]) -> dict[str, float]:
        
        adaptations = episode_data.get('adaptations', [])
        performance_improvements = episode_data.get('performance_improvements', [])
        stability_scores = episode_data.get('stability_scores', [])
        
        if not adaptations:
            return {}
        
        metrics = {
            'num_adaptations': len(adaptations),
            'avg_performance_improvement': np.mean(performance_improvements) if performance_improvements else 0.0,
            'avg_stability_score': np.mean(stability_scores) if stability_scores else 0.0,
            'adaptation_success_rate': self._compute_success_rate(performance_improvements),
            'adaptation_magnitude': np.mean([self._compute_magnitude(a) for a in adaptations])
        }
        
        self.update_history(metrics)
        
        return metrics
    
    def aggregate_metrics(self, num_episodes: int) -> dict[str, float]:
        
        if not self.metrics_history:
            return {}
        
        recent = self.metrics_history[-num_episodes:]
        
        aggregated = {
            'total_adaptations': sum(m['num_adaptations'] for m in recent),
            'mean_performance_improvement': np.mean([m['avg_performance_improvement'] for m in recent]),
            'mean_stability_score': np.mean([m['avg_stability_score'] for m in recent]),
            'overall_success_rate': np.mean([m['adaptation_success_rate'] for m in recent]),
            'mean_adaptation_magnitude': np.mean([m['adaptation_magnitude'] for m in recent])
        }
        
        return aggregated
    
    def _compute_success_rate(self, improvements: list[float]) -> float:
        
        if not improvements:
            return 0.0
        
        successful = sum(1 for imp in improvements if imp > 0)
        
        return successful / len(improvements)
    
    def _compute_magnitude(self, adaptation: dict[str, Any]) -> float:
        
        magnitude = 0.0
        
        for key, value in adaptation.items():
            if isinstance(value, (int, float)):
                magnitude += abs(value)
        
        return magnitude