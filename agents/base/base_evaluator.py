from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.metrics_history: list[dict[str, float]] = []
    
    @abstractmethod
    def compute_metrics(self, episode_data: dict[str, Any]) -> dict[str, float]:
        pass
    
    @abstractmethod
    def aggregate_metrics(self, num_episodes: int) -> dict[str, float]:
        pass
    
    def update_history(self, metrics: dict[str, float]) -> None:
        self.metrics_history.append(metrics)
    
    def reset_history(self) -> None:
        self.metrics_history = []
    
    def get_recent_performance(self, window: int = 100) -> dict[str, float]:
        if not self.metrics_history:
            return {}
        recent = self.metrics_history[-window:]
        avg_metrics = {}
        keys = recent[0].keys()
        for key in keys:
            avg_metrics[key] = sum(m[key] for m in recent) / len(recent)
        return avg_metrics