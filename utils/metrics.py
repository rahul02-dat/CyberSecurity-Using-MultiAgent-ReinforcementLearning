import numpy as np
from typing import Any, Dict, List


def compute_running_average(values: List[float], window: int = 100) -> float:
    
    if not values:
        return 0.0
    
    recent = values[-window:]
    return np.mean(recent)


def compute_exponential_moving_average(
    values: List[float], 
    alpha: float = 0.1
) -> List[float]:
    
    if not values:
        return []
    
    ema = [values[0]]
    
    for value in values[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])
    
    return ema


def compute_reward_statistics(rewards: List[float]) -> Dict[str, float]:
    
    if not rewards:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'total': 0.0
        }
    
    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'total': np.sum(rewards)
    }


def compute_success_rate(outcomes: List[bool]) -> float:
    
    if not outcomes:
        return 0.0
    
    return sum(outcomes) / len(outcomes)


def normalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    
    normalized = {}
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            normalized[key] = float(value)
        else:
            normalized[key] = value
    
    return normalized


def aggregate_agent_metrics(
    agent_metrics: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    
    all_keys = set()
    for metrics in agent_metrics.values():
        all_keys.update(metrics.keys())
    
    aggregated = {}
    
    for key in all_keys:
        values = [metrics.get(key, 0.0) for metrics in agent_metrics.values()]
        aggregated[f'mean_{key}'] = np.mean(values)
        aggregated[f'std_{key}'] = np.std(values)
    
    return aggregated


class MetricsTracker:
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def add(self, name: str, value: float) -> None:
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
    
    def get(self, name: str) -> List[float]:
        
        return self.metrics.get(name, [])
    
    def get_latest(self, name: str, default: float = 0.0) -> float:
        
        values = self.metrics.get(name, [])
        return values[-1] if values else default
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        
        values = self.metrics.get(name, [])
        return compute_reward_statistics(values)
    
    def reset(self) -> None:
        
        self.metrics = {}