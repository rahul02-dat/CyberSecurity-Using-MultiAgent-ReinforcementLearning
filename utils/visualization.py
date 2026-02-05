import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
import os


sns.set_style("whitegrid")


def plot_training_curves(
    metrics: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[str] = None
) -> None:
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        axes[idx].plot(values, label=metric_name)
        axes[idx].set_xlabel('Step')
        axes[idx].set_ylabel(metric_name)
        axes[idx].set_title(f'{metric_name} over time')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_agent_comparison(
    agent_metrics: Dict[str, Dict[str, float]],
    metric_name: str,
    title: str = "Agent Performance Comparison",
    save_path: Optional[str] = None
) -> None:
    
    agents = list(agent_metrics.keys())
    values = [metrics.get(metric_name, 0.0) for metrics in agent_metrics.values()]
    
    plt.figure(figsize=(10, 6))
    plt.bar(agents, values)
    plt.xlabel('Agent')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> None:
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_reward_distribution(
    rewards: List[float],
    title: str = "Reward Distribution",
    save_path: Optional[str] = None
) -> None:
    
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_multiple_agents_performance(
    metrics_over_time: Dict[str, List[float]],
    title: str = "Multi-Agent Performance",
    save_path: Optional[str] = None
) -> None:
    
    plt.figure(figsize=(12, 6))
    
    for agent_name, values in metrics_over_time.items():
        plt.plot(values, label=agent_name, alpha=0.8)
    
    plt.xlabel('Episode')
    plt.ylabel('Performance Metric')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()