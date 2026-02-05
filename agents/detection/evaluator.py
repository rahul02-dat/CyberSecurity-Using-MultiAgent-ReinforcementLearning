from agents.base.base_evaluator import BaseEvaluator
from typing import Any
import numpy as np


class DetectionEvaluator(BaseEvaluator):
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.num_classes = 5
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def compute_metrics(self, episode_data: dict[str, Any]) -> dict[str, float]:
        
        predictions = episode_data.get('predictions', [])
        true_labels = episode_data.get('true_labels', [])
        confidences = episode_data.get('confidences', [])
        rewards = episode_data.get('rewards', [])
        
        if not predictions or not true_labels:
            return {}
        
        predictions_array = np.array(predictions)
        true_labels_array = np.array(true_labels)
        
        accuracy = np.mean(predictions_array == true_labels_array)
        
        tp = np.sum((predictions_array > 0) & (true_labels_array > 0))
        fp = np.sum((predictions_array > 0) & (true_labels_array == 0))
        fn = np.sum((predictions_array == 0) & (true_labels_array > 0))
        tn = np.sum((predictions_array == 0) & (true_labels_array == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'avg_reward': np.mean(rewards) if rewards else 0.0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0.0
        }
        
        self.update_history(metrics)
        
        return metrics
    
    def aggregate_metrics(self, num_episodes: int) -> dict[str, float]:
        
        if not self.metrics_history:
            return {}
        
        recent = self.metrics_history[-num_episodes:]
        
        aggregated = {
            'mean_accuracy': np.mean([m['accuracy'] for m in recent]),
            'mean_precision': np.mean([m['precision'] for m in recent]),
            'mean_recall': np.mean([m['recall'] for m in recent]),
            'mean_f1_score': np.mean([m['f1_score'] for m in recent]),
            'mean_confidence': np.mean([m['avg_confidence'] for m in recent]),
            'mean_reward': np.mean([m['avg_reward'] for m in recent]),
            'accuracy_std': np.std([m['accuracy'] for m in recent])
        }
        
        return aggregated
    
    def update_confusion_matrix(self, predictions: list[int], true_labels: list[int]) -> None:
        
        for pred, true in zip(predictions, true_labels):
            self.confusion_matrix[true, pred] += 1
    
    def get_confusion_matrix(self) -> np.ndarray:
        return self.confusion_matrix
    
    def reset_confusion_matrix(self) -> None:
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))