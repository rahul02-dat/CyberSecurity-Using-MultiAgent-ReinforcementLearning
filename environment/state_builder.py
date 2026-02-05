import numpy as np
from typing import Any


class StateBuilder:
    
    def __init__(self, feature_dim: int = 50):
        self.feature_dim = feature_dim
    
    def build_monitoring_state(
        self, 
        traffic: np.ndarray, 
        attack: np.ndarray
    ) -> dict[str, Any]:
        
        combined = traffic + attack
        combined = np.clip(combined, 0, 1)
        
        state = {
            'feature_vector': combined,
            'traffic_intensity': traffic.mean(),
            'anomaly_score': self._compute_anomaly_score(combined)
        }
        
        return state
    
    def build_detection_state(
        self,
        compressed_features: np.ndarray,
        monitoring_info: dict[str, Any]
    ) -> dict[str, Any]:
        
        state = {
            'compressed_features': compressed_features,
            'num_features': len(compressed_features),
            'feature_variance': compressed_features.std(),
            'monitoring_confidence': monitoring_info.get('confidence', 0.5)
        }
        
        return state
    
    def build_response_state(
        self,
        detection_result: dict[str, Any],
        system_state: dict[str, Any]
    ) -> dict[str, Any]:
        
        state_vector = []
        
        predicted_class = detection_result.get('predicted_class', 0)
        class_one_hot = np.zeros(5)
        class_one_hot[predicted_class] = 1.0
        state_vector.extend(class_one_hot)
        
        state_vector.append(detection_result.get('confidence', 0.0))
        
        state_vector.append(system_state.get('infection_level', 0.0))
        state_vector.append(system_state.get('resource_usage', 0.5))
        state_vector.append(system_state.get('critical_assets_at_risk', 0.0))
        
        state_vector.extend(system_state.get('recent_attack_history', [0.0] * 5))
        
        state = {
            'state': np.array(state_vector),
            'predicted_class': predicted_class,
            'confidence': detection_result.get('confidence', 0.0),
            'infection_level': system_state.get('infection_level', 0.0)
        }
        
        return state
    
    def build_policy_adaptation_state(
        self,
        performance_data: dict[str, Any]
    ) -> dict[str, Any]:
        
        state = {
            'performance_data': performance_data
        }
        
        return state
    
    def _compute_anomaly_score(self, features: np.ndarray) -> float:
        
        mean = features.mean()
        std = features.std()
        max_val = features.max()
        
        anomaly_score = (max_val - mean) / (std + 1e-6)
        
        anomaly_score = min(1.0, anomaly_score / 3.0)
        
        return anomaly_score
    
    def compress_features(
        self, 
        full_features: np.ndarray, 
        selected_indices: np.ndarray
    ) -> np.ndarray:
        
        compressed = full_features[selected_indices]
        
        return compressed