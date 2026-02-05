import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Optional
import json
import os
from agents.base.base_agent import BaseAgent
from agents.policy_adaptation.model import MetaControllerNetwork
from agents.policy_adaptation.reward import PolicyAdaptationReward


class PolicyAdaptationAgent(BaseAgent):
    
    def __init__(
        self, 
        agent_id: str, 
        observation_space: dict[str, Any], 
        action_space: dict[str, Any],
        learning_rate: float = 1e-4
    ):
        super().__init__(agent_id, observation_space, action_space)
        
        self.performance_metrics_dim = observation_space['performance_metrics_dim']
        self.num_hyperparams = action_space['num_hyperparams']
        
        self.meta_controller = MetaControllerNetwork(
            performance_metrics_dim=self.performance_metrics_dim,
            num_hyperparams=self.num_hyperparams
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.meta_controller.parameters(), lr=learning_rate)
        
        self.reward_function = PolicyAdaptationReward()
        
        self.conservativeness = 1.0
        self.magnitude_scaling = 1.0
        
        self.hyperparam_bounds = {
            'monitoring_lr': (1e-5, 1e-2),
            'detection_lr': (1e-5, 1e-2),
            'response_lr': (1e-5, 1e-2),
            'monitoring_features': (5, 50),
            'detection_confidence': (0.5, 0.95),
            'response_temperature': (0.5, 2.0)
        }
    
    def select_action(self, observation: dict[str, Any], training: bool = True) -> Any:
        
        performance_vector = self._build_performance_vector(observation['performance_data'])
        performance_tensor = torch.FloatTensor(performance_vector).to(self.device)
        
        with torch.no_grad():
            hyperparam_adjustments, confidence = self.meta_controller(performance_tensor.unsqueeze(0))
            
            hyperparam_adjustments = hyperparam_adjustments.squeeze(0)
            confidence_score = confidence.item()
        
        adaptations = self._decode_hyperparams(hyperparam_adjustments)
        
        adaptations = self._apply_conservativeness(adaptations, confidence_score)
        
        return {
            'hyperparameter_adaptations': adaptations,
            'confidence': confidence_score,
            'raw_adjustments': hyperparam_adjustments.cpu().numpy()
        }
    
    def _build_performance_vector(self, performance_data: dict[str, Any]) -> list[float]:
        
        vector = []
        
        metrics = [
            'monitoring_avg_reward',
            'monitoring_information_gain',
            'detection_accuracy',
            'detection_f1_score',
            'response_avg_reward',
            'response_efficiency_ratio'
        ]
        
        for metric in metrics:
            vector.append(performance_data.get(metric, 0.0))
        
        variance_metrics = [
            'monitoring_reward_std',
            'detection_accuracy_std',
            'response_reward_stability'
        ]
        
        for metric in variance_metrics:
            vector.append(performance_data.get(metric, 0.0))
        
        return vector
    
    def _decode_hyperparams(self, adjustments: torch.Tensor) -> dict[str, Any]:
        
        adjustments_np = adjustments.cpu().numpy()
        
        adaptations = {
            'monitoring_lr_multiplier': 1.0 + adjustments_np[0] * 0.5 * self.magnitude_scaling,
            'detection_lr_multiplier': 1.0 + adjustments_np[1] * 0.5 * self.magnitude_scaling,
            'response_lr_multiplier': 1.0 + adjustments_np[2] * 0.5 * self.magnitude_scaling,
            'monitoring_features_delta': int(adjustments_np[3] * 5 * self.magnitude_scaling),
            'detection_confidence_delta': adjustments_np[4] * 0.1 * self.magnitude_scaling,
            'response_temperature_delta': adjustments_np[5] * 0.3 * self.magnitude_scaling
        }
        
        return adaptations
    
    def _apply_conservativeness(
        self, 
        adaptations: dict[str, Any], 
        confidence: float
    ) -> dict[str, Any]:
        
        scaling_factor = confidence / self.conservativeness
        
        conservative_adaptations = {}
        for key, value in adaptations.items():
            if isinstance(value, (int, float)):
                if 'multiplier' in key:
                    conservative_adaptations[key] = 1.0 + (value - 1.0) * scaling_factor
                else:
                    conservative_adaptations[key] = value * scaling_factor
            else:
                conservative_adaptations[key] = value
        
        return conservative_adaptations
    
    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        
        states = torch.FloatTensor(batch['states']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        
        hyperparam_adjustments, confidences = self.meta_controller(states)
        
        prediction_loss = nn.MSELoss()(
            hyperparam_adjustments, 
            torch.zeros_like(hyperparam_adjustments)
        )
        
        reward_prediction_loss = nn.MSELoss()(
            confidences.squeeze(), 
            (rewards + 1.0) / 2.0
        )
        
        total_loss = prediction_loss + reward_prediction_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_controller.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'prediction_loss': prediction_loss.item(),
            'reward_prediction_loss': reward_prediction_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def save(self, path: str) -> None:
        torch.save({
            'meta_controller': self.meta_controller.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'conservativeness': self.conservativeness,
            'magnitude_scaling': self.magnitude_scaling
        }, path)
    
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.meta_controller.load_state_dict(checkpoint['meta_controller'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.conservativeness = checkpoint['conservativeness']
        self.magnitude_scaling = checkpoint['magnitude_scaling']
    
    def get_state_dict(self) -> dict[str, Any]:
        return {
            'meta_controller': self.meta_controller.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'conservativeness': self.conservativeness,
            'magnitude_scaling': self.magnitude_scaling
        }