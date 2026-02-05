import gymnasium as gym
import numpy as np
from typing import Any, Optional
from environment.traffic_generator import TrafficGenerator
from environment.attack_simulator import AttackSimulator
from environment.state_builder import StateBuilder
from environment.cost_model import CostModel


class NetworkEnvironment(gym.Env):
    
    def __init__(
        self,
        feature_dim: int = 50,
        max_steps: int = 1000,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.max_steps = max_steps
        
        self.traffic_generator = TrafficGenerator(feature_dim=feature_dim, seed=seed)
        self.attack_simulator = AttackSimulator(feature_dim=feature_dim, seed=seed)
        self.state_builder = StateBuilder(feature_dim=feature_dim)
        self.cost_model = CostModel()
        
        self.observation_space = gym.spaces.Dict({
            'feature_vector': gym.spaces.Box(low=0, high=1, shape=(feature_dim,), dtype=np.float32),
            'compressed_features': gym.spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32),
            'system_state': gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        })
        
        self.action_space = gym.spaces.Dict({
            'monitoring': gym.spaces.MultiBinary(feature_dim),
            'detection': gym.spaces.Discrete(5),
            'response': gym.spaces.Discrete(6)
        })
        
        self.current_step = 0
        self.infection_level = 0.0
        self.resource_usage = 0.5
        self.critical_assets_at_risk = False
        self.recent_attack_history = [0] * 5
        
        self.total_damage = 0.0
        self.total_cost = 0.0
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict[str, Any], dict[str, Any]]:
        
        super().reset(seed=seed)
        
        if seed is not None:
            self.traffic_generator.rng = np.random.RandomState(seed)
            self.attack_simulator.rng = np.random.RandomState(seed + 1)
        
        self.current_step = 0
        self.infection_level = 0.0
        self.resource_usage = 0.5
        self.critical_assets_at_risk = False
        self.recent_attack_history = [0] * 5
        
        self.total_damage = 0.0
        self.total_cost = 0.0
        
        observation = self._get_initial_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, actions: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        
        traffic = self.traffic_generator.generate(1)[0]
        attack_vector, attack_type = self.attack_simulator.generate_attack(self.infection_level)
        
        monitoring_state = self.state_builder.build_monitoring_state(traffic, attack_vector)
        
        selected_indices = actions.get('monitoring_selected_indices', np.arange(20))
        compressed_features = self.state_builder.compress_features(
            monitoring_state['feature_vector'],
            selected_indices
        )
        
        detection_state = self.state_builder.build_detection_state(
            compressed_features,
            {'confidence': 0.8}
        )
        
        predicted_class = actions.get('detection_class', 0)
        detection_confidence = actions.get('detection_confidence', 0.7)
        
        system_state = {
            'infection_level': self.infection_level,
            'resource_usage': self.resource_usage,
            'critical_assets_at_risk': self.critical_assets_at_risk,
            'recent_attack_history': self.recent_attack_history
        }
        
        response_state = self.state_builder.build_response_state(
            {'predicted_class': predicted_class, 'confidence': detection_confidence},
            system_state
        )
        
        response_action = actions.get('response_action', 0)
        
        monitoring_cost = self.cost_model.compute_monitoring_cost(
            len(selected_indices),
            self.feature_dim
        )
        detection_cost = self.cost_model.compute_detection_cost(5)
        response_cost = self.cost_model.compute_response_cost(response_action)
        
        self.total_cost += monitoring_cost + detection_cost + response_cost
        
        response_effectiveness = self._compute_response_effectiveness(
            response_action,
            attack_type,
            predicted_class
        )
        
        damage = self.cost_model.compute_damage(
            attack_type,
            self.infection_level,
            response_effectiveness
        )
        self.total_damage += damage
        
        self._update_infection_level(attack_type, response_effectiveness)
        
        self.resource_usage = self.cost_model.compute_total_resource_usage(
            monitoring_cost,
            detection_cost,
            response_cost
        )
        
        self._update_attack_history(attack_type)
        
        reward = self._compute_global_reward(damage, monitoring_cost, detection_cost, response_cost)
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = self.infection_level >= 1.0
        
        observation = self._get_observation(monitoring_state, compressed_features, system_state)
        info = self._get_info()
        info['attack_type'] = attack_type
        info['predicted_class'] = predicted_class
        info['response_action'] = response_action
        info['damage'] = damage
        
        return observation, reward, terminated, truncated, info
    
    def _get_initial_observation(self) -> dict[str, Any]:
        
        traffic = self.traffic_generator.generate(1)[0]
        attack_vector = np.zeros(self.feature_dim)
        
        monitoring_state = self.state_builder.build_monitoring_state(traffic, attack_vector)
        compressed_features = monitoring_state['feature_vector'][:20]
        
        system_state = {
            'infection_level': self.infection_level,
            'resource_usage': self.resource_usage,
            'critical_assets_at_risk': self.critical_assets_at_risk,
            'recent_attack_history': self.recent_attack_history
        }
        
        return self._get_observation(monitoring_state, compressed_features, system_state)
    
    def _get_observation(
        self,
        monitoring_state: dict[str, Any],
        compressed_features: np.ndarray,
        system_state: dict[str, Any]
    ) -> dict[str, Any]:
        
        system_state_vector = np.array([
            system_state['infection_level'],
            system_state['resource_usage'],
            float(system_state['critical_assets_at_risk'])
        ] + system_state['recent_attack_history'][:5])
        
        observation = {
            'feature_vector': monitoring_state['feature_vector'].astype(np.float32),
            'compressed_features': compressed_features.astype(np.float32),
            'system_state': system_state_vector.astype(np.float32)
        }
        
        return observation
    
    def _compute_response_effectiveness(
        self,
        response_action: int,
        true_attack_type: int,
        predicted_attack_type: int
    ) -> float:
        
        if true_attack_type == 0:
            return 0.0
        
        if predicted_attack_type != true_attack_type:
            return 0.0
        
        effectiveness_map = {
            0: 0.0,
            1: 0.2,
            2: 0.7,
            3: 0.5,
            4: 0.6,
            5: 0.95
        }
        
        return effectiveness_map.get(response_action, 0.0)
    
    def _update_infection_level(self, attack_type: int, response_effectiveness: float) -> None:
        
        if attack_type != 0:
            infection_increase = 0.01 * (1.0 - response_effectiveness)
            self.infection_level = min(1.0, self.infection_level + infection_increase)
        else:
            infection_decrease = 0.005
            self.infection_level = max(0.0, self.infection_level - infection_decrease)
    
    def _update_attack_history(self, attack_type: int) -> None:
        
        self.recent_attack_history.append(attack_type)
        self.recent_attack_history = self.recent_attack_history[-5:]
    
    def _compute_global_reward(
        self,
        damage: float,
        monitoring_cost: float,
        detection_cost: float,
        response_cost: float
    ) -> float:
        
        total_cost = monitoring_cost + detection_cost + response_cost
        
        reward = -damage - 0.1 * total_cost
        
        if self.infection_level < 0.2:
            reward += 1.0
        elif self.infection_level > 0.8:
            reward -= 5.0
        
        return reward
    
    def _get_info(self) -> dict[str, Any]:
        
        return {
            'step': self.current_step,
            'infection_level': self.infection_level,
            'resource_usage': self.resource_usage,
            'total_damage': self.total_damage,
            'total_cost': self.total_cost
        }