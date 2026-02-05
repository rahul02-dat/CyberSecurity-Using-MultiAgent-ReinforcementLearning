import numpy as np
from typing import Any


class EpsilonGreedy:
    
    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
    
    def select_action(self, q_values: np.ndarray, training: bool = True) -> int:
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(len(q_values))
        else:
            return np.argmax(q_values)
    
    def decay(self) -> None:
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_epsilon(self) -> float:
        return self.epsilon


class BoltzmannExploration:
    
    def __init__(self, temperature: float = 1.0, temperature_decay: float = 0.995):
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = 0.1
    
    def select_action(self, q_values: np.ndarray, training: bool = True) -> int:
        
        if not training:
            return np.argmax(q_values)
        
        scaled_q = q_values / self.temperature
        exp_q = np.exp(scaled_q - np.max(scaled_q))
        probabilities = exp_q / np.sum(exp_q)
        
        return np.random.choice(len(q_values), p=probabilities)
    
    def decay(self) -> None:
        
        self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)
    
    def get_temperature(self) -> float:
        return self.temperature


class OrnsteinUhlenbeckNoise:
    
    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2
    ):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        
        self.state = np.ones(action_dim) * mu
    
    def sample(self) -> np.ndarray:
        
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        
        return self.state
    
    def reset(self) -> None:
        
        self.state = np.ones(self.action_dim) * self.mu