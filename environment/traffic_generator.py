import numpy as np
from typing import Any


class TrafficGenerator:
    
    def __init__(self, feature_dim: int = 50, seed: int = 42):
        self.feature_dim = feature_dim
        self.rng = np.random.RandomState(seed)
        
        self.normal_traffic_mean = 0.3
        self.normal_traffic_std = 0.1
        
        self.traffic_patterns = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'burst': 1.0
        }
        
        self.current_pattern = 'medium'
        self.pattern_duration = 0
        self.pattern_timer = 0
    
    def generate(self, num_packets: int = 1) -> np.ndarray:
        
        traffic = np.zeros((num_packets, self.feature_dim))
        
        for i in range(num_packets):
            packet = self._generate_packet()
            traffic[i] = packet
        
        return traffic
    
    def _generate_packet(self) -> np.ndarray:
        
        packet = self.rng.normal(
            self.normal_traffic_mean,
            self.normal_traffic_std,
            self.feature_dim
        )
        
        packet = np.clip(packet, 0, 1)
        
        intensity = self.traffic_patterns[self.current_pattern]
        packet *= intensity
        
        self._update_pattern()
        
        return packet
    
    def _update_pattern(self) -> None:
        
        self.pattern_timer += 1
        
        if self.pattern_duration == 0:
            self.pattern_duration = self.rng.randint(50, 200)
        
        if self.pattern_timer >= self.pattern_duration:
            self.pattern_timer = 0
            self.pattern_duration = 0
            
            patterns = list(self.traffic_patterns.keys())
            self.current_pattern = self.rng.choice(patterns)
    
    def set_pattern(self, pattern: str) -> None:
        
        if pattern in self.traffic_patterns:
            self.current_pattern = pattern
            self.pattern_timer = 0
            self.pattern_duration = 0
    
    def get_current_intensity(self) -> float:
        
        return self.traffic_patterns[self.current_pattern]