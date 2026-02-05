import numpy as np
from typing import Any, Optional


class AttackSimulator:
    
    def __init__(self, feature_dim: int = 50, seed: int = 42):
        self.feature_dim = feature_dim
        self.rng = np.random.RandomState(seed)
        
        self.attack_types = {
            0: 'Normal',
            1: 'DDoS',
            2: 'Injection',
            3: 'Malware',
            4: 'Probe'
        }
        
        self.attack_probabilities = {
            0: 0.7,
            1: 0.1,
            2: 0.08,
            3: 0.07,
            4: 0.05
        }
        
        self.attack_signatures = self._initialize_signatures()
        
        self.current_campaign = None
        self.campaign_duration = 0
        self.campaign_timer = 0
    
    def _initialize_signatures(self) -> dict[int, np.ndarray]:
        
        signatures = {}
        
        signatures[0] = np.zeros(self.feature_dim)
        
        signatures[1] = np.zeros(self.feature_dim)
        signatures[1][:10] = self.rng.uniform(0.8, 1.0, 10)
        
        signatures[2] = np.zeros(self.feature_dim)
        signatures[2][10:20] = self.rng.uniform(0.7, 0.95, 10)
        
        signatures[3] = np.zeros(self.feature_dim)
        signatures[3][20:35] = self.rng.uniform(0.75, 1.0, 15)
        
        signatures[4] = np.zeros(self.feature_dim)
        signatures[4][35:45] = self.rng.uniform(0.6, 0.9, 10)
        
        return signatures
    
    def generate_attack(self, infection_level: float = 0.0) -> tuple[np.ndarray, int]:
        
        if self.current_campaign is not None:
            attack_type = self.current_campaign
            self._update_campaign()
        else:
            attack_probs = list(self.attack_probabilities.values())
            attack_probs[0] *= (1.0 - infection_level)
            attack_probs = np.array(attack_probs)
            attack_probs /= attack_probs.sum()
            
            attack_type = self.rng.choice(list(self.attack_types.keys()), p=attack_probs)
            
            if attack_type != 0 and self.rng.random() < 0.1:
                self._start_campaign(attack_type)
        
        attack_vector = self._generate_attack_vector(attack_type, infection_level)
        
        return attack_vector, attack_type
    
    def _generate_attack_vector(self, attack_type: int, infection_level: float) -> np.ndarray:
        
        base_signature = self.attack_signatures[attack_type].copy()
        
        noise = self.rng.normal(0, 0.1, self.feature_dim)
        
        attack_vector = base_signature + noise
        
        if attack_type != 0:
            intensity = 0.5 + 0.5 * infection_level
            attack_vector *= intensity
        
        attack_vector = np.clip(attack_vector, 0, 1)
        
        return attack_vector
    
    def _start_campaign(self, attack_type: int) -> None:
        
        self.current_campaign = attack_type
        self.campaign_duration = self.rng.randint(20, 100)
        self.campaign_timer = 0
    
    def _update_campaign(self) -> None:
        
        self.campaign_timer += 1
        
        if self.campaign_timer >= self.campaign_duration:
            self.current_campaign = None
            self.campaign_duration = 0
            self.campaign_timer = 0
    
    def get_signature(self, attack_type: int) -> np.ndarray:
        
        return self.attack_signatures.get(attack_type, np.zeros(self.feature_dim))
    
    def is_campaign_active(self) -> bool:
        
        return self.current_campaign is not None
    
    def get_current_campaign(self) -> Optional[int]:
        
        return self.current_campaign