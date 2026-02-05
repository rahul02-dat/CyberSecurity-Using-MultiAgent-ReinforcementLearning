import torch
from typing import Any


class ResponseReward:
    
    def __init__(
        self,
        max_damage: float = 100.0,
        action_costs: dict[int, float] = None
    ):
        self.max_damage = max_damage
        
        if action_costs is None:
            self.action_costs = {
                0: 0.0,
                1: 1.0,
                2: 10.0,
                3: 5.0,
                4: 15.0,
                5: 50.0
            }
        else:
            self.action_costs = action_costs
        
        self.action_names = {
            0: 'DoNothing',
            1: 'Log',
            2: 'BlockIP',
            3: 'RateLimit',
            4: 'Honeypot',
            5: 'ShutDownSubnet'
        }
        
        self.action_effectiveness = {
            0: 0.0,
            1: 0.1,
            2: 0.7,
            3: 0.5,
            4: 0.6,
            5: 0.95
        }
    
    def compute(
        self,
        action: int,
        attack_class: int,
        attack_confidence: float,
        system_state: dict[str, Any]
    ) -> float:
        
        potential_damage = self._estimate_damage(attack_class, attack_confidence, system_state)
        
        mitigation_cost = self.action_costs.get(action, 0.0)
        
        effectiveness = self._compute_effectiveness(action, attack_class)
        
        damage_prevented = potential_damage * effectiveness
        
        reward = damage_prevented - mitigation_cost
        
        if action == 0 and attack_class > 0:
            reward -= potential_damage * 0.5
        
        if action == 5 and attack_class == 0:
            reward -= 100.0
        
        return reward
    
    def _estimate_damage(
        self,
        attack_class: int,
        confidence: float,
        system_state: dict[str, Any]
    ) -> float:
        
        base_damages = {
            0: 0.0,
            1: 30.0,
            2: 50.0,
            3: 80.0,
            4: 20.0
        }
        
        base_damage = base_damages.get(attack_class, 0.0)
        
        damage = base_damage * confidence
        
        infection_level = system_state.get('infection_level', 0.0)
        damage *= (1.0 + infection_level)
        
        critical_assets = system_state.get('critical_assets_at_risk', False)
        if critical_assets:
            damage *= 1.5
        
        return min(damage, self.max_damage)
    
    def _compute_effectiveness(self, action: int, attack_class: int) -> float:
        
        base_effectiveness = self.action_effectiveness.get(action, 0.0)
        
        if attack_class == 0:
            return 0.0
        
        effectiveness_matrix = {
            1: {1: 0.9, 2: 0.3, 3: 0.2, 4: 0.6},
            2: {1: 0.8, 2: 0.5, 3: 0.6, 4: 0.7},
            3: {1: 0.7, 2: 0.4, 3: 0.3, 4: 0.5},
            4: {1: 0.5, 2: 0.7, 3: 0.8, 4: 0.9},
            5: {1: 0.95, 2: 0.95, 3: 0.95, 4: 0.95}
        }
        
        if action in effectiveness_matrix and attack_class in effectiveness_matrix[action]:
            return effectiveness_matrix[action][attack_class]
        
        return base_effectiveness