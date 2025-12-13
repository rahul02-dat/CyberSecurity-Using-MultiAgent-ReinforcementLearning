import json
from typing import Dict, Any, List
from src.envs.marl_env import MARLCyberEnv


class SyntheticScenarioEnv(MARLCyberEnv):
    
    def __init__(self, config_path: str = "/Users/rahulmac/Documents/Projects/projects/CyberSecurity/configs/env.yml", scenario_path: str = None):
        super().__init__(config_path)
        
        self.scenario = None
        self.attack_schedule = []
        
        if scenario_path:
            self.load_scenario(scenario_path)
    
    def load_scenario(self, scenario_path: str):
        with open(scenario_path, 'r') as f:
            self.scenario = json.load(f)
        
        self.episode_length = self.scenario.get('episode_length', self.episode_length)
        self.attack_schedule = self._build_attack_schedule()
    
    def _build_attack_schedule(self) -> List[str]:
        schedule = ["none"] * self.episode_length
        
        for seq in self.scenario.get('attack_sequence', []):
            start, end = seq['step_range']
            attack_type = seq['attack_type']
            
            for step in range(start, min(end, self.episode_length)):
                schedule[step] = attack_type
        
        return schedule
    
    def step(self, actions: Dict[str, Dict[str, Any]]):
        if self.attack_schedule and self.step_count < len(self.attack_schedule):
            scheduled_attack = self.attack_schedule[self.step_count]
            
            if "attacker" not in actions:
                actions["attacker"] = {}
            actions["attacker"]["attack"] = scheduled_attack
        
        return super().step(actions)


class SimplePatternEnv(MARLCyberEnv):
    
    def __init__(self, config_path: str = "/Users/rahulmac/Documents/Projects/projects/CyberSecurity/configs/env.yml", pattern: str = "alternating"):
        super().__init__(config_path)
        self.pattern = pattern
        self.attack_types = ["scan", "ddos", "bruteforce"]
        self.attack_index = 0
    
    def step(self, actions: Dict[str, Dict[str, Any]]):
        if "attacker" not in actions:
            actions["attacker"] = {}
        
        if self.pattern == "alternating":
            if self.step_count % 20 < 10:
                actions["attacker"]["attack"] = "none"
            else:
                attack_idx = (self.step_count // 20) % len(self.attack_types)
                actions["attacker"]["attack"] = self.attack_types[attack_idx]
        
        elif self.pattern == "random":
            import random
            if random.random() < 0.3:
                actions["attacker"]["attack"] = random.choice(self.attack_types)
            else:
                actions["attacker"]["attack"] = "none"
        
        elif self.pattern == "burst":
            if 30 <= self.step_count < 50:
                actions["attacker"]["attack"] = "ddos"
            else:
                actions["attacker"]["attack"] = "none"
        
        else:
            actions["attacker"]["attack"] = "none"
        
        return super().step(actions)