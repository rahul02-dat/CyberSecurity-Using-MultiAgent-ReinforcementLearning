import numpy as np
from typing import Dict, Tuple, Any, List
import yaml


class MARLCyberEnv:
    
    def __init__(self, config_path: str = "/Users/rahulmac/Documents/Projects/projects/CyberSecurity/configs/env.yml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.agents = self.config['environment']['agents']
        self.episode_length = self.config['environment']['episode_length']
        self.state_dim = self.config['environment']['state_dim']
        
        self.rewards_config = self.config['rewards']
        self.state_bounds = self.config['state_bounds']
        self.attack_params = self.config['attack_params']
        self.baseline = self.config['baseline_state']
        self.noise_std = self.config['noise']['std']
        
        self.step_count = 0
        self.current_state = None
        self.current_attack = "none"
        self.attack_active = False
        
    def reset(self) -> Dict[str, np.ndarray]:
        self.step_count = 0
        self.current_attack = "none"
        self.attack_active = False
        
        self.current_state = np.array([
            self.baseline['flows_per_sec'],
            self.baseline['avg_pkt_size'],
            self.baseline['failed_logins'],
            self.baseline['entropy']
        ], dtype=np.float32)
        
        obs = {agent: self.current_state.copy() for agent in self.agents}
        return obs
    
    def step(self, actions: Dict[str, Dict[str, Any]]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict]
    ]:
        self.step_count += 1
        
        attacker_action = actions.get("attacker", {}).get("attack", "none")
        detection_action = actions.get("detection", {})
        response_action = actions.get("response", {})
        
        self.current_attack = attacker_action
        self.attack_active = attacker_action != "none"
        
        self._update_state(attacker_action, response_action)
        
        rewards = self._compute_rewards(detection_action, response_action)
        
        done = self.step_count >= self.episode_length
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        
        obs = {agent: self.current_state.copy() for agent in self.agents}
        
        infos = {
            agent: {
                "step": self.step_count,
                "attack_type": self.current_attack,
                "attack_active": self.attack_active
            }
            for agent in self.agents
        }
        
        return obs, rewards, dones, infos
    
    def _update_state(self, attack_type: str, response_action: Dict[str, Any]):
        state = self.current_state.copy()
        
        if attack_type != "none" and attack_type in self.attack_params:
            params = self.attack_params[attack_type]
            
            if 'flows_multiplier' in params:
                state[0] *= params['flows_multiplier']
            if 'avg_pkt_decrease' in params:
                state[1] *= params['avg_pkt_decrease']
            if 'failed_logins_increase' in params:
                state[2] += params['failed_logins_increase']
            if 'entropy_increase' in params:
                state[3] += params['entropy_increase']
            if 'entropy_decrease' in params:
                state[3] *= params['entropy_decrease']
        
        response_act = response_action.get("action", "ALLOW")
        if response_act == "BLOCK" and self.attack_active:
            state[0] = self.baseline['flows_per_sec']
            state[2] = self.baseline['failed_logins']
        
        noise = np.random.normal(0, self.noise_std, size=self.state_dim)
        state = state * (1 + noise)
        
        state[0] = np.clip(state[0], self.state_bounds['flows_per_sec'][0], 
                          self.state_bounds['flows_per_sec'][1])
        state[1] = np.clip(state[1], self.state_bounds['avg_pkt_size'][0],
                          self.state_bounds['avg_pkt_size'][1])
        state[2] = np.clip(state[2], self.state_bounds['failed_logins'][0],
                          self.state_bounds['failed_logins'][1])
        state[3] = np.clip(state[3], self.state_bounds['entropy'][0],
                          self.state_bounds['entropy'][1])
        
        self.current_state = state
    
    def _compute_rewards(
        self, 
        detection_action: Dict[str, Any],
        response_action: Dict[str, Any]
    ) -> Dict[str, float]:
        rewards = {}
        
        detection_flag = detection_action.get("flag", False)
        
        if self.attack_active and detection_flag:
            rewards["detection"] = self.rewards_config['detection']['correct']
        elif not self.attack_active and detection_flag:
            rewards["detection"] = self.rewards_config['detection']['false_positive']
        elif self.attack_active and not detection_flag:
            rewards["detection"] = self.rewards_config['detection']['missed_attack']
        else:
            rewards["detection"] = 0.0
        
        response_act = response_action.get("action", "ALLOW")
        
        if self.attack_active and response_act == "BLOCK":
            rewards["response"] = self.rewards_config['response']['correct_block']
        elif not self.attack_active and response_act == "BLOCK":
            rewards["response"] = self.rewards_config['response']['wrong_block']
        elif self.attack_active and response_act == "ALLOW":
            rewards["response"] = self.rewards_config['response']['allow_attack']
        else:
            rewards["response"] = 0.0
        
        rewards["attacker"] = self.rewards_config['attacker']['default']
        
        return rewards