import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
from typing import Dict, Any, Tuple
import json
from collections import deque

from src.envs.marl_env import MARLCyberEnv
from src.agents.detection_agent import DetectionRLModel
from src.agents.response_agent import ResponseRLModel
from src.agents.attacker_agent import AttackerRLModel
from src.agents.safety_meta_agent import SafetyMetaRLModel


class SafetyMetaMultiAgentEnv(MultiAgentEnv):
    
    def __init__(self, env_config):
        super().__init__()
        
        config_path = env_config.get("config_path", "configs/env.yml")
        safety_config_path = env_config.get("safety_config_path", "configs/safety_policy.yaml")
        
        self.env = MARLCyberEnv(config_path=config_path)
        
        with open(safety_config_path, 'r') as f:
            self.safety_config = yaml.safe_load(f)
        
        self.possible_agents = ["detection", "response", "attacker", "safety"]
        self.agents = ["detection", "response", "attacker", "safety"]
        
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True
        
        self.observation_spaces = {
            "detection": spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32),
            "response": spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32),
            "attacker": spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32),
            "safety": spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        }
        
        self.action_spaces = {
            "detection": spaces.MultiDiscrete([2, 10]),
            "response": spaces.MultiDiscrete([3, 3]),
            "attacker": spaces.Discrete(6),
            "safety": spaces.Discrete(4)
        }
        
        self.attack_map = ["none", "scan", "bruteforce", "ddos", "stealth_scan", "ip_rotate"]
        
        self.fp_window = self.safety_config['false_positive_window']
        self.fn_window = self.safety_config['false_negative_window']
        
        self.recent_fp = deque(maxlen=self.fp_window)
        self.recent_fn = deque(maxlen=self.fn_window)
        
        self.time_since_last_block = 0
        self.last_detection_action = {"flag": False, "confidence": 0.0}
        self.last_response_action = {"action": "ALLOW", "duration_bin": 0}
        self.last_safety_action = 0
        self.current_attacker_action = "none"
        self.attack_active = False
        self.attacker_evasion_steps = 0
        
        self.safety_reward_config = self.safety_config['safety_rewards']
        
    def observation_space(self, agent_id):
        return self.observation_spaces[agent_id]
    
    def action_space(self, agent_id):
        return self.action_spaces[agent_id]
    
    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        
        self.recent_fp.clear()
        self.recent_fn.clear()
        self.time_since_last_block = 0
        self.last_detection_action = {"flag": False, "confidence": 0.0}
        self.last_response_action = {"action": "ALLOW", "duration_bin": 0}
        self.last_safety_action = 0
        self.current_attacker_action = "none"
        self.attack_active = False
        self.attacker_evasion_steps = 0
        
        base_state = obs.get("detection", obs.get("attacker", np.array([0, 0, 0, 0]))).astype(np.float32)
        
        attacker_obs = np.array([
            base_state[0],
            base_state[2],
            base_state[3],
            float(self.time_since_last_block)
        ], dtype=np.float32)
        
        safety_obs = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ], dtype=np.float32)
        
        rllib_obs = {
            "detection": base_state,
            "response": np.concatenate([base_state, [0.0, 0.0]]).astype(np.float32),
            "attacker": attacker_obs,
            "safety": safety_obs
        }
        
        return rllib_obs, {}
    
    def step(self, action_dict: Dict[str, np.ndarray]):
        attacker_action_idx = int(action_dict.get("attacker", 0))
        attacker_action = self.attack_map[attacker_action_idx]
        self.current_attacker_action = attacker_action
        self.attack_active = attacker_action != "none"
        
        detection_action_raw = action_dict.get("detection", np.array([0, 0]))
        flag_idx = int(detection_action_raw[0])
        confidence_idx = int(detection_action_raw[1])
        
        detection_action = {
            "flag": bool(flag_idx),
            "confidence": confidence_idx / 10.0
        }
        
        self.last_detection_action = detection_action
        
        response_action_raw = action_dict.get("response", np.array([0, 0]))
        action_idx = int(response_action_raw[0])
        duration_idx = int(response_action_raw[1])
        
        action_map_response = {0: "ALLOW", 1: "BLOCK", 2: "QUARANTINE"}
        response_action = {
            "action": action_map_response[action_idx],
            "duration_bin": duration_idx
        }
        
        self.last_response_action = response_action
        
        safety_action_idx = int(action_dict.get("safety", 0))
        self.last_safety_action = safety_action_idx
        
        safety_action_map = {0: "ALLOW", 1: "BLOCK", 2: "QUARANTINE", 3: "ROLLBACK"}
        final_safety_decision = safety_action_map[safety_action_idx]
        
        if final_safety_decision == "ROLLBACK":
            final_action = "ALLOW"
        else:
            final_action = final_safety_decision
        
        final_response_action = {
            "action": final_action,
            "duration_bin": response_action["duration_bin"]
        }
        
        if final_response_action["action"] in ["BLOCK", "QUARANTINE"]:
            self.time_since_last_block = 0
        else:
            self.time_since_last_block += 1
        
        actions = {
            "detection": detection_action,
            "response": final_response_action,
            "attacker": {"attack": attacker_action}
        }
        
        obs, rewards, dones, infos = self.env.step(actions)
        
        if self.attack_active and not detection_action["flag"]:
            self.recent_fn.append(1)
        else:
            self.recent_fn.append(0)
            
        if not self.attack_active and detection_action["flag"]:
            self.recent_fp.append(1)
        else:
            self.recent_fp.append(0)
        
        fp_rate = sum(self.recent_fp) / len(self.recent_fp) if len(self.recent_fp) > 0 else 0.0
        fn_rate = sum(self.recent_fn) / len(self.recent_fn) if len(self.recent_fn) > 0 else 0.0
        
        base_state = obs.get("detection", obs.get("attacker", np.array([0, 0, 0, 0]))).astype(np.float32)
        
        attacker_obs = np.array([
            base_state[0],
            base_state[2],
            base_state[3],
            float(self.time_since_last_block)
        ], dtype=np.float32)
        
        safety_obs = np.array([
            float(detection_action["flag"]),
            detection_action["confidence"],
            float(action_idx) / 2.0,
            fp_rate,
            fn_rate,
            float(self.time_since_last_block) / 100.0
        ], dtype=np.float32)
        
        rllib_obs = {
            "detection": base_state,
            "response": np.concatenate([
                base_state,
                [float(detection_action["flag"]), detection_action["confidence"]]
            ]).astype(np.float32),
            "attacker": attacker_obs,
            "safety": safety_obs
        }
        
        attacker_reward = self._compute_attacker_reward(detection_action, final_response_action)
        
        safety_reward = self._compute_safety_reward(
            detection_action, response_action, final_safety_decision
        )
        
        rllib_rewards = {
            "detection": rewards["detection"],
            "response": rewards["response"],
            "attacker": attacker_reward,
            "safety": safety_reward
        }
        
        rllib_dones = {
            "detection": dones["detection"],
            "response": dones["response"],
            "attacker": dones["attacker"],
            "safety": dones["detection"],
            "__all__": dones["__all__"]
        }
        
        rllib_infos = {
            "detection": infos["detection"],
            "response": infos["response"],
            "attacker": {
                "attack_type": self.current_attacker_action,
                "evasion_steps": self.attacker_evasion_steps,
                "blocked": final_response_action["action"] in ["BLOCK", "QUARANTINE"]
            },
            "safety": {
                "final_decision": final_safety_decision,
                "original_response": response_action["action"],
                "fp_rate": fp_rate,
                "fn_rate": fn_rate
            }
        }
        
        rllib_truncated = {
            "detection": False,
            "response": False,
            "attacker": False,
            "safety": False,
            "__all__": False
        }
        
        return rllib_obs, rllib_rewards, rllib_dones, rllib_truncated, rllib_infos
    
    def _compute_attacker_reward(self, detection_action, response_action):
        reward = -1.0
        
        if not self.attack_active:
            return 0.0
        
        is_blocked = response_action["action"] in ["BLOCK", "QUARANTINE"]
        is_detected = detection_action["flag"]
        
        if is_blocked:
            reward += -5.0
            self.attacker_evasion_steps = 0
        else:
            if not is_detected:
                self.attacker_evasion_steps += 1
            else:
                self.attacker_evasion_steps = 0
        
        if not is_blocked and self.attack_active:
            reward += 5.0
        
        if self.attacker_evasion_steps >= 5:
            reward += 2.0
        
        return reward
    
    def _compute_safety_reward(self, detection_action, response_action, safety_decision):
        reward = 0.0
        
        response_proposed_block = response_action["action"] in ["BLOCK", "QUARANTINE"]
        safety_allowed_block = safety_decision in ["BLOCK", "QUARANTINE"]
        safety_rollback = safety_decision == "ROLLBACK"
        
        if self.attack_active:
            if safety_allowed_block:
                reward += self.safety_reward_config['correct_block']
            elif safety_rollback:
                reward += self.safety_reward_config['late_rollback']
            else:
                reward += self.safety_reward_config['indecision_penalty']
        else:
            if not safety_allowed_block:
                reward += self.safety_reward_config['correct_allow']
            else:
                reward += self.safety_reward_config['false_positive_block']
        
        return reward


def train_safety_meta_policy(pretrained_checkpoint: str = None):
    with open("configs/safety_policy.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    ray.init(ignore_reinit_error=True)
    
    ModelCatalog.register_custom_model("detection_model", DetectionRLModel)
    ModelCatalog.register_custom_model("response_model", ResponseRLModel)
    ModelCatalog.register_custom_model("attacker_model", AttackerRLModel)
    ModelCatalog.register_custom_model("safety_model", SafetyMetaRLModel)
    
    results_dir = os.path.abspath(config['logging']['results_dir'])
    os.makedirs(results_dir, exist_ok=True)
    
    def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        return f"{agent_id}_policy"
    
    safety_cfg = config['safety_policy']
    
    base_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=SafetyMetaMultiAgentEnv,
            env_config={
                "config_path": "configs/env.yml",
                "safety_config_path": "configs/safety_policy.yaml"
            }
        )
        .framework("torch")
        .resources(
            num_cpus_per_worker=1,
        )
        .training(
            train_batch_size=safety_cfg['train_batch_size'],
            minibatch_size=safety_cfg['sgd_minibatch_size'],
            num_epochs=safety_cfg['num_sgd_iter'],
            lr=float(safety_cfg['learning_rate']),
            gamma=safety_cfg['gamma'],
            lambda_=safety_cfg['lambda_'],
            clip_param=safety_cfg['clip_param'],
            vf_clip_param=10.0,
            entropy_coeff=safety_cfg['entropy_coeff'],
        )
        .multi_agent(
            policies={
                "detection_policy": (
                    None,
                    spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32),
                    spaces.MultiDiscrete([2, 10]),
                    {
                        "model": {
                            "custom_model": "detection_model",
                            "fcnet_hiddens": [64, 64],
                        }
                    }
                ),
                "response_policy": (
                    None,
                    spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32),
                    spaces.MultiDiscrete([3, 3]),
                    {
                        "model": {
                            "custom_model": "response_model",
                            "fcnet_hiddens": [64, 64],
                        }
                    }
                ),
                "attacker_policy": (
                    None,
                    spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32),
                    spaces.Discrete(6),
                    {
                        "model": {
                            "custom_model": "attacker_model",
                            "fcnet_hiddens": [64, 64],
                        }
                    }
                ),
                "safety_policy": (
                    None,
                    spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
                    spaces.Discrete(4),
                    {
                        "model": {
                            "custom_model": "safety_model",
                            "fcnet_hiddens": [32, 32],
                        }
                    }
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["safety_policy"],
        )
        .debugging(log_level="WARN")
    )
    
    num_iterations = safety_cfg['training']['num_iterations']
    checkpoint_freq = safety_cfg['training']['checkpoint_freq']
    
    print("="*80)
    print("Starting Safety Meta-Policy Training")
    print("="*80)
    print("Detection, Response, and Attacker agents are FROZEN")
    print("Only Safety Meta-Policy will be trained")
    print("="*80)
    
    trainer = base_config.build_algo()
    
    if pretrained_checkpoint:
        print(f"\nLoading pre-trained agents from: {pretrained_checkpoint}")
        try:
            import pickle
            checkpoint_file = os.path.join(pretrained_checkpoint, "algorithm_state.pkl")
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'rb') as f:
                    saved_state = pickle.load(f)
                
                for policy_id in ["detection_policy", "response_policy", "attacker_policy"]:
                    if policy_id in saved_state.get("worker", {}).get("state", {}):
                        trainer.get_policy(policy_id).set_state(
                            saved_state["worker"]["state"][policy_id]
                        )
                print("✓ Pre-trained agents loaded successfully")
                print("✓ Agents frozen for safety meta-policy training")
            else:
                print("⚠ Checkpoint file not found")
        except Exception as e:
            print(f"⚠ Could not load checkpoint: {e}")
    
    for policy_id in ["detection_policy", "response_policy", "attacker_policy"]:
        trainer.get_policy(policy_id).config["lr"] = 0.0
    
    metrics_log = []
    
    for iteration in range(1, num_iterations + 1):
        result = trainer.train()
        
        safety_reward = result['env_runners']['policy_reward_mean'].get('safety_policy', 0)
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}/{num_iterations} | Safety reward: {safety_reward:.2f}")
        
        metrics_log.append({
            "iteration": iteration,
            "safety_reward": safety_reward
        })
        
        if iteration % checkpoint_freq == 0:
            checkpoint_path = trainer.save(results_dir)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    final_checkpoint = trainer.save(results_dir)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Final checkpoint: {final_checkpoint}")
    
    metrics_file = os.path.join(results_dir, "safety_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Metrics saved: {metrics_file}")
    
    ray.shutdown()
    
    return trainer, metrics_log


if __name__ == "__main__":
    import glob
    
    checkpoint_pattern = os.path.abspath("results/phase3_adversarial/*/checkpoint_*")
    checkpoints = sorted(glob.glob(checkpoint_pattern))
    
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f"\nFound Phase 3 checkpoint: {latest_checkpoint}")
        print("Loading pre-trained agents for safety training...\n")
        train_safety_meta_policy(pretrained_checkpoint=latest_checkpoint)
    else:
        print("\n" + "="*80)
        print("WARNING: No Phase 3 checkpoint found!")
        print("="*80)
        print("Please run Phase 3 training first or specify checkpoint manually")
        print("="*80)
        print("\nStarting from scratch...")
        train_safety_meta_policy()