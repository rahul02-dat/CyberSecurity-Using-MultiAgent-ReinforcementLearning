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

from src.envs.marl_env import MARLCyberEnv
from src.agents.detection_agent import DetectionRLModel
from src.agents.response_agent import ResponseRLModel
from src.agents.attacker_agent import AttackerRLModel


class AdversarialMultiAgentEnv(MultiAgentEnv):
    
    def __init__(self, env_config):
        super().__init__()
        
        config_path = env_config.get("config_path", "configs/env.yml")
        adv_config_path = env_config.get("adv_config_path", "configs/adversarial_training.yaml")
        
        self.env = MARLCyberEnv(config_path=config_path)
        
        with open(adv_config_path, 'r') as f:
            self.adv_config = yaml.safe_load(f)
        
        self.possible_agents = ["detection", "response", "attacker"]
        self.agents = ["detection", "response", "attacker"]
        
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True
        
        self.observation_spaces = {
            "detection": spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32),
            "response": spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32),
            "attacker": spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        }
        
        self.action_spaces = {
            "detection": spaces.MultiDiscrete([2, 10]),
            "response": spaces.MultiDiscrete([3, 3]),
            "attacker": spaces.Discrete(6)
        }
        
        self.last_detection_action = {"flag": False, "confidence": 0.0}
        self.time_since_last_block = 0
        self.attacker_evasion_steps = 0
        self.current_attacker_action = "none"
        self.attack_active = False
        
        self.attack_map = ["none", "scan", "bruteforce", "ddos", "stealth_scan", "ip_rotate"]
        
        self.attacker_reward_config = self.adv_config['attacker_rewards']
        self.evasion_threshold = self.attacker_reward_config['evasion_threshold']
        
    def observation_space(self, agent_id):
        return self.observation_spaces[agent_id]
    
    def action_space(self, agent_id):
        return self.action_spaces[agent_id]
    
    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        
        self.last_detection_action = {"flag": False, "confidence": 0.0}
        self.time_since_last_block = 0
        self.attacker_evasion_steps = 0
        self.current_attacker_action = "none"
        self.attack_active = False
        
        base_state = obs.get("detection", obs.get("attacker", np.array([0, 0, 0, 0]))).astype(np.float32)
        
        attacker_obs = np.array([
            base_state[0],
            base_state[2],
            base_state[3],
            float(self.time_since_last_block)
        ], dtype=np.float32)
        
        rllib_obs = {
            "detection": base_state,
            "response": np.concatenate([
                base_state,
                [0.0, 0.0]
            ]).astype(np.float32),
            "attacker": attacker_obs
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
        
        action_map = {0: "ALLOW", 1: "BLOCK", 2: "QUARANTINE"}
        response_action = {
            "action": action_map[action_idx],
            "duration_bin": duration_idx
        }
        
        if response_action["action"] in ["BLOCK", "QUARANTINE"]:
            self.time_since_last_block = 0
        else:
            self.time_since_last_block += 1
        
        actions = {
            "detection": detection_action,
            "response": response_action,
            "attacker": {"attack": attacker_action}
        }
        
        obs, rewards, dones, infos = self.env.step(actions)
        
        base_state = obs.get("detection", obs.get("attacker", np.array([0, 0, 0, 0]))).astype(np.float32)
        
        attacker_obs = np.array([
            base_state[0],
            base_state[2],
            base_state[3],
            float(self.time_since_last_block)
        ], dtype=np.float32)
        
        rllib_obs = {
            "detection": base_state,
            "response": np.concatenate([
                base_state,
                [float(detection_action["flag"]), detection_action["confidence"]]
            ]).astype(np.float32),
            "attacker": attacker_obs
        }
        
        attacker_reward = self._compute_attacker_reward(
            detection_action, response_action
        )
        
        rllib_rewards = {
            "detection": rewards["detection"],
            "response": rewards["response"],
            "attacker": attacker_reward
        }
        
        rllib_dones = {
            "detection": dones["detection"],
            "response": dones["response"],
            "attacker": dones["attacker"],
            "__all__": dones["__all__"]
        }
        
        rllib_infos = {
            "detection": infos["detection"],
            "response": infos["response"],
            "attacker": {
                "attack_type": self.current_attacker_action,
                "evasion_steps": self.attacker_evasion_steps,
                "blocked": response_action["action"] in ["BLOCK", "QUARANTINE"]
            }
        }
        
        rllib_truncated = {
            "detection": False,
            "response": False,
            "attacker": False,
            "__all__": False
        }
        
        return rllib_obs, rllib_rewards, rllib_dones, rllib_truncated, rllib_infos
    
    def _compute_attacker_reward(self, detection_action, response_action):
        reward = self.attacker_reward_config['timestep_penalty']
        
        if not self.attack_active:
            return 0.0
        
        is_blocked = response_action["action"] in ["BLOCK", "QUARANTINE"]
        is_detected = detection_action["flag"]
        
        if is_blocked:
            reward += self.attacker_reward_config['blocked_penalty']
            self.attacker_evasion_steps = 0
        else:
            if not is_detected:
                self.attacker_evasion_steps += 1
            else:
                self.attacker_evasion_steps = 0
        
        if not is_blocked and self.attack_active:
            reward += self.attacker_reward_config['attack_success']
        
        if self.attacker_evasion_steps >= self.evasion_threshold:
            reward += self.attacker_reward_config['evade_detection_bonus']
        
        return reward


def train_adversarial_marl(pretrained_checkpoint: str = None):
    with open("configs/adversarial_training.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    ray.init(ignore_reinit_error=True)
    
    ModelCatalog.register_custom_model("detection_model", DetectionRLModel)
    ModelCatalog.register_custom_model("response_model", ResponseRLModel)
    ModelCatalog.register_custom_model("attacker_model", AttackerRLModel)
    
    results_dir = os.path.abspath(config['logging']['results_dir'])
    os.makedirs(results_dir, exist_ok=True)
    
    def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        return f"{agent_id}_policy"
    
    attacker_config = config['adversarial_training']['attacker']
    defender_config = config['adversarial_training']['defenders']
    
    base_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=AdversarialMultiAgentEnv,
            env_config={
                "config_path": "configs/env.yml",
                "adv_config_path": "configs/adversarial_training.yaml"
            }
        )
        .framework("torch")
        .resources(
            num_cpus_per_worker=1,
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
            },
            policy_mapping_fn=policy_mapping_fn,
        )
        .debugging(log_level="WARN")
    )
    
    num_rounds = config['adversarial_training']['num_rounds']
    attacker_episodes = config['adversarial_training']['attacker_episodes_per_round']
    defender_episodes = config['adversarial_training']['defender_episodes_per_round']
    
    metrics_log = []
    
    print("="*80)
    print("Starting Adversarial Co-Evolution Training")
    print("="*80)
    
    trainer = None
    
    for round_idx in range(num_rounds):
        print(f"\n{'='*80}")
        print(f"Round {round_idx + 1}/{num_rounds}")
        print(f"{'='*80}")
        
        print(f"\n[Round {round_idx + 1}] Phase 1: Training Attacker (Defenders Frozen)")
        
        attacker_train_config = base_config.copy()
        attacker_train_config.training(
            train_batch_size=attacker_config['train_batch_size'],
            minibatch_size=attacker_config['sgd_minibatch_size'],
            num_epochs=attacker_config['num_sgd_iter'],
            lr=float(attacker_config['learning_rate']),
            gamma=attacker_config['gamma'],
            lambda_=attacker_config['lambda_'],
            clip_param=attacker_config['clip_param'],
            vf_clip_param=10.0,
            entropy_coeff=attacker_config['entropy_coeff'],
        )
        attacker_train_config.multi_agent(
            policies_to_train=["attacker_policy"]
        )
        
        if trainer is None:
            trainer = attacker_train_config.build()
            
            if pretrained_checkpoint:
                print(f"\n  Loading pre-trained defenders from: {pretrained_checkpoint}")
                try:
                    state = trainer.__getstate__()
                    
                    import pickle
                    checkpoint_file = os.path.join(pretrained_checkpoint, "algorithm_state.pkl")
                    if os.path.exists(checkpoint_file):
                        with open(checkpoint_file, 'rb') as f:
                            saved_state = pickle.load(f)
                        
                        for policy_id in ["detection_policy", "response_policy"]:
                            if policy_id in saved_state.get("worker", {}).get("state", {}):
                                trainer.get_policy(policy_id).set_state(
                                    saved_state["worker"]["state"][policy_id]
                                )
                        print("  ✓ Defenders loaded successfully")
                    else:
                        print("  ⚠ Checkpoint file not found, starting from scratch")
                except Exception as e:
                    print(f"  ⚠ Could not load checkpoint: {e}")
                    print("  Starting adversarial training from scratch...")
        else:
            for policy_id in ["detection_policy", "response_policy"]:
                trainer.get_policy(policy_id).config["lr"] = 0.0
            trainer.get_policy("attacker_policy").config["lr"] = float(attacker_config['learning_rate'])
        
        for _ in range(attacker_episodes):
            result = trainer.train()
        
        attacker_reward = result['env_runners']['policy_reward_mean'].get('attacker_policy', 0)
        print(f"  Attacker reward: {attacker_reward:.2f}")
        
        print(f"\n[Round {round_idx + 1}] Phase 2: Training Defenders (Attacker Frozen)")
        
        defender_train_config = base_config.copy()
        defender_train_config.training(
            train_batch_size=defender_config['train_batch_size'],
            minibatch_size=defender_config['sgd_minibatch_size'],
            num_epochs=defender_config['num_sgd_iter'],
            lr=float(defender_config['learning_rate']),
            gamma=defender_config['gamma'],
            lambda_=defender_config['lambda_'],
            clip_param=defender_config['clip_param'],
            vf_clip_param=10.0,
            entropy_coeff=defender_config['entropy_coeff'],
        )
        defender_train_config.multi_agent(
            policies_to_train=["detection_policy", "response_policy"]
        )
        
        trainer.get_policy("attacker_policy").config["lr"] = 0.0
        for policy_id in ["detection_policy", "response_policy"]:
            trainer.get_policy(policy_id).config["lr"] = float(defender_config['learning_rate'])
        
        for _ in range(defender_episodes):
            result = trainer.train()
        
        detection_reward = result['env_runners']['policy_reward_mean'].get('detection_policy', 0)
        response_reward = result['env_runners']['policy_reward_mean'].get('response_policy', 0)
        
        print(f"  Detection reward: {detection_reward:.2f}")
        print(f"  Response reward: {response_reward:.2f}")
        
        round_metrics = {
            "round": round_idx + 1,
            "attacker_reward": attacker_reward,
            "detection_reward": detection_reward,
            "response_reward": response_reward
        }
        metrics_log.append(round_metrics)
        
        if (round_idx + 1) % config['logging']['checkpoint_freq'] == 0:
            checkpoint_path = trainer.save(results_dir)
            print(f"\n  Checkpoint saved: {checkpoint_path}")
    
    final_checkpoint = trainer.save(results_dir)
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Final checkpoint: {final_checkpoint}")
    
    metrics_file = os.path.join(results_dir, "adversarial_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Metrics saved: {metrics_file}")
    
    ray.shutdown()
    
    return trainer, metrics_log


if __name__ == "__main__":
    import glob
    
    checkpoint_pattern = os.path.abspath("results/phase2_training/marl_cybersecurity_phase2/*/checkpoint_*")
    checkpoints = sorted(glob.glob(checkpoint_pattern))
    
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f"\nFound Phase 2 checkpoint: {latest_checkpoint}")
        print("Loading pre-trained defenders for adversarial training...\n")
        train_adversarial_marl(pretrained_checkpoint=latest_checkpoint)
    else:
        print("\n" + "="*80)
        print("WARNING: No Phase 2 checkpoint found!")
        print("="*80)
        print("Please run Phase 2 training first:")
        print("  python src/rl/trainer_rllib.py")
        print("\nOr specify checkpoint path manually:")
        print("  train_adversarial_marl(pretrained_checkpoint='path/to/checkpoint')")
        print("="*80)