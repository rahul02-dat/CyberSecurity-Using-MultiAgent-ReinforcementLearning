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
from typing import Dict, Any, Tuple

from src.envs.synthetic_env import SyntheticScenarioEnv
from src.agents.detection_agent import DetectionRLModel
from src.agents.response_agent import ResponseRLModel
from src.rl.callbacks import MARLMetricsCallback


class RLlibMultiAgentEnv(MultiAgentEnv):
    
    def __init__(self, env_config):
        super().__init__()
        
        scenario_path = env_config.get("scenario_path", "data/synthetic/example_scenario.json")
        config_path = env_config.get("config_path", "configs/env.yml")
        
        self.env = SyntheticScenarioEnv(
            config_path=config_path,
            scenario_path=scenario_path
        )
        
        # Only use detection and response agents for RLlib training
        self._agent_ids = {"detection", "response"}
        self.possible_agents = ["detection", "response"]
        self.agents = ["detection", "response"]
        
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True
        
        self.observation_spaces = {
            "detection": spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32),
            "response": spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)
        }
        
        self.action_spaces = {
            "detection": spaces.MultiDiscrete([2, 10]),
            "response": spaces.MultiDiscrete([3, 3])
        }
        
        self.last_detection_action = {"flag": False, "confidence": 0.0}
        
    def observation_space(self, agent_id):
        return self.observation_spaces[agent_id]
    
    def action_space(self, agent_id):
        return self.action_spaces[agent_id]
        
    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        
        # obs is a dict where each agent key maps to the same state
        # Extract state from one of the agents and reshape for each agent's expected input
        base_state = obs.get("detection", obs.get("attacker", np.array([0, 0, 0, 0]))).astype(np.float32)
        
        rllib_obs = {
            "detection": base_state,
            "response": np.concatenate([
                base_state,
                [0.0, 0.0]
            ]).astype(np.float32)
        }
        
        self.last_detection_action = {"flag": False, "confidence": 0.0}
        
        return rllib_obs, {}
    
    def step(self, action_dict: Dict[str, np.ndarray]):
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
        
        actions = {
            "detection": detection_action,
            "response": response_action
        }
        
        obs, rewards, dones, infos = self.env.step(actions)
        
        # obs is a dict where each agent key maps to the same state
        base_state = obs.get("detection", obs.get("attacker", np.array([0, 0, 0, 0]))).astype(np.float32)
        
        rllib_obs = {
            "detection": base_state,
            "response": np.concatenate([
                base_state,
                [float(detection_action["flag"]), detection_action["confidence"]]
            ]).astype(np.float32)
        }
        
        rllib_rewards = {
            "detection": rewards.get("detection", 0.0),
            "response": rewards.get("response", 0.0)
        }
        
        rllib_dones = {
            "detection": dones.get("detection", False),
            "response": dones.get("response", False),
            "__all__": dones.get("__all__", False)
        }
        
        rllib_infos = {
            "detection": infos.get("detection", {}),
            "response": infos.get("response", {})
        }
        
        rllib_truncated = {
            "detection": False,
            "response": False,
            "__all__": False
        }
        
        return rllib_obs, rllib_rewards, rllib_dones, rllib_truncated, rllib_infos


def train_marl_agents(
    num_iterations: int = 100,
    checkpoint_freq: int = 10,
    results_dir: str = None
):
    if results_dir is None:
        results_dir = os.path.abspath("results/phase2_training")
    
    ray.init(ignore_reinit_error=True)
    
    ModelCatalog.register_custom_model("detection_model", DetectionRLModel)
    ModelCatalog.register_custom_model("response_model", ResponseRLModel)
    
    def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        return f"{agent_id}_policy"
    
    config = (
        PPOConfig()
        .environment(
            env=RLlibMultiAgentEnv,
            env_config={
                "config_path": "configs/env.yml",
                "scenario_path": "data/synthetic/example_scenario.json"
            }
        )
        .framework("torch")
        .env_runners(
            num_env_runners=2,
            num_envs_per_env_runner=1,
        )
        .training(
            train_batch_size=2000,
            minibatch_size=128,
            num_epochs=10,
            lr=5e-5,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
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
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["detection_policy", "response_policy"],
        )
        .callbacks(MARLMetricsCallback)
        .debugging(log_level="WARN")
    )
    
    os.makedirs(results_dir, exist_ok=True)
    
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            stop={"training_iteration": num_iterations},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=checkpoint_freq,
                checkpoint_at_end=True,
            ),
            storage_path=results_dir,
            name="marl_cybersecurity_phase2",
        ),
    )
    
    results = tuner.fit()
    
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    
    if best_result and best_result.checkpoint:
        print(f"Best checkpoint: {best_result.checkpoint}")
        if "episode_reward_mean" in best_result.metrics:
            print(f"Best reward: {best_result.metrics['episode_reward_mean']:.2f}")
    else:
        print("No successful trials completed")
    
    ray.shutdown()
    
    return best_result


if __name__ == "__main__":
    train_marl_agents(num_iterations=100, checkpoint_freq=10)