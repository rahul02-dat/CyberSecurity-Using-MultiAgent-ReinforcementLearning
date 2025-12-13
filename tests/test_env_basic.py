import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.envs.marl_env import MARLCyberEnv
from src.envs.synthetic_env import SyntheticScenarioEnv, SimplePatternEnv
from src.utils.seed import set_seed


def test_env_initialization():
    env = MARLCyberEnv(config_path="configs/env.yml")
    assert env.episode_length == 100
    assert env.state_dim == 4
    assert len(env.agents) == 3
    assert "detection" in env.agents
    assert "response" in env.agents
    assert "attacker" in env.agents


def test_env_reset():
    set_seed(42)
    env = MARLCyberEnv(config_path="configs/env.yml")
    obs = env.reset()
    
    assert len(obs) == 3
    assert "detection" in obs
    assert "response" in obs
    assert "attacker" in obs
    
    for agent, ob in obs.items():
        assert ob.shape == (4,)
        assert ob.dtype == np.float32


def test_env_step():
    set_seed(42)
    env = MARLCyberEnv(config_path="configs/env.yml")
    obs = env.reset()
    
    actions = {
        "detection": {"flag": False, "confidence": 0.0},
        "response": {"action": "ALLOW", "duration_bin": 0},
        "attacker": {"attack": "none"}
    }
    
    obs, rewards, dones, infos = env.step(actions)
    
    assert len(obs) == 3
    assert len(rewards) == 3
    assert len(dones) == 4
    assert "__all__" in dones
    assert len(infos) == 3


def test_env_episode_completion():
    set_seed(42)
    env = MARLCyberEnv(config_path="configs/env.yml")
    obs = env.reset()
    
    actions = {
        "detection": {"flag": False, "confidence": 0.0},
        "response": {"action": "ALLOW", "duration_bin": 0},
        "attacker": {"attack": "none"}
    }
    
    done = False
    step_count = 0
    
    while not done:
        obs, rewards, dones, infos = env.step(actions)
        done = dones["__all__"]
        step_count += 1
    
    assert step_count == 100


def test_synthetic_scenario_env():
    set_seed(42)
    env = SyntheticScenarioEnv(
        config_path="configs/env.yml",
        scenario_path="data/synthetic/example_scenario.json"
    )
    
    assert env.scenario is not None
    assert len(env.attack_schedule) == 100
    
    obs = env.reset()
    assert len(obs) == 3


def test_simple_pattern_env():
    set_seed(42)
    env = SimplePatternEnv(config_path="configs/env.yml", pattern="burst")
    
    obs = env.reset()
    assert len(obs) == 3
    
    actions = {
        "detection": {"flag": False, "confidence": 0.0},
        "response": {"action": "ALLOW", "duration_bin": 0}
    }
    
    obs, rewards, dones, infos = env.step(actions)
    assert "attacker" in infos


def test_attack_state_changes():
    set_seed(42)
    env = MARLCyberEnv(config_path="configs/env.yml")
    obs_init = env.reset()
    
    initial_flows = obs_init["detection"][0]
    
    actions = {
        "detection": {"flag": False, "confidence": 0.0},
        "response": {"action": "ALLOW", "duration_bin": 0},
        "attacker": {"attack": "ddos"}
    }
    
    obs, rewards, dones, infos = env.step(actions)
    
    assert obs["detection"][0] > initial_flows


if __name__ == "__main__":
    pytest.main([__file__, "-v"])