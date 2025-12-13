import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.envs.synthetic_env import SyntheticScenarioEnv
from src.agents.detection_rule import RuleBasedDetectionAgent
from src.agents.response_rule import RuleBasedResponseAgent
from src.utils.seed import set_seed
from src.utils.metrics import MetricsTracker


def run_episode():
    set_seed(42)
    
    env = SyntheticScenarioEnv(
        config_path="configs/env.yml",
        scenario_path="data/synthetic/example_scenario.json"
    )
    
    detection_agent = RuleBasedDetectionAgent()
    response_agent = RuleBasedResponseAgent()
    
    metrics = MetricsTracker()
    
    obs = env.reset()
    done = False
    
    print("Starting episode...")
    print("-" * 80)
    
    while not done:
        detection_action = detection_agent.act(obs["detection"])
        response_action = response_agent.act(obs["response"], detection_action)
        
        actions = {
            "detection": detection_action,
            "response": response_action
        }
        
        obs, rewards, dones, infos = env.step(actions)
        
        metrics.update(
            step=infos["detection"]["step"],
            attack_type=infos["detection"]["attack_type"],
            attack_active=infos["detection"]["attack_active"],
            detection_flag=detection_action["flag"],
            response_action=response_action["action"],
            rewards=rewards
        )
        
        if infos["detection"]["step"] % 10 == 0:
            print(f"Step {infos['detection']['step']:3d} | "
                  f"Attack: {infos['detection']['attack_type']:12s} | "
                  f"Detected: {detection_action['flag']} | "
                  f"Response: {response_action['action']:10s} | "
                  f"R_det: {rewards['detection']:6.1f} | "
                  f"R_resp: {rewards['response']:6.1f}")
        
        done = dones["__all__"]
    
    print("-" * 80)
    print("Episode completed!")
    print("\nFinal Metrics:")
    summary = metrics.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    run_episode()