from typing import Dict, Any
from environment.network_env import NetworkEnvironment
from utils.logging import Logger


def evaluate_agents(
    agents: Dict[str, Any],
    evaluators: Dict[str, Any],
    env: NetworkEnvironment,
    num_episodes: int = 50,
    logger: Logger = None
) -> Dict[str, Any]:
    
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False
        step = 0
        
        while not done and step < env.max_steps:
            monitoring_obs = {'feature_vector': obs['feature_vector']}
            monitoring_action = agents['monitoring'].select_action(monitoring_obs, training=False)
            
            selected_indices = monitoring_action['selected_indices']
            compressed_features = obs['feature_vector'][selected_indices]
            
            detection_obs = {'compressed_features': compressed_features}
            detection_action = agents['detection'].select_action(detection_obs, training=False)
            
            import numpy as np
            response_obs = {
                'state': np.concatenate([
                    np.eye(5)[detection_action['predicted_class']],
                    [detection_action['confidence']],
                    obs['system_state']
                ])
            }
            response_action = agents['response'].select_action(response_obs, training=False)
            
            actions = {
                'monitoring_selected_indices': selected_indices,
                'detection_class': detection_action['predicted_class'],
                'detection_confidence': detection_action['confidence'],
                'response_action': response_action['action']
            }
            
            next_obs, reward, terminated, truncated, info = env.step(actions)
            
            episode_reward += reward
            done = terminated or truncated
            
            obs = next_obs
            step += 1
        
        total_rewards.append(episode_reward)
        
        if logger and episode % 10 == 0:
            logger.info(f"Eval Episode {episode}: Reward = {episode_reward:.2f}")
    
    import numpy as np
    evaluation_results = {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'min_reward': np.min(total_rewards),
        'max_reward': np.max(total_rewards)
    }
    
    return evaluation_results