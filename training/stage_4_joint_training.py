from typing import Dict, Any
import numpy as np
from environment.network_env import NetworkEnvironment
from utils.logging import Logger


def joint_training(
    agents: Dict[str, Any],
    evaluators: Dict[str, Any],
    env: NetworkEnvironment,
    num_episodes: int = 100,
    logger: Logger = None
) -> Dict[str, Any]:
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False
        step = 0
        
        trajectories = {
            'monitoring': [],
            'detection': [],
            'response': []
        }
        
        while not done and step < env.max_steps:
            monitoring_obs = {'feature_vector': obs['feature_vector']}
            monitoring_action = agents['monitoring'].select_action(monitoring_obs, training=True)
            
            selected_indices = monitoring_action['selected_indices']
            compressed_features = obs['feature_vector'][selected_indices]
            
            detection_obs = {'compressed_features': compressed_features}
            detection_action = agents['detection'].select_action(detection_obs, training=True)
            
            response_obs = {
                'state': np.concatenate([
                    np.eye(5)[detection_action['predicted_class']],
                    [detection_action['confidence']],
                    obs['system_state']
                ])
            }
            response_action = agents['response'].select_action(response_obs, training=True)
            
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
        
        if logger and episode % 10 == 0:
            logger.info(f"Joint Episode {episode}: Total Reward = {episode_reward:.2f}")
    
    results = {
        'joint_training_complete': True,
        'episodes_completed': num_episodes
    }
    
    return results