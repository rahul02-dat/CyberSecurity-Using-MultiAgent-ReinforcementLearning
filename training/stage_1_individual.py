from typing import Dict, Any
import numpy as np
from environment.network_env import NetworkEnvironment
from utils.logging import Logger
from utils.metrics import MetricsTracker


def train_individual_agents(
    agents: Dict[str, Any],
    evaluators: Dict[str, Any],
    env: NetworkEnvironment,
    num_episodes: int = 100,
    logger: Logger = None
) -> Dict[str, Any]:
    
    metrics_tracker = MetricsTracker()
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False
        step = 0
        
        episode_data = {
            'monitoring': {'rewards': [], 'selected_features_counts': [], 'information_gains': []},
            'detection': {'predictions': [], 'true_labels': [], 'confidences': [], 'rewards': []},
            'response': {'actions': [], 'rewards': [], 'damages_prevented': [], 'costs_incurred': []}
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
            
            episode_data['monitoring']['rewards'].append(reward * 0.3)
            episode_data['monitoring']['selected_features_counts'].append(len(selected_indices))
            episode_data['monitoring']['information_gains'].append(0.5)
            
            episode_data['detection']['predictions'].append(detection_action['predicted_class'])
            episode_data['detection']['true_labels'].append(info.get('attack_type', 0))
            episode_data['detection']['confidences'].append(detection_action['confidence'])
            episode_data['detection']['rewards'].append(reward * 0.4)
            
            episode_data['response']['actions'].append(response_action['action'])
            episode_data['response']['rewards'].append(reward * 0.3)
            episode_data['response']['damages_prevented'].append(info.get('damage', 0) * 0.5)
            episode_data['response']['costs_incurred'].append(5.0)
            
            obs = next_obs
            step += 1
        
        for agent_name in ['monitoring', 'detection', 'response']:
            metrics = evaluators[agent_name].compute_metrics(episode_data[agent_name])
            metrics_tracker.add(f'{agent_name}_reward', metrics.get('avg_reward', 0))
        
        if logger and episode % 10 == 0:
            logger.info(f"Episode {episode}: Total Reward = {episode_reward:.2f}")
    
    results = {
        'monitoring_metrics': evaluators['monitoring'].aggregate_metrics(num_episodes),
        'detection_metrics': evaluators['detection'].aggregate_metrics(num_episodes),
        'response_metrics': evaluators['response'].aggregate_metrics(num_episodes)
    }
    
    return results