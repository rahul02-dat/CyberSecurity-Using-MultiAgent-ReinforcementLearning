import os
import sys
import json
import torch
import numpy as np
from typing import Dict, Any

from environment.network_env import NetworkEnvironment
from agents.monitoring.agent import MonitoringAgent
from agents.monitoring.evaluator import MonitoringEvaluator
from agents.monitoring.adaptation import MonitoringAdaptation
from agents.detection.agent import DetectionAgent
from agents.detection.evaluator import DetectionEvaluator
from agents.detection.adaptation import DetectionAdaptation
from agents.response.agent import ResponseAgent
from agents.response.evaluator import ResponseEvaluator
from agents.response.adaptation import ResponseAdaptation
from agents.policy_adaptation.agent import PolicyAdaptationAgent
from agents.policy_adaptation.evaluator import PolicyAdaptationEvaluator
from agents.policy_adaptation.adaptation import PolicyAdaptationAdaptation

from training.stage_1_individual import train_individual_agents
from training.checkpoint_manager import CheckpointManager
from utils.logging import Logger, setup_directories
from utils.seeding import set_seed
from utils.metrics import MetricsTracker
from coordination.message_bus import MessageBus
from coordination.shared_memory import AgentStateRegistry
from rl.policy_store import PolicyStore


def initialize_agents() -> Dict[str, Any]:
    
    monitoring_agent = MonitoringAgent(
        agent_id="monitoring",
        observation_space={'feature_vector_dim': 50},
        action_space={'max_features': 50},
        num_features_to_select=20,
        learning_rate=3e-4
    )
    
    detection_agent = DetectionAgent(
        agent_id="detection",
        observation_space={'compressed_feature_dim': 20},
        action_space={'num_classes': 5},
        learning_rate=3e-4,
        use_ollama=False
    )
    
    response_agent = ResponseAgent(
        agent_id="response",
        observation_space={'state_dim': 14},
        action_space={'num_actions': 6},
        learning_rate=3e-4
    )
    
    policy_adaptation_agent = PolicyAdaptationAgent(
        agent_id="policy_adaptation",
        observation_space={'performance_metrics_dim': 9},
        action_space={'num_hyperparams': 6},
        learning_rate=1e-4
    )
    
    return {
        'monitoring': monitoring_agent,
        'detection': detection_agent,
        'response': response_agent,
        'policy_adaptation': policy_adaptation_agent
    }


def initialize_evaluators() -> Dict[str, Any]:
    
    return {
        'monitoring': MonitoringEvaluator('monitoring'),
        'detection': DetectionEvaluator('detection'),
        'response': ResponseEvaluator('response'),
        'policy_adaptation': PolicyAdaptationEvaluator('policy_adaptation')
    }


def initialize_adaptations() -> Dict[str, Any]:
    
    return {
        'monitoring': MonitoringAdaptation('monitoring'),
        'detection': DetectionAdaptation('detection'),
        'response': ResponseAdaptation('response'),
        'policy_adaptation': PolicyAdaptationAdaptation('policy_adaptation')
    }


def save_performance_metrics(
    metrics: Dict[str, Any],
    output_dir: str = "performance/matrices"
) -> None:
    
    for agent_type, agent_metrics in metrics.items():
        filepath = os.path.join(output_dir, f"{agent_type}_metrics.json")
        
        existing_data = {}
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
        
        existing_data.update(agent_metrics)
        
        with open(filepath, 'w') as f:
            json.dump(existing_data, f, indent=2)


def main():
    
    setup_directories()
    
    set_seed(42)
    
    logger = Logger(log_dir="output/logs", experiment_name="marl_cybersec")
    logger.info("Starting MARL-Cybersec System")
    
    env = NetworkEnvironment(feature_dim=50, max_steps=1000, seed=42)
    logger.info("Environment initialized")
    
    agents = initialize_agents()
    logger.info(f"Initialized {len(agents)} agents")
    
    evaluators = initialize_evaluators()
    adaptations = initialize_adaptations()
    
    checkpoint_manager = CheckpointManager()
    policy_store = PolicyStore()
    
    for agent_id, agent in agents.items():
        policy_store.register_policy(agent_id, agent)
    
    message_bus = MessageBus()
    agent_registry = AgentStateRegistry()
    
    for agent_id in agents.keys():
        agent_registry.register_agent(agent_id)
    
    logger.info("Starting Stage 1: Individual Agent Training")
    
    stage_1_results = train_individual_agents(
        agents=agents,
        evaluators=evaluators,
        env=env,
        num_episodes=100,
        logger=logger
    )
    
    logger.info("Stage 1 completed")
    logger.info(f"Results: {stage_1_results}")
    
    save_performance_metrics(stage_1_results)
    
    checkpoint_manager.save_checkpoint(
        epoch=1,
        agents=agents,
        metrics=stage_1_results
    )
    
    policy_store.save_all(epoch=1)
    
    logger.info("Training completed successfully")
    
    logger.info("System Evaluation:")
    for agent_type, metrics in stage_1_results.items():
        logger.info(f"\n{agent_type}:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric_name}: {value:.4f}")
            else:
                logger.info(f"  {metric_name}: {value}")
    
    metrics_file = logger.save_metrics()
    logger.info(f"Metrics saved to {metrics_file}")
    
    return stage_1_results


if __name__ == "__main__":
    try:
        results = main()
        print("\n" + "="*50)
        print("MARL-Cybersec Training Complete!")
        print("="*50)
        print(f"\nResults Summary:")
        for agent_type, metrics in results.items():
            print(f"\n{agent_type.upper()}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)