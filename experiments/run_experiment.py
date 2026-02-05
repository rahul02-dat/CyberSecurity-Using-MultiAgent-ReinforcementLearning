import yaml
import sys
from pathlib import Path
from typing import Dict, Any
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import initialize_agents, initialize_evaluators, initialize_adaptations
from environment.network_env import NetworkEnvironment
from training.continual_loop import continual_learning_loop
from utils.logging import Logger, setup_directories
from utils.seeding import set_seed
from training.checkpoint_manager import CheckpointManager


def load_config(config_path: str) -> Dict[str, Any]:
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def run_experiment(config_path: str) -> Dict[str, Any]:
    
    config = load_config(config_path)
    
    experiment_name = config['experiment']['name']
    seed = config['experiment']['seed']
    
    setup_directories()
    set_seed(seed)
    
    logger = Logger(
        log_dir="output/logs",
        experiment_name=experiment_name
    )
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Configuration: {config_path}")
    
    env_config = config['environment']
    env = NetworkEnvironment(
        feature_dim=env_config['feature_dim'],
        max_steps=env_config['max_steps'],
        seed=seed
    )
    
    agents = initialize_agents()
    evaluators = initialize_evaluators()
    adaptations = initialize_adaptations()
    
    checkpoint_manager = CheckpointManager()
    
    training_config = config['training']
    
    results = continual_learning_loop(
        agents=agents,
        evaluators=evaluators,
        adaptations=adaptations,
        env=env,
        num_cycles=training_config['num_episodes'] // 100,
        episodes_per_cycle=100,
        logger=logger
    )
    
    checkpoint_manager.save_checkpoint(
        epoch=training_config['num_episodes'],
        agents=agents,
        metrics=results['final_performance']
    )
    
    logger.info("Experiment completed successfully")
    logger.save_metrics()
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <config_path>")
        print("Example: python run_experiment.py experiments/configs/baseline.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    results = run_experiment(config_path)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nFinal Performance:")
    for key, value in results['final_performance'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")