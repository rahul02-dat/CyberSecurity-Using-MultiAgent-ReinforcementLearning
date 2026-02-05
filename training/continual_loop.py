from typing import Dict, Any
from environment.network_env import NetworkEnvironment
from training.stage_1_individual import train_individual_agents
from training.stage_2_evaluation import evaluate_agents
from training.stage_3_self_adaptation import apply_self_adaptation
from training.stage_4_joint_training import joint_training
from utils.logging import Logger


def continual_learning_loop(
    agents: Dict[str, Any],
    evaluators: Dict[str, Any],
    adaptations: Dict[str, Any],
    env: NetworkEnvironment,
    num_cycles: int = 10,
    episodes_per_cycle: int = 50,
    logger: Logger = None
) -> Dict[str, Any]:
    
    all_results = []
    
    for cycle in range(num_cycles):
        if logger:
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting Continual Learning Cycle {cycle + 1}/{num_cycles}")
            logger.info(f"{'='*50}\n")
        
        if logger:
            logger.info("Phase 1: Individual Training")
        individual_results = train_individual_agents(
            agents=agents,
            evaluators=evaluators,
            env=env,
            num_episodes=episodes_per_cycle,
            logger=logger
        )
        
        if logger:
            logger.info("Phase 2: Evaluation")
        eval_results = evaluate_agents(
            agents=agents,
            evaluators=evaluators,
            env=env,
            num_episodes=episodes_per_cycle // 2,
            logger=logger
        )
        
        if logger:
            logger.info("Phase 3: Self-Adaptation")
        adaptation_results = apply_self_adaptation(
            agents=agents,
            adaptations=adaptations,
            performance_metrics=individual_results,
            logger=logger
        )
        
        if logger:
            logger.info("Phase 4: Joint Training")
        joint_results = joint_training(
            agents=agents,
            evaluators=evaluators,
            env=env,
            num_episodes=episodes_per_cycle,
            logger=logger
        )
        
        cycle_results = {
            'cycle': cycle,
            'individual': individual_results,
            'evaluation': eval_results,
            'adaptation': adaptation_results,
            'joint': joint_results
        }
        
        all_results.append(cycle_results)
        
        if logger:
            logger.info(f"\nCycle {cycle + 1} completed")
            logger.info(f"Mean Evaluation Reward: {eval_results['mean_reward']:.2f}")
    
    return {
        'num_cycles': num_cycles,
        'cycle_results': all_results,
        'final_performance': all_results[-1]['evaluation'] if all_results else {}
    }