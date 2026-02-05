from typing import Dict, Any
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logging import Logger


def apply_self_adaptation(
    agents: Dict[str, Any],
    adaptations: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    logger: Logger = None
) -> Dict[str, Any]:
    
    adaptation_results = {}
    
    for agent_name in ['monitoring', 'detection', 'response']:
        if agent_name not in agents or agent_name not in adaptations:
            continue
        
        agent = agents[agent_name]
        adaptation_module = adaptations[agent_name]
        
        agent_metrics = performance_metrics.get(f'{agent_name}_metrics', {})
        
        proposed_adaptations = adaptation_module.propose_adaptations(agent_metrics)
        
        adaptation_module.apply_adaptations(proposed_adaptations, agent)
        
        adaptation_results[agent_name] = proposed_adaptations
        
        if logger:
            logger.info(f"Applied adaptations to {agent_name}: {proposed_adaptations}")
    
    return adaptation_results