from typing import Any, Dict, List, Optional
import threading


class Synchronizer:
    
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        
        self.agent_actions: Dict[str, Any] = {}
        self.agent_ready: Dict[str, bool] = {}
        
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        
        self.round_number = 0
    
    def submit_action(self, agent_id: str, action: Any) -> None:
        
        with self.condition:
            self.agent_actions[agent_id] = action
            self.agent_ready[agent_id] = True
            
            self.condition.notify_all()
    
    def wait_for_all_agents(self, timeout: Optional[float] = None) -> bool:
        
        with self.condition:
            while len(self.agent_ready) < self.num_agents:
                result = self.condition.wait(timeout=timeout)
                if not result and timeout is not None:
                    return False
            
            return True
    
    def get_all_actions(self) -> Dict[str, Any]:
        
        with self.lock:
            return self.agent_actions.copy()
    
    def reset_round(self) -> None:
        
        with self.lock:
            self.agent_actions.clear()
            self.agent_ready.clear()
            self.round_number += 1
    
    def get_round_number(self) -> int:
        
        with self.lock:
            return self.round_number
    
    def is_agent_ready(self, agent_id: str) -> bool:
        
        with self.lock:
            return self.agent_ready.get(agent_id, False)


class TurnBasedCoordinator:
    
    def __init__(self, agent_order: List[str]):
        self.agent_order = agent_order
        self.current_agent_index = 0
        
        self.lock = threading.Lock()
    
    def get_current_agent(self) -> str:
        
        with self.lock:
            return self.agent_order[self.current_agent_index]
    
    def advance_turn(self) -> str:
        
        with self.lock:
            self.current_agent_index = (self.current_agent_index + 1) % len(self.agent_order)
            return self.agent_order[self.current_agent_index]
    
    def reset(self) -> None:
        
        with self.lock:
            self.current_agent_index = 0