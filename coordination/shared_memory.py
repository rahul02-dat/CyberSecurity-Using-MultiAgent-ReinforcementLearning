from typing import Any, Dict, Optional
import threading


class SharedMemory:
    
    def __init__(self):
        self.memory: Dict[str, Any] = {}
        self.lock = threading.Lock()
    
    def write(self, key: str, value: Any) -> None:
        
        with self.lock:
            self.memory[key] = value
    
    def read(self, key: str, default: Any = None) -> Any:
        
        with self.lock:
            return self.memory.get(key, default)
    
    def update(self, updates: Dict[str, Any]) -> None:
        
        with self.lock:
            self.memory.update(updates)
    
    def delete(self, key: str) -> None:
        
        with self.lock:
            if key in self.memory:
                del self.memory[key]
    
    def clear(self) -> None:
        
        with self.lock:
            self.memory.clear()
    
    def keys(self) -> list[str]:
        
        with self.lock:
            return list(self.memory.keys())
    
    def snapshot(self) -> Dict[str, Any]:
        
        with self.lock:
            return self.memory.copy()


class AgentStateRegistry:
    
    def __init__(self):
        self.shared_memory = SharedMemory()
    
    def register_agent(self, agent_id: str, initial_state: Optional[Dict[str, Any]] = None) -> None:
        
        state = initial_state if initial_state is not None else {}
        self.shared_memory.write(f"agent_{agent_id}", state)
    
    def update_agent_state(self, agent_id: str, state_updates: Dict[str, Any]) -> None:
        
        current_state = self.shared_memory.read(f"agent_{agent_id}", {})
        current_state.update(state_updates)
        self.shared_memory.write(f"agent_{agent_id}", current_state)
    
    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        
        return self.shared_memory.read(f"agent_{agent_id}", {})
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        
        snapshot = self.shared_memory.snapshot()
        
        return {
            key.replace("agent_", ""): value 
            for key, value in snapshot.items() 
            if key.startswith("agent_")
        }