from abc import ABC, abstractmethod
from typing import Any, Optional
import torch


class BaseAgent(ABC):
    
    def __init__(self, agent_id: str, observation_space: dict[str, Any], action_space: dict[str, Any]):
        self.agent_id = agent_id
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def select_action(self, observation: dict[str, Any], training: bool = True) -> Any:
        pass
    
    @abstractmethod
    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass
    
    @abstractmethod
    def get_state_dict(self) -> dict[str, Any]:
        pass