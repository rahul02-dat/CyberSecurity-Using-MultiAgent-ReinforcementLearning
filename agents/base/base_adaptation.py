from abc import ABC, abstractmethod
from typing import Any


class BaseAdaptation(ABC):
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.adaptation_history: list[dict[str, Any]] = []
    
    @abstractmethod
    def propose_adaptations(self, performance_data: dict[str, Any]) -> dict[str, Any]:
        pass
    
    @abstractmethod
    def apply_adaptations(self, adaptations: dict[str, Any], agent: Any) -> None:
        pass
    
    def record_adaptation(self, adaptation: dict[str, Any]) -> None:
        self.adaptation_history.append(adaptation)
    
    def get_adaptation_history(self) -> list[dict[str, Any]]:
        return self.adaptation_history