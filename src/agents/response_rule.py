import numpy as np
from typing import Dict, Any


class RuleBasedResponseAgent:
    
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        high_confidence_threshold: float = 0.8
    ):
        self.confidence_threshold = confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
    
    def act(
        self, 
        observation: np.ndarray, 
        detection_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        flag = detection_result.get("flag", False)
        confidence = detection_result.get("confidence", 0.0)
        
        if not flag:
            return {
                "action": "ALLOW",
                "duration_bin": 0
            }
        
        if confidence >= self.high_confidence_threshold:
            action = "BLOCK"
            duration_bin = 2
        elif confidence >= self.confidence_threshold:
            action = "QUARANTINE"
            duration_bin = 1
        else:
            action = "ALLOW"
            duration_bin = 0
        
        return {
            "action": action,
            "duration_bin": duration_bin
        }