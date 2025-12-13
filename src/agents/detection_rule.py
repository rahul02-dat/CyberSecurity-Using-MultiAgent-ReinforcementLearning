import numpy as np
from typing import Dict, Any


class RuleBasedDetectionAgent:
    
    def __init__(
        self,
        flow_threshold: float = 150.0,
        failed_login_threshold: float = 10.0,
        entropy_threshold: float = 6.5,
        pkt_size_low_threshold: float = 400.0
    ):
        self.flow_threshold = flow_threshold
        self.failed_login_threshold = failed_login_threshold
        self.entropy_threshold = entropy_threshold
        self.pkt_size_low_threshold = pkt_size_low_threshold
    
    def act(self, observation: np.ndarray) -> Dict[str, Any]:
        flows_per_sec = observation[0]
        avg_pkt_size = observation[1]
        failed_logins = observation[2]
        entropy = observation[3]
        
        score = 0.0
        
        if flows_per_sec > self.flow_threshold:
            score += 0.4
        
        if failed_logins > self.failed_login_threshold:
            score += 0.3
        
        if entropy > self.entropy_threshold:
            score += 0.2
        
        if avg_pkt_size < self.pkt_size_low_threshold:
            score += 0.1
        
        flag = score >= 0.5
        confidence = min(score, 1.0)
        
        return {
            "flag": flag,
            "confidence": confidence
        }