import json
from typing import Dict, Any, Optional
import numpy as np
from src.llm.llm_interface import LMStudioInterface


class StructuredExplanationExtractor:
    
    def __init__(
        self,
        llm_interface: Optional[LMStudioInterface] = None,
        enable_llm: bool = True
    ):
        self.enable_llm = enable_llm
        
        if enable_llm:
            if llm_interface is None:
                self.llm = LMStudioInterface()
            else:
                self.llm = llm_interface
        else:
            self.llm = None
    
    def extract_explanation(
        self,
        obs: np.ndarray,
        detection_action: Dict[str, Any],
        response_action: Dict[str, Any],
        safety_decision: str,
        recent_fp_rate: float,
        recent_fn_rate: float,
        time_since_last_block: int,
        attack_active: bool
    ) -> Dict[str, Any]:
        
        state = {
            "flows_per_sec": float(obs[0]),
            "avg_pkt_size": float(obs[1]) if len(obs) > 1 else 0.0,
            "failed_logins": float(obs[2]) if len(obs) > 2 else 0.0,
            "entropy": float(obs[3]) if len(obs) > 3 else 0.0
        }
        
        detection = {
            "flag": detection_action.get("flag", False),
            "confidence": detection_action.get("confidence", 0.0)
        }
        
        response = {
            "action": response_action.get("action", "ALLOW"),
            "duration": response_action.get("duration_bin", 0)
        }
        
        recent_metrics = {
            "false_positive_rate": recent_fp_rate,
            "false_negative_rate": recent_fn_rate,
            "time_since_last_block": time_since_last_block
        }
        
        if self.enable_llm and self.llm is not None:
            try:
                explanation = self.llm.explain_decision(
                    state=state,
                    detection=detection,
                    safety_decision=safety_decision,
                    response_action=response,
                    recent_metrics=recent_metrics
                )
                
                if explanation:
                    explanation["timestamp"] = time_since_last_block
                    explanation["actual_attack"] = attack_active
                    return explanation
                    
            except Exception:
                pass
        
        return self._generate_rule_based_explanation(
            state, detection, safety_decision, response, recent_metrics, attack_active
        )
    
    def _generate_rule_based_explanation(
        self,
        state: Dict[str, float],
        detection: Dict[str, Any],
        safety_decision: str,
        response: Dict[str, Any],
        recent_metrics: Dict[str, float],
        attack_active: bool
    ) -> Dict[str, Any]:
        
        reasons = []
        risk_level = "LOW"
        rollback_condition = "N/A"
        
        if safety_decision == "BLOCK":
            summary = "Traffic blocked due to detected attack patterns"
            risk_level = "HIGH"
            
            if state["flows_per_sec"] > 150:
                reasons.append(f"High traffic flow: {state['flows_per_sec']:.0f} flows/sec")
            if state["failed_logins"] > 10:
                reasons.append(f"Excessive failed logins: {state['failed_logins']:.0f}")
            if state["entropy"] > 6.5:
                reasons.append(f"Elevated entropy: {state['entropy']:.2f}")
            if detection["confidence"] > 0.7:
                reasons.append(f"High detection confidence: {detection['confidence']:.2f}")
                
        elif safety_decision == "QUARANTINE":
            summary = "Traffic quarantined for monitoring"
            risk_level = "MEDIUM"
            
            reasons.append(f"Medium confidence detection: {detection['confidence']:.2f}")
            reasons.append("Suspicious patterns require observation")
            
        elif safety_decision == "ROLLBACK":
            summary = "Previous block rolled back as false positive"
            risk_level = "LOW"
            rollback_condition = f"False positive rate: {recent_metrics['false_positive_rate']:.2f}"
            
            reasons.append(f"High FP rate: {recent_metrics['false_positive_rate']:.2f}")
            reasons.append("Legitimate traffic incorrectly blocked")
            reasons.append("Safety override activated")
            
        else:
            summary = "Traffic allowed - normal patterns"
            risk_level = "LOW"
            
            reasons.append("Network metrics within normal range")
            reasons.append(f"Low detection confidence: {detection['confidence']:.2f}")
            
        if not reasons:
            reasons = ["Decision based on aggregated threat assessment"]
        
        confidence = 0.85 if attack_active == (safety_decision in ["BLOCK", "QUARANTINE"]) else 0.70
        
        return {
            "summary": summary,
            "key_reasons": reasons[:5],
            "risk_level": risk_level,
            "rollback_condition": rollback_condition,
            "confidence_in_decision": confidence,
            "timestamp": recent_metrics["time_since_last_block"],
            "actual_attack": attack_active
        }
    
    def format_for_audit(self, explanation: Dict[str, Any]) -> str:
        lines = [
            f"Summary: {explanation['summary']}",
            f"Risk Level: {explanation['risk_level']}",
            f"Confidence: {explanation['confidence_in_decision']:.2f}",
            "Key Reasons:"
        ]
        
        for i, reason in enumerate(explanation['key_reasons'], 1):
            lines.append(f"  {i}. {reason}")
        
        if explanation['rollback_condition'] != "N/A":
            lines.append(f"Rollback Condition: {explanation['rollback_condition']}")
        
        return "\n".join(lines)
    
    def save_explanation(self, explanation: Dict[str, Any], filepath: str):
        with open(filepath, 'w') as f:
            json.dump(explanation, f, indent=2)
    
    def clear_cache(self):
        if self.llm:
            self.llm.clear_cache()