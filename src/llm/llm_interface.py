import json
import requests
from typing import Dict, Any, Optional
import time
import hashlib


class LMStudioInterface:
    
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "openai/gpt-oss-20b",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 10
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.cache = {}
        
    def _create_cache_key(self, input_data: Dict[str, Any]) -> str:
        json_str = json.dumps(input_data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _build_prompt(self, input_data: Dict[str, Any], decision: str) -> str:
        prompt = f"""You are a cybersecurity audit assistant. Analyze this security decision and explain it.

Input:
{json.dumps(input_data, indent=2)}

The system made the decision: {decision}

Explain this decision. Output ONLY valid JSON with this exact structure:
{{
  "summary": "Brief explanation in 1-2 sentences",
  "key_reasons": ["reason 1", "reason 2", "reason 3"],
  "risk_level": "LOW or MEDIUM or HIGH",
  "rollback_condition": "Condition that triggered rollback or N/A",
  "confidence_in_decision": 0.85
}}

JSON output:"""
        return prompt
    
    def explain_decision(
        self,
        state: Dict[str, float],
        detection: Dict[str, Any],
        safety_decision: str,
        response_action: Dict[str, Any],
        recent_metrics: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        
        input_data = {
            "state": state,
            "detection": detection,
            "safety_decision": safety_decision,
            "response_action": response_action,
            "recent_metrics": recent_metrics
        }
        
        cache_key = self._create_cache_key(input_data)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        prompt = self._build_prompt(input_data, safety_decision)
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a cybersecurity audit assistant. Output only valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                },
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                return self._fallback_explanation(safety_decision)
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            parsed = self._parse_llm_output(content)
            
            if parsed:
                self.cache[cache_key] = parsed
                return parsed
            else:
                return self._fallback_explanation(safety_decision)
                
        except requests.exceptions.Timeout:
            return self._fallback_explanation(safety_decision)
        except requests.exceptions.ConnectionError:
            return self._fallback_explanation(safety_decision)
        except Exception:
            return self._fallback_explanation(safety_decision)
    
    def _parse_llm_output(self, content: str) -> Optional[Dict[str, Any]]:
        try:
            content = content.strip()
            
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()
            
            parsed = json.loads(content)
            
            required_keys = ["summary", "key_reasons", "risk_level", "rollback_condition", "confidence_in_decision"]
            if not all(k in parsed for k in required_keys):
                return None
            
            if parsed["risk_level"] not in ["LOW", "MEDIUM", "HIGH"]:
                return None
            
            if not isinstance(parsed["key_reasons"], list):
                return None
            
            if not isinstance(parsed["confidence_in_decision"], (int, float)):
                return None
            
            return parsed
            
        except json.JSONDecodeError:
            return None
        except Exception:
            return None
    
    def _fallback_explanation(self, decision: str) -> Dict[str, Any]:
        explanations = {
            "ALLOW": {
                "summary": "Traffic allowed based on normal network patterns and low threat indicators.",
                "key_reasons": [
                    "Network metrics within normal thresholds",
                    "Detection confidence below alert level",
                    "No significant anomalies detected"
                ],
                "risk_level": "LOW",
                "rollback_condition": "N/A",
                "confidence_in_decision": 0.75
            },
            "BLOCK": {
                "summary": "Traffic blocked due to high confidence attack detection and anomalous patterns.",
                "key_reasons": [
                    "Attack patterns detected with high confidence",
                    "Network metrics exceed normal thresholds",
                    "Immediate threat mitigation required"
                ],
                "risk_level": "HIGH",
                "rollback_condition": "N/A",
                "confidence_in_decision": 0.85
            },
            "QUARANTINE": {
                "summary": "Traffic quarantined for monitoring due to suspicious but inconclusive patterns.",
                "key_reasons": [
                    "Medium confidence detection signals",
                    "Suspicious patterns require monitoring",
                    "Balanced security and availability approach"
                ],
                "risk_level": "MEDIUM",
                "rollback_condition": "N/A",
                "confidence_in_decision": 0.70
            },
            "ROLLBACK": {
                "summary": "Previous block rolled back due to false positive detection and legitimate traffic indicators.",
                "key_reasons": [
                    "High false positive rate detected",
                    "Traffic patterns indicate legitimate activity",
                    "Safety override to restore service"
                ],
                "risk_level": "LOW",
                "rollback_condition": "False positive rate exceeded threshold",
                "confidence_in_decision": 0.80
            }
        }
        
        return explanations.get(decision, explanations["ALLOW"])
    
    def clear_cache(self):
        self.cache.clear()