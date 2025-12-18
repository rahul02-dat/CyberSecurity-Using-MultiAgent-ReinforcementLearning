from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import Any
import json
import os


class MARLMetricsCallback(DefaultCallbacks):
    
    def __init__(self):
        super().__init__()
        self.explanations = []
        self.enable_explanations = os.environ.get("ENABLE_LLM_EXPLANATIONS", "true").lower() == "true"
        
        if self.enable_explanations:
            try:
                from src.llm.structured_extractor import StructuredExplanationExtractor
                self.explainer = StructuredExplanationExtractor(enable_llm=True)
            except Exception:
                self.explainer = None
                self.enable_explanations = False
    
    def on_episode_start(
        self,
        *,
        algorithm: Any = None,
        episode: Any = None,
        env_runner: Any = None,
        **kwargs
    ) -> None:
        pass
    
    def on_episode_step(
        self,
        *,
        algorithm: Any = None,
        episode: Any = None,
        env_runner: Any = None,
        **kwargs
    ) -> None:
        if self.enable_explanations and self.explainer and episode:
            try:
                info = episode.last_info_for()
                
                if info and "safety" in info:
                    obs = episode.last_obs_for("safety")
                    
                    if obs is not None and len(obs) >= 6:
                        detection_flag = bool(obs[0] > 0.5)
                        detection_confidence = float(obs[1])
                        response_action_encoded = float(obs[2])
                        fp_rate = float(obs[3])
                        fn_rate = float(obs[4])
                        time_since_block = int(float(obs[5]) * 100)
                        
                        safety_info = info.get("safety", {})
                        safety_decision = safety_info.get("final_decision", "ALLOW")
                        
                        detection_action = {
                            "flag": detection_flag,
                            "confidence": detection_confidence
                        }
                        
                        response_map = {0.0: "ALLOW", 0.5: "BLOCK", 1.0: "QUARANTINE"}
                        closest_action = min(response_map.keys(), key=lambda x: abs(x - response_action_encoded))
                        
                        response_action = {
                            "action": response_map[closest_action],
                            "duration_bin": 0
                        }
                        
                        attack_active = info.get("detection", {}).get("attack_active", False)
                        
                        base_obs = episode.last_obs_for("detection")
                        if base_obs is not None:
                            explanation = self.explainer.extract_explanation(
                                obs=base_obs,
                                detection_action=detection_action,
                                response_action=response_action,
                                safety_decision=safety_decision,
                                recent_fp_rate=fp_rate,
                                recent_fn_rate=fn_rate,
                                time_since_last_block=time_since_block,
                                attack_active=attack_active
                            )
                            
                            self.explanations.append(explanation)
            except Exception:
                pass
    
    def on_episode_end(
        self,
        *,
        algorithm: Any = None,
        episode: Any = None,
        env_runner: Any = None,
        **kwargs
    ) -> None:
        if self.enable_explanations and len(self.explanations) > 0:
            try:
                log_dir = "results/explanations"
                os.makedirs(log_dir, exist_ok=True)
                
                episode_id = episode.episode_id if episode else "unknown"
                filepath = os.path.join(log_dir, f"episode_{episode_id}_explanations.json")
                
                with open(filepath, 'w') as f:
                    json.dump(self.explanations, f, indent=2)
                
                self.explanations = []
            except Exception:
                pass