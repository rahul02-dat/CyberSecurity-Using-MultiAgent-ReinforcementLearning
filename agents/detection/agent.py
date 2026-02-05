import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Optional
from agents.base.base_agent import BaseAgent
from agents.detection.model import DetectionPolicyNetwork
from agents.detection.reward import DetectionReward
import requests
import json


class DetectionAgent(BaseAgent):
    
    def __init__(
        self, 
        agent_id: str, 
        observation_space: dict[str, Any], 
        action_space: dict[str, Any],
        learning_rate: float = 3e-4,
        use_ollama: bool = False,
        ollama_url: str = "http://localhost:11434"
    ):
        super().__init__(agent_id, observation_space, action_space)
        
        self.input_dim = observation_space['compressed_feature_dim']
        self.num_classes = 5
        
        self.policy_network = DetectionPolicyNetwork(
            input_dim=self.input_dim,
            num_classes=self.num_classes
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        self.reward_function = DetectionReward()
        
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url
        self.confidence_threshold = 0.7
        
        self.class_weights = torch.ones(self.num_classes).to(self.device)
    
    def select_action(self, observation: dict[str, Any], training: bool = True) -> Any:
        
        feature_vector = torch.FloatTensor(observation['compressed_features']).to(self.device)
        
        with torch.no_grad():
            class_logits, _, confidence = self.policy_network(feature_vector.unsqueeze(0))
            
            class_probs = torch.softmax(class_logits, dim=-1)
            predicted_class = torch.argmax(class_probs, dim=-1).item()
            confidence_score = confidence.item()
        
        use_llm = False
        llm_response = None
        
        if self.use_ollama and confidence_score < self.confidence_threshold:
            try:
                llm_response = self._query_ollama(observation)
                if llm_response and 'class' in llm_response:
                    predicted_class = llm_response['class']
                    use_llm = True
            except Exception:
                pass
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'class_probabilities': class_probs.squeeze(0).cpu().numpy(),
            'used_llm': use_llm,
            'llm_response': llm_response
        }
    
    def _query_ollama(self, observation: dict[str, Any]) -> Optional[dict[str, Any]]:
        
        feature_summary = self._summarize_features(observation['compressed_features'])
        
        prompt = f"""Analyze this network traffic pattern:
        
Features: {feature_summary}

Classify as one of: Normal(0), DDoS(1), Injection(2), Malware(3), Probe(4)

Respond with JSON: {{"class": <number>, "reasoning": "<brief explanation>"}}"""
        
        payload = {
            "model": "llama2",
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                try:
                    parsed = json.loads(result['response'])
                    return parsed
                except json.JSONDecodeError:
                    return None
        
        return None
    
    def _summarize_features(self, features: list[float]) -> str:
        
        import numpy as np
        features_array = np.array(features)
        
        summary = f"mean={features_array.mean():.3f}, std={features_array.std():.3f}, "
        summary += f"max={features_array.max():.3f}, min={features_array.min():.3f}"
        
        return summary
    
    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        
        class_logits, values, confidences = self.policy_network(states)
        
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        classification_loss = criterion(class_logits, actions)
        
        value_loss = nn.MSELoss()(values.squeeze(), rewards)
        
        confidence_target = (actions == torch.argmax(class_logits, dim=1)).float()
        confidence_loss = nn.BCELoss()(confidences.squeeze(), confidence_target)
        
        total_loss = classification_loss + 0.5 * value_loss + 0.3 * confidence_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'classification_loss': classification_loss.item(),
            'value_loss': value_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def save(self, path: str) -> None:
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'class_weights': self.class_weights,
            'confidence_threshold': self.confidence_threshold
        }, path)
    
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.class_weights = checkpoint['class_weights']
        self.confidence_threshold = checkpoint['confidence_threshold']
    
    def get_state_dict(self) -> dict[str, Any]:
        return {
            'policy_network': self.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'class_weights': self.class_weights.cpu(),
            'confidence_threshold': self.confidence_threshold
        }