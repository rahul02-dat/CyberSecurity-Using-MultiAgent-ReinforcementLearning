import requests
import json
from typing import Any, Optional, Dict


class OllamaClient:
    
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama2:7b"):
        self.base_url = base_url
        self.default_model = default_model
        self.timeout = 30
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        stream: bool = False,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        
        model_name = model or self.default_model
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream
        }
        
        if format:
            payload["format"] = format
        
        if options:
            payload["options"] = options
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {
                "error": str(e),
                "response": None
            }
    
    def chat(
        self,
        messages: list[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        
        model_name = model or self.default_model
        
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": stream
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {
                "error": str(e),
                "message": None
            }
    
    def list_models(self) -> Dict[str, Any]:
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {
                "error": str(e),
                "models": []
            }
    
    def pull_model(self, model: str) -> Dict[str, Any]:
        
        payload = {"name": model}
        
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=300
            )
            
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def is_available(self) -> bool:
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False