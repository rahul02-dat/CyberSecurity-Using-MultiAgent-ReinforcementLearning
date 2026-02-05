import torch
from typing import Any, Dict, Optional
import os


class PolicyStore:
    
    def __init__(self, base_dir: str = "output/policies"):
        self.base_dir = base_dir
        self.policies: Dict[str, Any] = {}
        
        os.makedirs(base_dir, exist_ok=True)
    
    def register_policy(self, agent_id: str, policy: Any) -> None:
        
        self.policies[agent_id] = policy
    
    def get_policy(self, agent_id: str) -> Optional[Any]:
        
        return self.policies.get(agent_id)
    
    def save_policy(self, agent_id: str, epoch: int) -> str:
        
        if agent_id not in self.policies:
            raise ValueError(f"Policy for agent {agent_id} not registered")
        
        agent_dir = os.path.join(self.base_dir, agent_id)
        os.makedirs(agent_dir, exist_ok=True)
        
        filepath = os.path.join(agent_dir, f"policy_epoch_{epoch}.pt")
        
        policy = self.policies[agent_id]
        if hasattr(policy, 'save'):
            policy.save(filepath)
        else:
            torch.save(policy, filepath)
        
        return filepath
    
    def load_policy(self, agent_id: str, epoch: int) -> None:
        
        if agent_id not in self.policies:
            raise ValueError(f"Policy for agent {agent_id} not registered")
        
        agent_dir = os.path.join(self.base_dir, agent_id)
        filepath = os.path.join(agent_dir, f"policy_epoch_{epoch}.pt")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Policy file not found: {filepath}")
        
        policy = self.policies[agent_id]
        if hasattr(policy, 'load'):
            policy.load(filepath)
        else:
            self.policies[agent_id] = torch.load(filepath)
    
    def save_all(self, epoch: int) -> Dict[str, str]:
        
        filepaths = {}
        for agent_id in self.policies.keys():
            filepath = self.save_policy(agent_id, epoch)
            filepaths[agent_id] = filepath
        
        return filepaths
    
    def load_latest(self) -> None:
        
        for agent_id in self.policies.keys():
            agent_dir = os.path.join(self.base_dir, agent_id)
            
            if not os.path.exists(agent_dir):
                continue
            
            policy_files = [f for f in os.listdir(agent_dir) if f.startswith("policy_epoch_")]
            
            if not policy_files:
                continue
            
            epochs = [int(f.split("_")[-1].split(".")[0]) for f in policy_files]
            latest_epoch = max(epochs)
            
            self.load_policy(agent_id, latest_epoch)