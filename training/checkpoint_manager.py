import torch
import os
import json
from typing import Any, Dict, Optional, List
from datetime import datetime


class CheckpointManager:
    
    def __init__(self, checkpoint_dir: str = "output/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.metadata_file = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
        self.metadata: List[Dict[str, Any]] = []
        
        self._load_metadata()
    
    def save_checkpoint(
        self,
        epoch: int,
        agents: Dict[str, Any],
        optimizers: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> str:
        
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        checkpoint = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'agents': {},
            'optimizers': {},
            'metrics': metrics or {},
            'extra_data': extra_data or {}
        }
        
        for agent_id, agent in agents.items():
            if hasattr(agent, 'get_state_dict'):
                checkpoint['agents'][agent_id] = agent.get_state_dict()
            else:
                checkpoint['agents'][agent_id] = agent.state_dict()
        
        if optimizers:
            for opt_id, optimizer in optimizers.items():
                checkpoint['optimizers'][opt_id] = optimizer.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        self.metadata.append({
            'epoch': epoch,
            'checkpoint_name': checkpoint_name,
            'checkpoint_path': checkpoint_path,
            'timestamp': checkpoint['timestamp'],
            'metrics': metrics
        })
        
        self._save_metadata()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        epoch: int,
        agents: Dict[str, Any],
        optimizers: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        
        for agent_id, agent in agents.items():
            if agent_id in checkpoint['agents']:
                if hasattr(agent, 'load_state_dict'):
                    agent.load_state_dict(checkpoint['agents'][agent_id])
                else:
                    agent.load_state_dict(checkpoint['agents'][agent_id])
        
        if optimizers and 'optimizers' in checkpoint:
            for opt_id, optimizer in optimizers.items():
                if opt_id in checkpoint['optimizers']:
                    optimizer.load_state_dict(checkpoint['optimizers'][opt_id])
        
        return checkpoint
    
    def load_latest_checkpoint(
        self,
        agents: Dict[str, Any],
        optimizers: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        
        if not self.metadata:
            return None
        
        latest = max(self.metadata, key=lambda x: x['epoch'])
        
        return self.load_checkpoint(latest['epoch'], agents, optimizers)
    
    def get_available_checkpoints(self) -> List[Dict[str, Any]]:
        
        return self.metadata.copy()
    
    def delete_checkpoint(self, epoch: int) -> None:
        
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        self.metadata = [m for m in self.metadata if m['epoch'] != epoch]
        self._save_metadata()
    
    def _load_metadata(self) -> None:
        
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
    
    def _save_metadata(self) -> None:
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)