import numpy as np
import torch
from typing import Dict, Any
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import gymnasium as gym


class ResponseRLModel(TorchModelV2, torch.nn.Module):
    
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        torch.nn.Module.__init__(self)
        
        hidden_dim = model_config.get("fcnet_hiddens", [64, 64])[0]
        
        self.shared_layers = torch.nn.Sequential(
            torch.nn.Linear(6, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        
        self.action_logits = torch.nn.Linear(hidden_dim, 3)
        self.duration_logits = torch.nn.Linear(hidden_dim, 3)
        
        self.value_branch = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        
        self._features = None
        
    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: list,
        seq_lens: TensorType,
    ) -> tuple[TensorType, list]:
        obs = input_dict["obs"].float()
        
        self._features = self.shared_layers(obs)
        
        action_logits = self.action_logits(self._features)
        duration_logits = self.duration_logits(self._features)
        
        logits = torch.cat([action_logits, duration_logits], dim=-1)
        
        return logits, state
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None
        return torch.squeeze(self.value_branch(self._features), -1)