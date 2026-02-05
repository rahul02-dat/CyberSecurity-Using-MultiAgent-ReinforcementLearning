import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int) -> None:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_random_state() -> dict:
    
    return {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def set_random_state(state: dict) -> None:
    
    random.setstate(state['random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


class SeedManager:
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.seed_counter = 0
    
    def get_next_seed(self) -> int:
        
        seed = self.base_seed + self.seed_counter
        self.seed_counter += 1
        return seed
    
    def reset(self) -> None:
        
        self.seed_counter = 0