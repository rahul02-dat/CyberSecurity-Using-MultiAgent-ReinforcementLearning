from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import Any


class MARLMetricsCallback(DefaultCallbacks):
    """Callback for tracking MARL metrics in the new RLlib API stack."""
    
    def on_episode_start(
        self,
        *,
        algorithm: Any = None,
        episode: Any = None,
        env_runner: Any = None,
        **kwargs
    ) -> None:
        """Called at the start of an episode."""
        pass
    
    def on_episode_step(
        self,
        *,
        algorithm: Any = None,
        episode: Any = None,
        env_runner: Any = None,
        **kwargs
    ) -> None:
        """Called at each step of an episode."""
        pass
    
    def on_episode_end(
        self,
        *,
        algorithm: Any = None,
        episode: Any = None,
        env_runner: Any = None,
        **kwargs
    ) -> None:
        """Called at the end of an episode."""
        pass

