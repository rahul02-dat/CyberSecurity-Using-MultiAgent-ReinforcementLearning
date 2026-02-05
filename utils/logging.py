import os
import json
import logging
from typing import Any, Optional
from datetime import datetime


class Logger:
    
    def __init__(self, log_dir: str = "output/logs", experiment_name: str = "default"):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(
            log_dir, 
            f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(experiment_name)
        
        self.metrics_history: list[dict[str, Any]] = []
    
    def info(self, message: str) -> None:
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        self.logger.error(message)
    
    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        
        self.metrics_history.append({
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        metrics_str = ', '.join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                 for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metrics_str}")
    
    def save_metrics(self, filepath: Optional[str] = None) -> str:
        
        if filepath is None:
            filepath = os.path.join(
                self.log_dir,
                f"{self.experiment_name}_metrics.json"
            )
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        return filepath
    
    def load_metrics(self, filepath: str) -> None:
        
        with open(filepath, 'r') as f:
            self.metrics_history = json.load(f)


def setup_directories() -> None:
    
    directories = [
        "output",
        "output/logs",
        "output/policies",
        "output/checkpoints",
        "performance/matrices",
        "performance/history"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        if directory == "performance/matrices":
            default_files = [
                "monitoring_metrics.json",
                "detection_metrics.json",
                "response_metrics.json",
                "global_metrics.json"
            ]
            
            for filename in default_files:
                filepath = os.path.join(directory, filename)
                if not os.path.exists(filepath):
                    with open(filepath, 'w') as f:
                        json.dump({}, f)