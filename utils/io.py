"""I/O utilities for loading/saving configs, checkpoints, and data.

Provides utilities for file operations and configuration management.
"""

from pathlib import Path
import yaml
import json
from typing import Dict, Any


def load_yaml(path: str) -> Dict:
    """Load YAML configuration file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Dictionary with config contents
        
    TODO: Implement YAML loading
    TODO: Support _base_ inheritance for nested configs
    """
    pass


def save_yaml(data: Dict, path: str):
    """Save dictionary as YAML file.
    
    TODO: Implement YAML saving
    """
    pass


def load_json(path: str) -> Any:
    """Load JSON file.
    
    TODO: Implement JSON loading with error handling
    """
    pass


def save_json(data: Any, path: str, indent: int = 2):
    """Save data as JSON file.
    
    TODO: Implement JSON saving
    """
    pass


def ensure_dir(path: str):
    """Create directory if it doesn't exist.
    
    TODO: Create parent directories as needed
    """
    pass
