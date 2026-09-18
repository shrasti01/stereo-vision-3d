"""
I/O Utilities Module

Utility functions for file handling, directory management, and data persistence.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Any
import json
import yaml


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if needed."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(directory: str, extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) -> List[Path]:
    """List all image files in a directory."""
    path = Path(directory)
    if not path.exists():
        return []
    
    images = []
    for ext in extensions:
        images.extend(path.glob(f'*{ext}'))
        images.extend(path.glob(f'*{ext.upper()}'))
    
    return sorted(images)


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_yaml(filepath: str) -> Any:
    """Load YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Any, filepath: str):
    """Save data to YAML file."""
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def save_numpy_array(array: np.ndarray, filepath: str):
    """Save numpy array to .npy file."""
    np.save(filepath, array)


def load_numpy_array(filepath: str) -> np.ndarray:
    """Load numpy array from .npy file."""
    return np.load(filepath)


def save_image(image: np.ndarray, filepath: str, create_dirs: bool = True):
    """Save image with optional directory creation."""
    if create_dirs:
        ensure_dir(str(Path(filepath).parent))
    cv2.imwrite(filepath, image)


def load_image(filepath: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Load image with error handling."""
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}")
        return None
    
    image = cv2.imread(str(path), flags)
    if image is None:
        print(f"Error: Failed to load image: {filepath}")
        return None
    
    return image


def get_image_info(image: np.ndarray) -> dict:
    """Get basic image information."""
    return {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "size": image.size,
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "mean": float(np.mean(image)),
        "std": float(np.std(image))
    }


def create_output_structure(base_dir: str) -> dict:
    """Create standard output directory structure."""
    base = Path(base_dir)
    dirs = {
        "calibration": base / "calibration",
        "rectification": base / "rectification",
        "disparity": base / "disparity",
        "depth": base / "depth",
        "measurements": base / "measurements",
        "evaluation": base / "evaluation",
        "visualizations": base / "visualizations"
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    return {k: str(v) for k, v in dirs.items()}


def find_calibration_pairs(left_dir: str, right_dir: str) -> List[Tuple[str, str]]:
    """Find matching calibration image pairs."""
    left_path = Path(left_dir)
    right_path = Path(right_dir)
    
    if not left_path.exists() or not right_path.exists():
        return []
    
    left_images = {}
    for ext in ('.png', '.jpg', '.jpeg'):
        for img in left_path.glob(f'*{ext}'):
            left_images[img.stem] = img
    
    pairs = []
    for ext in ('.png', '.jpg', '.jpeg'):
        for img in right_path.glob(f'*{ext}'):
            if img.stem in left_images:
                pairs.append((str(left_images[img.stem]), str(img)))
    
    return sorted(pairs)


def find_stereo_pairs(stereo_dir: str) -> List[Tuple[str, str]]:
    """Find matching stereo image pairs in left/right subdirectories."""
    stereo_path = Path(stereo_dir)
    left_dir = stereo_path / "left"
    right_dir = stereo_path / "right"
    
    return find_calibration_pairs(str(left_dir), str(right_dir))


class ConfigManager:
    """Configuration manager for the stereo vision system."""
    
    DEFAULT_CONFIG = {
        "calibration": {
            "pattern_size": [9, 6],
            "square_size_mm": 25.0,
            "min_valid_images": 10
        },
        "preprocessing": {
            "apply_clahe": True,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid": [8, 8],
            "apply_denoise": False,
            "denoise_strength": 10,
            "max_dimension": 1280
        },
        "rectification": {
            "alpha": 0.0
        },
        "matching": {
            "algorithm": "sgbm",
            "sgbm": {
                "min_disparity": 0,
                "num_disparities": 64,
                "block_size": 5,
                "uniqueness_ratio": 10,
                "speckle_window_size": 100,
                "speckle_range": 32
            },
            "bm": {
                "num_disparities": 64,
                "block_size": 15
            }
        },
        "depth": {
            "min_disparity": 1.0,
            "max_depth_mm": 10000.0
        },
        "measurement": {
            "min_roi_area": 100,
            "depth_tolerance_mm": 50.0
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and Path(config_path).exists():
            self.load(config_path)
    
    def load(self, filepath: str):
        """Load configuration from file."""
        loaded = load_yaml(filepath)
        if loaded:
            self._merge_config(self.config, loaded)
    
    def save(self, filepath: str):
        """Save configuration to file."""
        save_yaml(self.config, filepath)
    
    def _merge_config(self, base: dict, override: dict):
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'matching.sgbm.block_size')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        target = self.config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value