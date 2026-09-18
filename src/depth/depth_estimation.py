"""
Depth Estimation Module

This module converts disparity maps to depth/distance maps using the
stereo depth equation: Z = (f * B) / d

Where:
- Z = depth/distance (in same units as baseline)
- f = focal length in pixels
- B = stereo baseline (distance between cameras)
- d = disparity in pixels
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from src.calibration.camera_calibration import StereoCalibrationResult
from src.matching.stereo_matching import DisparityResult


@dataclass
class DepthResult:
    """Result of depth estimation."""
    depth_map: np.ndarray           # Depth in mm (or same unit as baseline)
    depth_normalized: np.ndarray    # Normalized 0-255 for visualization
    depth_color: np.ndarray         # Color-mapped visualization
    focal_length_px: float          # Focal length in pixels
    baseline_mm: float              # Baseline in mm
    valid_mask: np.ndarray          # Boolean mask of valid depth pixels
    stats: Dict[str, float]         # Statistics


def extract_focal_length(camera_matrix: np.ndarray) -> float:
    """
    Extract focal length from camera matrix.
    
    Args:
        camera_matrix: 3x3 camera matrix
        
    Returns:
        Focal length in pixels (average of fx and fy)
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    return (fx + fy) / 2.0


def extract_baseline(stereo_calib: StereoCalibrationResult) -> float:
    """
    Extract baseline (translation between cameras) from stereo calibration.
    
    Args:
        stereo_calib: Stereo calibration result
        
    Returns:
        Baseline magnitude in mm
    """
    if stereo_calib.T is None:
        raise ValueError("Stereo calibration missing translation vector")
    
    # T is translation from left to right camera
    # Baseline is the magnitude of translation
    baseline = np.linalg.norm(stereo_calib.T)
    return float(baseline)


def disparity_to_depth(disparity: np.ndarray,
                       focal_length_px: float,
                       baseline_mm: float,
                       min_disparity: float = 1.0,
                       max_depth_mm: float = 10000.0) -> np.ndarray:
    """
    Convert disparity map to depth map using Z = (f * B) / d.
    
    Args:
        disparity: Disparity map (float, pixels)
        focal_length_px: Focal length in pixels
        baseline_mm: Baseline in mm
        min_disparity: Minimum valid disparity (avoid division by zero)
        max_depth_mm: Maximum depth to clamp (for visualization)
        
    Returns:
        Depth map in mm
    """
    depth = np.zeros_like(disparity, dtype=np.float32)
    
    # Valid disparity mask
    valid = (disparity > min_disparity) & np.isfinite(disparity)
    
    # Apply depth formula: Z = (f * B) / d
    depth[valid] = (focal_length_px * baseline_mm) / disparity[valid]
    
    # Clamp to max depth
    depth = np.clip(depth, 0, max_depth_mm)
    
    return depth


def depth_to_disparity(depth: np.ndarray,
                       focal_length_px: float,
                       baseline_mm: float) -> np.ndarray:
    """
    Convert depth map back to disparity (inverse of disparity_to_depth).
    
    Args:
        depth: Depth map in mm
        focal_length_px: Focal length in pixels
        baseline_mm: Baseline in mm
        
    Returns:
        Disparity map in pixels
    """
    disparity = np.zeros_like(depth, dtype=np.float32)
    valid = depth > 0
    disparity[valid] = (focal_length_px * baseline_mm) / depth[valid]
    return disparity


def estimate_depth_from_disparity(disparity_result: DisparityResult,
                                  stereo_calib: StereoCalibrationResult,
                                  min_disparity: float = 1.0,
                                  max_depth_mm: float = 10000.0) -> DepthResult:
    """
    Complete depth estimation from disparity result and stereo calibration.
    
    Args:
        disparity_result: Result from stereo matching
        stereo_calib: Stereo calibration result
        min_disparity: Minimum valid disparity
        max_depth_mm: Maximum depth for clamping
        
    Returns:
        DepthResult with depth map and statistics
    """
    # Extract camera parameters
    focal_length = extract_focal_length(stereo_calib.left_camera_matrix)
    baseline = extract_baseline(stereo_calib)
    
    # Convert disparity to depth
    depth_map = disparity_to_depth(
        disparity_result.disparity,
        focal_length,
        baseline,
        min_disparity,
        max_depth_mm
    )
    
    # Valid mask
    valid_mask = (disparity_result.disparity > min_disparity) & np.isfinite(disparity_result.disparity)
    
    # Normalize for visualization
    depth_vis = np.zeros_like(depth_map)
    if np.any(valid_mask):
        valid_depths = depth_map[valid_mask]
        min_d = np.min(valid_depths)
        max_d = np.max(valid_depths)
        if max_d > min_d:
            depth_vis[valid_mask] = 255 * (depth_map[valid_mask] - min_d) / (max_d - min_d)
    
    depth_normalized = depth_vis.astype(np.uint8)
    
    # Color map
    depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    depth_color[~valid_mask] = [0, 0, 0]
    
    # Statistics
    valid_depths = depth_map[valid_mask]
    stats = {}
    if len(valid_depths) > 0:
        stats = {
            "min_depth_mm": float(np.min(valid_depths)),
            "max_depth_mm": float(np.max(valid_depths)),
            "mean_depth_mm": float(np.mean(valid_depths)),
            "median_depth_mm": float(np.median(valid_depths)),
            "std_depth_mm": float(np.std(valid_depths)),
            "valid_pixel_count": int(np.sum(valid_mask)),
            "total_pixels": int(depth_map.size),
            "valid_ratio": float(np.sum(valid_mask)) / depth_map.size
        }
    
    return DepthResult(
        depth_map=depth_map,
        depth_normalized=depth_normalized,
        depth_color=depth_color,
        focal_length_px=focal_length,
        baseline_mm=baseline,
        valid_mask=valid_mask,
        stats=stats
    )


def estimate_depth_from_Q(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Alternative depth estimation using reprojection matrix Q.
    
    The Q matrix from stereoRectify encodes the relationship:
    [X, Y, Z, W]^T = Q * [x, y, disparity, 1]^T
    
    Args:
        disparity: Disparity map
        Q: 4x4 reprojection matrix from stereoRectify
        
    Returns:
        Depth map (Z coordinate) in mm
    """
    h, w = disparity.shape
    
    # Create coordinate grids
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    # Stack coordinates: [x, y, disparity, 1]
    points_4d = np.stack([
        x_coords.astype(np.float32),
        y_coords.astype(np.float32),
        disparity.astype(np.float32),
        np.ones_like(disparity, dtype=np.float32)
    ], axis=-1)  # Shape: (h, w, 4)
    
    # Apply Q matrix
    # Q is 4x4, points are (h, w, 4) -> result (h, w, 4)
    points_3d = cv2.transform(points_4d.reshape(-1, 4), Q.T).reshape(h, w, 4)
    
    # Extract Z (depth) - divide by W for homogeneous coordinates
    # Q matrix convention: [X, Y, Z, W] -> Z/W is actual depth
    depth = points_3d[:, :, 2] / points_3d[:, :, 3]
    
    return depth


def compute_depth_error(estimated_depth: np.ndarray,
                        ground_truth_depth: np.ndarray,
                        valid_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute depth estimation errors against ground truth.
    
    Args:
        estimated_depth: Estimated depth map
        ground_truth_depth: Ground truth depth map
        valid_mask: Optional mask for valid pixels
        
    Returns:
        Dictionary of error metrics
    """
    if valid_mask is None:
        valid_mask = (estimated_depth > 0) & (ground_truth_depth > 0) & \
                     np.isfinite(estimated_depth) & np.isfinite(ground_truth_depth)
    
    est_valid = estimated_depth[valid_mask]
    gt_valid = ground_truth_depth[valid_mask]
    
    if len(est_valid) == 0:
        return {"error": "No valid pixels for comparison"}
    
    # Absolute error
    abs_error = np.abs(est_valid - gt_valid)
    
    # Relative error
    rel_error = abs_error / (gt_valid + 1e-6)
    
    # Metrics
    metrics = {
        "mae_mm": float(np.mean(abs_error)),           # Mean Absolute Error
        "rmse_mm": float(np.sqrt(np.mean((est_valid - gt_valid) ** 2))),  # RMSE
        "median_abs_error_mm": float(np.median(abs_error)),
        "mean_rel_error": float(np.mean(rel_error)),
        "median_rel_error": float(np.median(rel_error)),
        "max_abs_error_mm": float(np.max(abs_error)),
        "valid_pixels": int(np.sum(valid_mask)),
        "accuracy_1mm": float(np.sum(abs_error < 1.0) / len(abs_error) * 100),
        "accuracy_5mm": float(np.sum(abs_error < 5.0) / len(abs_error) * 100),
        "accuracy_10mm": float(np.sum(abs_error < 10.0) / len(abs_error) * 100),
        "accuracy_5pct": float(np.sum(rel_error < 0.05) / len(rel_error) * 100),
    }
    
    return metrics


def save_depth_result(result: DepthResult, output_dir: str, prefix: str = "depth"):
    """Save depth result to files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save raw depth as 32-bit float (in mm)
    np.save(str(out_path / f"{prefix}_raw.npy"), result.depth_map)
    
    # Save normalized
    cv2.imwrite(str(out_path / f"{prefix}_normalized.png"), result.depth_normalized)
    
    # Save color
    cv2.imwrite(str(out_path / f"{prefix}_color.png"), result.depth_color)
    
    # Save metadata
    import json
    meta = {
        "focal_length_px": result.focal_length_px,
        "baseline_mm": result.baseline_mm,
        "stats": result.stats
    }
    with open(out_path / f"{prefix}_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)


def load_depth_raw(filepath: str) -> np.ndarray:
    """Load raw depth map from .npy file."""
    return np.load(filepath)