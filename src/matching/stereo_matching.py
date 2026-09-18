"""
Stereo Matching Module

This module computes disparity maps from rectified stereo image pairs
using various stereo matching algorithms (SGBM, BM, etc.).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class MatchingAlgorithm(Enum):
    """Supported stereo matching algorithms."""
    SGBM = "sgbm"
    BM = "bm"
    SGBM_3WAY = "sgbm_3way"


@dataclass
class SGBMParams:
    """Parameters for Semi-Global Block Matching."""
    min_disparity: int = 0
    num_disparities: int = 64
    block_size: int = 5
    P1: int = 0
    P2: int = 0
    disp12_max_diff: int = 1
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 32
    pre_filter_cap: int = 63
    mode: int = cv2.STEREO_SGBM_MODE_SGBM
    
    def __post_init__(self):
        # Auto-compute P1, P2 if not set
        if self.P1 == 0:
            self.P1 = 8 * 3 * self.block_size ** 2
        if self.P2 == 0:
            self.P2 = 32 * 3 * self.block_size ** 2
        
        # Ensure num_disparities is divisible by 16
        if self.num_disparities % 16 != 0:
            self.num_disparities = ((self.num_disparities // 16) + 1) * 16
        
        # Ensure block_size is odd
        if self.block_size % 2 == 0:
            self.block_size += 1


@dataclass
class BMParams:
    """Parameters for Block Matching (BM)."""
    num_disparities: int = 64
    block_size: int = 15
    pre_filter_type: int = cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE
    pre_filter_size: int = 9
    pre_filter_cap: int = 31
    texture_threshold: int = 10
    uniqueness_ratio: int = 15
    speckle_window_size: int = 100
    speckle_range: int = 32
    disp12_max_diff: int = 1
    min_disparity: int = 0
    
    def __post_init__(self):
        if self.num_disparities % 16 != 0:
            self.num_disparities = ((self.num_disparities // 16) + 1) * 16
        if self.block_size % 2 == 0:
            self.block_size += 1
        if self.block_size < 5:
            self.block_size = 5
        if self.block_size > 255:
            self.block_size = 255


@dataclass
class DisparityResult:
    """Result of stereo matching."""
    disparity: np.ndarray
    disparity_normalized: np.ndarray
    disparity_color: np.ndarray
    algorithm: str
    parameters: Dict[str, Any]
    compute_time_ms: float
    valid_roi: Tuple[int, int, int, int]


def create_sgbm_matcher(params: SGBMParams) -> cv2.StereoSGBM:
    """Create SGBM stereo matcher with given parameters."""
    return cv2.StereoSGBM_create(
        minDisparity=params.min_disparity,
        numDisparities=params.num_disparities,
        blockSize=params.block_size,
        P1=params.P1,
        P2=params.P2,
        disp12MaxDiff=params.disp12_max_diff,
        uniquenessRatio=params.uniqueness_ratio,
        speckleWindowSize=params.speckle_window_size,
        speckleRange=params.speckle_range,
        preFilterCap=params.pre_filter_cap,
        mode=params.mode
    )


def create_bm_matcher(params: BMParams) -> cv2.StereoBM:
    """Create BM stereo matcher with given parameters."""
    matcher = cv2.StereoBM_create(
        numDisparities=params.num_disparities,
        blockSize=params.block_size
    )
    matcher.setPreFilterType(params.pre_filter_type)
    matcher.setPreFilterSize(params.pre_filter_size)
    matcher.setPreFilterCap(params.pre_filter_cap)
    matcher.setTextureThreshold(params.texture_threshold)
    matcher.setUniquenessRatio(params.uniqueness_ratio)
    matcher.setSpeckleWindowSize(params.speckle_window_size)
    matcher.setSpeckleRange(params.speckle_range)
    matcher.setDisp12MaxDiff(params.disp12_max_diff)
    matcher.setMinDisparity(params.min_disparity)
    return matcher


def create_sgbm_3way_matcher(params: SGBMParams) -> Tuple[cv2.StereoSGBM, cv2.StereoSGBM]:
    """Create left and right SGBM matchers for 3-way matching (left-right consistency)."""
    left_matcher = create_sgbm_matcher(params)
    
    # Create right matcher with same parameters
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    
    # Create WLS filter for disparity refinement
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)
    
    return left_matcher, right_matcher, wls_filter


def compute_disparity_sgbm(left: np.ndarray, 
                           right: np.ndarray,
                           params: SGBMParams) -> DisparityResult:
    """
    Compute disparity using SGBM algorithm.
    
    Args:
        left: Left rectified image (grayscale)
        right: Right rectified image (grayscale)
        params: SGBM parameters
        
    Returns:
        DisparityResult
    """
    import time
    start = time.time()
    
    matcher = create_sgbm_matcher(params)
    disparity = matcher.compute(left, right).astype(np.float32) / 16.0
    
    compute_time = (time.time() - start) * 1000
    
    return _postprocess_disparity(disparity, left.shape, params, "SGBM", compute_time)


def compute_disparity_bm(left: np.ndarray,
                         right: np.ndarray,
                         params: BMParams) -> DisparityResult:
    """
    Compute disparity using BM algorithm.
    
    Args:
        left: Left rectified image (grayscale)
        right: Right rectified image (grayscale)
        params: BM parameters
        
    Returns:
        DisparityResult
    """
    import time
    start = time.time()
    
    matcher = create_bm_matcher(params)
    disparity = matcher.compute(left, right).astype(np.float32) / 16.0
    
    compute_time = (time.time() - start) * 1000
    
    return _postprocess_disparity(disparity, left.shape, params, "BM", compute_time)


def compute_disparity_sgbm_3way(left: np.ndarray,
                                 right: np.ndarray,
                                 params: SGBMParams) -> DisparityResult:
    """
    Compute disparity using SGBM with left-right consistency check and WLS filtering.
    
    Args:
        left: Left rectified image (grayscale)
        right: Right rectified image (grayscale)
        params: SGBM parameters
        
    Returns:
        DisparityResult
    """
    import time
    start = time.time()
    
    try:
        left_matcher, right_matcher, wls_filter = create_sgbm_3way_matcher(params)
        
        # Compute left and right disparities
        disp_left = left_matcher.compute(left, right).astype(np.float32) / 16.0
        disp_right = right_matcher.compute(right, left).astype(np.float32) / 16.0
        
        # Filter with WLS
        disparity = wls_filter.filter(disp_left, left, None, disp_right)
        
        compute_time = (time.time() - start) * 1000
        
        return _postprocess_disparity(disparity, left.shape, params, "SGBM_3WAY", compute_time)
    except AttributeError:
        # ximgproc not available, fall back to regular SGBM
        print("Warning: cv2.ximgproc not available, falling back to SGBM")
        return compute_disparity_sgbm(left, right, params)


def _postprocess_disparity(disparity: np.ndarray,
                           image_shape: Tuple[int, int],
                           params: Any,
                           algorithm: str,
                           compute_time: float) -> DisparityResult:
    """Post-process disparity map: normalize, colorize, compute valid ROI."""
    h, w = image_shape
    
    # Create mask for valid disparities
    valid_mask = (disparity > params.min_disparity) & (disparity < params.min_disparity + params.num_disparities)
    
    # Normalize for visualization
    disparity_vis = np.zeros_like(disparity)
    if np.any(valid_mask):
        min_disp = np.min(disparity[valid_mask])
        max_disp = np.max(disparity[valid_mask])
        if max_disp > min_disp:
            disparity_vis[valid_mask] = 255 * (disparity[valid_mask] - min_disp) / (max_disp - min_disp)
    
    disparity_normalized = disparity_vis.astype(np.uint8)
    
    # Color map
    disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    disparity_color[~valid_mask] = [0, 0, 0]
    
    # Valid ROI
    if np.any(valid_mask):
        y_coords, x_coords = np.where(valid_mask)
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        valid_roi = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    else:
        valid_roi = (0, 0, w, h)
    
    # Store parameters as dict
    param_dict = {}
    for key in dir(params):
        if not key.startswith('_'):
            val = getattr(params, key)
            if not callable(val):
                param_dict[key] = val
    
    return DisparityResult(
        disparity=disparity,
        disparity_normalized=disparity_normalized,
        disparity_color=disparity_color,
        algorithm=algorithm,
        parameters=param_dict,
        compute_time_ms=compute_time,
        valid_roi=valid_roi
    )


def compute_disparity(left: np.ndarray,
                      right: np.ndarray,
                      algorithm: MatchingAlgorithm = MatchingAlgorithm.SGBM,
                      params: Optional[SGBMParams] = None,
                      bm_params: Optional[BMParams] = None) -> DisparityResult:
    """
    Compute disparity map from rectified stereo pair.
    
    Args:
        left: Left rectified image (grayscale)
        right: Right rectified image (grayscale)
        algorithm: Matching algorithm to use
        params: SGBM parameters (used for SGBM and SGBM_3WAY)
        bm_params: BM parameters (used for BM)
        
    Returns:
        DisparityResult
    """
    # Ensure grayscale
    if len(left.shape) == 3:
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    if len(right.shape) == 3:
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    
    # Ensure same size
    if left.shape != right.shape:
        raise ValueError(f"Image size mismatch: left {left.shape}, right {right.shape}")
    
    if algorithm == MatchingAlgorithm.SGBM:
        if params is None:
            params = SGBMParams()
        return compute_disparity_sgbm(left, right, params)
    
    elif algorithm == MatchingAlgorithm.BM:
        if bm_params is None:
            bm_params = BMParams()
        return compute_disparity_bm(left, right, bm_params)
    
    elif algorithm == MatchingAlgorithm.SGBM_3WAY:
        if params is None:
            params = SGBMParams()
        return compute_disparity_sgbm_3way(left, right, params)
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def filter_disparity(disparity: np.ndarray,
                     method: str = "median",
                     kernel_size: int = 5,
                     speckle_window: int = 100,
                     speckle_range: int = 32) -> np.ndarray:
    """
    Filter disparity map to remove noise.
    
    Args:
        disparity: Raw disparity map
        method: Filter method ('median', 'bilateral', 'speckle')
        kernel_size: Kernel size for median/bilateral
        speckle_window: Speckle filter window size
        speckle_range: Speckle filter range
        
    Returns:
        Filtered disparity map
    """
    filtered = disparity.copy()
    
    if method == "median":
        # Median filter preserves edges better
        valid = filtered > 0
        filtered[valid] = cv2.medianBlur(filtered[valid].astype(np.float32), kernel_size)
    
    elif method == "bilateral":
        # Bilateral filter for edge-preserving smoothing
        valid = filtered > 0
        filtered[valid] = cv2.bilateralFilter(
            filtered[valid].astype(np.float32), 
            kernel_size, 75, 75
        )
    
    elif method == "speckle":
        # Remove small speckles
        # Convert to 16-bit for OpenCV filter
        disp_16 = (filtered * 16).astype(np.int16)
        cv2.filterSpeckles(disp_16, 0, speckle_window, speckle_range)
        filtered = disp_16.astype(np.float32) / 16.0
    
    return filtered


def save_disparity_result(result: DisparityResult, output_dir: str, prefix: str = "disparity"):
    """Save disparity result to files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save raw disparity as 16-bit PNG (x16 scale for OpenCV compatibility)
    disp_16 = (result.disparity * 16).astype(np.int16)
    cv2.imwrite(str(out_path / f"{prefix}_raw.png"), disp_16)
    
    # Save normalized
    cv2.imwrite(str(out_path / f"{prefix}_normalized.png"), result.disparity_normalized)
    
    # Save color
    cv2.imwrite(str(out_path / f"{prefix}_color.png"), result.disparity_color)
    
    # Save parameters
    import json
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return tuple(convert(x) for x in obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj
    
    with open(out_path / f"{prefix}_params.json", 'w') as f:
        json.dump({
            "algorithm": result.algorithm,
            "parameters": convert(result.parameters),
            "compute_time_ms": convert(result.compute_time_ms),
            "valid_roi": convert(result.valid_roi)
        }, f, indent=2)


def load_disparity_raw(filepath: str) -> np.ndarray:
    """Load raw disparity from 16-bit PNG."""
    disp_16 = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if disp_16 is None:
        raise ValueError(f"Could not load disparity from {filepath}")
    return disp_16.astype(np.float32) / 16.0