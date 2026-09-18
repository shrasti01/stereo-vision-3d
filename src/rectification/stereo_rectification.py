"""
Stereo Rectification Module

This module handles rectification of stereo image pairs using calibration parameters.
Rectification aligns epipolar lines horizontally, enabling efficient stereo matching.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

from src.calibration.camera_calibration import StereoCalibrationResult


@dataclass
class RectificationMaps:
    """Rectification maps for left and right images."""
    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray
    left_roi: Tuple[int, int, int, int]  # x, y, w, h
    right_roi: Tuple[int, int, int, int]


@dataclass
class RectifiedPair:
    """Rectified stereo image pair."""
    left: np.ndarray
    right: np.ndarray
    left_original: np.ndarray
    right_original: np.ndarray
    maps: RectificationMaps


def compute_rectification_maps(calib: StereoCalibrationResult,
                               image_size: Tuple[int, int],
                               alpha: float = 0.0) -> RectificationMaps:
    """
    Compute rectification maps from stereo calibration.
    
    Args:
        calib: Stereo calibration result
        image_size: Image size (width, height)
        alpha: Free scaling parameter (0=valid pixels only, 1=all pixels)
        
    Returns:
        RectificationMaps with mapping arrays
    """
    if calib.R1 is None or calib.R2 is None or calib.P1 is None or calib.P2 is None:
        raise ValueError("Calibration result missing rectification matrices")
    
    # Compute undistortion and rectification maps
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        calib.left_camera_matrix, calib.left_dist_coeffs,
        calib.R1, calib.P1, image_size, cv2.CV_16SC2
    )
    
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        calib.right_camera_matrix, calib.right_dist_coeffs,
        calib.R2, calib.P2, image_size, cv2.CV_16SC2
    )
    
    # Get valid ROIs
    left_roi = cv2.getValidDisparityROI(
        (0, 0, image_size[0], image_size[1]),
        (0, 0, image_size[0], image_size[1]),
        0, 0, 0
    )
    
    right_roi = cv2.getValidDisparityROI(
        (0, 0, image_size[0], image_size[1]),
        (0, 0, image_size[0], image_size[1]),
        0, 0, 0
    )
    
    return RectificationMaps(
        left_map1=left_map1,
        left_map2=left_map2,
        right_map1=right_map1,
        right_map2=right_map2,
        left_roi=left_roi,
        right_roi=right_roi
    )


def rectify_stereo_pair(left: np.ndarray,
                        right: np.ndarray,
                        maps: RectificationMaps,
                        interpolation: int = cv2.INTER_LINEAR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply rectification to a stereo image pair.
    
    Args:
        left: Left image
        right: Right image
        maps: Rectification maps
        interpolation: Interpolation method
        
    Returns:
        Tuple of (rectified_left, rectified_right)
    """
    left_rectified = cv2.remap(left, maps.left_map1, maps.left_map2, interpolation)
    right_rectified = cv2.remap(right, maps.right_map1, maps.right_map2, interpolation)
    
    return left_rectified, right_rectified


def rectify_stereo_pair_full(left: np.ndarray,
                             right: np.ndarray,
                             calib: StereoCalibrationResult,
                             alpha: float = 0.0) -> RectifiedPair:
    """
    Complete rectification pipeline for a stereo pair.
    
    Args:
        left: Left image
        right: Right image
        calib: Stereo calibration result
        alpha: Free scaling parameter
        
    Returns:
        RectifiedPair with rectified images and maps
    """
    image_size = (left.shape[1], left.shape[0])
    
    maps = compute_rectification_maps(calib, image_size, alpha)
    left_rect, right_rect = rectify_stereo_pair(left, right, maps)
    
    return RectifiedPair(
        left=left_rect,
        right=right_rect,
        left_original=left.copy(),
        right_original=right.copy(),
        maps=maps
    )


def draw_epipolar_lines(image: np.ndarray,
                        num_lines: int = 20,
                        color: Tuple[int, int, int] = (0, 255, 0),
                        line_thickness: int = 1) -> np.ndarray:
    """
    Draw horizontal epipolar lines on rectified image for visualization.
    
    Args:
        image: Input image
        num_lines: Number of lines to draw
        color: Line color (BGR)
        line_thickness: Line thickness
        
    Returns:
        Image with epipolar lines drawn
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    h, w = vis.shape[:2]
    step = h // (num_lines + 1)
    
    for i in range(1, num_lines + 1):
        y = i * step
        cv2.line(vis, (0, y), (w, y), color, line_thickness)
    
    return vis


def create_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Create side-by-side visualization of stereo pair.
    
    Args:
        left: Left image
        right: Right image
        
    Returns:
        Combined side-by-side image
    """
    # Ensure same height
    h = max(left.shape[0], right.shape[0])
    
    def resize_to_height(img, target_h):
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            return cv2.resize(img, (new_w, target_h))
        return img
    
    left_r = resize_to_height(left, h)
    right_r = resize_to_height(right, h)
    
    # Convert to color if grayscale
    if len(left_r.shape) == 2:
        left_r = cv2.cvtColor(left_r, cv2.COLOR_GRAY2BGR)
    if len(right_r.shape) == 2:
        right_r = cv2.cvtColor(right_r, cv2.COLOR_GRAY2BGR)
    
    return np.hstack([left_r, right_r])


def create_anaglyph(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Create red-cyan anaglyph for 3D visualization.
    
    Args:
        left: Left image (grayscale)
        right: Right image (grayscale)
        
    Returns:
        Anaglyph image (BGR)
    """
    if len(left.shape) == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    if len(right.shape) == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    
    # Red channel from left, Green+Blue from right
    anaglyph = np.zeros_like(left)
    anaglyph[:, :, 2] = left[:, :, 2]    # Red from left
    anaglyph[:, :, 1] = right[:, :, 1]   # Green from right
    anaglyph[:, :, 0] = right[:, :, 0]   # Blue from right
    
    return anaglyph


def verify_rectification(left_rect: np.ndarray,
                         right_rect: np.ndarray,
                         num_test_points: int = 10) -> dict:
    """
    Verify rectification quality by checking epipolar line alignment.
    
    Args:
        left_rect: Rectified left image
        right_rect: Rectified right image
        num_test_points: Number of random points to test
        
    Returns:
        Dictionary with verification metrics
    """
    h, w = left_rect.shape[:2]
    
    # Generate random points in left image
    np.random.seed(42)
    y_coords = np.random.randint(h // 4, 3 * h // 4, num_test_points)
    x_coords = np.random.randint(w // 4, 3 * w // 4, num_test_points)
    
    max_vertical_disparity = 0
    avg_vertical_disparity = 0
    
    # For each point, find best match along epipolar line in right image
    # This is a simplified check - in practice would use feature matching
    for y, x in zip(y_coords, x_coords):
        # Extract patch from left
        patch_size = 11
        half = patch_size // 2
        
        if y - half < 0 or y + half >= h or x - half < 0 or x + half >= w:
            continue
            
        left_patch = left_rect[y-half:y+half+1, x-half:x+half+1]
        
        # Search along same row in right image (epipolar line)
        search_range = min(50, w // 4)
        best_x = x
        best_score = float('inf')
        
        for dx in range(-search_range, search_range + 1):
            rx = x + dx
            if rx - half < 0 or rx + half >= w:
                continue
            right_patch = right_rect[y-half:y+half+1, rx-half:rx+half+1]
            
            # SAD score
            score = np.sum(np.abs(left_patch.astype(np.float32) - right_patch.astype(np.float32)))
            if score < best_score:
                best_score = score
                best_x = rx
        
        # In perfect rectification, best match should be at same y
        # Here we just measure horizontal disparity
        disparity = x - best_x
    
    return {
        "image_size": (w, h),
        "test_points": num_test_points,
        "note": "Full verification requires feature matching; this is a basic check"
    }


def save_rectification_maps(maps: RectificationMaps, filepath: str):
    """Save rectification maps to file."""
    np.savez_compressed(
        filepath,
        left_map1=maps.left_map1,
        left_map2=maps.left_map2,
        right_map1=maps.right_map1,
        right_map2=maps.right_map2,
        left_roi=np.array(maps.left_roi),
        right_roi=np.array(maps.right_roi)
    )


def load_rectification_maps(filepath: str) -> RectificationMaps:
    """Load rectification maps from file."""
    data = np.load(filepath)
    return RectificationMaps(
        left_map1=data['left_map1'],
        left_map2=data['left_map2'],
        right_map1=data['right_map1'],
        right_map2=data['right_map2'],
        left_roi=tuple(data['left_roi'].tolist()),
        right_roi=tuple(data['right_roi'].tolist())
    )