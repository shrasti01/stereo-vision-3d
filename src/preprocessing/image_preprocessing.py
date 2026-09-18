"""
Image Validation and Preprocessing Module

This module handles loading, validating, and preprocessing stereo image pairs
before they enter the calibration and stereo matching pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ImagePair:
    """Container for a validated stereo image pair."""
    left: np.ndarray
    right: np.ndarray
    left_path: str
    right_path: str
    width: int
    height: int


@dataclass
class ValidationResult:
    """Result of image pair validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    image_pair: Optional[ImagePair] = None


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array, or None if failed
    """
    path = Path(image_path)
    if not path.exists():
        print(f"Error: File not found: {image_path}")
        return None
    
    image = cv2.imread(str(path))
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return None
    
    return image


def validate_image_pair(left_path: str, right_path: str) -> ValidationResult:
    """
    Validate a stereo image pair for processing.
    
    Checks:
    - Both files exist and can be loaded
    - Both images have same dimensions
    - Both images are valid (not empty, not corrupted)
    - Images are grayscale or can be converted
    
    Args:
        left_path: Path to left image
        right_path: Path to right image
        
    Returns:
        ValidationResult with validation status and image pair if valid
    """
    errors = []
    warnings = []
    
    left_img = load_image(left_path)
    right_img = load_image(right_path)
    
    if left_img is None:
        errors.append(f"Left image failed to load: {left_path}")
    if right_img is None:
        errors.append(f"Right image failed to load: {right_path}")
    
    if left_img is not None and right_img is not None:
        if left_img.shape != right_img.shape:
            errors.append(
                f"Image dimensions mismatch: "
                f"Left {left_img.shape} vs Right {right_img.shape}"
            )
        
        if left_img.size == 0 or right_img.size == 0:
            errors.append("One or both images are empty")
        
        h, w = left_img.shape[:2]
        if h < 100 or w < 100:
            warnings.append(f"Images are very small ({w}x{h}), results may be poor")
        
        if len(left_img.shape) == 3:
            warnings.append("Left image is color, will be converted to grayscale")
        if len(right_img.shape) == 3:
            warnings.append("Right image is color, will be converted to grayscale")
    
    is_valid = len(errors) == 0
    
    image_pair = None
    if is_valid and left_img is not None and right_img is not None:
        image_pair = ImagePair(
            left=left_img,
            right=right_img,
            left_path=left_path,
            right_path=right_path,
            width=left_img.shape[1],
            height=left_img.shape[0]
        )
    
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        image_pair=image_pair
    )


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if it's color.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def preprocess_image(image: np.ndarray, 
                     apply_clahe: bool = True,
                     clahe_clip_limit: float = 2.0,
                     clahe_tile_grid: Tuple[int, int] = (8, 8),
                     apply_denoise: bool = False,
                     denoise_strength: float = 10) -> np.ndarray:
    """
    Preprocess a single image for stereo matching.
    
    Args:
        image: Input image (grayscale)
        apply_clahe: Whether to apply CLAHE histogram equalization
        clahe_clip_limit: CLAHE clip limit
        clahe_tile_grid: CLAHE tile grid size
        apply_denoise: Whether to apply denoising
        denoise_strength: Denoising strength
        
    Returns:
        Preprocessed image
    """
    processed = image.copy()
    
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, 
                                tileGridSize=clahe_tile_grid)
        processed = clahe.apply(processed)
    
    if apply_denoise:
        processed = cv2.fastNlMeansDenoising(
            processed, None, denoise_strength, 7, 21
        )
    
    return processed


def preprocess_stereo_pair(left: np.ndarray, 
                           right: np.ndarray,
                           apply_clahe: bool = True,
                           apply_denoise: bool = False,
                           clahe_clip_limit: float = 2.0,
                           clahe_tile_grid: Tuple[int, int] = (8, 8),
                           denoise_strength: float = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a stereo image pair identically.
    
    Args:
        left: Left image
        right: Right image
        apply_clahe: Whether to apply CLAHE
        apply_denoise: Whether to apply denoising
        clahe_clip_limit: CLAHE clip limit
        clahe_tile_grid: CLAHE tile grid size
        denoise_strength: Denoising strength
        
    Returns:
        Tuple of (preprocessed_left, preprocessed_right)
    """
    left_gray = convert_to_grayscale(left)
    right_gray = convert_to_grayscale(right)
    
    left_processed = preprocess_image(left_gray, apply_clahe, clahe_clip_limit, clahe_tile_grid, apply_denoise, denoise_strength)
    right_processed = preprocess_image(right_gray, apply_clahe, clahe_clip_limit, clahe_tile_grid, apply_denoise, denoise_strength)
    
    return left_processed, right_processed


def resize_image_pair(left: np.ndarray, 
                      right: np.ndarray,
                      max_dimension: int = 1280) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Resize stereo pair maintaining aspect ratio if too large.
    
    Args:
        left: Left image
        right: Right image
        max_dimension: Maximum width or height
        
    Returns:
        Tuple of (resized_left, resized_right, scale_factor)
    """
    h, w = left.shape[:2]
    scale = 1.0
    
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        left_resized = cv2.resize(left, (new_w, new_h), interpolation=cv2.INTER_AREA)
        right_resized = cv2.resize(right, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return left_resized, right_resized, scale
    
    return left, right, scale


def load_and_preprocess_pair(left_path: str, 
                             right_path: str,
                             apply_clahe: bool = True,
                             apply_denoise: bool = False,
                             max_dimension: int = 1280) -> ValidationResult:
    """
    Complete pipeline: load, validate, and preprocess a stereo pair.
    
    Args:
        left_path: Path to left image
        right_path: Path to right image
        apply_clahe: Whether to apply CLAHE
        apply_denoise: Whether to apply denoising
        max_dimension: Maximum dimension for resizing
        
    Returns:
        ValidationResult with preprocessed image pair
    """
    result = validate_image_pair(left_path, right_path)
    
    if not result.is_valid or result.image_pair is None:
        return result
    
    pair = result.image_pair
    
    left_processed, right_processed, scale = resize_image_pair(
        pair.left, pair.right, max_dimension
    )
    
    left_processed, right_processed = preprocess_stereo_pair(
        left_processed, right_processed, apply_clahe, apply_denoise,
        clahe_clip_limit=2.0, clahe_tile_grid=(8, 8), denoise_strength=10
    )
    
    pair.left = left_processed
    pair.right = right_processed
    pair.width = left_processed.shape[1]
    pair.height = left_processed.shape[0]
    
    if scale != 1.0:
        result.warnings.append(f"Images resized by factor {scale:.3f}")
    
    return result


def find_stereo_pairs(calib_dir: str, stereo_dir: str) -> List[Tuple[str, str]]:
    """
    Find matching stereo image pairs in directories.
    
    Args:
        calib_dir: Calibration images directory (unused, kept for compatibility)
        stereo_dir: Stereo images directory with left/right subdirs
        
    Returns:
        List of (left_path, right_path) tuples
    """
    stereo_path = Path(stereo_dir)
    left_dir = stereo_path / "left"
    right_dir = stereo_path / "right"
    
    if not left_dir.exists() or not right_dir.exists():
        return []
    
    left_images = sorted(list(left_dir.glob("*.png")) + 
                         list(left_dir.glob("*.jpg")) + 
                         list(left_dir.glob("*.jpeg")))
    right_images = sorted(list(right_dir.glob("*.png")) + 
                          list(right_dir.glob("*.jpg")) + 
                          list(right_dir.glob("*.jpeg")))
    
    pairs = []
    right_stems = {img.stem: img for img in right_images}
    
    for left_img in left_images:
        if left_img.stem in right_stems:
            pairs.append((str(left_img), str(right_stems[left_img.stem])))
    
    return pairs