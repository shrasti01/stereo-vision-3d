"""
Visualization Utilities Module

Functions for creating visualizations of stereo vision results.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import matplotlib.pyplot as plt


def normalize_for_display(image: np.ndarray, 
                          min_val: Optional[float] = None,
                          max_val: Optional[float] = None) -> np.ndarray:
    """
    Normalize image to 0-255 range for display.
    
    Args:
        image: Input image
        min_val: Minimum value (None = auto)
        max_val: Maximum value (None = auto)
        
    Returns:
        Normalized uint8 image
    """
    if min_val is None:
        min_val = np.min(image)
    if max_val is None:
        max_val = np.max(image)
    
    if max_val <= min_val:
        return np.zeros_like(image, dtype=np.uint8)
    
    normalized = np.clip((image - min_val) / (max_val - min_val) * 255, 0, 255)
    return normalized.astype(np.uint8)


def apply_colormap(image: np.ndarray, 
                   colormap: int = cv2.COLORMAP_JET,
                   mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply colormap to grayscale image.
    
    Args:
        image: Grayscale image (0-255)
        colormap: OpenCV colormap
        mask: Optional mask (valid pixels)
        
    Returns:
        Color-mapped image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    colored = cv2.applyColorMap(image, colormap)
    
    if mask is not None:
        colored[~mask] = [0, 0, 0]
    
    return colored


def create_side_by_side(images: List[np.ndarray],
                        labels: Optional[List[str]] = None,
                        scale: float = 1.0) -> np.ndarray:
    """
    Create side-by-side comparison of multiple images.
    
    Args:
        images: List of images
        labels: Optional labels for each image
        scale: Scale factor for output
        
    Returns:
        Combined image
    """
    # Ensure all images have same height
    heights = [img.shape[0] for img in images]
    max_h = max(heights)
    
    resized = []
    for img in images:
        if img.shape[0] != max_h:
            scale_h = max_h / img.shape[0]
            new_w = int(img.shape[1] * scale_h)
            img = cv2.resize(img, (new_w, max_h))
        
        # Ensure 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        resized.append(img)
    
    combined = np.hstack(resized)
    
    # Add labels if provided
    if labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7 * scale
        thickness = max(1, int(2 * scale))
        x_offset = 0
        
        for i, (img, label) in enumerate(zip(resized, labels)):
            cv2.putText(combined, label, (x_offset + 10, 30), 
                       font, font_scale, (255, 255, 255), thickness)
            cv2.putText(combined, label, (x_offset + 10, 30), 
                       font, font_scale, (0, 0, 0), thickness - 1)
            x_offset += img.shape[1]
    
    return combined


def create_stereo_comparison(left: np.ndarray,
                             right: np.ndarray,
                             disparity: Optional[np.ndarray] = None,
                             depth: Optional[np.ndarray] = None,
                             disparity_right: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create comprehensive stereo vision comparison visualization.
    
    Args:
        left: Left rectified image
        right: Right rectified image
        disparity: Disparity map
        depth: Depth map
        disparity_right: Right disparity map
        
    Returns:
        Combined visualization image
    """
    # Prepare base images
    if len(left.shape) == 2:
        left_vis = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    else:
        left_vis = left.copy()
    
    if len(right.shape) == 2:
        right_vis = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    else:
        right_vis = right.copy()
    
    row1 = np.hstack([left_vis, right_vis])
    
    row2_images = []
    
    # Disparity
    if disparity is not None:
        disp_vis = normalize_for_display(disparity)
        disp_color = apply_colormap(disp_vis)
        cv2.putText(disp_color, "Disparity", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        row2_images.append(disp_color)
    
    # Depth
    if depth is not None:
        depth_vis = normalize_for_display(depth)
        depth_color = apply_colormap(depth_vis)
        cv2.putText(depth_color, "Depth", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        row2_images.append(depth_color)
    
    # Right disparity
    if disparity_right is not None:
        disp_r_vis = normalize_for_display(disparity_right)
        disp_r_color = apply_colormap(disp_r_vis)
        cv2.putText(disp_r_color, "Right Disparity", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        row2_images.append(disp_r_color)
    
    # Epipolar line check
    if disparity is not None:
        epipolar_vis = draw_epipolar_lines(left_vis.copy(), num_lines=15)
        cv2.putText(epipolar_vis, "Epipolar Check", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        row2_images.append(epipolar_vis)
    
    if row2_images:
        row2 = np.hstack(row2_images)
        # Match width of row1
        if row2.shape[1] != row1.shape[1]:
            scale = row1.shape[1] / row2.shape[1]
            row2 = cv2.resize(row2, (row1.shape[1], int(row2.shape[0] * scale)))
        
        combined = np.vstack([row1, row2])
    else:
        combined = row1
    
    return combined


def draw_epipolar_lines(image: np.ndarray,
                        num_lines: int = 20,
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 1) -> np.ndarray:
    """Draw horizontal epipolar lines on image."""
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    h, w = vis.shape[:2]
    step = h // (num_lines + 1)
    
    for i in range(1, num_lines + 1):
        y = i * step
        cv2.line(vis, (0, y), (w, y), color, thickness)
    
    return vis


def draw_measurements(image: np.ndarray,
                      measurements: List,
                      color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw distance measurements on image.
    
    Args:
        image: Base image
        measurements: List of DistanceMeasurement objects
        color: Drawing color
        
    Returns:
        Annotated image
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    for m in measurements:
        roi = m.roi
        
        if roi.method.value == "rectangle":
            x, y, w, h = roi.params
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            label_pos = (x, max(0, y - 10))
        
        elif roi.method.value == "polygon":
            points = np.array(roi.params, dtype=np.int32)
            cv2.polylines(vis, [points], True, color, 2)
            label_pos = (points[0][0], points[0][1] - 10)
        
        elif roi.method.value == "center_point":
            x, y, radius = roi.params
            cv2.circle(vis, (x, y), radius, color, 2)
            label_pos = (x + radius + 5, y)
        
        else:
            label_pos = (10, 30)
        
        label = roi.label or "ROI"
        text = f"{label}: {m.distance_mm:.1f} mm"
        cv2.putText(vis, text, label_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return vis


def create_error_heatmap(estimated: np.ndarray,
                         ground_truth: np.ndarray,
                         valid_mask: np.ndarray,
                         max_error: float = 100.0) -> np.ndarray:
    """
    Create error heatmap between estimated and ground truth.
    
    Args:
        estimated: Estimated depth map
        ground_truth: Ground truth depth map
        valid_mask: Valid pixel mask
        max_error: Maximum error for color scaling (mm)
        
    Returns:
        Error heatmap visualization
    """
    error = np.zeros_like(estimated)
    error[valid_mask] = np.abs(estimated[valid_mask] - ground_truth[valid_mask])
    
    # Normalize error
    error_vis = np.clip(error / max_error * 255, 0, 255).astype(np.uint8)
    error_color = apply_colormap(error_vis, cv2.COLORMAP_HOT, valid_mask)
    
    return error_color


def plot_depth_profile(depth_map: np.ndarray,
                       row: int,
                       title: str = "Depth Profile",
                       save_path: Optional[str] = None):
    """
    Plot depth profile along a row.
    
    Args:
        depth_map: Depth map
        row: Row index
        title: Plot title
        save_path: Optional path to save plot
    """
    profile = depth_map[row, :]
    valid = profile > 0
    x = np.arange(len(profile))
    
    plt.figure(figsize=(10, 4))
    plt.plot(x[valid], profile[valid], 'b-', linewidth=1)
    plt.xlabel('Pixel Column')
    plt.ylabel('Depth (mm)')
    plt.title(f'{title} - Row {row}')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_disparity_histogram(disparity: np.ndarray,
                             valid_mask: np.ndarray,
                             bins: int = 50,
                             title: str = "Disparity Distribution",
                             save_path: Optional[str] = None):
    """Plot histogram of disparity values."""
    valid_disp = disparity[valid_mask]
    
    plt.figure(figsize=(8, 5))
    plt.hist(valid_disp, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Disparity (pixels)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_error_distribution(errors: np.ndarray,
                            title: str = "Error Distribution",
                            save_path: Optional[str] = None):
    """Plot error distribution."""
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Error (mm)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.2f}')
    plt.axvline(np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_visualization_grid(output_dir: str,
                            left: np.ndarray,
                            right: np.ndarray,
                            disparity: Optional[np.ndarray] = None,
                            depth: Optional[np.ndarray] = None,
                            measurements: Optional[List] = None,
                            prefix: str = "result"):
    """
    Save a complete visualization grid to disk.
    
    Args:
        output_dir: Output directory
        left: Left image
        right: Right image
        disparity: Disparity map
        depth: Depth map
        measurements: Optional measurements
        prefix: File prefix
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Individual images
    cv2.imwrite(str(out_path / f"{prefix}_left.png"), left)
    cv2.imwrite(str(out_path / f"{prefix}_right.png"), right)
    
    if disparity is not None:
        disp_vis = normalize_for_display(disparity)
        cv2.imwrite(str(out_path / f"{prefix}_disparity_gray.png"), disp_vis)
        cv2.imwrite(str(out_path / f"{prefix}_disparity_color.png"), 
                   apply_colormap(disp_vis))
    
    if depth is not None:
        depth_vis = normalize_for_display(depth)
        cv2.imwrite(str(out_path / f"{prefix}_depth_gray.png"), depth_vis)
        cv2.imwrite(str(out_path / f"{prefix}_depth_color.png"), 
                   apply_colormap(depth_vis))
    
    # Combined visualization
    combined = create_stereo_comparison(left, right, disparity, depth)
    if measurements:
        combined = draw_measurements(combined, measurements)
    
    cv2.imwrite(str(out_path / f"{prefix}_combined.png"), combined)
    
    print(f"Visualizations saved to {output_dir}")