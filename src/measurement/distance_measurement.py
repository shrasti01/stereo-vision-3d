"""
Distance Measurement Module

This module provides tools for selecting regions of interest (ROIs) in the
depth map and measuring distances to objects.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum


class SelectionMethod(Enum):
    """Method for ROI selection."""
    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    CENTER_POINT = "center_point"
    CONTOUR = "contour"
    THRESHOLD = "threshold"


@dataclass
class ROI:
    """Region of Interest definition."""
    method: SelectionMethod
    # For rectangle: (x, y, w, h)
    # For polygon: list of (x, y) points
    # For center_point: (x, y)
    # For contour: contour index
    # For threshold: (min_depth, max_depth)
    params: Union[Tuple[int, int, int, int], List[Tuple[int, int]], Tuple[int, int], int, Tuple[float, float]]
    label: str = ""


@dataclass
class DistanceMeasurement:
    """Result of distance measurement for an ROI."""
    roi: ROI
    distance_mm: float              # Median distance in mm
    distance_mean_mm: float         # Mean distance in mm
    distance_std_mm: float          # Standard deviation in mm
    distance_min_mm: float          # Minimum distance in mm
    distance_max_mm: float          # Maximum distance in mm
    valid_pixel_count: int          # Number of valid depth pixels in ROI
    total_pixels: int               # Total pixels in ROI
    confidence: float               # Ratio of valid to total pixels
    depth_values: np.ndarray        # All valid depth values in ROI


def create_rect_roi(x: int, y: int, w: int, h: int, label: str = "") -> ROI:
    """Create a rectangular ROI."""
    return ROI(SelectionMethod.RECTANGLE, (x, y, w, h), label)


def create_polygon_roi(points: List[Tuple[int, int]], label: str = "") -> ROI:
    """Create a polygon ROI."""
    return ROI(SelectionMethod.POLYGON, points, label)


def create_center_point_roi(x: int, y: int, radius: int = 5, label: str = "") -> ROI:
    """Create a circular ROI around a center point."""
    return ROI(SelectionMethod.CENTER_POINT, (x, y, radius), label)


def create_contour_roi(contour_index: int, label: str = "") -> ROI:
    """Create ROI from contour index."""
    return ROI(SelectionMethod.CONTOUR, contour_index, label)


def create_threshold_roi(min_depth: float, max_depth: float, label: str = "") -> ROI:
    """Create ROI based on depth threshold."""
    return ROI(SelectionMethod.THRESHOLD, (min_depth, max_depth), label)


def get_roi_mask(roi: ROI, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Generate binary mask for an ROI.
    
    Args:
        roi: ROI definition
        image_shape: (height, width) of image
        
    Returns:
        Binary mask (same size as image)
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if roi.method == SelectionMethod.RECTANGLE:
        x, y, rw, rh = roi.params
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))
        cv2.rectangle(mask, (x, y), (x + rw, y + rh), 255, -1)
    
    elif roi.method == SelectionMethod.POLYGON:
        points = np.array(roi.params, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    
    elif roi.method == SelectionMethod.CENTER_POINT:
        x, y, radius = roi.params
        cv2.circle(mask, (x, y), radius, 255, -1)
    
    elif roi.method == SelectionMethod.THRESHOLD:
        # This is handled separately with depth map
        pass
    
    return mask


def measure_distance_in_roi(depth_map: np.ndarray,
                            valid_mask: np.ndarray,
                            roi: ROI) -> DistanceMeasurement:
    """
    Measure distance statistics within an ROI.
    
    Args:
        depth_map: Depth map in mm
        valid_mask: Boolean mask of valid depth pixels
        roi: Region of interest
        
    Returns:
        DistanceMeasurement
    """
    h, w = depth_map.shape[:2]
    
    # Generate ROI mask
    if roi.method == SelectionMethod.THRESHOLD:
        min_d, max_d = roi.params
        roi_mask = (depth_map >= min_d) & (depth_map <= max_d) & valid_mask
    else:
        roi_mask = get_roi_mask(roi, (h, w)).astype(bool) & valid_mask
    
    # Extract depth values in ROI
    depth_values = depth_map[roi_mask]
    total_pixels = int(np.sum(get_roi_mask(roi, (h, w)) > 0)) if roi.method != SelectionMethod.THRESHOLD else int(np.sum(roi_mask))
    valid_count = len(depth_values)
    
    if valid_count == 0:
        return DistanceMeasurement(
            roi=roi,
            distance_mm=0.0,
            distance_mean_mm=0.0,
            distance_std_mm=0.0,
            distance_min_mm=0.0,
            distance_max_mm=0.0,
            valid_pixel_count=0,
            total_pixels=total_pixels,
            confidence=0.0,
            depth_values=np.array([])
        )
    
    confidence = valid_count / max(total_pixels, 1)
    
    return DistanceMeasurement(
        roi=roi,
        distance_mm=float(np.median(depth_values)),
        distance_mean_mm=float(np.mean(depth_values)),
        distance_std_mm=float(np.std(depth_values)),
        distance_min_mm=float(np.min(depth_values)),
        distance_max_mm=float(np.max(depth_values)),
        valid_pixel_count=valid_count,
        total_pixels=total_pixels,
        confidence=confidence,
        depth_values=depth_values
    )


def measure_multiple_rois(depth_map: np.ndarray,
                          valid_mask: np.ndarray,
                          rois: List[ROI]) -> List[DistanceMeasurement]:
    """
    Measure distances for multiple ROIs.
    
    Args:
        depth_map: Depth map in mm
        valid_mask: Boolean mask of valid depth pixels
        rois: List of ROIs
        
    Returns:
        List of DistanceMeasurement
    """
    return [measure_distance_in_roi(depth_map, valid_mask, roi) for roi in rois]


def find_objects_by_depth(depth_map: np.ndarray,
                          valid_mask: np.ndarray,
                          min_area: int = 100,
                          depth_tolerance: float = 50.0) -> List[ROI]:
    """
    Automatically find objects by clustering similar depth values.
    
    Args:
        depth_map: Depth map in mm
        valid_mask: Boolean mask of valid depth pixels
        min_area: Minimum contour area
        depth_tolerance: Depth range for clustering (mm)
        
    Returns:
        List of ROIs (contours)
    """
    # Normalize depth for thresholding
    valid_depths = depth_map[valid_mask]
    if len(valid_depths) == 0:
        return []
    
    # Simple approach: threshold at different depth levels
    rois = []
    depth_min = np.min(valid_depths)
    depth_max = np.max(valid_depths)
    
    # Create depth slices
    num_slices = 5
    for i in range(num_slices):
        slice_min = depth_min + (depth_max - depth_min) * i / num_slices
        slice_max = depth_min + (depth_max - depth_min) * (i + 1) / num_slices
        
        # Create binary mask for this depth slice
        slice_mask = (depth_map >= slice_min) & (depth_map <= slice_max) & valid_mask
        slice_mask = slice_mask.astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for j, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area >= min_area:
                rois.append(create_contour_roi(len(rois), f"object_{len(rois)}_slice{i}_cnt{j}"))
    
    return rois


def draw_roi_visualization(image: np.ndarray,
                           measurement: DistanceMeasurement,
                           color: Tuple[int, int, int] = (0, 255, 0),
                           thickness: int = 2) -> np.ndarray:
    """
    Draw ROI and distance measurement on image.
    
    Args:
        image: Base image (BGR)
        measurement: Distance measurement result
        color: Drawing color
        thickness: Line thickness
        
    Returns:
        Annotated image
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    roi = measurement.roi
    
    if roi.method == SelectionMethod.RECTANGLE:
        x, y, w, h = roi.params
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        label_pos = (x, y - 10)
    
    elif roi.method == SelectionMethod.POLYGON:
        points = np.array(roi.params, dtype=np.int32)
        cv2.polylines(vis, [points], True, color, thickness)
        label_pos = (points[0][0], points[0][1] - 10)
    
    elif roi.method == SelectionMethod.CENTER_POINT:
        x, y, radius = roi.params
        cv2.circle(vis, (x, y), radius, color, thickness)
        label_pos = (x + radius + 5, y)
    
    elif roi.method == SelectionMethod.CONTOUR:
        # Contour drawing requires original contour - not stored in ROI
        label_pos = (10, 30)
    
    else:
        label_pos = (10, 30)
    
    # Draw distance text
    label = roi.label or "ROI"
    text = f"{label}: {measurement.distance_mm:.1f} mm (±{measurement.distance_std_mm:.1f})"
    cv2.putText(vis, text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return vis


def save_measurements(measurements: List[DistanceMeasurement], 
                      output_dir: str,
                      prefix: str = "measurements"):
    """Save measurements to CSV and JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # CSV
    csv_path = out_path / f"{prefix}.csv"
    with open(csv_path, 'w') as f:
        f.write("label,method,distance_mm,distance_mean_mm,distance_std_mm,")
        f.write("distance_min_mm,distance_max_mm,valid_pixels,total_pixels,confidence\n")
        
        for m in measurements:
            roi = m.roi
            f.write(f'"{roi.label}","{roi.method.value}",')
            f.write(f"{m.distance_mm:.2f},{m.distance_mean_mm:.2f},{m.distance_std_mm:.2f},")
            f.write(f"{m.distance_min_mm:.2f},{m.distance_max_mm:.2f},")
            f.write(f"{m.valid_pixel_count},{m.total_pixels},{m.confidence:.4f}\n")
    
    # JSON with full data
    import json
    json_data = []
    for m in measurements:
        json_data.append({
            "roi": {
                "method": m.roi.method.value,
                "params": m.roi.params,
                "label": m.roi.label
            },
            "distance_mm": m.distance_mm,
            "distance_mean_mm": m.distance_mean_mm,
            "distance_std_mm": m.distance_std_mm,
            "distance_min_mm": m.distance_min_mm,
            "distance_max_mm": m.distance_max_mm,
            "valid_pixel_count": m.valid_pixel_count,
            "total_pixels": m.total_pixels,
            "confidence": m.confidence
        })
    
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
    
    with open(out_path / f"{prefix}.json", 'w') as f:
        json.dump(convert(json_data), f, indent=2)


def interactive_roi_selection(image: np.ndarray, 
                              window_name: str = "Select ROI") -> List[ROI]:
    """
    Interactive ROI selection using mouse (for testing).
    Note: This requires GUI and blocks execution.
    
    Args:
        image: Image to select ROIs on
        window_name: Window title
        
    Returns:
        List of selected ROIs
    """
    rois = []
    current_roi = []
    drawing = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, current_roi, rois
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_roi = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            pass  # Could draw preview
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            current_roi.append((x, y))
            if len(current_roi) == 2:
                x1, y1 = current_roi[0]
                x2, y2 = current_roi[1]
                roi = create_rect_roi(min(x1, x2), min(y1, y2), 
                                      abs(x2 - x1), abs(y2 - y1), f"roi_{len(rois)}")
                rois.append(roi)
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("Click and drag to select rectangles. Press 'q' to finish.")
    while True:
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        # Draw existing ROIs
        for roi in rois:
            if roi.method == SelectionMethod.RECTANGLE:
                x, y, w, h = roi.params
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow(window_name, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return rois