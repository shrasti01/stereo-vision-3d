"""
Accuracy Evaluation Module

This module provides comprehensive error analysis for the stereo vision system,
including depth accuracy, disparity quality metrics, and calibration verification.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class DisparityQualityMetrics:
    """Quality metrics for disparity map."""
    valid_pixel_ratio: float
    mean_disparity: float
    std_disparity: float
    min_disparity: float
    max_disparity: float
    speckle_count: int
    texture_coverage: float
    left_right_consistency: float


@dataclass
class DepthAccuracyMetrics:
    """Accuracy metrics for depth estimation."""
    mae_mm: float
    rmse_mm: float
    median_abs_error_mm: float
    mean_rel_error: float
    median_rel_error: float
    accuracy_1mm: float
    accuracy_5mm: float
    accuracy_10mm: float
    accuracy_5pct: float
    valid_pixels: int


@dataclass
class CalibrationQualityMetrics:
    """Quality metrics for calibration."""
    mono_reprojection_error_left: float
    mono_reprojection_error_right: float
    stereo_reprojection_error: float
    baseline_mm: float
    focal_length_px: float
    principal_point_offset_px: Tuple[float, float]


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    disparity_quality: DisparityQualityMetrics
    depth_accuracy: Optional[DepthAccuracyMetrics]
    calibration_quality: CalibrationQualityMetrics
    measurements: List[Dict[str, Any]]
    summary: str


def evaluate_disparity_quality(disparity: np.ndarray,
                               valid_mask: np.ndarray,
                               left_image: np.ndarray,
                               right_image: np.ndarray,
                               disparity_right: Optional[np.ndarray] = None) -> DisparityQualityMetrics:
    """
    Evaluate quality of a disparity map.
    
    Args:
        disparity: Disparity map
        valid_mask: Valid pixel mask
        left_image: Left rectified image
        right_image: Right rectified image
        disparity_right: Right disparity map (for consistency check)
        
    Returns:
        DisparityQualityMetrics
    """
    valid_pixels = disparity[valid_mask]
    total_pixels = disparity.size
    
    # Basic statistics
    valid_ratio = np.sum(valid_mask) / total_pixels
    mean_disp = np.mean(valid_pixels) if len(valid_pixels) > 0 else 0
    std_disp = np.std(valid_pixels) if len(valid_pixels) > 0 else 0
    min_disp = np.min(valid_pixels) if len(valid_pixels) > 0 else 0
    max_disp = np.max(valid_pixels) if len(valid_pixels) > 0 else 0
    
    # Speckle detection (isolated pixels with very different disparity)
    speckle_count = 0
    if len(valid_pixels) > 0:
        kernel = np.ones((3, 3), np.uint8)
        # Find isolated pixels
        disp_8u = ((disparity - min_disp) / (max_disp - min_disp + 1e-6) * 255).astype(np.uint8)
        disp_8u[~valid_mask] = 0
        median_filtered = cv2.medianBlur(disp_8u, 3)
        diff = cv2.absdiff(disp_8u, median_filtered)
        speckle_mask = (diff > 20) & valid_mask
        speckle_count = int(np.sum(speckle_mask))
    
    # Texture coverage (using gradient magnitude)
    if len(left_image.shape) == 3:
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_image
    
    grad_x = cv2.Sobel(left_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(left_gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    texture_threshold = np.percentile(gradient_mag[valid_mask], 25) if np.any(valid_mask) else 0
    texture_coverage = float(np.sum((gradient_mag > texture_threshold) & valid_mask) / max(np.sum(valid_mask), 1))
    
    # Left-right consistency
    lr_consistency = 0.0
    if disparity_right is not None:
        # Warp right disparity to left view and compare
        h, w = disparity.shape
        valid_lr = valid_mask & (disparity_right > 0)
        if np.any(valid_lr):
            # Simple consistency: at same pixel, disparities should be similar
            # (in rectified images, disparity is horizontal shift)
            consistency_diff = np.abs(disparity[valid_lr] - disparity_right[valid_lr])
            lr_consistency = float(1.0 - np.mean(consistency_diff) / (max_disp + 1e-6))
            lr_consistency = max(0.0, min(1.0, lr_consistency))
    
    return DisparityQualityMetrics(
        valid_pixel_ratio=float(valid_ratio),
        mean_disparity=float(mean_disp),
        std_disparity=float(std_disp),
        min_disparity=float(min_disp),
        max_disparity=float(max_disp),
        speckle_count=speckle_count,
        texture_coverage=texture_coverage,
        left_right_consistency=lr_consistency
    )


def evaluate_depth_accuracy(estimated_depth: np.ndarray,
                            ground_truth_depth: np.ndarray,
                            valid_mask: Optional[np.ndarray] = None) -> DepthAccuracyMetrics:
    """
    Evaluate depth accuracy against ground truth.
    
    Args:
        estimated_depth: Estimated depth map (mm)
        ground_truth_depth: Ground truth depth map (mm)
        valid_mask: Optional valid pixel mask
        
    Returns:
        DepthAccuracyMetrics
    """
    if valid_mask is None:
        valid_mask = (estimated_depth > 0) & (ground_truth_depth > 0) & \
                     np.isfinite(estimated_depth) & np.isfinite(ground_truth_depth)
    
    est_valid = estimated_depth[valid_mask]
    gt_valid = ground_truth_depth[valid_mask]
    
    if len(est_valid) == 0:
        return DepthAccuracyMetrics(
            mae_mm=0, rmse_mm=0, median_abs_error_mm=0,
            mean_rel_error=0, median_rel_error=0,
            accuracy_1mm=0, accuracy_5mm=0, accuracy_10mm=0, accuracy_5pct=0,
            valid_pixels=0
        )
    
    abs_error = np.abs(est_valid - gt_valid)
    rel_error = abs_error / (gt_valid + 1e-6)
    
    return DepthAccuracyMetrics(
        mae_mm=float(np.mean(abs_error)),
        rmse_mm=float(np.sqrt(np.mean((est_valid - gt_valid) ** 2))),
        median_abs_error_mm=float(np.median(abs_error)),
        mean_rel_error=float(np.mean(rel_error)),
        median_rel_error=float(np.median(rel_error)),
        accuracy_1mm=float(np.sum(abs_error < 1.0) / len(abs_error) * 100),
        accuracy_5mm=float(np.sum(abs_error < 5.0) / len(abs_error) * 100),
        accuracy_10mm=float(np.sum(abs_error < 10.0) / len(abs_error) * 100),
        accuracy_5pct=float(np.sum(rel_error < 0.05) / len(rel_error) * 100),
        valid_pixels=int(np.sum(valid_mask))
    )


def evaluate_calibration_quality(stereo_calib: Any,
                                 left_calib: Any,
                                 right_calib: Any) -> CalibrationQualityMetrics:
    """
    Evaluate calibration quality.
    
    Args:
        stereo_calib: Stereo calibration result
        left_calib: Left camera calibration
        right_calib: Right camera calibration
        
    Returns:
        CalibrationQualityMetrics
    """
    # Focal length
    fx = stereo_calib.left_camera_matrix[0, 0]
    fy = stereo_calib.left_camera_matrix[1, 1]
    focal_length = (fx + fy) / 2.0
    
    # Principal point offset from image center
    cx = stereo_calib.left_camera_matrix[0, 2]
    cy = stereo_calib.left_camera_matrix[1, 2]
    img_w = stereo_calib.image_size[0]
    img_h = stereo_calib.image_size[1]
    pp_offset = (cx - img_w / 2, cy - img_h / 2)
    
    # Baseline
    baseline = np.linalg.norm(stereo_calib.T) if stereo_calib.T is not None else 0
    
    return CalibrationQualityMetrics(
        mono_reprojection_error_left=left_calib.reprojection_error,
        mono_reprojection_error_right=right_calib.reprojection_error,
        stereo_reprojection_error=stereo_calib.reprojection_error,
        baseline_mm=float(baseline),
        focal_length_px=float(focal_length),
        principal_point_offset_px=(float(pp_offset[0]), float(pp_offset[1]))
    )


def evaluate_measurements(measurements: List[Any],
                          ground_truth_distances: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Evaluate distance measurements against ground truth.
    
    Args:
        measurements: List of DistanceMeasurement objects
        ground_truth_distances: Dict mapping label -> true distance (mm)
        
    Returns:
        List of evaluation results per measurement
    """
    results = []
    
    for m in measurements:
        label = m.roi.label
        gt_distance = ground_truth_distances.get(label)
        
        result = {
            "label": label,
            "estimated_mm": m.distance_mm,
            "estimated_mean_mm": m.distance_mean_mm,
            "std_mm": m.distance_std_mm,
            "confidence": m.confidence,
            "valid_pixels": m.valid_pixel_count
        }
        
        if gt_distance is not None:
            abs_error = abs(m.distance_mm - gt_distance)
            rel_error = abs_error / (gt_distance + 1e-6)
            result["ground_truth_mm"] = gt_distance
            result["abs_error_mm"] = abs_error
            result["rel_error"] = rel_error
            result["within_1mm"] = abs_error < 1.0
            result["within_5mm"] = abs_error < 5.0
            result["within_10mm"] = abs_error < 10.0
            result["within_5pct"] = rel_error < 0.05
        else:
            result["ground_truth_mm"] = None
            result["abs_error_mm"] = None
            result["rel_error"] = None
        
        results.append(result)
    
    return results


def generate_evaluation_report(disparity_result: Any,
                               depth_result: Any,
                               stereo_calib: Any,
                               left_calib: Any,
                               right_calib: Any,
                               measurements: List[Any],
                               ground_truth_depth: Optional[np.ndarray] = None,
                               ground_truth_distances: Optional[Dict[str, float]] = None,
                               left_rect: Optional[np.ndarray] = None,
                               right_rect: Optional[np.ndarray] = None,
                               disparity_right: Optional[np.ndarray] = None) -> EvaluationReport:
    """
    Generate complete evaluation report.
    
    Args:
        disparity_result: DisparityResult from stereo matching
        depth_result: DepthResult from depth estimation
        stereo_calib: StereoCalibrationResult
        left_calib: Left CalibrationResult
        right_calib: Right CalibrationResult
        measurements: List of DistanceMeasurement
        ground_truth_depth: Optional ground truth depth map
        ground_truth_distances: Optional dict of ground truth distances
        left_rect: Left rectified image
        right_rect: Right rectified image
        disparity_right: Right disparity map
        
    Returns:
        EvaluationReport
    """
    # Disparity quality
    disp_quality = evaluate_disparity_quality(
        disparity_result.disparity,
        disparity_result.disparity > 0,
        left_rect if left_rect is not None else np.zeros_like(disparity_result.disparity),
        right_rect if right_rect is not None else np.zeros_like(disparity_result.disparity),
        disparity_right
    )
    
    # Depth accuracy
    depth_accuracy = None
    if ground_truth_depth is not None and depth_result is not None:
        depth_accuracy = evaluate_depth_accuracy(
            depth_result.depth_map,
            ground_truth_depth,
            depth_result.valid_mask
        )
    
    # Calibration quality
    calib_quality = evaluate_calibration_quality(stereo_calib, left_calib, right_calib)
    
    # Measurements evaluation
    meas_results = []
    if ground_truth_distances:
        meas_results = evaluate_measurements(measurements, ground_truth_distances)
    else:
        for m in measurements:
            meas_results.append({
                "label": m.roi.label,
                "estimated_mm": m.distance_mm,
                "estimated_mean_mm": m.distance_mean_mm,
                "std_mm": m.distance_std_mm,
                "confidence": m.confidence,
                "valid_pixels": m.valid_pixel_count
            })
    
    # Generate summary
    summary_lines = [
        "=" * 60,
        "STEREO VISION 3D DISTANCE MEASUREMENT - EVALUATION REPORT",
        "=" * 60,
        "",
        "DISPARITY QUALITY:",
        f"  Valid Pixel Ratio: {disp_quality.valid_pixel_ratio*100:.1f}%",
        f"  Mean Disparity: {disp_quality.mean_disparity:.2f} px",
        f"  Disparity Range: [{disp_quality.min_disparity:.2f}, {disp_quality.max_disparity:.2f}] px",
        f"  Speckle Count: {disp_quality.speckle_count}",
        f"  Texture Coverage: {disp_quality.texture_coverage*100:.1f}%",
        f"  Left-Right Consistency: {disp_quality.left_right_consistency*100:.1f}%",
        "",
        "CALIBRATION QUALITY:",
        f"  Left Mono Reprojection Error: {calib_quality.mono_reprojection_error_left:.4f} px",
        f"  Right Mono Reprojection Error: {calib_quality.mono_reprojection_error_right:.4f} px",
        f"  Stereo Reprojection Error: {calib_quality.stereo_reprojection_error:.4f} px",
        f"  Focal Length: {calib_quality.focal_length_px:.2f} px",
        f"  Baseline: {calib_quality.baseline_mm:.2f} mm",
        f"  Principal Point Offset: ({calib_quality.principal_point_offset_px[0]:.2f}, {calib_quality.principal_point_offset_px[1]:.2f}) px",
    ]
    
    if depth_accuracy:
        summary_lines.extend([
            "",
            "DEPTH ACCURACY (vs Ground Truth):",
            f"  MAE: {depth_accuracy.mae_mm:.2f} mm",
            f"  RMSE: {depth_accuracy.rmse_mm:.2f} mm",
            f"  Median Abs Error: {depth_accuracy.median_abs_error_mm:.2f} mm",
            f"  Mean Rel Error: {depth_accuracy.mean_rel_error*100:.2f}%",
            f"  Accuracy <1mm: {depth_accuracy.accuracy_1mm:.1f}%",
            f"  Accuracy <5mm: {depth_accuracy.accuracy_5mm:.1f}%",
            f"  Accuracy <10mm: {depth_accuracy.accuracy_10mm:.1f}%",
            f"  Accuracy <5%: {depth_accuracy.accuracy_5pct:.1f}%",
            f"  Valid Pixels: {depth_accuracy.valid_pixels}",
        ])
    
    if measurements:
        summary_lines.extend(["", "MEASUREMENTS:"])
        for r in meas_results:
            summary_lines.append(f"  {r['label']}: {r['estimated_mm']:.1f} mm ± {r['std_mm']:.1f} mm (conf: {r['confidence']*100:.1f}%)")
            if r.get('ground_truth_mm') is not None:
                summary_lines.append(f"    GT: {r['ground_truth_mm']:.1f} mm, Error: {r['abs_error_mm']:.1f} mm ({r['rel_error']*100:.1f}%)")
    
    summary = "\n".join(summary_lines)
    
    return EvaluationReport(
        disparity_quality=disp_quality,
        depth_accuracy=depth_accuracy,
        calibration_quality=calib_quality,
        measurements=meas_results,
        summary=summary
    )


def save_evaluation_report(report: EvaluationReport, output_dir: str, prefix: str = "evaluation"):
    """Save evaluation report to files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary text
    with open(out_path / f"{prefix}_report.txt", 'w') as f:
        f.write(report.summary)
    
    # Save detailed JSON
    data = {
        "disparity_quality": {
            "valid_pixel_ratio": report.disparity_quality.valid_pixel_ratio,
            "mean_disparity": report.disparity_quality.mean_disparity,
            "std_disparity": report.disparity_quality.std_disparity,
            "min_disparity": report.disparity_quality.min_disparity,
            "max_disparity": report.disparity_quality.max_disparity,
            "speckle_count": report.disparity_quality.speckle_count,
            "texture_coverage": report.disparity_quality.texture_coverage,
            "left_right_consistency": report.disparity_quality.left_right_consistency
        },
        "calibration_quality": {
            "mono_reprojection_error_left": report.calibration_quality.mono_reprojection_error_left,
            "mono_reprojection_error_right": report.calibration_quality.mono_reprojection_error_right,
            "stereo_reprojection_error": report.calibration_quality.stereo_reprojection_error,
            "baseline_mm": report.calibration_quality.baseline_mm,
            "focal_length_px": report.calibration_quality.focal_length_px,
            "principal_point_offset_px": report.calibration_quality.principal_point_offset_px
        },
        "measurements": report.measurements
    }
    
    if report.depth_accuracy:
        data["depth_accuracy"] = {
            "mae_mm": report.depth_accuracy.mae_mm,
            "rmse_mm": report.depth_accuracy.rmse_mm,
            "median_abs_error_mm": report.depth_accuracy.median_abs_error_mm,
            "mean_rel_error": report.depth_accuracy.mean_rel_error,
            "median_rel_error": report.depth_accuracy.median_rel_error,
            "accuracy_1mm": report.depth_accuracy.accuracy_1mm,
            "accuracy_5mm": report.depth_accuracy.accuracy_5mm,
            "accuracy_10mm": report.depth_accuracy.accuracy_10mm,
            "accuracy_5pct": report.depth_accuracy.accuracy_5pct,
            "valid_pixels": report.depth_accuracy.valid_pixels
        }
    
    with open(out_path / f"{prefix}_report.json", 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Evaluation report saved to {output_dir}")


def theoretical_depth_error(focal_length_px: float,
                            baseline_mm: float,
                            disparity: float,
                            disparity_error_px: float = 0.5) -> float:
    """
    Calculate theoretical depth error based on disparity quantization error.
    
    From Z = f*B/d, error propagation gives:
    ΔZ = (f*B/d²) * Δd = Z²/(f*B) * Δd
    
    Args:
        focal_length_px: Focal length in pixels
        baseline_mm: Baseline in mm
        disparity: Disparity in pixels
        disparity_error_px: Disparity error (typically 0.5-1 pixel for subpixel)
        
    Returns:
        Theoretical depth error in mm
    """
    if disparity <= 0:
        return float('inf')
    
    depth = (focal_length_px * baseline_mm) / disparity
    depth_error = (depth ** 2) / (focal_length_px * baseline_mm) * disparity_error_px
    return depth_error


def print_theoretical_error_table(focal_length_px: float, baseline_mm: float):
    """Print table of theoretical depth errors at different distances."""
    print("\nTheoretical Depth Error (delta_d = 0.5 px):")
    print(f"  Focal Length: {focal_length_px:.1f} px, Baseline: {baseline_mm:.1f} mm")
    print("  " + "-" * 50)
    print(f"  {'Distance (mm)':>15} | {'Disparity (px)':>15} | {'Error (mm)':>12} | {'Error (%)':>10}")
    print("  " + "-" * 50)
    
    for dist in [500, 1000, 1500, 2000, 3000, 5000, 10000]:
        disp = (focal_length_px * baseline_mm) / dist
        error = theoretical_depth_error(focal_length_px, baseline_mm, disp, 0.5)
        pct = (error / dist) * 100
        print(f"  {dist:>15} | {disp:>15.2f} | {error:>12.2f} | {pct:>9.2f}%")