"""
Main CLI Entry Point for Stereo Vision 3D Distance Measurement System

This module provides the command-line interface for the complete stereo vision pipeline.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.image_preprocessing import (
    load_and_preprocess_pair, find_stereo_pairs, ValidationResult
)
from src.calibration.camera_calibration import (
    run_calibration_pipeline, load_stereo_calibration, load_calibration,
    StereoCalibrationResult, CalibrationResult, ChessboardDetector
)
from src.rectification.stereo_rectification import (
    rectify_stereo_pair_full, compute_rectification_maps,
    create_side_by_side, create_anaglyph, save_rectification_maps
)
from src.matching.stereo_matching import (
    compute_disparity, SGBMParams, BMParams, MatchingAlgorithm,
    save_disparity_result, filter_disparity
)
from src.depth.depth_estimation import (
    estimate_depth_from_disparity, save_depth_result, compute_depth_error
)
from src.measurement.distance_measurement import (
    create_rect_roi, create_center_point_roi, create_polygon_roi,
    measure_distance_in_roi, measure_multiple_rois,
    draw_roi_visualization, save_measurements
)
from src.evaluation.accuracy_evaluation import (
    generate_evaluation_report, save_evaluation_report,
    print_theoretical_error_table
)
from src.utils.io_utils import (
    ensure_dir, find_stereo_pairs as find_pairs,
    create_output_structure, ConfigManager
)
from src.utils.visualization import (
    save_visualization_grid, create_stereo_comparison,
    normalize_for_display, apply_colormap
)


def run_calibration(args) -> int:
    """Run camera calibration."""
    print("Running stereo calibration...")
    
    if not os.path.exists(args.left_calib) or not os.path.exists(args.right_calib):
        print(f"Error: Calibration directories not found:")
        print(f"  Left: {args.left_calib}")
        print(f"  Right: {args.right_calib}")
        return 1
    
    result = run_calibration_pipeline(
        left_calib_dir=args.left_calib,
        right_calib_dir=args.right_calib,
        pattern_size=tuple(args.pattern_size),
        square_size=args.square_size,
        output_dir=args.output_dir,
        visualize=args.visualize
    )
    
    if result.success:
        print("\nCalibration successful!")
        print(f"  Baseline: {np.linalg.norm(result.T):.2f} mm")
        print(f"  Focal length: {(result.left_camera_matrix[0,0] + result.left_camera_matrix[1,1])/2:.2f} px")
        print(f"  Stereo reprojection error: {result.reprojection_error:.4f} px")
        return 0
    else:
        print(f"\nCalibration failed: {result.errors}")
        return 1


def run_stereo_pipeline(args) -> int:
    """Run complete stereo vision pipeline."""
    print("=" * 60)
    print("STEREO VISION 3D DISTANCE MEASUREMENT PIPELINE")
    print("=" * 60)
    
    # Load or run calibration
    calib_dir = Path(args.output_dir) / "calibration"
    calib_file = calib_dir / "stereo_calibration.json"
    
    stereo_calib = None
    left_calib = None
    right_calib = None
    
    if calib_file.exists() and not args.force_calibration:
        print(f"\nLoading calibration from {calib_file}")
        stereo_calib = load_stereo_calibration(str(calib_file))
        left_calib = load_calibration(str(calib_dir / "left_calibration.json"))
        right_calib = load_calibration(str(calib_dir / "right_calibration.json"))
        
        # Verify calibration is valid (has required matrices)
        if stereo_calib.left_camera_matrix is None or stereo_calib.T is None:
            print("Loaded calibration incomplete, need to recalibrate...")
            stereo_calib = None
    
    if stereo_calib is None:
        if not os.path.exists(args.left_calib) or not os.path.exists(args.right_calib):
            print(f"Error: Calibration directories required:")
            print(f"  Left: {args.left_calib}")
            print(f"  Right: {args.right_calib}")
            return 1
        
        print("\nRunning calibration...")
        result = run_calibration_pipeline(
            left_calib_dir=args.left_calib,
            right_calib_dir=args.right_calib,
            pattern_size=tuple(args.pattern_size),
            square_size=args.square_size,
            output_dir=args.output_dir,
            visualize=args.visualize
        )
        
        if not result.success:
            print(f"Calibration failed: {result.errors}")
            return 1
        
        stereo_calib = result
        left_calib = load_calibration(str(calib_dir / "left_calibration.json"))
        right_calib = load_calibration(str(calib_dir / "right_calibration.json"))
    
    # Find stereo pairs
    if args.left_image and args.right_image:
        # Single pair specified
        pairs = [(args.left_image, args.right_image)]
    else:
        # Find all pairs in stereo directory
        pairs = find_pairs(args.stereo_dir)
    
    if not pairs:
        print(f"Error: No stereo pairs found in {args.stereo_dir}")
        print("Expected structure: stereo_dir/left/*.png and stereo_dir/right/*.png")
        return 1
    
    print(f"\nFound {len(pairs)} stereo pair(s)")
    
    # Setup matching parameters
    sgbm_params = SGBMParams(
        min_disparity=args.min_disp,
        num_disparities=args.num_disp,
        block_size=args.block_size,
        uniqueness_ratio=args.uniqueness_ratio,
        speckle_window_size=args.speckle_window,
        speckle_range=args.speckle_range
    )
    
    # Process each pair
    for idx, (left_path, right_path) in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"Processing pair {idx+1}/{len(pairs)}: {Path(left_path).name}")
        print(f"{'='*60}")
        
        # Load and preprocess
        print("\n[1/6] Loading and preprocessing images...")
        result = load_and_preprocess_pair(
            left_path, right_path,
            apply_clahe=args.clahe,
            apply_denoise=args.denoise,
            max_dimension=args.max_dim
        )
        
        if not result.is_valid:
            print(f"  Validation failed: {result.errors}")
            continue
        
        pair = result.image_pair
        left_img, right_img = pair.left, pair.right
        
        # Save preprocessed
        out_dir = Path(args.output_dir) / "preprocessed"
        ensure_dir(str(out_dir))
        cv2.imwrite(str(out_dir / f"{Path(left_path).stem}_left.png"), left_img)
        cv2.imwrite(str(out_dir / f"{Path(right_path).stem}_right.png"), right_img)
        
        # Rectification
        print("\n[2/6] Rectifying stereo pair...")
        rectified = rectify_stereo_pair_full(
            left_img, right_img, stereo_calib, alpha=args.alpha
        )
        left_rect, right_rect = rectified.left, rectified.right
        
        # Save rectified
        rect_dir = Path(args.output_dir) / "rectified"
        ensure_dir(str(rect_dir))
        cv2.imwrite(str(rect_dir / f"{Path(left_path).stem}_left_rect.png"), left_rect)
        cv2.imwrite(str(rect_dir / f"{Path(right_path).stem}_right_rect.png"), right_rect)
        
        # Side-by-side for verification
        sbS = create_side_by_side(left_rect, right_rect)
        cv2.imwrite(str(rect_dir / f"{Path(left_path).stem}_sidebyside.png"), sbS)
        
        # Stereo matching
        print("\n[3/6] Computing disparity map...")
        algorithm = MatchingAlgorithm(args.algorithm)
        disp_result = compute_disparity(
            left_rect, right_rect,
            algorithm=algorithm,
            params=sgbm_params
        )
        
        # Filter disparity
        if args.filter_speckle:
            disp_result.disparity = filter_disparity(
                disp_result.disparity, method="speckle",
                speckle_window=args.speckle_window,
                speckle_range=args.speckle_range
            )
        
        # Save disparity
        disp_dir = Path(args.output_dir) / "disparity"
        ensure_dir(str(disp_dir))
        save_disparity_result(disp_result, str(disp_dir), Path(left_path).stem)
        
        # Depth estimation
        print("\n[4/6] Estimating depth map...")
        depth_result = estimate_depth_from_disparity(
            disp_result, stereo_calib,
            min_disparity=args.min_disp,
            max_depth_mm=args.max_depth
        )
        
        # Save depth
        depth_dir = Path(args.output_dir) / "depth"
        ensure_dir(str(depth_dir))
        save_depth_result(depth_result, str(depth_dir), Path(left_path).stem)
        
        # Measurements
        print("\n[5/6] Measuring distances...")
        measurements = []
        
        if args.roi:
            # User-specified ROIs
            for roi_spec in args.roi:
                # Format: x,y,w,ht,label or x,y,r,label for center point
                parts = roi_spec.split(',')
                if len(parts) == 5:
                    x, y, w, h, label = parts
                    roi = create_rect_roi(int(x), int(y), int(w), int(h), label)
                elif len(parts) == 4:
                    x, y, r, label = parts
                    roi = create_center_point_roi(int(x), int(y), int(r), label)
                else:
                    print(f"  Invalid ROI format: {roi_spec}")
                    continue
                
                m = measure_distance_in_roi(depth_result.depth_map, depth_result.valid_mask, roi)
                measurements.append(m)
                print(f"  {m.roi.label}: {m.distance_mm:.1f} mm ± {m.distance_std_mm:.1f} mm (conf: {m.confidence*100:.1f}%)")
        else:
            # Auto-detect objects
            print("  Auto-detecting objects...")
            # Use threshold-based ROIs at different depth ranges
            valid_depths = depth_result.depth_map[depth_result.valid_mask]
            if len(valid_depths) > 0:
                d_min, d_max = np.min(valid_depths), np.max(valid_depths)
                num_zones = 3
                for i in range(num_zones):
                    z_min = d_min + (d_max - d_min) * i / num_zones
                    z_max = d_min + (d_max - d_min) * (i + 1) / num_zones
                    roi = create_rect_roi(0, 0, left_rect.shape[1], left_rect.shape[0], f"zone_{i}")
                    # Override measure to use threshold
                    from src.measurement.distance_measurement import SelectionMethod, ROI
                    roi = ROI(SelectionMethod.THRESHOLD, (z_min, z_max), f"zone_{i}")
                    m = measure_distance_in_roi(depth_result.depth_map, depth_result.valid_mask, roi)
                    if m.valid_pixel_count > 100:
                        measurements.append(m)
                        print(f"  {m.roi.label}: {m.distance_mm:.1f} mm ± {m.distance_std_mm:.1f} mm")
        
        # Save measurements
        if measurements:
            meas_dir = Path(args.output_dir) / "measurements"
            ensure_dir(str(meas_dir))
            save_measurements(measurements, str(meas_dir), Path(left_path).stem)
        
        # Evaluation
        print("\n[6/6] Generating evaluation report...")
        report = generate_evaluation_report(
            disp_result, depth_result, stereo_calib, left_calib, right_calib,
            measurements, None, None, left_rect, right_rect
        )
        
        eval_dir = Path(args.output_dir) / "evaluation"
        ensure_dir(str(eval_dir))
        save_evaluation_report(report, str(eval_dir), Path(left_path).stem)
        
        print(report.summary)
        
        # Visualizations
        print("\nSaving visualizations...")
        vis_dir = Path(args.output_dir) / "visualizations"
        ensure_dir(str(vis_dir))
        save_visualization_grid(
            str(vis_dir), left_rect, right_rect,
            disp_result.disparity, depth_result.depth_map,
            measurements, Path(left_path).stem
        )
        
        # Theoretical error analysis
        print_theoretical_error_table(depth_result.focal_length_px, depth_result.baseline_mm)
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_dir}")
    
    return 0


def run_disparity_only(args) -> int:
    """Run only disparity computation on rectified images."""
    print("Running disparity computation...")
    
    if not os.path.exists(args.left) or not os.path.exists(args.right):
        print(f"Error: Images not found")
        return 1
    
    left = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
    
    sgbm_params = SGBMParams(
        min_disparity=args.min_disp,
        num_disparities=args.num_disp,
        block_size=args.block_size
    )
    
    result = compute_disparity(left, right, MatchingAlgorithm.SGBM, sgbm_params)
    
    ensure_dir(args.output_dir)
    save_disparity_result(result, args.output_dir, "disparity")
    
    print(f"Disparity computed in {result.compute_time_ms:.1f} ms")
    print(f"Valid pixels: {np.sum(result.disparity > 0)} / {result.disparity.size}")
    
    return 0


def run_depth_only(args) -> int:
    """Run depth estimation from existing disparity and calibration."""
    print("Running depth estimation...")
    
    # Load disparity
    if args.disparity.endswith('.npy'):
        disp = np.load(args.disparity)
    elif args.disparity.endswith('.png'):
        disp = cv2.imread(args.disparity, cv2.IMREAD_UNCHANGED)
        if disp is not None and (disp.dtype == np.int16 or disp.dtype == np.uint16):
            disp = disp.astype(np.float32) / 16.0
    else:
        disp = cv2.imread(args.disparity, cv2.IMREAD_UNCHANGED)
    
    if disp is None:
        print("Error: Could not load disparity")
        return 1
    
    # Load calibration
    calib = load_stereo_calibration(args.calibration)
    
    # Create dummy DisparityResult
    from src.matching.stereo_matching import DisparityResult
    disp_result = DisparityResult(
        disparity=disp,
        disparity_normalized=normalize_for_display(disp),
        disparity_color=apply_colormap(normalize_for_display(disp)),
        algorithm="loaded",
        parameters={},
        compute_time_ms=0,
        valid_roi=(0, 0, disp.shape[1], disp.shape[0])
    )
    
    depth_result = estimate_depth_from_disparity(
        disp_result, calib,
        min_disparity=args.min_disp,
        max_depth_mm=args.max_depth
    )
    
    ensure_dir(args.output_dir)
    save_depth_result(depth_result, args.output_dir, "depth")
    
    print(f"Depth estimated: {depth_result.stats.get('mean_depth_mm', 0):.1f} mm mean")
    print(f"Valid pixels: {depth_result.stats.get('valid_pixel_count', 0)} / {depth_result.stats.get('total_pixels', 1)}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Stereo Vision 3D Distance Measurement System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run calibration
  python -m src.cli.main calibrate --left-calib data/calibration/left --right-calib data/calibration/right
  
  # Run full pipeline
  python -m src.cli.main pipeline --stereo-dir data/stereo --left-calib data/calibration/left --right-calib data/calibration/right
  
  # Run with specific ROI
  python -m src.cli.main pipeline --stereo-dir data/stereo --roi "100,100,200,200,object1"
  
  # Compute disparity only
  python -m src.cli.main disparity --left outputs/rectified/left_rect.png --right outputs/rectified/right_rect.png
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Calibration command
    calib_parser = subparsers.add_parser('calibrate', help='Run stereo calibration')
    calib_parser.add_argument('--left-calib', required=True, help='Left calibration images directory')
    calib_parser.add_argument('--right-calib', required=True, help='Right calibration images directory')
    calib_parser.add_argument('--pattern-size', nargs=2, type=int, default=[9, 6], help='Chessboard pattern size (cols rows)')
    calib_parser.add_argument('--square-size', type=float, default=25.0, help='Chessboard square size in mm')
    calib_parser.add_argument('--output-dir', default='outputs', help='Output directory')
    calib_parser.add_argument('--visualize', action='store_true', help='Save corner detection visualizations')
    
    # Full pipeline command
    pipe_parser = subparsers.add_parser('pipeline', help='Run complete stereo vision pipeline')
    pipe_parser.add_argument('--stereo-dir', default='data/stereo', help='Stereo images directory (left/right subdirs)')
    pipe_parser.add_argument('--left-calib', default='data/calibration/left', help='Left calibration directory')
    pipe_parser.add_argument('--right-calib', default='data/calibration/right', help='Right calibration directory')
    pipe_parser.add_argument('--left-image', help='Specific left image (optional)')
    pipe_parser.add_argument('--right-image', help='Specific right image (optional)')
    pipe_parser.add_argument('--pattern-size', nargs=2, type=int, default=[9, 6], help='Chessboard pattern size')
    pipe_parser.add_argument('--square-size', type=float, default=25.0, help='Chessboard square size in mm')
    pipe_parser.add_argument('--output-dir', default='outputs', help='Output directory')
    pipe_parser.add_argument('--force-calibration', action='store_true', help='Force recalibration')
    pipe_parser.add_argument('--visualize', action='store_true', help='Save calibration visualizations')
    
    # Preprocessing options
    pipe_parser.add_argument('--clahe', action='store_true', default=True, help='Apply CLAHE')
    pipe_parser.add_argument('--no-clahe', dest='clahe', action='store_false', help='Disable CLAHE')
    pipe_parser.add_argument('--denoise', action='store_true', help='Apply denoising')
    pipe_parser.add_argument('--max-dim', type=int, default=1280, help='Max image dimension')
    
    # Matching options
    pipe_parser.add_argument('--algorithm', choices=['sgbm', 'bm', 'sgbm_3way'], default='sgbm', help='Matching algorithm')
    pipe_parser.add_argument('--min-disp', type=int, default=0, help='Minimum disparity')
    pipe_parser.add_argument('--num-disp', type=int, default=64, help='Number of disparities (multiple of 16)')
    pipe_parser.add_argument('--block-size', type=int, default=5, help='Block size (odd)')
    pipe_parser.add_argument('--uniqueness-ratio', type=int, default=10, help='Uniqueness ratio')
    pipe_parser.add_argument('--speckle-window', type=int, default=100, help='Speckle window size')
    pipe_parser.add_argument('--speckle-range', type=int, default=32, help='Speckle range')
    pipe_parser.add_argument('--filter-speckle', action='store_true', help='Filter speckles after matching')
    
    # Rectification options
    pipe_parser.add_argument('--alpha', type=float, default=0.0, help='Rectification alpha (0-1)')
    
    # Depth options
    pipe_parser.add_argument('--max-depth', type=float, default=10000.0, help='Max depth in mm')
    
    # ROI options
    pipe_parser.add_argument('--roi', action='append', help='ROI specification: x,y,w,h,label or x,y,r,label')
    
    # Disparity only command
    disp_parser = subparsers.add_parser('disparity', help='Compute disparity from rectified images')
    disp_parser.add_argument('--left', required=True, help='Left rectified image')
    disp_parser.add_argument('--right', required=True, help='Right rectified image')
    disp_parser.add_argument('--output-dir', default='outputs/disparity', help='Output directory')
    disp_parser.add_argument('--min-disp', type=int, default=0)
    disp_parser.add_argument('--num-disp', type=int, default=64)
    disp_parser.add_argument('--block-size', type=int, default=5)
    
    # Depth only command
    depth_parser = subparsers.add_parser('depth', help='Compute depth from disparity and calibration')
    depth_parser.add_argument('--disparity', required=True, help='Disparity map (.npy or .png)')
    depth_parser.add_argument('--calibration', required=True, help='Stereo calibration JSON')
    depth_parser.add_argument('--output-dir', default='outputs/depth', help='Output directory')
    depth_parser.add_argument('--min-disp', type=float, default=1.0)
    depth_parser.add_argument('--max-depth', type=float, default=10000.0)
    
    args = parser.parse_args()
    
    if args.command == 'calibrate':
        return run_calibration(args)
    elif args.command == 'pipeline':
        return run_stereo_pipeline(args)
    elif args.command == 'disparity':
        return run_disparity_only(args)
    elif args.command == 'depth':
        return run_depth_only(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())