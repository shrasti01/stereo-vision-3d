"""
Camera Calibration Module

This module handles chessboard corner detection and intrinsic camera calibration
for mono and stereo cameras.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import json


@dataclass
class CalibrationResult:
    """Result of camera calibration."""
    success: bool
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    rvecs: Optional[List[np.ndarray]] = None
    tvecs: Optional[List[np.ndarray]] = None
    reprojection_error: float = 0.0
    image_size: Tuple[int, int] = (0, 0)
    num_valid_images: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class StereoCalibrationResult:
    """Result of stereo calibration."""
    success: bool
    left_camera_matrix: Optional[np.ndarray] = None
    left_dist_coeffs: Optional[np.ndarray] = None
    right_camera_matrix: Optional[np.ndarray] = None
    right_dist_coeffs: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None  # Rotation between cameras
    T: Optional[np.ndarray] = None  # Translation between cameras (baseline)
    E: Optional[np.ndarray] = None  # Essential matrix
    F: Optional[np.ndarray] = None  # Fundamental matrix
    R1: Optional[np.ndarray] = None  # Rectification transform left
    R2: Optional[np.ndarray] = None  # Rectification transform right
    P1: Optional[np.ndarray] = None  # Projection matrix left
    P2: Optional[np.ndarray] = None  # Projection matrix right
    Q: Optional[np.ndarray] = None   # Disparity-to-depth mapping matrix
    reprojection_error: float = 0.0
    image_size: Tuple[int, int] = (0, 0)
    num_valid_pairs: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ChessboardDetector:
    """Detects chessboard corners in calibration images."""
    
    def __init__(self, 
                 pattern_size: Tuple[int, int] = (9, 6),
                 square_size: float = 25.0):
        """
        Initialize chessboard detector.
        
        Args:
            pattern_size: Number of inner corners (width, height)
            square_size: Physical size of chessboard square in mm
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    def detect_corners(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect chessboard corners in a single image.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            Tuple of (found, corners) where corners is refined subpixel corners
        """
        gray = image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        found, corners = cv2.findChessboardCorners(
            gray, self.pattern_size, None
        )
        
        if found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), self.criteria
            )
            return True, corners_refined
        
        return False, None
    
    def get_object_points(self) -> np.ndarray:
        """
        Generate 3D object points for the chessboard pattern.
        
        Returns:
            Array of 3D points in world coordinates (mm)
        """
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def process_calibration_directory(self, 
                                      directory: str,
                                      visualize: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
        """
        Process all images in a calibration directory.
        
        Args:
            directory: Path to calibration images
            visualize: Whether to save visualization images
            
        Returns:
            Tuple of (object_points_list, image_points_list, image_size)
        """
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        image_size = None
        
        path = Path(directory)
        if not path.exists():
            print(f"Error: Calibration directory not found: {directory}")
            return [], [], (0, 0)
        
        images = sorted(list(path.glob("*.png")) + 
                       list(path.glob("*.jpg")) + 
                       list(path.glob("*.jpeg")))
        
        if not images:
            print(f"Warning: No images found in {directory}")
            return [], [], (0, 0)
        
        objp = self.get_object_points()
        
        vis_dir = Path(directory) / "visualizations"
        if visualize:
            vis_dir.mkdir(exist_ok=True)
        
        for idx, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            if image_size is None:
                image_size = (img.shape[1], img.shape[0])
            
            found, corners = self.detect_corners(img)
            
            if found:
                objpoints.append(objp)
                imgpoints.append(corners)
                
                if visualize:
                    vis_img = img.copy()
                    cv2.drawChessboardCorners(vis_img, self.pattern_size, corners, found)
                    cv2.imwrite(str(vis_dir / f"corners_{img_path.stem}.png"), vis_img)
                print(f"  Found corners in {img_path.name}")
            else:
                print(f"  No corners found in {img_path.name}")
        
        return objpoints, imgpoints, image_size


def calibrate_camera(objpoints: List[np.ndarray],
                     imgpoints: List[np.ndarray],
                     image_size: Tuple[int, int]) -> CalibrationResult:
    """
    Perform camera calibration from detected corners.
    
    Args:
        objpoints: List of 3D object points
        imgpoints: List of 2D image points
        image_size: Image size (width, height)
        
    Returns:
        CalibrationResult with camera matrix and distortion coefficients
    """
    if len(objpoints) < 3:
        return CalibrationResult(
            success=False,
            errors=[f"Need at least 3 valid images, got {len(objpoints)}"]
        )
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    
    return CalibrationResult(
        success=True,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
        reprojection_error=mean_error,
        image_size=image_size,
        num_valid_images=len(objpoints)
    )


def stereo_calibrate(objpoints: List[np.ndarray],
                     imgpoints_left: List[np.ndarray],
                     imgpoints_right: List[np.ndarray],
                     left_calib: CalibrationResult,
                     right_calib: CalibrationResult,
                     image_size: Tuple[int, int],
                     flags: int = cv2.CALIB_FIX_INTRINSIC) -> StereoCalibrationResult:
    """
    Perform stereo calibration.
    
    Args:
        objpoints: List of 3D object points
        imgpoints_left: List of 2D points from left camera
        imgpoints_right: List of 2D points from right camera
        left_calib: Left camera calibration result
        right_calib: Right camera calibration result
        image_size: Image size (width, height)
        flags: Calibration flags
        
    Returns:
        StereoCalibrationResult
    """
    if len(objpoints) < 3:
        return StereoCalibrationResult(
            success=False,
            errors=[f"Need at least 3 valid stereo pairs, got {len(objpoints)}"]
        )
    
    if len(imgpoints_left) != len(objpoints) or len(imgpoints_right) != len(objpoints):
        return StereoCalibrationResult(
            success=False,
            errors=["Mismatch in number of object/image points"]
        )
    
    # Perform stereo calibration
    ret, left_cam, left_dist, right_cam, right_dist, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        left_calib.camera_matrix, left_calib.dist_coeffs,
        right_calib.camera_matrix, right_calib.dist_coeffs,
        image_size, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=flags
    )
    
    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        left_cam, left_dist, right_cam, right_dist,
        image_size, R, T, alpha=0  # alpha=0 for no black borders
    )
    
    # Calculate reprojection error for stereo
    total_error = 0
    for i in range(len(objpoints)):
        # Project to left camera
        imgpoints_l_proj, _ = cv2.projectPoints(
            objpoints[i], np.zeros(3), np.zeros(3), left_cam, left_dist
        )
        error_l = cv2.norm(imgpoints_left[i], imgpoints_l_proj, cv2.NORM_L2) / len(imgpoints_l_proj)
        
        # Project to right camera (using R, T)
        imgpoints_r_proj, _ = cv2.projectPoints(
            objpoints[i], R, T, right_cam, right_dist
        )
        error_r = cv2.norm(imgpoints_right[i], imgpoints_r_proj, cv2.NORM_L2) / len(imgpoints_r_proj)
        
        total_error += (error_l + error_r) / 2
    
    mean_error = total_error / len(objpoints)
    
    return StereoCalibrationResult(
        success=True,
        left_camera_matrix=left_cam,
        left_dist_coeffs=left_dist,
        right_camera_matrix=right_cam,
        right_dist_coeffs=right_dist,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        reprojection_error=mean_error,
        image_size=image_size,
        num_valid_pairs=len(objpoints)
    )


def save_calibration(result: CalibrationResult, filepath: str):
    """Save mono calibration result to JSON."""
    data = {
        "camera_matrix": result.camera_matrix.tolist() if result.camera_matrix is not None else None,
        "dist_coeffs": result.dist_coeffs.tolist() if result.dist_coeffs is not None else None,
        "reprojection_error": result.reprojection_error,
        "image_size": result.image_size,
        "num_valid_images": result.num_valid_images
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_stereo_calibration(result: StereoCalibrationResult, filepath: str):
    """Save stereo calibration result to JSON."""
    def arr_to_list(arr):
        return arr.tolist() if arr is not None else None
    
    data = {
        "left_camera_matrix": arr_to_list(result.left_camera_matrix),
        "left_dist_coeffs": arr_to_list(result.left_dist_coeffs),
        "right_camera_matrix": arr_to_list(result.right_camera_matrix),
        "right_dist_coeffs": arr_to_list(result.right_dist_coeffs),
        "R": arr_to_list(result.R),
        "T": arr_to_list(result.T),
        "E": arr_to_list(result.E),
        "F": arr_to_list(result.F),
        "R1": arr_to_list(result.R1),
        "R2": arr_to_list(result.R2),
        "P1": arr_to_list(result.P1),
        "P2": arr_to_list(result.P2),
        "Q": arr_to_list(result.Q),
        "reprojection_error": result.reprojection_error,
        "image_size": result.image_size,
        "num_valid_pairs": result.num_valid_pairs
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_calibration(filepath: str) -> CalibrationResult:
    """Load mono calibration from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    result = CalibrationResult(success=True)
    result.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64) if data["camera_matrix"] else None
    result.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64) if data["dist_coeffs"] else None
    result.reprojection_error = data["reprojection_error"]
    result.image_size = tuple(data["image_size"])
    result.num_valid_images = data["num_valid_images"]
    return result


def load_stereo_calibration(filepath: str) -> StereoCalibrationResult:
    """Load stereo calibration from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    def list_to_arr(lst):
        return np.array(lst, dtype=np.float64) if lst is not None else None
    
    result = StereoCalibrationResult(success=True)
    result.left_camera_matrix = list_to_arr(data["left_camera_matrix"])
    result.left_dist_coeffs = list_to_arr(data["left_dist_coeffs"])
    result.right_camera_matrix = list_to_arr(data["right_camera_matrix"])
    result.right_dist_coeffs = list_to_arr(data["right_dist_coeffs"])
    result.R = list_to_arr(data["R"])
    result.T = list_to_arr(data["T"])
    result.E = list_to_arr(data["E"])
    result.F = list_to_arr(data["F"])
    result.R1 = list_to_arr(data["R1"])
    result.R2 = list_to_arr(data["R2"])
    result.P1 = list_to_arr(data["P1"])
    result.P2 = list_to_arr(data["P2"])
    result.Q = list_to_arr(data["Q"])
    result.reprojection_error = data["reprojection_error"]
    result.image_size = tuple(data["image_size"])
    result.num_valid_pairs = data["num_valid_pairs"]
    return result


def run_calibration_pipeline(left_calib_dir: str,
                             right_calib_dir: str,
                             pattern_size: Tuple[int, int] = (9, 6),
                             square_size: float = 25.0,
                             output_dir: str = "outputs/calibration",
                             visualize: bool = False) -> StereoCalibrationResult:
    """
    Run complete stereo calibration pipeline.
    
    Args:
        left_calib_dir: Left camera calibration images directory
        right_calib_dir: Right camera calibration images directory
        pattern_size: Chessboard pattern size (inner corners)
        square_size: Chessboard square size in mm
        output_dir: Output directory for calibration files
        visualize: Whether to save corner detection visualizations
        
    Returns:
        StereoCalibrationResult
    """
    print("=" * 60)
    print("STEREO CAMERA CALIBRATION PIPELINE")
    print("=" * 60)
    
    detector = ChessboardDetector(pattern_size, square_size)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process left camera
    print(f"\n[1/4] Processing LEFT camera images from: {left_calib_dir}")
    objpoints_left, imgpoints_left, img_size_left = detector.process_calibration_directory(
        left_calib_dir, visualize
    )
    
    # Process right camera
    print(f"\n[2/4] Processing RIGHT camera images from: {right_calib_dir}")
    objpoints_right, imgpoints_right, img_size_right = detector.process_calibration_directory(
        right_calib_dir, visualize
    )
    
    if not objpoints_left or not objpoints_right:
        return StereoCalibrationResult(
            success=False,
            errors=["No valid calibration images found"]
        )
    
    if img_size_left != img_size_right:
        return StereoCalibrationResult(
            success=False,
            errors=[f"Image size mismatch: left {img_size_left}, right {img_size_right}"]
        )
    
    image_size = img_size_left
    
    # Calibrate left camera
    print("\n[3/4] Calibrating LEFT camera...")
    left_calib = calibrate_camera(objpoints_left, imgpoints_left, image_size)
    if not left_calib.success:
        return StereoCalibrationResult(success=False, errors=left_calib.errors)
    print(f"  Reprojection error: {left_calib.reprojection_error:.4f} pixels")
    
    # Calibrate right camera
    print("\n[3/4] Calibrating RIGHT camera...")
    right_calib = calibrate_camera(objpoints_right, imgpoints_right, image_size)
    if not right_calib.success:
        return StereoCalibrationResult(success=False, errors=right_calib.errors)
    print(f"  Reprojection error: {right_calib.reprojection_error:.4f} pixels")
    
    # Stereo calibration
    print("\n[4/4] Performing stereo calibration...")
    # Use common object points (assuming same chessboard used for both)
    # We need to match the pairs - use minimum of both
    num_pairs = min(len(objpoints_left), len(objpoints_right))
    stereo_result = stereo_calibrate(
        objpoints_left[:num_pairs],
        imgpoints_left[:num_pairs],
        imgpoints_right[:num_pairs],
        left_calib,
        right_calib,
        image_size
    )
    
    if stereo_result.success:
        print(f"  Stereo reprojection error: {stereo_result.reprojection_error:.4f} pixels")
        print(f"  Baseline (T): {stereo_result.T.flatten() if stereo_result.T is not None else 'N/A'}")
        
        # Save results
        save_stereo_calibration(stereo_result, str(output_path / "stereo_calibration.json"))
        save_calibration(left_calib, str(output_path / "left_calibration.json"))
        save_calibration(right_calib, str(output_path / "right_calibration.json"))
        print(f"\n  Calibration saved to: {output_dir}")
    else:
        print(f"  Stereo calibration failed: {stereo_result.errors}")
    
    return stereo_result