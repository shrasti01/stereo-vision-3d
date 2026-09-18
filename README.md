# Stereo Vision 3D Distance Measurement System

A complete stereo computer vision pipeline for estimating real-world distances of objects from stereo image pairs. Built for a college Computer Vision project (2nd/3rd year B.Tech level).

## Overview

This system implements the complete stereo vision pipeline:

```
Left Image + Right Image
        |
        v
Image Validation / Preprocessing
        |
        v
Camera Calibration (Chessboard)
        |
        v
Stereo Rectification
        |
        v
Stereo Matching (SGBM/BM)
        |
        v
Disparity Map
        |
        v
Depth Estimation (Z = f*B/d)
        |
        v
Object / ROI Selection
        |
        v
Distance Measurement
        |
        v
Accuracy Evaluation
        |
        v
Results + Saved Outputs
```

## Core Theory

### Stereo Depth Equation

The fundamental equation for stereo depth estimation:

**Z = (f × B) / d**

Where:
- **Z** = Depth/distance to object (mm)
- **f** = Focal length in pixels (from camera calibration)
- **B** = Stereo baseline - distance between camera centers (mm)
- **d** = Disparity - horizontal pixel shift between left/right images

This equation derives from similar triangles in the stereo geometry:
- Triangle 1: Object point, left camera center, right camera center
- Triangle 2: Image points on left/right sensors, focal point

### Calibration

Camera calibration determines:
- **Intrinsic parameters**: Focal length (fx, fy), principal point (cx, cy), distortion coefficients
- **Extrinsic parameters**: Rotation (R) and translation (T) between cameras
- **Baseline**: |T| = distance between camera centers

### Rectification

Stereo rectification warps images so that:
- Epipolar lines become horizontal scanlines
- Corresponding points lie on same row (y_left = y_right)
- Enables efficient 1D stereo matching along horizontal lines

### Stereo Matching

**Semi-Global Block Matching (SGBM)** - Primary algorithm:
- Combines local matching with global smoothness constraints
- Uses dynamic programming along multiple paths
- Produces dense disparity maps with good edge preservation

**Block Matching (BM)** - Simpler alternative:
- Sliding window correlation
- Faster but noisier results

## Project Structure

```
stereo_vision_3d/
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── src/
│   ├── __init__.py
│   ├── cli/
│   │   └── main.py            # CLI commands
│   ├── calibration/
│   │   ├── camera_calibration.py  # Chessboard detection, calibration
│   │   └── __init__.py
│   ├── preprocessing/
│   │   ├── image_preprocessing.py # Load, validate, preprocess
│   │   └── __init__.py
│   ├── rectification/
│   │   ├── stereo_rectification.py # Rectify stereo pairs
│   │   └── __init__.py
│   ├── matching/
│   │   ├── stereo_matching.py   # SGBM, BM disparity computation
│   │   └── __init__.py
│   ├── depth/
│   │   ├── depth_estimation.py  # Disparity -> depth conversion
│   │   └── __init__.py
│   ├── measurement/
│   │   ├── distance_measurement.py # ROI selection, distance measurement
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── accuracy_evaluation.py # Error analysis, reporting
│   │   └── __init__.py
│   └── utils/
│       ├── io_utils.py          # File I/O, config management
│       ├── visualization.py     # Visualization utilities
│       └── __init__.py
├── data/
│   ├── calibration/
│   │   ├── left/              # Left camera calibration images
│   │   └── right/             # Right camera calibration images
│   ├── stereo/
│   │   ├── left/              # Left stereo images
│   │   └── right/             # Right stereo images
│   └── ground_truth/          # Optional ground truth data
└── outputs/                   # Generated results
    ├── calibration/
    ├── preprocessed/
    ├── rectified/
    ├── disparity/
    ├── depth/
    ├── measurements/
    ├── evaluation/
    └── visualizations/
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Camera Calibration

First, capture chessboard images with both cameras:
- Print a chessboard pattern (e.g., 9×6 inner corners, 25mm squares)
- Capture 15-20 pairs from different angles/distances
- Save as `data/calibration/left/*.png` and `data/calibration/right/*.png`

Run calibration:
```bash
python main.py calibrate \
    --left-calib data/calibration/left \
    --right-calib data/calibration/right \
    --pattern-size 9 6 \
    --square-size 25.0 \
    --output-dir outputs \
    --visualize
```

Outputs:
- `outputs/calibration/stereo_calibration.json` - Full stereo calibration
- `outputs/calibration/left_calibration.json` - Left camera intrinsics
- `outputs/calibration/right_calibration.json` - Right camera intrinsics
- `outputs/calibration/visualizations/` - Corner detection images

### 2. Full Pipeline (Calibration + Stereo Matching + Depth + Measurement)

Capture stereo image pairs:
- Save as `data/stereo/left/*.png` and `data/stereo/right/*.png`
- Filenames must match (e.g., `left/scene1.png`, `right/scene1.png`)

Run pipeline:
```bash
python main.py pipeline \
    --stereo-dir data/stereo \
    --left-calib data/calibration/left \
    --right-calib data/calibration/right \
    --output-dir outputs \
    --algorithm sgbm \
    --num-disp 64 \
    --block-size 5 \
    --roi "100,100,200,200,object1" \
    --roi "300,150,50,target"
```

Options:
- `--algorithm`: `sgbm` (default), `bm`, or `sgbm_3way` (with left-right check)
- `--num-disp`: Number of disparities (must be multiple of 16)
- `--block-size`: Matching block size (odd, typically 5-15)
- `--roi`: Region of interest - format `x,y,w,h,label` (rect) or `x,y,r,label` (circle)
- `--filter-speckle`: Enable speckle filtering
- `--max-depth`: Maximum depth in mm (default 10000)

### 3. Disparity Only (on pre-rectified images)

```bash
python main.py disparity \
    --left outputs/rectified/scene1_left_rect.png \
    --right outputs/rectified/scene1_right_rect.png \
    --output-dir outputs/disparity
```

### 4. Depth Only (from existing disparity + calibration)

```bash
python main.py depth \
    --disparity outputs/disparity/scene1_raw.npy \
    --calibration outputs/calibration/stereo_calibration.json \
    --output-dir outputs/depth
```

## Output Files

Each run generates organized outputs in `outputs/`:

| Directory | Contents |
|-----------|----------|
| `calibration/` | Camera matrices, distortion, stereo params (JSON) |
| `preprocessed/` | CLAHE-enhanced, resized input images |
| `rectified/` | Rectified left/right, side-by-side comparison |
| `disparity/` | Raw disparity (16-bit PNG, NPY), normalized, color-mapped |
| `depth/` | Depth map (mm, NPY), normalized, color-mapped, metadata |
| `measurements/` | CSV + JSON with distance measurements |
| `evaluation/` | Text report + JSON with accuracy metrics |
| `visualizations/` | Combined comparison images |

## ROI Specification

Regions of Interest for distance measurement:

```bash
# Rectangle: x,y,width,height,label
--roi "100,100,200,150,box"

# Center point (circular): x,y,radius,label
--roi "320,240,20,target"
```

Multiple ROIs supported (repeat `--roi`).

## Configuration

Create `config.yaml` for persistent settings:

```yaml
calibration:
  pattern_size: [9, 6]
  square_size_mm: 25.0
  
preprocessing:
  apply_clahe: true
  max_dimension: 1280
  
matching:
  algorithm: sgbm
  sgbm:
    num_disparities: 64
    block_size: 5
```

Use with: `--config config.yaml` (if implemented)

## Accuracy Evaluation

The system provides comprehensive evaluation:

- **Disparity Quality**: Valid pixel ratio, speckle count, texture coverage, left-right consistency
- **Depth Accuracy**: MAE, RMSE, relative error, accuracy thresholds (1mm, 5mm, 10mm, 5%)
- **Calibration Quality**: Reprojection errors, baseline, focal length
- **Measurement Error**: Per-ROI comparison with ground truth
- **Theoretical Error**: Depth error bounds from disparity quantization

## Theoretical Error Analysis

From error propagation on Z = f×B/d:

**ΔZ = (Z² / (f×B)) × Δd**

Where Δd ≈ 0.5-1 pixel (subpixel interpolation limit).

Example with f=800px, B=120mm:
| Distance | Disparity | Depth Error (Δd=0.5px) |
|----------|-----------|------------------------|
| 500 mm   | 192 px    | ~0.8 mm (0.16%)        |
| 1000 mm  | 96 px     | ~3.2 mm (0.32%)        |
| 2000 mm  | 48 px     | ~12.8 mm (0.64%)       |
| 5000 mm  | 19 px     | ~80 mm (1.6%)          |

**Key insight**: Error grows quadratically with distance. For accurate measurements, keep objects within 2-3m for typical webcam baselines.

## Requirements

- Python 3.8+
- OpenCV 4.8+ (opencv-python)
- NumPy 1.24+
- PyYAML 6.0+
- Matplotlib 3.7+ (optional, for plotting)

Optional for advanced features:
- opencv-contrib-python (for ximgproc: WLS filter, right matcher)

## Troubleshooting

### "No corners found" during calibration
- Ensure chessboard is fully visible, well-lit, not blurred
- Check pattern_size matches your printed chessboard (inner corners!)
- Try different angles/distances
- Use `--visualize` to debug corner detection

### "Image size mismatch"
- Left/right images must have identical resolution
- Calibration and stereo images must match in size
- Use `--max-dim` to resize consistently

### Poor disparity quality
- Increase `--num-disp` for farther objects
- Adjust `--block-size` (larger = smoother, smaller = more detail)
- Ensure good texture in scene (avoid plain walls)
- Check rectification quality (epipolar lines should be horizontal)

### Depth values seem wrong
- Verify calibration: check baseline (T vector) is reasonable
- Check focal length: should be ~0.8-1.2 × image width in pixels
- Ensure disparity is in pixels (not 16× scaled)
- Check min_disparity > 0 to avoid division by zero

## Academic Context

This project demonstrates:
1. **Camera calibration** (Zhang's method via OpenCV)
2. **Epipolar geometry** and stereo rectification
3. **Dense stereo matching** (SGBM algorithm)
4. **Depth from disparity** (triangulation)
5. **Error analysis** and accuracy evaluation
6. **Software engineering**: modular design, CLI, documentation

## References

- OpenCV Documentation: Camera Calibration, Stereo Vision
- Hirschmüller, H. (2008). "Stereo Processing by Semiglobal Matching and Mutual Information"
- Zhang, Z. (2000). "A Flexible New Technique for Camera Calibration"
- Hartley & Zisserman: "Multiple View Geometry in Computer Vision"

## License

Educational project - free for academic use.