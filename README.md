# Stereo Vision 3D Distance Measurement System

## Project Description

A computer vision system that estimates real-world distances of objects from stereo image pairs using OpenCV and Python. The system implements the complete stereo vision pipeline: camera calibration using chessboard patterns, stereo rectification for epipolar alignment, Semi-Global Block Matching (SGBM) for dense disparity computation, and depth estimation using the fundamental stereo equation Z = (f x B) / d. The project supports region-of-interest selection for measuring distances to specific objects and provides comprehensive accuracy evaluation with error analysis. This system demonstrates practical applications of stereo vision in robotics, autonomous navigation, and 3D scene understanding without relying on deep learning methods.

## Features

- Camera calibration with chessboard detection
- Stereo rectification and epipolar alignment
- SGBM and Block Matching disparity computation
- Real-time depth map generation
- ROI-based distance measurement
- Accuracy evaluation and error analysis
- Command-line interface

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run full pipeline
python main.py pipeline --stereo-dir data/stereo --left-calib data/calibration/left --right-calib data/calibration/right

# Compute disparity only
python main.py disparity --left left_rect.png --right right_rect.png

# Compute depth from disparity
python main.py depth --disparity disparity.png --calibration stereo_calibration.json
```

## Pipeline

```
Input Images -> Preprocessing -> Calibration -> Rectification -> Matching -> Disparity -> Depth -> Measurement -> Output
```

## Technologies

- Python 3.8+
- OpenCV 4.8+
- NumPy
- Matplotlib (optional)

## License

Educational project - free for academic use.