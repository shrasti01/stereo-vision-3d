#!/usr/bin/env python3
"""
Stereo Vision 3D Distance Measurement System

A complete stereo computer vision pipeline for estimating object distances
from stereo image pairs.

Usage:
    python main.py calibrate --left-calib data/calibration/left --right-calib data/calibration/right
    python main.py pipeline --stereo-dir data/stereo --left-calib data/calibration/left --right-calib data/calibration/right
    python main.py disparity --left left_rect.png --right right_rect.png
    python main.py depth --disparity disparity.npy --calibration stereo_calibration.json

Author: College Computer Vision Project
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cli.main import main

if __name__ == '__main__':
    sys.exit(main())