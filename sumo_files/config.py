"""
Configuration constants for the V2X simulation.
"""
import numpy as np

# Communication and sensor parameters
COMM_RANGE = 50.0
SENSOR_RANGE = 60.0

# Noise parameters
POS_NOISE = 2.0
LENGTH_NOISE = 0.2
WIDTH_NOISE = 0.2
ANGLE_NOISE = 5.0
SPEED_NOISE = 0.5  # m/s
HEADING_NOISE = 3.0  # degrees

# Vehicle physical parameters
REAL_LENGTH = 4.5
REAL_WIDTH = 1.8

# Random seed for reproducibility
np.random.seed(0)

# IOU threshold for fusion algorithms
DEFAULT_IOU_THRESHOLD = 0.5

# Evaluation threshold
IOU_EVALUATION_THRESHOLD = 0.5
