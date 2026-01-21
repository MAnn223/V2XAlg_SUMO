"""
V2X Algorithm Simulation Package

This package provides a modular framework for evaluating different
fusion algorithms that reduce redundancy in vehicle-to-vehicle communication.
"""

from .simulation import Simulation
from .detection import Sensor, Detection
from .communication import CommunicationManager
from .fusion_algorithms import (
    FusionAlgorithm,
    NoFusion,
    NMSFusion,
    WeightedNMSFusion
)
from .evaluation import Evaluator, calculate_iou
from .visualization import Visualizer

__all__ = [
    'Simulation',
    'Sensor',
    'Detection',
    'CommunicationManager',
    'FusionAlgorithm',
    'NoFusion',
    'NMSFusion',
    'WeightedNMSFusion',
    'Evaluator',
    'calculate_iou',
    'Visualizer'
]
