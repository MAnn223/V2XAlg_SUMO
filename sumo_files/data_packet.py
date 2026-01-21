"""
Data packet structure similar to SDSM (Sensor Data Sharing Message) format.
Contains vehicle state information and object detections.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import traci
import math


@dataclass
class VehicleState:
    """Vehicle state information (ego vehicle's own state)."""
    vehicle_id: str
    position: Tuple[float, float]  # (x, y)
    speed: float  # m/s
    heading: float  # degrees
    acceleration: float  # m/s^2
    timestamp: float
    covariance: Dict[str, float] = field(default_factory=dict)


@dataclass
class ObjectDetection:
    """Single object detection within a data packet."""
    object_id: str
    position: Tuple[float, float]  # (x, y)
    speed: float  # m/s
    heading: float  # degrees
    length: float  # m
    width: float  # m
    detected_by: str  # ID of vehicle that made this detection
    confidence: float  # 0.0 to 1.0
    covariance: Dict[str, float] = field(default_factory=dict)


@dataclass
class DataPacket:
    """
    SDSM-style data packet containing vehicle state and object detections.
    Similar to CPX-SDSM format but adapted for this simulation.
    """
    vehicle_id: str  # ID of vehicle sending this packet
    vehicle_state: VehicleState
    detections: List[ObjectDetection] = field(default_factory=list)
    timestamp: float = 0.0
    packet_id: int = 0  # Unique packet identifier
    
    def to_dict(self):
        """Convert packet to dictionary format."""
        return {
            'vehicle_id': self.vehicle_id,
            'vehicle_state': {
                'vehicle_id': self.vehicle_state.vehicle_id,
                'position': self.vehicle_state.position,
                'speed': self.vehicle_state.speed,
                'heading': self.vehicle_state.heading,
                'acceleration': self.vehicle_state.acceleration,
                'timestamp': self.vehicle_state.timestamp,
                'covariance': self.vehicle_state.covariance
            },
            'detections': [
                {
                    'object_id': det.object_id,
                    'position': det.position,
                    'speed': det.speed,
                    'heading': det.heading,
                    'length': det.length,
                    'width': det.width,
                    'confidence': det.confidence,
                    'covariance': det.covariance,
                    'detected_by': det.detected_by
                }
                for det in self.detections
            ],
            'timestamp': self.timestamp,
            'packet_id': self.packet_id
        }
    
    @classmethod
    def from_dict(cls, d):
        """Create DataPacket from dictionary."""
        vehicle_state = VehicleState(
            vehicle_id=d['vehicle_state']['vehicle_id'],
            position=tuple(d['vehicle_state']['position']),
            speed=d['vehicle_state']['speed'],
            heading=d['vehicle_state']['heading'],
            acceleration=d['vehicle_state']['acceleration'],
            timestamp=d['vehicle_state']['timestamp'],
            covariance=d['vehicle_state'].get('covariance', {})
        )
        
        detections = [
            ObjectDetection(
                object_id=det['object_id'],
                position=tuple(det['position']),
                speed=det['speed'],
                heading=det['heading'],
                length=det['length'],
                width=det['width'],
                confidence=det['confidence'],
                covariance=det.get('covariance', {}),
                detected_by=det['detected_by']
            )
            for det in d['detections']
        ]
        
        return cls(
            vehicle_id=d['vehicle_id'],
            vehicle_state=vehicle_state,
            detections=detections,
            timestamp=d['timestamp'],
            packet_id=d.get('packet_id', 0)
        )
    
    def get_total_size(self):
        """
        Estimate packet size (for redundancy metrics).
        Returns approximate size in bytes.
        """
        # Rough estimate: vehicle state + detections
        base_size = 100  # Vehicle state overhead
        detection_size = 80  # Per detection
        return base_size + len(self.detections) * detection_size
    
    def get_unique_objects(self):
        """Get set of unique object IDs in this packet."""
        return {det.object_id for det in self.detections}
