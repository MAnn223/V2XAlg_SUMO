"""
Evaluation module for measuring redundancy reduction and information loss.
"""
from typing import List, Dict, Set
from data_packet import DataPacket, ObjectDetection
import numpy as np
import math
import traci
from config import COMM_RANGE, REAL_LENGTH, REAL_WIDTH


class Evaluator:
    """Evaluates redundancy reduction algorithm performance."""
    
    def __init__(self):
        pass
    
    def evaluate_redundancy_reduction(self, original_packets: List[DataPacket], 
                                     reduced_packets: List[DataPacket]) -> Dict:
        """
        Evaluate how much redundancy was reduced.
        
        Args:
            original_packets: List of packets before redundancy reduction
            reduced_packets: List of packets after redundancy reduction
            
        Returns:
            Dictionary with redundancy reduction metrics
        """
        # Calculate total packet size
        original_size = sum(p.get_total_size() for p in original_packets)
        reduced_size = sum(p.get_total_size() for p in reduced_packets)
        
        # Calculate number of packets
        original_count = len(original_packets)
        reduced_count = len(reduced_packets)
        
        # Calculate total detections
        original_detections = sum(len(p.detections) for p in original_packets)
        reduced_detections = sum(len(p.detections) for p in reduced_packets)
        
        # Calculate unique objects
        original_objects = set()
        for p in original_packets:
            original_objects.update(p.get_unique_objects())
        
        reduced_objects = set()
        for p in reduced_packets:
            reduced_objects.update(p.get_unique_objects())
        
        # Calculate redundancy reduction percentages
        size_reduction = ((original_size - reduced_size) / original_size * 100) if original_size > 0 else 0.0
        packet_reduction = ((original_count - reduced_count) / original_count * 100) if original_count > 0 else 0.0
        detection_reduction = ((original_detections - reduced_detections) / original_detections * 100) if original_detections > 0 else 0.0
        
        # Information preservation (coverage of unique objects)
        object_coverage = (len(reduced_objects) / len(original_objects) * 100) if original_objects else 100.0
        
        return {
            'original_packets': original_count,
            'reduced_packets': reduced_count,
            'packet_reduction_pct': packet_reduction,
            'original_size_bytes': original_size,
            'reduced_size_bytes': reduced_size,
            'size_reduction_pct': size_reduction,
            'original_detections': original_detections,
            'reduced_detections': reduced_detections,
            'detection_reduction_pct': detection_reduction,
            'original_unique_objects': len(original_objects),
            'reduced_unique_objects': len(reduced_objects),
            'object_coverage_pct': object_coverage,
            'information_loss_pct': 100.0 - object_coverage
        }
    
    def evaluate_information_loss(self, reduced_packets: List[DataPacket], 
                                  ground_truth_objects: Dict[str, Dict]) -> Dict:
        """
        Evaluate information loss by comparing reduced packets to ground truth.
        
        Args:
            reduced_packets: List of packets after redundancy reduction
            ground_truth_objects: Dictionary mapping object_id to ground truth data
                                 Format: {obj_id: {'position': (x, y), 'speed': v, 'heading': h}}
            
        Returns:
            Dictionary with information loss metrics
        """
        if not reduced_packets:
            return {
                'detected_objects': 0,
                'total_objects': len(ground_truth_objects),
                'detection_rate': 0.0,
                'position_error_mean': 0.0,
                'position_error_std': 0.0,
                'speed_error_mean': 0.0,
                'speed_error_std': 0.0,
                'heading_error_mean': 0.0,
                'heading_error_std': 0.0
            }
        
        # Collect all detections from reduced packets
        all_detections = {}
        for packet in reduced_packets:
            for det in packet.detections:
                # Keep best detection per object (highest confidence)
                if det.object_id not in all_detections or det.confidence > all_detections[det.object_id].confidence:
                    all_detections[det.object_id] = det
        
        detected_objects = set(all_detections.keys())
        total_objects = set(ground_truth_objects.keys())
        
        # Calculate detection rate
        detection_rate = (len(detected_objects & total_objects) / len(total_objects) * 100) if total_objects else 0.0
        
        # Calculate errors for detected objects
        position_errors = []
        speed_errors = []
        heading_errors = []
        
        for obj_id in detected_objects & total_objects:
            det = all_detections[obj_id]
            gt = ground_truth_objects[obj_id]
            
            # Position error
            pos_error = math.dist(det.position, gt['position'])
            position_errors.append(pos_error)
            
            # Speed error
            speed_error = abs(det.speed - gt['speed'])
            speed_errors.append(speed_error)
            
            # Heading error
            heading_diff = abs(det.heading - gt['heading'])
            heading_diff = min(heading_diff, 360 - heading_diff)  # Wrap around
            heading_errors.append(heading_diff)
        
        return {
            'detected_objects': len(detected_objects & total_objects),
            'total_objects': len(total_objects),
            'detection_rate': detection_rate,
            'position_error_mean': np.mean(position_errors) if position_errors else 0.0,
            'position_error_std': np.std(position_errors) if position_errors else 0.0,
            'speed_error_mean': np.mean(speed_errors) if speed_errors else 0.0,
            'speed_error_std': np.std(speed_errors) if speed_errors else 0.0,
            'heading_error_mean': np.mean(heading_errors) if heading_errors else 0.0,
            'heading_error_std': np.std(heading_errors) if heading_errors else 0.0
        }
    
    def get_ground_truth_objects(self, veh_ids: List[str], ego_id: str, 
                                 positions: Dict[str, tuple]) -> Dict[str, Dict]:
        """
        Get ground truth object information for vehicles in communication range.
        
        Args:
            veh_ids: List of all vehicle IDs
            ego_id: Ego vehicle ID
            positions: Dictionary mapping vehicle IDs to positions
            
        Returns:
            Dictionary mapping object_id to ground truth data
        """
        ground_truth = {}
        ego_pos = positions.get(ego_id)
        
        if ego_pos is None:
            return ground_truth
        
        for veh_id in veh_ids:
            if veh_id == ego_id:
                continue
            
            veh_pos = positions.get(veh_id)
            if veh_pos is None:
                continue
            
            # Only include vehicles in communication range
            if math.dist(ego_pos, veh_pos) <= COMM_RANGE:
                try:
                    veh_speed = traci.vehicle.getSpeed(veh_id)
                    veh_angle = traci.vehicle.getAngle(veh_id)
                    
                    ground_truth[veh_id] = {
                        'position': veh_pos,
                        'speed': veh_speed,
                        'heading': veh_angle,
                        'length': REAL_LENGTH,
                        'width': REAL_WIDTH
                    }
                except:
                    continue
        
        return ground_truth
    
    def compute_statistics(self, metrics_list: List[Dict]) -> Dict:
        """
        Compute aggregate statistics from a list of metric dictionaries.
        
        Args:
            metrics_list: List of metric dictionaries from multiple timesteps
            
        Returns:
            Dictionary with aggregate statistics
        """
        if not metrics_list:
            return {}
        
        # Aggregate metrics
        aggregated = {
            'packet_reduction_pct': [],
            'size_reduction_pct': [],
            'detection_reduction_pct': [],
            'object_coverage_pct': [],
            'information_loss_pct': [],
            'detection_rate': [],
            'position_error_mean': [],
            'speed_error_mean': [],
            'heading_error_mean': []
        }
        
        for metrics in metrics_list:
            for key in aggregated.keys():
                if key in metrics:
                    aggregated[key].append(metrics[key])
        
        # Calculate statistics
        stats = {}
        for key, values in aggregated.items():
            if values:
                stats[f'{key}_mean'] = float(np.mean(values))
                stats[f'{key}_std'] = float(np.std(values))
                stats[f'{key}_median'] = float(np.median(values))
                stats[f'{key}_min'] = float(np.min(values))
                stats[f'{key}_max'] = float(np.max(values))
        
        return stats
