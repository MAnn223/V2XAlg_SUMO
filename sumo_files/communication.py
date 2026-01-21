"""
Communication module for V2X message exchange.
"""
import math
from config import COMM_RANGE


class CommunicationManager:
    """Manages vehicle-to-vehicle communication."""
    
    def __init__(self, comm_range=COMM_RANGE):
        self.comm_range = comm_range
    
    def find_neighbors(self, veh_id, positions):
        """
        Find all vehicles within communication range.
        
        Args:
            veh_id: ID of the vehicle
            positions: Dictionary mapping vehicle IDs to (x, y) positions
            
        Returns:
            List of neighbor vehicle IDs
        """
        cur_pos = positions[veh_id]
        neighbors = []
        
        for n_id, n_pos in positions.items():
            if n_id == veh_id:
                continue
            if math.dist(cur_pos, n_pos) <= self.comm_range:
                neighbors.append(n_id)
        
        return neighbors
    
    def broadcast_detections(self, veh_detections, neighbor_detections):
        """
        Combine vehicle's own detections with detections from neighbors.
        
        Args:
            veh_detections: List of Detection objects from the vehicle
            neighbor_detections: List of lists of Detection objects from neighbors
            
        Returns:
            Combined list of Detection objects
        """
        all_detections = list(veh_detections)
        for n_detections in neighbor_detections:
            all_detections.extend(n_detections)
        return all_detections
