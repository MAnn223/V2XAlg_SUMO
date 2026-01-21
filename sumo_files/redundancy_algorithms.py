"""
Redundancy reduction algorithms for V2X networks.
These algorithms reduce redundant data transmission while preserving information.
"""
from abc import ABC, abstractmethod
from typing import List, Set
from data_packet import DataPacket, ObjectDetection
import math


class RedundancyReductionAlgorithm(ABC):
    """Base class for redundancy reduction algorithms."""
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def reduce_redundancy(self, packets: List[DataPacket], ego_id: str = None) -> List[DataPacket]:
        """
        Reduce redundancy in a list of data packets.
        
        Args:
            packets: List of DataPacket objects from ego and neighbors
            ego_id: Optional ego vehicle ID
            
        Returns:
            List of DataPacket objects with reduced redundancy
        """
        pass
    
    def _calculate_detection_similarity(self, det1: ObjectDetection, det2: ObjectDetection, 
                                       position_threshold: float = 5.0) -> float:
        """
        Calculate similarity between two detections (0.0 to 1.0).
        Higher value means more similar/redundant.
        
        Args:
            det1: First detection
            det2: Second detection
            position_threshold: Distance threshold in meters for considering detections similar
            
        Returns:
            Similarity score (0.0 = different, 1.0 = identical/redundant)
        """
        # Calculate position distance
        pos_dist = math.dist(det1.position, det2.position)
        
        # If too far apart, not similar
        if pos_dist > position_threshold:
            return 0.0
        
        # Calculate heading difference
        heading_diff = abs(det1.heading - det2.heading)
        heading_diff = min(heading_diff, 360 - heading_diff)  # Wrap around
        
        # Calculate speed difference
        speed_diff = abs(det1.speed - det2.speed)
        
        # Normalize differences
        pos_sim = 1.0 - (pos_dist / position_threshold)
        heading_sim = 1.0 - (heading_diff / 180.0)  # Max difference is 180 degrees
        speed_sim = 1.0 - min(speed_diff / 10.0, 1.0)  # Normalize by 10 m/s
        
        # Weighted combination
        similarity = 0.5 * pos_sim + 0.3 * heading_sim + 0.2 * speed_sim
        
        return max(0.0, min(1.0, similarity))


class GreedyRedundancyReduction(RedundancyReductionAlgorithm):
    """
    Greedy algorithm for redundancy reduction.
    Selects packets greedily to maximize information coverage while minimizing redundancy.
    """
    
    def __init__(self, similarity_threshold: float = 0.7, max_packets: int = None):
        """
        Initialize greedy redundancy reduction algorithm.
        
        Args:
            similarity_threshold: Threshold for considering detections redundant (0.0 to 1.0)
            max_packets: Maximum number of packets to keep (None = no limit)
        """
        super().__init__("Greedy")
        self.similarity_threshold = similarity_threshold
        self.max_packets = max_packets
    
    def reduce_redundancy(self, packets: List[DataPacket], ego_id: str = None) -> List[DataPacket]:
        """
        Greedily select packets to minimize redundancy.
        
        Strategy:
        1. Always include ego vehicle's packet (if present)
        2. Greedily add packets that add the most new information
        3. Skip packets that are highly redundant with already selected ones
        """
        if not packets:
            return []
        
        # Separate ego packet from neighbor packets
        ego_packet = None
        neighbor_packets = []
        
        for packet in packets:
            if packet.vehicle_id == ego_id:
                ego_packet = packet
            else:
                neighbor_packets.append(packet)
        
        # Start with ego packet if available
        selected_packets = []
        covered_objects: Set[str] = set()
        
        if ego_packet:
            selected_packets.append(ego_packet)
            covered_objects.update(ego_packet.get_unique_objects())
        
        # Greedily select neighbor packets
        remaining_packets = neighbor_packets.copy()
        
        while remaining_packets:
            best_packet = None
            best_score = -1
            best_idx = -1
            
            # Find packet that adds most new information
            for idx, packet in enumerate(remaining_packets):
                # Calculate information gain
                new_objects = packet.get_unique_objects() - covered_objects
                redundancy_score = 0.0
                
                # Calculate redundancy with already selected packets
                for selected in selected_packets:
                    redundancy = self._calculate_packet_redundancy(packet, selected)
                    redundancy_score += redundancy
                
                # Score = new information - redundancy penalty
                # Normalize by number of detections
                num_detections = len(packet.detections) if packet.detections else 1
                info_gain = len(new_objects) / num_detections
                redundancy_penalty = redundancy_score / len(selected_packets) if selected_packets else 0
                
                score = info_gain - (redundancy_penalty * 0.5)
                
                if score > best_score:
                    best_score = score
                    best_packet = packet
                    best_idx = idx
            
            # If no good packet found or max reached, stop
            if best_packet is None or best_score < 0:
                break
            
            if self.max_packets and len(selected_packets) >= self.max_packets:
                break
            
            # Add best packet
            selected_packets.append(best_packet)
            covered_objects.update(best_packet.get_unique_objects())
            remaining_packets.pop(best_idx)
        
        return selected_packets
    
    def _calculate_packet_redundancy(self, packet1: DataPacket, packet2: DataPacket) -> float:
        """
        Calculate redundancy between two packets (0.0 to 1.0).
        Higher value means more redundant.
        
        Args:
            packet1: First packet
            packet2: Second packet
            
        Returns:
            Redundancy score
        """
        if not packet1.detections or not packet2.detections:
            return 0.0
        
        total_similarity = 0.0
        comparisons = 0
        
        # Compare all detections between packets
        for det1 in packet1.detections:
            for det2 in packet2.detections:
                if det1.object_id == det2.object_id:
                    # Same object - calculate similarity
                    similarity = self._calculate_detection_similarity(det1, det2)
                    total_similarity += similarity
                    comparisons += 1
        
        if comparisons == 0:
            return 0.0
        
        return total_similarity / comparisons
