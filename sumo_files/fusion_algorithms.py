"""
Fusion algorithms for reducing redundancy in V2X detections.
"""
from abc import ABC, abstractmethod
from detection import Detection
from evaluation import calculate_iou


class FusionAlgorithm(ABC):
    """Base class for fusion algorithms."""
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def fuse(self, detections, ego_id=None):
        """
        Fuse detections to reduce redundancy.
        
        Args:
            detections: List of Detection objects
            ego_id: Optional ego vehicle ID
            
        Returns:
            List of fused Detection objects
        """
        pass


class NoFusion(FusionAlgorithm):
    """No fusion - returns only ego vehicle detections."""
    
    def __init__(self):
        super().__init__("No Fusion")
    
    def fuse(self, detections, ego_id=None):
        """Return only detections made by the ego vehicle."""
        if ego_id is None:
            return detections
        # Handle both Detection objects and dicts
        return [d for d in detections 
                if (hasattr(d, 'detected_by') and d.detected_by == ego_id) or
                   (isinstance(d, dict) and d.get('detected_by') == ego_id)]


class NMSFusion(FusionAlgorithm):
    """Non-Maximum Suppression fusion algorithm."""
    
    def __init__(self, iou_threshold=0.5):
        super().__init__("NMS")
        self.iou_threshold = iou_threshold
    
    def fuse(self, detections, ego_id=None):
        """
        Sort by confidence and keep most confident detection,
        remove all detections with IOU > threshold.
        """
        # Convert to dict format for IOU calculation
        detections_dict = [d.to_dict() if isinstance(d, Detection) else d 
                          for d in detections]
        
        sorted_detections = sorted(detections_dict, 
                                  key=lambda d: d['confidence'], 
                                  reverse=True)
        
        fused = []
        used = [False] * len(sorted_detections)
        
        for i in range(len(sorted_detections)):
            if used[i]:
                continue
            
            # Keep detection with highest confidence
            fused.append(sorted_detections[i])
            used[i] = True
            
            # Remove closeby detections
            for j in range(i + 1, len(sorted_detections)):
                if used[j]:
                    continue
                iou = calculate_iou(sorted_detections[i], sorted_detections[j])
                if iou > self.iou_threshold:
                    used[j] = True
        
        # Convert back to Detection objects if needed
        return [Detection.from_dict(d) if isinstance(d, dict) else d 
                for d in fused]


class WeightedNMSFusion(FusionAlgorithm):
    """Weighted Non-Maximum Suppression fusion algorithm."""
    
    def __init__(self, iou_threshold=0.5):
        super().__init__("Weighted NMS")
        self.iou_threshold = iou_threshold
    
    def fuse(self, detections, ego_id=None):
        """
        Group detections with IOU > threshold, find weighted average.
        """
        # Convert to dict format for IOU calculation
        detections_dict = [d.to_dict() if isinstance(d, Detection) else d 
                          for d in detections]
        
        sorted_detections = sorted(detections_dict,
                                  key=lambda d: d['confidence'],
                                  reverse=True)
        
        fused = []
        used = [False] * len(sorted_detections)
        
        for i in range(len(sorted_detections)):
            if used[i]:
                continue
            
            group = [sorted_detections[i]]
            used[i] = True
            
            # Create group of detections close to primary detection
            for j in range(i + 1, len(sorted_detections)):
                if used[j]:
                    continue
                iou = calculate_iou(sorted_detections[i], sorted_detections[j])
                if iou > self.iou_threshold:
                    group.append(sorted_detections[j])
                    used[j] = True
            
            # Find weights
            weights = []
            for det in group:
                w = det['confidence'] / det['covariance']['x']
                weights.append(w)
            
            total_weight = sum(weights)
            
            # Find fused positions
            x_fused = sum(det['pos'][0] * w for det, w in zip(group, weights)) / total_weight
            y_fused = sum(det['pos'][1] * w for det, w in zip(group, weights)) / total_weight
            length_fused = sum(det['length'] * w for det, w in zip(group, weights)) / total_weight
            width_fused = sum(det['width'] * w for det, w in zip(group, weights)) / total_weight
            angle_fused = sum(det['angle'] * w for det, w in zip(group, weights)) / total_weight
            
            # Compute weighted average covariance
            cov_x = sum(det['covariance']['x'] * w for det, w in zip(group, weights)) / total_weight
            cov_y = sum(det['covariance']['y'] * w for det, w in zip(group, weights)) / total_weight
            cov_length = sum(det['covariance']['length'] * w for det, w in zip(group, weights)) / total_weight
            cov_width = sum(det['covariance']['width'] * w for det, w in zip(group, weights)) / total_weight
            cov_angle = sum(det['covariance']['angle'] * w for det, w in zip(group, weights)) / total_weight
            
            fused_det = {
                'pos': (x_fused, y_fused),
                'length': length_fused,
                'width': width_fused,
                'angle': angle_fused,
                'confidence': max(det['confidence'] for det in group),
                'covariance': {
                    'x': cov_x,
                    'y': cov_y,
                    'length': cov_length,
                    'width': cov_width,
                    'angle': cov_angle
                },
                'obj_id': [det['obj_id'] for det in group],
                'detected_by': group[0]['detected_by']  # Keep original detected_by
            }
            
            fused.append(fused_det)
        
        # Convert back to Detection objects if needed
        return [Detection.from_dict(d) if isinstance(d, dict) else d 
                for d in fused]
