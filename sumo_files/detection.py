"""
Detection module for vehicle sensor simulation using SDSM-style data packets.
"""
import math
import traci
import numpy as np
from config import (
    SENSOR_RANGE, POS_NOISE, LENGTH_NOISE, WIDTH_NOISE, 
    ANGLE_NOISE, REAL_LENGTH, REAL_WIDTH, SPEED_NOISE, HEADING_NOISE
)
from data_packet import DataPacket, VehicleState, ObjectDetection


class Sensor:
    """Simulates vehicle sensor for detecting surrounding vehicles and creating data packets."""
    
    def __init__(self, sensor_range=SENSOR_RANGE, pos_noise=POS_NOISE,
                 length_noise=LENGTH_NOISE, width_noise=WIDTH_NOISE,
                 angle_noise=ANGLE_NOISE, speed_noise=SPEED_NOISE,
                 heading_noise=HEADING_NOISE):
        self.sensor_range = sensor_range
        self.pos_noise = pos_noise
        self.length_noise = length_noise
        self.width_noise = width_noise
        self.angle_noise = angle_noise
        self.speed_noise = speed_noise
        self.heading_noise = heading_noise
        self._packet_counter = 0
    
    def create_data_packet(self, ego_id, veh_ids, timestamp):
        """
        Creates a data packet (SDSM-style) with vehicle state and object detections.
        
        Args:
            ego_id: ID of the vehicle creating this packet
            veh_ids: List of all vehicle IDs in simulation
            timestamp: Current simulation time
            
        Returns:
            DataPacket object
        """
        # Get ego vehicle state
        ego_pos = traci.vehicle.getPosition(ego_id)
        ego_speed = traci.vehicle.getSpeed(ego_id)
        ego_angle = traci.vehicle.getAngle(ego_id)
        ego_accel = traci.vehicle.getAcceleration(ego_id)
        
        # Create vehicle state with noise
        pos_noise = np.random.normal(0, self.pos_noise, 2)
        noisy_pos = (ego_pos[0] + pos_noise[0], ego_pos[1] + pos_noise[1])
        noisy_speed = max(0, ego_speed + np.random.normal(0, self.speed_noise))
        noisy_heading = ego_angle + np.random.normal(0, self.heading_noise)
        noisy_accel = ego_accel + np.random.normal(0, 0.5)  # Acceleration noise
        
        vehicle_state = VehicleState(
            vehicle_id=ego_id,
            position=noisy_pos,
            speed=noisy_speed,
            heading=noisy_heading,
            acceleration=noisy_accel,
            timestamp=timestamp,
            covariance={
                'x': self.pos_noise**2,
                'y': self.pos_noise**2,
                'speed': self.speed_noise**2,
                'heading': self.heading_noise**2
            }
        )
        
        # Detect surrounding objects
        detections = []
        for veh_id in veh_ids:
            if veh_id == ego_id:
                continue  # Skip itself
            
            veh_pos = traci.vehicle.getPosition(veh_id)
            veh_angle = traci.vehicle.getAngle(veh_id)
            veh_speed = traci.vehicle.getSpeed(veh_id)
            
            # Ignore if outside sensor detection range
            dist = math.dist(ego_pos, veh_pos)
            if dist > self.sensor_range:
                continue
            
            # Add noise to measured position
            pos_noise = np.random.normal(0, self.pos_noise, 2)
            veh_pos_w_noise = (veh_pos[0] + pos_noise[0], veh_pos[1] + pos_noise[1])
            
            # Add noise to dimensions and state
            noisy_length = np.random.normal(REAL_LENGTH, self.length_noise)
            noisy_width = np.random.normal(REAL_WIDTH, self.width_noise)
            noisy_heading = veh_angle + np.random.normal(0, self.angle_noise)
            noisy_speed = max(0, veh_speed + np.random.normal(0, self.speed_noise))
            
            # Decrease confidence as distance increases
            confidence = max(0.1, 1.0 - (dist / self.sensor_range))
            
            # Create object detection
            detection = ObjectDetection(
                object_id=veh_id,
                position=veh_pos_w_noise,
                speed=noisy_speed,
                heading=noisy_heading,
                length=noisy_length,
                width=noisy_width,
                confidence=confidence,
                covariance={
                    'x': self.pos_noise**2,
                    'y': self.pos_noise**2,
                    'speed': self.speed_noise**2,
                    'heading': self.angle_noise**2,
                    'length': self.length_noise**2,
                    'width': self.width_noise**2
                },
                detected_by=ego_id
            )
            
            detections.append(detection)
        
        # Create data packet
        self._packet_counter += 1
        packet = DataPacket(
            vehicle_id=ego_id,
            vehicle_state=vehicle_state,
            detections=detections,
            timestamp=timestamp,
            packet_id=self._packet_counter
        )
        
        return packet
