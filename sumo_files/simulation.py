"""
Main simulation class for V2X redundancy reduction algorithm evaluation.
"""
import traci
from detection import Sensor
from communication import CommunicationManager
from redundancy_algorithms import RedundancyReductionAlgorithm
from evaluation import Evaluator
from data_packet import DataPacket
import numpy as np
import os


class Simulation:
    """Main simulation class for running V2X redundancy reduction algorithms."""
    
    def __init__(self, net_file, route_file, algorithms=None, verbose=True):
        """
        Initialize simulation.
        
        Args:
            net_file: Path to SUMO network file
            route_file: Path to SUMO route file
            algorithms: List of RedundancyReductionAlgorithm instances to evaluate
            verbose: Whether to print progress messages
        """
        self.net_file = net_file
        self.route_file = route_file
        self.verbose = verbose
        
        # Initialize components
        self.sensor = Sensor()
        self.comm_manager = CommunicationManager()
        self.evaluator = Evaluator()
        
        # Set up algorithms
        if algorithms is None:
            from redundancy_algorithms import GreedyRedundancyReduction
            self.algorithms = [
                GreedyRedundancyReduction(similarity_threshold=0.7)
            ]
        else:
            self.algorithms = algorithms
        
        # Results storage: {algorithm_name: {
        #   'redundancy_metrics': [...],
        #   'information_loss_metrics': [...],
        #   'timestep_metrics': [...],
        #   'vehicle_count': 0,
        #   'simulation_time': 0
        # }}
        self.results = {alg.name: {
            'redundancy_metrics': [],
            'information_loss_metrics': [],
            'timestep_metrics': [],
            'vehicle_count': 0,
            'simulation_time': 0
        } for alg in self.algorithms}
    
    def run(self):
        """Run the simulation."""
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        net_path = os.path.join(script_dir, self.net_file)
        route_path = os.path.join(script_dir, self.route_file)
        
        # Use sumo (headless) instead of sumo-gui for better performance
        traci.start(["sumo", "-n", net_path, "-r", route_path])
        
        try:
            # Run simulation as long as there are vehicles expected
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                self._process_timestep()
            
            if self.verbose:
                print("\nSimulation completed.")
            
            # Store final simulation metadata
            sim_time = traci.simulation.getTime()
            veh_ids = traci.vehicle.getIDList()
            for alg_name in self.results:
                self.results[alg_name]['vehicle_count'] = len(veh_ids)
                self.results[alg_name]['simulation_time'] = sim_time
            
            # Compute aggregate statistics
            self._compute_aggregate_stats()
        
        finally:
            traci.close()
    
    def _process_timestep(self):
        """Process a single simulation timestep."""
        sim_time = traci.simulation.getTime()
        
        # Get all vehicles
        veh_ids = traci.vehicle.getIDList()
        if not veh_ids:
            return
        
        # Get all positions
        positions = {}
        for veh_id in veh_ids:
            positions[veh_id] = traci.vehicle.getPosition(veh_id)
        
        # Each vehicle creates a data packet
        all_packets = {}
        for veh_id in veh_ids:
            packet = self.sensor.create_data_packet(veh_id, veh_ids, sim_time)
            all_packets[veh_id] = packet
            
            if self.verbose and packet.detections:
                print(f"[t={sim_time:.1f}] {veh_id} created packet with {len(packet.detections)} detections")
        
        # Each vehicle shares with neighbors and applies redundancy reduction algorithms
        for veh_id in veh_ids:
            neighbors = self.comm_manager.find_neighbors(veh_id, positions)
            
            if neighbors:
                if self.verbose:
                    print(f"  {veh_id} can communicate with: {neighbors}")
                
                # Collect packets from ego and neighbors
                neighbor_packets = [all_packets[n_id] for n_id in neighbors if n_id in all_packets]
                all_received_packets = [all_packets[veh_id]] + neighbor_packets
                
                # Filter out self-detections from neighbor packets (keep ego's own detections)
                filtered_packets = []
                for packet in all_received_packets:
                    if packet.vehicle_id == veh_id:
                        # Keep ego packet as-is
                        filtered_packets.append(packet)
                    else:
                        # Filter out detections of ego vehicle from neighbor packets
                        filtered_detections = [
                            det for det in packet.detections 
                            if det.object_id != veh_id
                        ]
                        if filtered_detections:
                            # Create new packet with filtered detections
                            filtered_packet = DataPacket(
                                vehicle_id=packet.vehicle_id,
                                vehicle_state=packet.vehicle_state,
                                detections=filtered_detections,
                                timestamp=packet.timestamp,
                                packet_id=packet.packet_id
                            )
                            filtered_packets.append(filtered_packet)
                
                if self.verbose:
                    total_detections = sum(len(p.detections) for p in filtered_packets)
                    print(f"  {veh_id}: {len(filtered_packets)} packets, {total_detections} total detections")
                
                # Get ground truth for evaluation
                ground_truth = self.evaluator.get_ground_truth_objects(veh_ids, veh_id, positions)
                
                # Evaluate each algorithm
                for algorithm in self.algorithms:
                    # Apply redundancy reduction
                    reduced_packets = algorithm.reduce_redundancy(filtered_packets, veh_id)
                    
                    if self.verbose:
                        reduced_detections = sum(len(p.detections) for p in reduced_packets)
                        print(f"  {veh_id}: {algorithm.name} -> {len(reduced_packets)} packets, "
                              f"{reduced_detections} detections")
                    
                    # Evaluate redundancy reduction
                    redundancy_metrics = self.evaluator.evaluate_redundancy_reduction(
                        filtered_packets, reduced_packets
                    )
                    
                    # Evaluate information loss
                    information_loss_metrics = self.evaluator.evaluate_information_loss(
                        reduced_packets, ground_truth
                    )
                    
                    # Store results
                    self.results[algorithm.name]['redundancy_metrics'].append(redundancy_metrics)
                    self.results[algorithm.name]['information_loss_metrics'].append(information_loss_metrics)
                    self.results[algorithm.name]['timestep_metrics'].append({
                        'timestep': sim_time,
                        'vehicle_id': veh_id,
                        **redundancy_metrics,
                        **information_loss_metrics
                    })
                    
                    if self.verbose:
                        print(f"    {algorithm.name} - Packet reduction: "
                              f"{redundancy_metrics['packet_reduction_pct']:.1f}%, "
                              f"Coverage: {redundancy_metrics['object_coverage_pct']:.1f}%")
        
        if self.verbose:
            print()
    
    def _compute_aggregate_stats(self):
        """Compute aggregate statistics from all timesteps."""
        for alg_name in self.results:
            redundancy_list = self.results[alg_name]['redundancy_metrics']
            information_list = self.results[alg_name]['information_loss_metrics']
            
            # Combine metrics for statistics
            combined_metrics = []
            for red_metrics, info_metrics in zip(redundancy_list, information_list):
                combined = {**red_metrics, **info_metrics}
                combined_metrics.append(combined)
            
            # Compute aggregate statistics
            aggregate_stats = self.evaluator.compute_statistics(combined_metrics)
            self.results[alg_name]['aggregate_stats'] = aggregate_stats
    
    def get_results(self):
        """Get simulation results."""
        return self.results
    
    def add_algorithm(self, algorithm):
        """Add a new redundancy reduction algorithm to evaluate."""
        if not isinstance(algorithm, RedundancyReductionAlgorithm):
            raise ValueError("Algorithm must be an instance of RedundancyReductionAlgorithm")
        self.algorithms.append(algorithm)
        self.results[algorithm.name] = {
            'redundancy_metrics': [],
            'information_loss_metrics': [],
            'timestep_metrics': [],
            'vehicle_count': 0,
            'simulation_time': 0
        }