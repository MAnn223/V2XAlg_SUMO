"""
Visualization module for plotting simulation results with redundancy reduction metrics.
"""
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Dict, List
from evaluation import Evaluator


class Visualizer:
    """Creates visualizations of simulation results."""
    
    def __init__(self, evaluator=None):
        self.evaluator = evaluator or Evaluator()
    
    def plot_results(self, results_dict: Dict, output_dir: str = '.', 
                    timestamp: str = None) -> str:
        """
        Plot comparison of different redundancy reduction algorithms.
        
        Args:
            results_dict: Dictionary mapping algorithm names to results
                         Format: {algorithm_name: {
                             'redundancy_metrics': [...],
                             'information_loss_metrics': [...],
                             'timestep_metrics': [...]
                         }}
            output_dir: Directory to save the plot
            timestamp: Optional timestamp string (if None, will generate)
            
        Returns:
            Path to saved plot file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_file = f"{output_dir}/redundancy_reduction_{timestamp}.png"
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 1.5, 1], width_ratios=[1, 1, 1])
        
        # Top row: Redundancy reduction metrics
        ax_packets = fig.add_subplot(gs[0, 0])
        ax_size = fig.add_subplot(gs[0, 1])
        ax_detections = fig.add_subplot(gs[0, 2])
        
        # Middle row: Information preservation metrics
        ax_coverage = fig.add_subplot(gs[1, 0])
        ax_detection_rate = fig.add_subplot(gs[1, 1])
        ax_errors = fig.add_subplot(gs[1, 2])
        
        # Bottom row: Summary statistics
        ax_summary = fig.add_subplot(gs[2, :])
        
        # Extract data for plotting
        algorithm_names = list(results_dict.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(algorithm_names)))
        
        # Prepare data
        packet_reductions = []
        size_reductions = []
        detection_reductions = []
        coverage_rates = []
        detection_rates = []
        position_errors = []
        
        for alg_name in algorithm_names:
            stats = results_dict[alg_name].get('aggregate_stats', {})
            
            packet_reductions.append(stats.get('packet_reduction_pct_mean', 0))
            size_reductions.append(stats.get('size_reduction_pct_mean', 0))
            detection_reductions.append(stats.get('detection_reduction_pct_mean', 0))
            coverage_rates.append(stats.get('object_coverage_pct_mean', 0))
            detection_rates.append(stats.get('detection_rate_mean', 0))
            position_errors.append(stats.get('position_error_mean_mean', 0))
        
        # Plot 1: Packet Reduction
        bars1 = ax_packets.bar(algorithm_names, packet_reductions, color=colors, alpha=0.7, edgecolor='black')
        ax_packets.set_ylabel('Packet Reduction (%)', fontsize=11, fontweight='bold')
        ax_packets.set_title('Network Packet Reduction', fontsize=12, fontweight='bold')
        ax_packets.grid(axis='y', alpha=0.3)
        ax_packets.set_ylim([0, max(packet_reductions) * 1.2 if packet_reductions else 100])
        self._add_value_labels(ax_packets, bars1)
        
        # Plot 2: Size Reduction
        bars2 = ax_size.bar(algorithm_names, size_reductions, color=colors, alpha=0.7, edgecolor='black')
        ax_size.set_ylabel('Size Reduction (%)', fontsize=11, fontweight='bold')
        ax_size.set_title('Data Size Reduction', fontsize=12, fontweight='bold')
        ax_size.grid(axis='y', alpha=0.3)
        ax_size.set_ylim([0, max(size_reductions) * 1.2 if size_reductions else 100])
        self._add_value_labels(ax_size, bars2)
        
        # Plot 3: Detection Reduction
        bars3 = ax_detections.bar(algorithm_names, detection_reductions, color=colors, alpha=0.7, edgecolor='black')
        ax_detections.set_ylabel('Detection Reduction (%)', fontsize=11, fontweight='bold')
        ax_detections.set_title('Detection Count Reduction', fontsize=12, fontweight='bold')
        ax_detections.grid(axis='y', alpha=0.3)
        ax_detections.set_ylim([0, max(detection_reductions) * 1.2 if detection_reductions else 100])
        self._add_value_labels(ax_detections, bars3)
        
        # Plot 4: Object Coverage
        bars4 = ax_coverage.bar(algorithm_names, coverage_rates, color=colors, alpha=0.7, edgecolor='black')
        ax_coverage.set_ylabel('Object Coverage (%)', fontsize=11, fontweight='bold')
        ax_coverage.set_title('Information Preservation (Coverage)', fontsize=12, fontweight='bold')
        ax_coverage.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect Coverage')
        ax_coverage.grid(axis='y', alpha=0.3)
        ax_coverage.set_ylim([0, 105])
        ax_coverage.legend()
        self._add_value_labels(ax_coverage, bars4)
        
        # Plot 5: Detection Rate
        bars5 = ax_detection_rate.bar(algorithm_names, detection_rates, color=colors, alpha=0.9, edgecolor='black', hatch='/')
        ax_detection_rate.set_ylabel('Detection Rate (%)', fontsize=11, fontweight='bold')
        ax_detection_rate.set_title('Ground Truth Detection Rate', fontsize=12, fontweight='bold')
        ax_detection_rate.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect Detection')
        ax_detection_rate.grid(axis='y', alpha=0.3)
        ax_detection_rate.set_ylim([0, 105])
        ax_detection_rate.legend()
        self._add_value_labels(ax_detection_rate, bars5)
        
        # Plot 6: Position Error
        bars6 = ax_errors.bar(algorithm_names, position_errors, color=colors, alpha=0.9, edgecolor='black', hatch='\\')
        ax_errors.set_ylabel('Mean Position Error (m)', fontsize=11, fontweight='bold')
        ax_errors.set_title('Information Accuracy (Position Error)', fontsize=12, fontweight='bold')
        ax_errors.grid(axis='y', alpha=0.3)
        if position_errors:
            ax_errors.set_ylim([0, max(position_errors) * 1.3])
        self._add_value_labels(ax_errors, bars6)
        
        # Summary text - get vehicle count and simulation time from first algorithm
        first_algorithm = list(results_dict.keys())[0]
        vehicle_count = results_dict[first_algorithm].get('vehicle_count', 0)
        simulation_time = results_dict[first_algorithm].get('simulation_time', 0)
        summary_text = self._generate_summary_text(results_dict, vehicle_count, simulation_time)
        ax_summary.axis('off')
        ax_summary.text(0.02, 0.5, summary_text, fontsize=10, 
                       family='monospace', va='center', ha='left')
        
        plt.suptitle('V2X Network Redundancy Reduction Algorithm Evaluation', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as '{output_file}'")
        plt.close()  # Close to free memory
        
        return output_file
    
    def _add_value_labels(self, ax, bars):
        """Add value labels on top of bars."""
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%' if height < 100 else f'{height:.0f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    def _generate_summary_text(self, results_dict: Dict, vehicle_count: int, simulation_time: float) -> str:
        """Generate summary text for the statistics panel."""
        if not results_dict:
            return f"\n    No data collected during simulation.\n    Vehicle Count: {vehicle_count}\n    Simulation Duration: {simulation_time:.1f} sec."
        
        summary_lines = ["    Algorithm Performance Summary:", ""]
        
        for alg_name, results in results_dict.items():
            stats = results.get('aggregate_stats', {})
            if not stats:
                continue
            
            summary_lines.append(f"    {alg_name}:")
            
            # Redundancy reduction metrics
            summary_lines.append(f"      Redundancy Reduction:")
            summary_lines.append(f"        - Packet Reduction: {stats.get('packet_reduction_pct_mean', 0):.1f}% "
                               f"(±{stats.get('packet_reduction_pct_std', 0):.1f}%)")
            summary_lines.append(f"        - Size Reduction: {stats.get('size_reduction_pct_mean', 0):.1f}% "
                               f"(±{stats.get('size_reduction_pct_std', 0):.1f}%)")
            summary_lines.append(f"        - Detection Reduction: {stats.get('detection_reduction_pct_mean', 0):.1f}% "
                               f"(±{stats.get('detection_reduction_pct_std', 0):.1f}%)")
            
            # Information preservation metrics
            summary_lines.append(f"      Information Preservation:")
            summary_lines.append(f"        - Object Coverage: {stats.get('object_coverage_pct_mean', 0):.1f}% "
                               f"(±{stats.get('object_coverage_pct_std', 0):.1f}%)")
            summary_lines.append(f"        - Detection Rate: {stats.get('detection_rate_mean', 0):.1f}% "
                               f"(±{stats.get('detection_rate_std', 0):.1f}%)")
            summary_lines.append(f"        - Information Loss: {stats.get('information_loss_pct_mean', 0):.1f}% "
                               f"(±{stats.get('information_loss_pct_std', 0):.1f}%)")
            
            # Accuracy metrics
            summary_lines.append(f"      Accuracy:")
            summary_lines.append(f"      Vehicle Count: {vehicle_count}")
            summary_lines.append(f"      Simulation Duration: {simulation_time:.1f} sec")
            summary_lines.append(f"        - Mean Position Error: {stats.get('position_error_mean_mean', 0):.2f}m "
                               f"(±{stats.get('position_error_mean_std', 0):.2f}m)")
            summary_lines.append(f"        - Mean Speed Error: {stats.get('speed_error_mean_mean', 0):.2f}m/s "
                               f"(±{stats.get('speed_error_mean_std', 0):.2f}m/s)")
            summary_lines.append(f"        - Mean Heading Error: {stats.get('heading_error_mean_mean', 0):.2f}° "
                               f"(±{stats.get('heading_error_mean_std', 0):.2f}°)")
            
            summary_lines.append("")
        
        # Calculate efficiency score (redundancy reduction vs information preservation trade-off)
        if len(results_dict) > 1:
            summary_lines.append("    Efficiency Score (Reduction % × Coverage % / 100):")
            for alg_name, results in results_dict.items():
                stats = results.get('aggregate_stats', {})
                reduction = stats.get('packet_reduction_pct_mean', 0)
                coverage = stats.get('object_coverage_pct_mean', 0)
                efficiency = (reduction * coverage) / 100.0
                summary_lines.append(f"      - {alg_name}: {efficiency:.1f}")
            summary_lines.append("")
        
        return "\n".join(summary_lines)
