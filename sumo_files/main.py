"""
Main entry point for running V2X simulation with redundancy reduction algorithms.
"""
from simulation import Simulation
from visualization import Visualizer
from evaluation import Evaluator
from redundancy_algorithms import GreedyRedundancyReduction


def main():
    """Run simulation and visualize results."""
    # Initialize simulation with network and route files
    sim = Simulation(
        net_file="net.net.xml",
        route_file="routes.rou.xml",
        verbose=True
    )
    
    # Optionally add more algorithms here
    # Example:
    # sim.add_algorithm(GreedyRedundancyReduction(similarity_threshold=0.8))
    # sim.add_algorithm(AnotherAlgorithm())
    
    # Run simulation
    print("Starting simulation...")
    print("=" * 60)
    sim.run()
    
    # Get results
    results = sim.get_results()
    
    # Visualize results
    print("\n" + "=" * 60)
    print("Generating visualization...")
    evaluator = Evaluator()
    visualizer = Visualizer(evaluator)
    
    # Save visualization with timestamp
    output_path = visualizer.plot_results(results, output_dir='.')
    print(f"Results visualization saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    for alg_name, alg_results in results.items():
        stats = alg_results.get('aggregate_stats', {})
        if stats:
            print(f"\n{alg_name}:")
            print(f"  Packet Reduction: {stats.get('packet_reduction_pct_mean', 0):.1f}% "
                  f"(±{stats.get('packet_reduction_pct_std', 0):.1f}%)")
            print(f"  Size Reduction: {stats.get('size_reduction_pct_mean', 0):.1f}% "
                  f"(±{stats.get('size_reduction_pct_std', 0):.1f}%)")
            print(f"  Object Coverage: {stats.get('object_coverage_pct_mean', 0):.1f}% "
                  f"(±{stats.get('object_coverage_pct_std', 0):.1f}%)")
            print(f"  Detection Rate: {stats.get('detection_rate_mean', 0):.1f}% "
                  f"(±{stats.get('detection_rate_std', 0):.1f}%)")
            print(f"  Information Loss: {stats.get('information_loss_pct_mean', 0):.1f}% "
                  f"(±{stats.get('information_loss_pct_std', 0):.1f}%)")


if __name__ == "__main__":
    main()
