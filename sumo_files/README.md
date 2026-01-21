# V2X Algorithm Simulation Framework

A modular framework for evaluating different fusion algorithms that reduce redundancy in vehicle-to-vehicle (V2X) message broadcasting.

## Structure

The codebase is organized into the following modules:

### Core Modules

- **`config.py`**: Configuration constants (communication range, sensor parameters, noise levels, etc.)
- **`detection.py`**: Vehicle detection simulation
  - `Detection`: Represents a single vehicle detection
  - `Sensor`: Simulates vehicle sensors with noise

### Communication

- **`communication.py`**: V2X communication management
  - `CommunicationManager`: Handles neighbor discovery and message broadcasting

### Algorithms

- **`fusion_algorithms.py`**: Fusion algorithm implementations
  - `FusionAlgorithm`: Abstract base class for all fusion algorithms
  - `NoFusion`: Baseline - returns only ego vehicle detections
  - `NMSFusion`: Non-Maximum Suppression fusion
  - `WeightedNMSFusion`: Weighted Non-Maximum Suppression fusion

### Evaluation

- **`evaluation.py`**: Performance metrics
  - `Evaluator`: Calculates IOU scores and statistical metrics
  - `calculate_iou()`: IOU calculation function

### Visualization

- **`visualization.py`**: Results plotting
  - `Visualizer`: Creates comparison plots and statistical summaries

### Simulation

- **`simulation.py`**: Main simulation engine
  - `Simulation`: Orchestrates the entire simulation and algorithm evaluation

### Entry Point

- **`main.py`**: Main script to run simulations

## Usage

### Basic Usage

Run the simulation with default algorithms:

```python
python main.py
```

### Custom Algorithms

To evaluate your own fusion algorithm:

```python
from simulation import Simulation
from fusion_algorithms import FusionAlgorithm
from visualization import Visualizer

# Define your custom algorithm
class MyCustomAlgorithm(FusionAlgorithm):
    def __init__(self):
        super().__init__("My Custom Algorithm")
    
    def fuse(self, detections, ego_id=None):
        # Your fusion logic here
        return fused_detections

# Create simulation with custom algorithms
sim = Simulation(
    net_file="net.net.xml",
    route_file="routes.rou.xml",
    algorithms=[
        NoFusion(),  # Baseline
        NMSFusion(),  # Existing algorithm
        MyCustomAlgorithm()  # Your algorithm
    ]
)

sim.run()
results = sim.get_results()

# Visualize
visualizer = Visualizer()
visualizer.plot_results(results)
```

### Adding Algorithms to Existing Simulation

```python
from fusion_algorithms import NMSFusion

sim = Simulation(net_file="net.net.xml", route_file="routes.rou.xml")
sim.add_algorithm(NMSFusion(iou_threshold=0.6))  # Custom threshold
sim.run()
```

## File Structure

```
sumo_files/
├── __init__.py           # Package initialization
├── config.py             # Configuration constants
├── detection.py          # Detection classes
├── communication.py      # Communication management
├── fusion_algorithms.py  # Fusion algorithm implementations
├── evaluation.py         # Evaluation metrics
├── visualization.py      # Plotting functionality
├── simulation.py         # Main simulation class
├── main.py              # Entry point
├── net.net.xml          # SUMO network file
└── routes.rou.xml       # SUMO route file
```

## Extending the Framework

### Creating a New Fusion Algorithm

1. Inherit from `FusionAlgorithm`:

```python
from fusion_algorithms import FusionAlgorithm

class MyAlgorithm(FusionAlgorithm):
    def __init__(self, param1, param2):
        super().__init__("My Algorithm Name")
        self.param1 = param1
        self.param2 = param2
    
    def fuse(self, detections, ego_id=None):
        # Your fusion logic
        # detections is a list of Detection objects
        # Return a list of fused Detection objects
        return fused_detections
```

2. Add it to your simulation:

```python
sim = Simulation(..., algorithms=[MyAlgorithm(param1=1, param2=2)])
```

### Custom Evaluation Metrics

Extend the `Evaluator` class to add custom metrics:

```python
from evaluation import Evaluator

class MyEvaluator(Evaluator):
    def compute_custom_metric(self, detections):
        # Your metric calculation
        return metric_value
```

## Configuration

Modify `config.py` to adjust:
- Communication range (`COMM_RANGE`)
- Sensor range (`SENSOR_RANGE`)
- Noise parameters (`POS_NOISE`, `LENGTH_NOISE`, etc.)
- IOU thresholds

## Output

The simulation generates:
- Console output with per-timestep statistics
- `fusion_comparison_iou.png`: Visualization comparing all algorithms
  - Box plots of IOU distributions
  - Time series of IOU scores per vehicle
  - Statistical summary with improvements
