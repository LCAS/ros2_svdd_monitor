# ROS2 SVDD Monitor

A ROS2 Humble package for real-time anomaly detection using Support Vector Data Description (SVDD). This package monitors robot behavior by analyzing IMU and velocity command data to detect anomalous patterns.

## Overview

The `ros2_svdd_monitor` package implements anomaly detection using One-Class SVM (Support Vector Data Description). It subscribes to `/imu` and `/cmd_vel` topics, extracts features from the sensor data, and uses a trained SVDD model to detect anomalies in robot behavior.

## Features

- **Real-time anomaly detection** using SVDD/One-Class SVM
- **Feature extraction** from IMU and cmd_vel data
- **Two-phase operation**: training and monitoring
- **Configurable parameters** for model training and detection
- **ROS2 integration** with standard message types
- **Comprehensive testing** with pytest

## Installation

### Prerequisites

- ROS2 Humble
- Python 3.8+
- pip

### Dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Build

```bash
cd /path/to/workspace
colcon build --packages-select ros2_svdd_monitor
source install/setup.bash
```

## Usage

### Training Phase

First, collect normal operation data and train the SVDD model:

```bash
ros2 run ros2_svdd_monitor train
```

The training node will:
1. Subscribe to `/imu` and `/cmd_vel` topics
2. Collect data for 60 seconds (configurable)
3. Extract features from the sensor data
4. Train the SVDD model
5. Save the trained model to `~/svdd_model.pkl`

**Important**: Ensure your robot is operating normally during the training phase, as the model will learn what "normal" behavior looks like.

### Monitoring Phase

After training, run the monitoring node to detect anomalies:

```bash
ros2 run ros2_svdd_monitor monitor
```

The monitoring node will:
1. Load the trained model
2. Subscribe to `/imu` and `/cmd_vel` topics
3. Continuously extract features and detect anomalies
4. Publish anomaly detection results to:
   - `/anomaly_detected` (std_msgs/Bool): True if anomaly detected
   - `/anomaly_score` (std_msgs/Float32): Anomaly score (negative = anomaly)

## Topics

### Subscribed Topics

- `/imu` (sensor_msgs/Imu): IMU sensor data
- `/cmd_vel` (geometry_msgs/Twist): Velocity commands

### Published Topics (Monitor Node)

- `/anomaly_detected` (std_msgs/Bool): Boolean indicating if anomaly is detected
- `/anomaly_score` (std_msgs/Float32): Anomaly score from the SVDD model

## Configuration

Configuration parameters are defined in `ros2_svdd_monitor/config.py`:

### Model Parameters
- `nu`: Upper bound on fraction of outliers (default: 0.1)
- `kernel`: Kernel type for SVM (default: 'rbf')
- `gamma`: Kernel coefficient (default: 'auto')

### Feature Parameters
- `window_size`: Number of samples for feature computation (default: 10)
- `feature_buffer_size`: Buffer size for streaming features (default: 100)

### Training Parameters
- `training_duration`: Duration in seconds for training data collection (default: 60.0)

### Monitoring Parameters
- `anomaly_threshold`: Decision threshold (default: 0.0)
- `publish_rate`: Rate for publishing results in Hz (default: 1.0)

## Architecture

### Modules

1. **config.py**: Configuration parameters for the SVDD system
2. **features.py**: Feature extraction from IMU and cmd_vel data
3. **svdd_model.py**: SVDD model implementation using scikit-learn's OneClassSVM
4. **train_node.py**: ROS2 node for training the SVDD model
5. **monitor_node.py**: ROS2 node for real-time anomaly detection

### Feature Extraction

The feature extractor computes statistical features from windowed sensor data:
- **IMU features**: mean, std, min, max of linear acceleration (x, y, z) and angular velocity (x, y, z)
- **cmd_vel features**: mean, std, min, max of linear velocity (x, y, z) and angular velocity (x, y, z)

Total: 48 features (6 dimensions × 4 statistics × 2 data sources)

## Testing

Run tests using pytest:

```bash
cd src/ros2_svdd_monitor
pytest
```

Or using colcon:

```bash
colcon test --packages-select ros2_svdd_monitor
colcon test-result --verbose
```

## Development

### Development Environment

This package includes a devcontainer configuration for development with VSCode. See the devcontainer documentation in `.devcontainer/` for setup instructions.

### Code Style

This package follows ROS2 Python style guidelines. Run linters:

```bash
ament_flake8 ros2_svdd_monitor
ament_pep257 ros2_svdd_monitor
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the LCAS ros2_pkg_template
- Uses scikit-learn for One-Class SVM implementation

## Troubleshooting

### No features collected during training

**Problem**: Training completes but reports "No features collected during training period!"

**Solution**: 
- Ensure `/imu` and `/cmd_vel` topics are actively publishing data
- Check topic names match your robot's configuration
- Verify data is being published at a reasonable rate (> 1 Hz)

### Model file not found

**Problem**: Monitor node reports "Model file not found"

**Solution**: 
- Run the training node first: `ros2 run ros2_svdd_monitor train`
- Check that the model was saved successfully
- Verify the model path in config.py matches your system

### Import errors

**Problem**: "ModuleNotFoundError" for numpy or sklearn

**Solution**: 
- Install dependencies: `pip install -r requirements.txt`
- Ensure your ROS2 workspace is properly sourced

## Contributing

Contributions are welcome! Please ensure:
- Code passes all tests
- New features include corresponding tests
- Code follows ROS2 Python style guidelines
