# ros2_svdd_monitor

[![Python Tests](https://github.com/LCAS/ros2_svdd_monitor/actions/workflows/pytest.yml/badge.svg)](https://github.com/LCAS/ros2_svdd_monitor/actions/workflows/pytest.yml)
[![ROS CI](https://github.com/LCAS/ros2_svdd_monitor/actions/workflows/ros-ci.yml/badge.svg)](https://github.com/LCAS/ros2_svdd_monitor/actions/workflows/ros-ci.yml)

SVDD-style anomaly detection for ROS 2 using proprioceptive sensors (IMU and cmd_vel).

This package implements Support Vector Data Description (SVDD) using OneClassSVM for real-time anomaly detection on robotic systems. It monitors the relationship between commanded velocities (`/cmd_vel`) and actual IMU measurements (`/imu`) to detect anomalous behavior that might indicate hardware failures, unexpected terrain, or other system issues.

## Features

- **Proprioceptive-only monitoring**: Uses only `/cmd_vel` and `/imu` topics (no external sensors required)
- **Feature engineering**: Extracts sliding-window features that capture the expected relationship between commanded motion and IMU response
- **SVDD anomaly detection**: Uses OneClassSVM as a proxy for SVDD to learn normal operation patterns
- **Real-time monitoring**: ROS 2 node that publishes anomaly detection results
- **Configurable**: Hyperparameters (window size, nu, gamma) configurable via YAML
- **Offline training**: Train models on recorded data before deployment

## Installation

### Prerequisites

- ROS 2 Humble
- Python 3.10+

### Build from Source

1. Clone this repository into your ROS 2 workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/LCAS/ros2_svdd_monitor.git
```

2. Install Python dependencies:

```bash
pip install -r ros2_svdd_monitor/requirements.txt
```

3. Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select ros2_svdd_monitor
source install/setup.bash
```

## Quick Start

### 1. Collect Training Data

First, record normal operation data. You can either:

**Option A: Use the built-in CSV recorder (recommended)**

```bash
# Start the CSV recorder
ros2 run ros2_svdd_monitor rosbag_to_csv training_data.csv

# In another terminal, operate your robot normally
# (or play a rosbag of normal operation)
ros2 bag play normal_operation.bag
```

**Option B: Record a rosbag and export to CSV later**

```bash
# Record a rosbag with /cmd_vel and /imu topics
ros2 bag record /cmd_vel /imu -o normal_operation

# Then use the CSV converter
ros2 run ros2_svdd_monitor rosbag_to_csv training_data.csv
ros2 bag play normal_operation
```

The CSV file should have the following columns:
```
timestamp,linear_x,linear_y,linear_z,angular_x,angular_y,angular_z,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
```

### 2. Train the SVDD Model

Train the anomaly detection model on your collected data:

```bash
ros2 run ros2_svdd_monitor train --csv training_data.csv --output-dir models/
```

This will:
- Load the training data from CSV
- Extract features using sliding windows
- Train a OneClassSVM model
- Save the trained model and scaler to `models/svdd_model.pkl` and `models/scaler.pkl`

### 3. Run the Monitor

Start the real-time anomaly monitor:

```bash
ros2 run ros2_svdd_monitor monitor
```

The monitor will:
- Subscribe to `/cmd_vel` and `/imu` topics
- Maintain sliding windows of recent data
- Extract features and run anomaly detection
- Publish results to `/svdd/anomaly` (Bool) and `/svdd/anomaly_score` (Float32)

### 4. Visualize Anomalies

Monitor the anomaly detection in real-time:

```bash
# Watch for anomalies
ros2 topic echo /svdd/anomaly

# View anomaly scores
ros2 topic echo /svdd/anomaly_score

# Use rqt_plot for visualization
rqt_plot /svdd/anomaly_score
```

## Configuration

The SVDD monitor is configured via `config/config.yaml`:

```yaml
# Sliding window parameters
window_size: 10  # Number of recent samples for feature extraction

# OneClassSVM hyperparameters
nu: 0.1  # Upper bound on fraction of outliers (0 < nu <= 1)
gamma: 'scale'  # RBF kernel coefficient

# Model paths
model_path: 'svdd_model.pkl'
scaler_path: 'scaler.pkl'

# Anomaly detection threshold
anomaly_threshold: 0.0  # Decision function threshold
```

### Hyperparameter Tuning

- **window_size**: Larger windows capture more temporal context but increase latency. Typical values: 5-20.
- **nu**: Upper bound on training errors. Lower values make the model more strict. Typical values: 0.05-0.2.
- **gamma**: Controls the RBF kernel width. Use 'scale' (default) or 'auto' for automatic selection, or specify a float value.
- **anomaly_threshold**: Adjust this based on your false positive/negative tolerance. Default is 0.0.

## How It Works

### Feature Extraction

The monitor extracts 20 features from sliding windows of `/cmd_vel` and `/imu` data:

**Command Velocity Features (8):**
- Mean, std, min, max of linear.x (forward velocity)
- Mean, std, min, max of angular.z (yaw rate)

**IMU Features (8):**
- Mean, std, min, max of accel.x (forward acceleration)
- Mean, std, min, max of gyro.z (yaw angular velocity)

**Cross-Features (4):**
- Correlation between cmd_vel.linear.x and imu.accel.x
- Correlation between cmd_vel.angular.z and imu.gyro.z
- Ratio of imu.accel to cmd_vel.linear
- Ratio of imu.gyro to cmd_vel.angular

These features capture the expected physical relationship between commanded motion and sensor response. Anomalies occur when this relationship breaks down (e.g., wheel slip, motor failure, unexpected obstacles).

### SVDD Model

The OneClassSVM is trained on features extracted from normal operation data. It learns a decision boundary that encompasses normal operation patterns. During monitoring, new feature vectors are classified as:

- **Inliers (normal)**: Decision function > threshold
- **Outliers (anomalies)**: Decision function < threshold

The decision function value is published as the anomaly score, allowing for tunable sensitivity.

## Example Use Cases

1. **Wheel slip detection**: Commanded forward motion without corresponding IMU acceleration
2. **Motor failure**: No angular velocity despite commanded turning
3. **Collision detection**: Unexpected deceleration (high negative acceleration)
4. **Terrain anomaly**: Unusual vibration patterns in IMU during motion
5. **Sensor failure**: Loss of correlation between command and measurement

## Testing

Run unit tests with pytest:

```bash
cd src/ros2_svdd_monitor
pytest test/ -v
```

Tests cover:
- Feature extraction correctness
- Model training and prediction
- Model save/load functionality
- Edge cases (empty windows, missing data)

## Development

This package includes a devcontainer for development:

1. Open the repository in VS Code
2. Click "Reopen in Container" when prompted
3. The container includes ROS 2 Humble and all dependencies

See the template README for more details on devcontainer usage.

## Topics

### Subscribed Topics

- `/cmd_vel` (geometry_msgs/Twist): Commanded velocity
- `/imu` (sensor_msgs/Imu): IMU measurements

### Published Topics

- `/svdd/anomaly` (std_msgs/Bool): True if anomaly detected
- `/svdd/anomaly_score` (std_msgs/Float32): Anomaly score (decision function value)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{ros2_svdd_monitor,
  title = {ros2_svdd_monitor: SVDD Anomaly Detection for ROS 2},
  author = {LCAS},
  year = {2025},
  url = {https://github.com/LCAS/ros2_svdd_monitor}
}
```

## Acknowledgments

This package was created from the [LCAS ROS 2 Package Template](https://github.com/LCAS/ros2_pkg_template).

## Troubleshooting

### Model file not found
Ensure you've trained a model first using the `train` command. Check that the model path in `config.yaml` points to the correct location.

### No anomaly detection
- Check that both `/cmd_vel` and `/imu` topics are publishing data
- Verify that the window has filled (requires `window_size` samples)
- Try adjusting the `anomaly_threshold` in the config

### Too many false positives
- Increase `anomaly_threshold` (more lenient)
- Increase `nu` during training (allows more outliers)
- Collect more diverse training data

### Too many false negatives
- Decrease `anomaly_threshold` (more strict)
- Decrease `nu` during training (stricter model)
- Ensure training data only contains normal operation

## References

- Tax, D. M., & Duin, R. P. (2004). Support vector data description. Machine learning, 54(1), 45-66.
- Schölkopf, B., et al. (2001). Estimating the support of a high-dimensional distribution. Neural computation, 13(7), 1443-1471.
