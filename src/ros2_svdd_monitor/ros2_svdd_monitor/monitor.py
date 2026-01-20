#!/usr/bin/env python3
"""
SVDD Anomaly Monitor Node

This ROS 2 node subscribes to /cmd_vel and /imu topics, maintains sliding windows,
extracts features, and publishes anomaly detection results.

Publishes:
    /svdd/anomaly (std_msgs/Bool): True if anomaly detected
    /svdd/anomaly_score (std_msgs/Float32): Anomaly score (decision function value)
"""

import os
import sys
import yaml
from collections import deque
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32
from ament_index_python.packages import get_package_share_directory

from ros2_svdd_monitor.svdd_model import SVDDModel
from ros2_svdd_monitor.features import extract_window_features


class SVDDMonitor(Node):
    """
    ROS 2 node for real-time SVDD-based anomaly detection.
    """

    def __init__(self, config_path=None):
        super().__init__('svdd_monitor')
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize sliding windows
        self.window_size = self.config['window_size']
        self.cmd_vel_window = deque(maxlen=self.window_size)
        self.imu_window = deque(maxlen=self.window_size)
        
        # Load trained model
        self.model = SVDDModel()
        model_path = os.path.expanduser(self.config['model_path'])
        scaler_path = os.path.expanduser(self.config['scaler_path'])
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found: {model_path}")
            self.get_logger().error("Please train a model first using: ros2 run ros2_svdd_monitor train")
            sys.exit(1)
        
        self.get_logger().info(f"Loading model from {model_path}...")
        self.model.load(model_path, scaler_path)
        self.get_logger().info("Model loaded successfully!")
        
        # Anomaly threshold(s)
        self.anomaly_threshold = self.config.get('anomaly_threshold', 0.0)
        self.enter_threshold = None
        self.exit_threshold = None

        # Optionally load auto-threshold saved at training time
        use_auto = self.config.get('use_auto_threshold', False)
        if use_auto:
            try:
                # Determine threshold file path
                threshold_path = self.config.get('threshold_path', 'threshold.yaml')
                if not os.path.isabs(threshold_path):
                    # Resolve relative to model directory if not absolute
                    model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else os.getcwd()
                    threshold_path = os.path.join(model_dir, threshold_path)

                if os.path.exists(threshold_path):
                    with open(threshold_path, 'r') as f:
                        tconf = yaml.safe_load(f)
                    if isinstance(tconf, dict):
                        if 'enter_threshold' in tconf and 'exit_threshold' in tconf:
                            self.enter_threshold = float(tconf['enter_threshold'])
                            self.exit_threshold = float(tconf['exit_threshold'])
                            self.get_logger().info(
                                f"Loaded enter/exit thresholds from {threshold_path}: enter={self.enter_threshold}, exit={self.exit_threshold}"
                            )
                        elif 'threshold' in tconf:
                            self.anomaly_threshold = float(tconf['threshold'])
                            self.get_logger().info(f"Loaded threshold from {threshold_path}: {self.anomaly_threshold}")
                        else:
                            self.get_logger().warn(f"Threshold file missing expected keys: {threshold_path}")
                else:
                    self.get_logger().warn(f"Auto-threshold enabled but file not found: {threshold_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load auto-threshold: {e}")
        
        # Subscribers with flexible QoS to handle different publishers
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        cmd_vel_topic = self.config.get('cmd_vel_topic', '/cmd_vel')
        imu_topic = self.config.get('imu_topic', '/imu')
        
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            cmd_vel_topic,
            self.cmd_vel_callback,
            qos_profile
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            qos_profile
        )
        
        # Publishers (latched) so last message is delivered to late subscribers
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        pub_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.anomaly_pub = self.create_publisher(Bool, '/svdd/anomaly', pub_qos)
        self.score_pub = self.create_publisher(Float32, '/svdd/anomaly_score', pub_qos)
        
        # Statistics
        self.message_count = 0
        self.anomaly_count = 0
        self.warmup_samples = 50  # Skip first N samples to let windows stabilize

        # Smoothing and hysteresis
        self.smoothing_alpha = float(self.config.get('smoothing_alpha', 0.2))
        self.hysteresis_abs_margin = float(self.config.get('hysteresis_abs_margin', 0.5))
        self.hysteresis_ratio = float(self.config.get('hysteresis_ratio', 0.05))
        self.enter_consecutive = int(self.config.get('enter_consecutive', 2))
        self.exit_consecutive = int(self.config.get('exit_consecutive', 2))
        self.smoothed_score = None
        self.in_anomaly = False
        self.consec_anom = 0
        self.consec_norm = 0
        
        self.get_logger().info("SVDD Monitor started!")
        self.get_logger().info(f"Window size: {self.window_size}")
        if self.enter_threshold is not None and self.exit_threshold is not None:
            self.get_logger().info(f"Enter/Exit thresholds: {self.enter_threshold} / {self.exit_threshold}")
        else:
            self.get_logger().info(f"Anomaly threshold: {self.anomaly_threshold}")
        self.get_logger().info(f"Subscribing to {cmd_vel_topic} and {imu_topic}")
        self.get_logger().info("Publishing to /svdd/anomaly and /svdd/anomaly_score")

    def load_config(self, config_path=None):
        """Load configuration from YAML file."""
        if config_path is None:
            # Search for config in common locations
            possible_paths = [
                'config/config.yaml',
                '../config/config.yaml',
                os.path.join(os.path.dirname(__file__), '../config/config.yaml'),
            ]
            
            # Add the installed share directory location
            try:
                share_dir = get_package_share_directory('ros2_svdd_monitor')
                possible_paths.insert(0, os.path.join(share_dir, 'config/config.yaml'))
            except Exception:
                pass
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if config_path is None:
                self.get_logger().error("Could not find config.yaml")
                sys.exit(1)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config

    def cmd_vel_callback(self, msg):
        """Handle incoming cmd_vel messages."""
        # Store as array: [linear.x, linear.y, linear.z, angular.x, angular.y, angular.z]
        cmd_vel_data = [
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
            msg.angular.x,
            msg.angular.y,
            msg.angular.z,
        ]
        self.cmd_vel_window.append(cmd_vel_data)
        
        # Check for anomaly if we have enough data, otherwise publish defaults
        if len(self.cmd_vel_window) >= self.window_size and len(self.imu_window) >= self.window_size:
            self.check_for_anomaly()
        else:
            self.publish_defaults()

    def imu_callback(self, msg):
        """Handle incoming IMU messages."""
        # Store as array: [accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z]
        imu_data = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ]
        self.imu_window.append(imu_data)
        
        # Check for anomaly if we have enough data, otherwise publish defaults
        if len(self.cmd_vel_window) >= self.window_size and len(self.imu_window) >= self.window_size:
            self.check_for_anomaly()
        else:
            self.publish_defaults()

    def publish_defaults(self):
        """Publish default messages so topics are active before detection starts."""
        anomaly_msg = Bool()
        anomaly_msg.data = False
        self.anomaly_pub.publish(anomaly_msg)
        score_msg = Float32()
        score_msg.data = 0.0
        self.score_pub.publish(score_msg)

    def check_for_anomaly(self):
        """
        Extract features from current windows and check for anomaly.
        """
        # Extract features from current windows
        cmd_vel_array = list(self.cmd_vel_window)
        imu_array = list(self.imu_window)
        
        features = extract_window_features(cmd_vel_array, imu_array)
        features = features.reshape(1, -1)  # Reshape for single prediction
        
        # Get anomaly score (decision function)
        raw_score = self.model.decision_function(
            features,
            scale=self.config.get('feature_scaling', True)
        )[0]
        # Smooth score with EMA to reduce flicker
        if self.smoothed_score is None:
            self.smoothed_score = float(raw_score)
        else:
            a = self.smoothing_alpha
            self.smoothed_score = float(a * raw_score + (1.0 - a) * self.smoothed_score)
        score = self.smoothed_score
        
        # Predict anomaly
        prediction = self.model.predict(
            features,
            scale=self.config.get('feature_scaling', True)
        )[0]
        
        # During warmup, publish but never flag anomaly
        if self.message_count < self.warmup_samples:
            is_anomaly = False
            self.consec_anom = 0
            self.consec_norm = 0
        else:
            # Hysteresis thresholds
            if self.enter_threshold is not None and self.exit_threshold is not None:
                enter_threshold = self.enter_threshold
                exit_threshold = self.exit_threshold
            else:
                # Symmetric margins around single threshold, clamped by abs_max
                base = self.anomaly_threshold
                margin = max(self.hysteresis_abs_margin, abs(base) * self.hysteresis_ratio)
                max_margin = float(self.config.get('hysteresis_abs_max', float('inf')))
                if np.isfinite(max_margin):
                    margin = min(margin, max_margin)
                enter_threshold = base - margin
                exit_threshold = base + margin

            # Update consecutive counters based on score relative to thresholds
            if not self.in_anomaly:
                # Candidate to enter anomaly when below enter_threshold
                if score < enter_threshold:
                    self.consec_anom += 1
                else:
                    self.consec_anom = 0
                if self.consec_anom >= self.enter_consecutive:
                    self.in_anomaly = True
                    self.consec_anom = 0
            else:
                # Candidate to exit anomaly when above exit_threshold
                if score > exit_threshold:
                    self.consec_norm += 1
                else:
                    self.consec_norm = 0
                if self.consec_norm >= self.exit_consecutive:
                    self.in_anomaly = False
                    self.consec_norm = 0

            is_anomaly = self.in_anomaly
        
        # Publish results
        anomaly_msg = Bool()
        anomaly_msg.data = bool(is_anomaly)
        self.anomaly_pub.publish(anomaly_msg)
        
        score_msg = Float32()
        score_msg.data = float(score)
        self.score_pub.publish(score_msg)
        
        # Update statistics
        self.message_count += 1
        if (self.message_count >= self.warmup_samples) and is_anomaly:
            self.anomaly_count += 1
        
        # Log periodically
        if self.message_count % 100 == 0:
            anomaly_rate = self.anomaly_count / self.message_count * 100
            self.get_logger().info(
                f"Processed {self.message_count} samples, "
                f"anomalies: {self.anomaly_count} ({anomaly_rate:.2f}%)"
            )
        
        # Log anomalies with detailed diagnostics
        if is_anomaly:
            # Log raw sensor data
            cmd_vel_mean = np.mean(cmd_vel_array, axis=0)
            imu_mean = np.mean(imu_array, axis=0)
            
            self.get_logger().warn(
                f"ANOMALY DETECTED! Score: {score:.6f} (threshold: {self.anomaly_threshold}) Prediction: {prediction}"
            )
            self.get_logger().warn(
                f"  cmd_vel (mean): linear=[{cmd_vel_mean[0]:.4f}, {cmd_vel_mean[1]:.4f}, {cmd_vel_mean[2]:.4f}] "
                f"angular=[{cmd_vel_mean[3]:.4f}, {cmd_vel_mean[4]:.4f}, {cmd_vel_mean[5]:.4f}]"
            )
            self.get_logger().warn(
                f"  imu (mean): accel=[{imu_mean[0]:.4f}, {imu_mean[1]:.4f}, {imu_mean[2]:.4f}] "
                f"gyro=[{imu_mean[3]:.4f}, {imu_mean[4]:.4f}, {imu_mean[5]:.4f}]"
            )
            self.get_logger().warn(
                f"  Features: {' '.join([f'{f:.4f}' for f in features[0][:5]])}... (showing first 5 of {len(features[0])})"
            )


def main(args=None):
    """Main entry point for the SVDD monitor node."""
    rclpy.init(args=args)
    
    # Check for config path argument
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    node = SVDDMonitor(config_path=config_path)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
