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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32

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
        model_path = self.config['model_path']
        scaler_path = self.config['scaler_path']
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found: {model_path}")
            self.get_logger().error("Please train a model first using: ros2 run ros2_svdd_monitor train")
            sys.exit(1)
        
        self.get_logger().info(f"Loading model from {model_path}...")
        self.model.load(model_path, scaler_path)
        self.get_logger().info("Model loaded successfully!")
        
        # Anomaly threshold
        self.anomaly_threshold = self.config.get('anomaly_threshold', 0.0)
        
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
        
        # Publishers
        self.anomaly_pub = self.create_publisher(Bool, '/svdd/anomaly', 10)
        self.score_pub = self.create_publisher(Float32, '/svdd/anomaly_score', 10)
        
        # Statistics
        self.message_count = 0
        self.anomaly_count = 0
        self.warmup_samples = 50  # Skip first N samples to let windows stabilize
        
        self.get_logger().info("SVDD Monitor started!")
        self.get_logger().info(f"Window size: {self.window_size}")
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
        
        # Check for anomaly if we have enough data
        if len(self.cmd_vel_window) >= self.window_size and len(self.imu_window) >= self.window_size:
            self.check_for_anomaly()

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
        
        # Check for anomaly if we have enough data
        if len(self.cmd_vel_window) >= self.window_size and len(self.imu_window) >= self.window_size:
            self.check_for_anomaly()

    def check_for_anomaly(self):
        """
        Extract features from current windows and check for anomaly.
        """
        # Skip anomaly detection during warmup period
        if self.message_count < self.warmup_samples:
            self.message_count += 1
            return
        
        # Extract features from current windows
        cmd_vel_array = list(self.cmd_vel_window)
        imu_array = list(self.imu_window)
        
        features = extract_window_features(cmd_vel_array, imu_array)
        features = features.reshape(1, -1)  # Reshape for single prediction
        
        # Get anomaly score (decision function)
        score = self.model.decision_function(
            features,
            scale=self.config.get('feature_scaling', True)
        )[0]
        
        # Predict anomaly
        prediction = self.model.predict(
            features,
            scale=self.config.get('feature_scaling', True)
        )[0]
        
        # Use only score threshold for anomaly detection (more reliable than binary prediction)
        is_anomaly = score < self.anomaly_threshold
        
        # Publish results
        anomaly_msg = Bool()
        anomaly_msg.data = bool(is_anomaly)
        self.anomaly_pub.publish(anomaly_msg)
        
        score_msg = Float32()
        score_msg.data = float(score)
        self.score_pub.publish(score_msg)
        
        # Update statistics
        self.message_count += 1
        if is_anomaly:
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
