"""Monitoring node for SVDD anomaly detection."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32
import os

from ros2_svdd_monitor.config import SVDDConfig
from ros2_svdd_monitor.features import FeatureExtractor
from ros2_svdd_monitor.svdd_model import SVDDModel


class MonitorNode(Node):
    """ROS2 node for SVDD-based anomaly detection."""
    
    def __init__(self):
        """Initialize monitoring node."""
        super().__init__('svdd_monitor')
        
        # Configuration
        self.config = SVDDConfig()
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(window_size=self.config.window_size)
        
        # SVDD model
        self.model = SVDDModel(
            nu=self.config.nu,
            kernel=self.config.kernel,
            gamma=self.config.gamma
        )
        
        # Load trained model
        if not os.path.exists(self.config.model_path):
            self.get_logger().error(f'Model file not found at {self.config.model_path}!')
            self.get_logger().error('Please run the training node first.')
            raise RuntimeError('Model file not found')
            
        if not self.model.load(self.config.model_path):
            self.get_logger().error('Failed to load model!')
            raise RuntimeError('Failed to load model')
            
        self.get_logger().info(f'Model loaded successfully from {self.config.model_path}')
        
        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu,
            self.config.imu_topic,
            self.imu_callback,
            10
        )
        
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            self.config.cmd_vel_topic,
            self.cmd_vel_callback,
            10
        )
        
        # Publishers
        self.anomaly_pub = self.create_publisher(Bool, '/anomaly_detected', 10)
        self.score_pub = self.create_publisher(Float32, '/anomaly_score', 10)
        
        # Timer for periodic anomaly checking
        self.check_timer = self.create_timer(
            1.0 / self.config.publish_rate,
            self.check_anomaly
        )
        
        self.get_logger().info('Monitoring started. Waiting for sensor data...')
        
    def imu_callback(self, msg):
        """
        Handle IMU messages.
        
        Args:
            msg: sensor_msgs/Imu message
        """
        self.feature_extractor.add_imu_data(msg)
        
    def cmd_vel_callback(self, msg):
        """
        Handle cmd_vel messages.
        
        Args:
            msg: geometry_msgs/Twist message
        """
        self.feature_extractor.add_cmd_vel_data(msg)
        
    def check_anomaly(self):
        """Check for anomalies in buffered data."""
        # Extract features
        features = self.feature_extractor.extract_features()
        
        if features is None:
            return
            
        try:
            # Predict anomaly
            predictions, scores = self.model.predict(features)
            
            # predictions: 1 for inliers, -1 for outliers
            # scores: negative values indicate outliers
            is_anomaly = predictions[0] == -1
            anomaly_score = float(scores[0])
            
            # Publish results
            anomaly_msg = Bool()
            anomaly_msg.data = is_anomaly
            self.anomaly_pub.publish(anomaly_msg)
            
            score_msg = Float32()
            score_msg.data = anomaly_score
            self.score_pub.publish(score_msg)
            
            if is_anomaly:
                self.get_logger().warn(f'Anomaly detected! Score: {anomaly_score:.4f}')
                
        except Exception as e:
            self.get_logger().error(f'Error during anomaly detection: {e}')


def main(args=None):
    """Main entry point for monitoring node."""
    import logging
    logger = logging.getLogger(__name__)
    
    rclpy.init(args=args)
    
    try:
        node = MonitorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f'Error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
