"""Training node for SVDD model."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
import numpy as np

from ros2_svdd_monitor.config import SVDDConfig
from ros2_svdd_monitor.features import FeatureExtractor
from ros2_svdd_monitor.svdd_model import SVDDModel


class TrainNode(Node):
    """ROS2 node for training SVDD model."""
    
    def __init__(self):
        """Initialize training node."""
        super().__init__('svdd_train')
        
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
        
        # Data storage
        self.features_list = []
        
        # Flags
        self.training_active = True
        
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
        
        # Timer for training completion
        self.training_timer = self.create_timer(
            self.config.training_duration,
            self.complete_training
        )
        
        self.get_logger().info(f'Training started. Collecting data for {self.config.training_duration} seconds...')
        
    def imu_callback(self, msg):
        """
        Handle IMU messages.
        
        Args:
            msg: sensor_msgs/Imu message
        """
        if not self.training_active:
            return
            
        self.feature_extractor.add_imu_data(msg)
        self.try_extract_features()
        
    def cmd_vel_callback(self, msg):
        """
        Handle cmd_vel messages.
        
        Args:
            msg: geometry_msgs/Twist message
        """
        if not self.training_active:
            return
            
        self.feature_extractor.add_cmd_vel_data(msg)
        
    def try_extract_features(self):
        """Try to extract features from buffered data."""
        features = self.feature_extractor.extract_features()
        if features is not None:
            self.features_list.append(features)
            
    def complete_training(self):
        """Complete training and save model."""
        self.training_active = False
        self.training_timer.cancel()
        
        if len(self.features_list) == 0:
            self.get_logger().error('No features collected during training period!')
            self.get_logger().info('Please ensure /imu and /cmd_vel topics are publishing data.')
            return
            
        self.get_logger().info(f'Training completed. Collected {len(self.features_list)} feature samples.')
        
        # Convert features to numpy array
        features_array = np.array(self.features_list)
        
        # Train the model
        self.get_logger().info('Training SVDD model...')
        success = self.model.train(features_array)
        
        if success:
            # Save the model
            self.get_logger().info(f'Saving model to {self.config.model_path}...')
            save_success = self.model.save(self.config.model_path)
            
            if save_success:
                self.get_logger().info('Model saved successfully!')
            else:
                self.get_logger().error('Failed to save model!')
        else:
            self.get_logger().error('Failed to train model!')


def main(args=None):
    """Main entry point for training node."""
    rclpy.init(args=args)
    node = TrainNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
