"""
Rosbag to CSV Converter

This utility helps export /cmd_vel and /imu topics from a ROS 2 bag to CSV format
for training the SVDD model.

Usage:
    ros2 run ros2_svdd_monitor rosbag_to_csv <bag_path> <output_csv>

Or use this as a reference to create a custom subscriber that writes synchronized
CSV rows during data collection.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
import csv
from collections import deque
import sys


class BagToCSVConverter(Node):
    """
    Node to subscribe to /cmd_vel and /imu and write synchronized data to CSV.
    
    This can be run live while recording a rosbag to create training data.
    """

    def __init__(self, output_csv_path):
        super().__init__('rosbag_to_csv_converter')
        
        self.output_csv_path = output_csv_path
        self.csv_file = open(output_csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header
        self.csv_writer.writerow([
            'timestamp',
            'linear_x', 'linear_y', 'linear_z',
            'angular_x', 'angular_y', 'angular_z',
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z'
        ])
        
        # Store latest messages
        self.latest_cmd_vel = None
        self.latest_imu = None
        
        # Subscriptions
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )
        
        self.get_logger().info(f'Recording to {output_csv_path}')
        self.get_logger().info('Subscribing to /cmd_vel and /imu')
        
        self.sample_count = 0

    def cmd_vel_callback(self, msg):
        """Store latest cmd_vel message."""
        self.latest_cmd_vel = msg
        self.try_write_row()

    def imu_callback(self, msg):
        """Store latest IMU message."""
        self.latest_imu = msg
        self.try_write_row()

    def try_write_row(self):
        """
        Write a row to CSV if we have both cmd_vel and IMU data.
        """
        if self.latest_cmd_vel is not None and self.latest_imu is not None:
            timestamp = self.get_clock().now().nanoseconds / 1e9
            
            row = [
                timestamp,
                self.latest_cmd_vel.linear.x,
                self.latest_cmd_vel.linear.y,
                self.latest_cmd_vel.linear.z,
                self.latest_cmd_vel.angular.x,
                self.latest_cmd_vel.angular.y,
                self.latest_cmd_vel.angular.z,
                self.latest_imu.linear_acceleration.x,
                self.latest_imu.linear_acceleration.y,
                self.latest_imu.linear_acceleration.z,
                self.latest_imu.angular_velocity.x,
                self.latest_imu.angular_velocity.y,
                self.latest_imu.angular_velocity.z,
            ]
            
            self.csv_writer.writerow(row)
            self.sample_count += 1
            
            if self.sample_count % 100 == 0:
                self.get_logger().info(f'Recorded {self.sample_count} samples')

    def __del__(self):
        """Close CSV file on cleanup."""
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
            self.get_logger().info(f'Closed CSV file with {self.sample_count} samples')


def main(args=None):
    """
    Main entry point for rosbag to CSV converter.
    
    Usage:
        python -m ros2_svdd_monitor.utils.rosbag_to_csv output.csv
    
    Then play your rosbag:
        ros2 bag play <bag_name>
    
    Or run this alongside your robot to collect training data.
    """
    if len(sys.argv) < 2:
        print("Usage: ros2 run ros2_svdd_monitor rosbag_to_csv <output_csv>")
        print("Then play your rosbag: ros2 bag play <bag_name>")
        return
    
    output_csv = sys.argv[1]
    
    rclpy.init(args=args)
    node = BagToCSVConverter(output_csv)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
