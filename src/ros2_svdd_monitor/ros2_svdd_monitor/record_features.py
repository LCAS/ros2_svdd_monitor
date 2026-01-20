#!/usr/bin/env python3
"""Record /cmd_vel and /imu for a duration and save sliding-window features.

Usage:
  python3 record_features.py --duration 30 --rate 10 --window-size 10 --out features.npz

The script subscribes to `/cmd_vel` and `/imu`, records messages, aligns them
to a time grid at `--rate` Hz, computes sliding-window features using
`extract_window_features`, and saves array `X` into an .npz file.
"""
import argparse
import time
import os
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu

# Ensure local package import works when running this script directly.
_here = os.path.abspath(os.path.dirname(__file__))
_pkg_parent = os.path.dirname(_here)  # .../src/ros2_svdd_monitor
_src_root = os.path.dirname(_pkg_parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

try:
    from ros2_svdd_monitor.features import extract_window_features
except Exception:
    from features import extract_window_features


class Recorder(Node):
    def __init__(self):
        super().__init__('msvdd_recorder')
        self.cmd_times = []
        self.cmd_vals = []
        self.imu_times = []
        self.imu_vals = []
        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel', self.cb_cmd, 10)
        self.sub_imu = self.create_subscription(Imu, '/imu', self.cb_imu, 10)

    def cb_cmd(self, msg: Twist):
        t = self.get_clock().now().nanoseconds * 1e-9
        v = [msg.linear.x, msg.linear.y, msg.linear.z,
             msg.angular.x, msg.angular.y, msg.angular.z]
        self.cmd_times.append(t)
        self.cmd_vals.append(v)

    def cb_imu(self, msg: Imu):
        t = self.get_clock().now().nanoseconds * 1e-9
        v = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
             msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        self.imu_times.append(t)
        self.imu_vals.append(v)


def align_and_extract(rec: Recorder, duration, rate, window_size):
    if len(rec.cmd_times) == 0 and len(rec.imu_times) == 0:
        raise RuntimeError('No messages recorded on /cmd_vel or /imu')

    start = min(rec.cmd_times[0] if rec.cmd_times else float('inf'),
                rec.imu_times[0] if rec.imu_times else float('inf'))
    end = max(rec.cmd_times[-1] if rec.cmd_times else 0.0,
              rec.imu_times[-1] if rec.imu_times else 0.0)

    # create time grid
    dt = 1.0 / float(rate)
    t_grid = np.arange(start, end + 1e-6, dt)

    # helper to get last-known value at or before t
    def last_value(times, vals, t):
        if len(times) == 0:
            return None
        idx = np.searchsorted(times, t) - 1
        if idx < 0:
            return None
        return vals[idx]

    cmd_aligned = []
    imu_aligned = []

    for t in t_grid:
        c = last_value(rec.cmd_times, rec.cmd_vals, t)
        m = last_value(rec.imu_times, rec.imu_vals, t)
        # if no value yet, use zeros
        if c is None:
            c = [0.0] * 6
        if m is None:
            m = [0.0] * 6
        cmd_aligned.append(c)
        imu_aligned.append(m)

    cmd_arr = np.array(cmd_aligned)
    imu_arr = np.array(imu_aligned)

    # sliding-window extraction
    features = []
    for i in range(len(t_grid)):
        start_idx = max(0, i - window_size + 1)
        cmd_win = cmd_arr[start_idx:i+1]
        imu_win = imu_arr[start_idx:i+1]
        f = extract_window_features(cmd_win, imu_win)
        features.append(f)

    X = np.array(features)
    return X


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--duration', type=float, default=30.0)
    p.add_argument('--rate', type=float, default=10.0)
    p.add_argument('--window-size', type=int, default=10)
    p.add_argument('--out', default='features.npz')
    args = p.parse_args()

    rclpy.init()
    rec = Recorder()
    print(f'Recording for {args.duration}s...')
    t0 = time.time()
    try:
        while time.time() - t0 < args.duration:
            rclpy.spin_once(rec, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    print('Processing...')
    X = align_and_extract(rec, args.duration, args.rate, args.window_size)
    np.savez(args.out, X=X)
    print(f'Saved features to {args.out}, shape={X.shape}')

    rec.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
