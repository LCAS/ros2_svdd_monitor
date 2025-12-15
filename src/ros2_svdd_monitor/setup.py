from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'ros2_svdd_monitor'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'scikit-learn',
        'numpy',
        'pandas',
        'joblib',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='LCAS',
    maintainer_email='lcas@lincoln.ac.uk',
    description='SVDD-style anomaly detection for ROS 2 using proprioceptive sensors',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = ros2_svdd_monitor.train:main',
            'monitor = ros2_svdd_monitor.monitor:main',
            'rosbag_to_csv = ros2_svdd_monitor.utils.rosbag_to_csv:main',
        ],
    },
)
