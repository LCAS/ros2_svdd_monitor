from setuptools import find_packages, setup

package_name = 'ros2_svdd_monitor'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
    ],
    zip_safe=True,
    maintainer='L-CAS',
    maintainer_email='lcas@lincoln.ac.uk',
    description='ROS2 package for anomaly detection using Support Vector Data Description (SVDD)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = ros2_svdd_monitor.train_node:main',
            'monitor = ros2_svdd_monitor.monitor_node:main',
        ],
    },
)
