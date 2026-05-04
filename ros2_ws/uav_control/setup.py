from setuptools import find_packages, setup

package_name = 'uav_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='presrobb',
    maintainer_email='robbp1@duq.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'figure8 = uav_control.figure8:main',
            'attacker = uav_control.attacker:main',
            'm1 = uav_control.m1:main',
            'm2 = uav_control.m2:main',
            'm3 = uav_control.m3:main',
            'rl_env = uav_control.rl_env:main',
            'uav_env = uav_control.uav_env:main',
            'uav_training = uav_control.uav_training:main',
            'dqn = uav_control.dqn:main',
            'sac = uav_control.sac:main',
            'sac_test = uav_control.sac_test:main',
            'sac_post_test = uav_control.sac_post_test:main',
        ],
    },
)
