#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
import numpy as np
import time
from std_srvs.srv import Empty
import subprocess
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
class sac(Node,gym.Env):
    def __init__(self):
        super().__init__('sac')
        
        # ROS 2
        
        self.odom_sub = self.create_subscription(Odometry, '/model/crazyflie/odometry', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/crazyflie/gazebo/command/twist', 10)
        #self.timer = self.create_timer(0.1, self.step)
        
        # Gym Environment
        
        self.terminated = False
        self.truncated = False
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_z = 0.0
        self.x = np.random.uniform(-2, 2)
        self.y = np.random.uniform(-2, 2)
        self.z = np.random.uniform(1, 2)
        self.agent_location = np.array([self.x, self.y, self.z])
        
        self.tx = np.random.uniform(-2, 2)
        self.ty = np.random.uniform(-2, 2)
        self.tz = np.random.uniform(1, 2)
        self.target_location = np.array([self.tx, self.ty, self.tz])
        
        self.observation_space = spaces.Box(
            low = np.array([-1, -1, -1, 0, 0, 0, 0], dtype = np.float32),
            high = np.array([1, 1, 1, 1, 1, 1, 1], dtype = np.float32),
            dtype = np.float32
        )
        
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(3,), dtype=np.float32)
        
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        
        self.kp = np.random.uniform(0,2)
        self.ki = np.random.uniform(0,2)
        self.kd = np.random.uniform(0,2)
        
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        
        self.integral = np.zeros(3)
        self.ts = 0.1
        
        self.step_counter = 0
        
        self.ep_counter = 0
        
        self.reward = 0 
        
        self.episodes = 10
        
        self.hover_counter = 0
        
        self.obs = np.zeros(1)
        
        # Agent
        
        self.lr = 0.01
        self.discount_factor = 0.95
        
        self.training_error = []
        
        self.ep_reward = 0
        
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_success = []
        self.episode_kp = []
        self.episode_ki = []
        self.episode_kd = []
        
        self.dx_norm = (self.tx-self.x)/4
        self.dy_norm = (self.ty-self.y)/4
        self.dz_norm = (self.tz-self.z)
        
        self.vel_mag = 0.0
    
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        #self.get_logger().info(f"[ODOM] x= {self.x:.2f}, y= {self.y:.2f}, z= {self.z:.2f}")
        self.vx = msg.twist.twist.linear.x
        self.vy = msg.twist.twist.linear.y
        self.vz = msg.twist.twist.linear.z
        
    def get_obs(self):
        self.dx_norm = (self.tx-self.x)/4
        self.dy_norm = (self.ty-self.y)/4
        self.dz_norm = (self.tz-self.z)
        return np.array([
            self.dx_norm, self.dy_norm, self.dz_norm,
            self.kp/2, self.ki/2, self.kd/2, 
            self.vel_mag/3
        ], dtype = np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.truncated = False
        self.terminated = False
        self.reward = 0
        self.ep_reward = 0
        self.integral = np.zeros(3)
        self.hover_counter = 0
        self.step_counter = 0
        
        self.kp = np.random.uniform(0,2)
        self.ki = np.random.uniform(0,2)
        self.kd = np.random.uniform(0,2)
        
        self.x = np.random.uniform(-2, 2)
        self.y = np.random.uniform(-2, 2)
        self.z = np.random.uniform(1, 2)
        self.agent_location = np.array([self.x, self.y, self.z])
        
        self.tx = np.random.uniform(-2, 2)
        self.ty = np.random.uniform(-2, 2)
        self.tz = np.random.uniform(1, 2)
        self.target_location = np.array([self.tx, self.ty, self.tz])
        
        self.vel_mag = 0.0
        subprocess.run([
            "gz", "service",
            "-s", "/world/crazyflie_world/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000",
            "--req",
            f'name: "crazyflie", position: {{x: {self.x}, y: {self.y}, z: {self.z}}}, orientation: {{x: 0, y: 0, z: 0, w: 1}}'
        ])
        self.obs = self.get_obs()
        
        info = {}
        
        return self.obs, info
    
    def step(self, action):
        
        dkp, dki, dkd = action
        
        self.kp += dkp
        self.ki += dki
        self.kd += dkd
        
        self.kp = np.clip(self.kp, 0.0, 2.0)
        self.ki = np.clip(self.ki, 0.0, 2.0)
        self.kd = np.clip(self.kd, 0.0, 2.0)
        
        old_dist = np.linalg.norm(np.array([self.dx, self.dy, self.dz]))
        
        for _ in range(10):
        
            self.dx = self.tx - self.x
            self.dy = self.ty - self.y
            self.dz = self.tz - self.z
       
            self.integral += np.array([self.dx,self.dy,self.dz])*self.ts
            self.integral = np.clip(self.integral, -0.5, 0.5)
        
            out_x = (self.kp * self.dx) + (self.ki * self.integral[0]) - (self.kd * self.vx)
            out_y = (self.kp * self.dy) + (self.ki * self.integral[1]) - (self.kd * self.vy)
            out_z = (self.kp * self.dz) + (self.ki * self.integral[2]) - (self.kd * self.vz)
            
            cmd = Twist()
            cmd.linear.x = np.clip(out_x, -1.0, 1.0)
            cmd.linear.y = np.clip(out_y, -1.0, 1.0)
            cmd.linear.z = np.clip(out_z, -1.0, 1.0)
            self.cmd_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.1)
        
            dist = np.linalg.norm(np.array([self.dx, self.dy, self.dz]))
            self.vel_mag = np.linalg.norm([self.vx, self.vy, self.vz])
        
            if dist <= 0.1 and self.vel_mag <= 0.1:
                self.hover_counter += 1
            else:
                self.hover_counter = 0.0
        
        self.step_counter += 1
        
        if self.step_counter >= 100:
            self.reward = -50.0
            self.ep_reward += self.reward
            self.truncated = True
            print("Miss")    
        elif self.hover_counter >= 5:
            self.reward = 50.0
            self.ep_reward += self.reward
            self.terminated = True
            print("Goal")
        elif self.z < 0.1:
            self.reward = -50.0
            self.ep_reward += self.reward
            self.terminated = True
            print("Crashed")
        else:
            self.reward = (old_dist-dist)
            
            self.reward -= 0.01
            
        
        
        """
        else:
            self.reward = -(dist*0.2) - (0.1*self.vel_mag)
            self.ep_reward += self.reward
        """    
        
        
        self.obs = self.get_obs()
        
        info = {}
        return self.obs, self.reward, self.terminated, self.truncated, info

def main(args=None):
    rclpy.init(args=args)
    
    env = sac()
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="sac_mk2_uav"
    )
    
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        buffer_size=100000,
        tensorboard_log = "./tb_logs2/sac/", 
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto"
    )
    
    model.learn(total_timesteps=100000, callback=checkpoint_callback)
    model.save("crazyflie_sac_mk3")
    rclpy.shutdown()

if __name__ == '__main__':
    main()
