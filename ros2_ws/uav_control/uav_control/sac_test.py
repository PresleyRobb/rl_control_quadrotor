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
class sac_training(Node,gym.Env):
    def __init__(self):
        super().__init__('sac_test')
        
        # ROS 2
        
        self.odom_sub = self.create_subscription(Odometry, '/model/crazyflie/odometry', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/crazyflie/gazebo/command/twist', 10)
        #self.timer = self.create_timer(0.1, self.step)
        
        # Gym Environment
        
        self.terminated = False
        self.truncated = False
        self.goal = False
        
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
        
        self.kp_dict = []
        self.ki_dict = []
        self.kd_dict = []
    
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
        self.goal = False
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
            self.reward = -10
            self.ep_reward += self.reward
            self.truncated = True
            print("Miss")
            print(f"Kp: {self.kp}, Ki: {self.ki}, Kd: {self.kd}")    
        elif self.hover_counter >= 5:
            self.reward = 20
            self.ep_reward += self.reward
            self.terminated = True
            self.goal = True
            print("Goal")
            print(f"Kp: {self.kp}, Ki: {self.ki}, Kd: {self.kd}") 
        elif self.z < 0.1:
            self.reward = -10
            self.ep_reward += self.reward
            self.terminated = True
            print("Crashed")
            print(f"Kp: {self.kp}, Ki: {self.ki}, Kd: {self.kd}")
        else:
            self.reward = -(dist*0.2) - (0.1*self.vel_mag)
            self.ep_reward += self.reward
            
        
        
        self.obs = self.get_obs()
        
        info = {}
        return self.obs, self.reward, self.terminated, self.truncated, info

def main(args=None):
    rclpy.init(args=args)
    
    env = sac_training()
    
    model = SAC.load("models/sac_uav_100000_steps")
    KP = 0
    KI = 0
    KD = 0
    done_counter = 0
    for ep in range(100):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        print(f"Starting episode {ep+1}")
        
        while not done:
            action,_ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            done = terminated or truncated
            if env.goal: 
                done_counter +=1
                KP += env.kp
                KI += env.ki
                KD += env.kd
                
                env.kp_dict.append(env.kp)
                env.ki_dict.append(env.ki)
                env.kd_dict.append(env.kd)
            
        print(f"Episode {ep+1} complete, Total Reward: {total_reward:.2f}")
    mean_KP = KP/done_counter
    mean_KI = KI/done_counter
    mean_KD = KD/done_counter
    print(f"KP: {mean_KP}, KI: {mean_KI}, KD: {mean_KD}")
    
    kp_std_dev = 0
    for i in env.kp_dict:
        kp_std_dev += (i - mean_KP)**2
    kp_std_dev = kp_std_dev/done_counter
    
    ki_std_dev = 0
    for i in env.ki_dict:
        ki_std_dev += (i - mean_KI)**2
    ki_std_dev = ki_std_dev/done_counter
    
    kd_std_dev = 0
    for i in env.kd_dict:
        kd_std_dev += (i - mean_KD)**2
    kd_std_dev = kd_std_dev/done_counter
    
    print(f"KP: {mean_KP}, KI: {mean_KI}, KD: {mean_KD}")
    
    print(f"KP standard deviation: {kp_std_dev}")
    print(f"KI standard deviation: {ki_std_dev}")
    print(f"KD standard deviation: {kd_std_dev}")

    
    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
