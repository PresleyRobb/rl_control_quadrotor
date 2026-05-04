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

class uav_env(Node, gym.Env):
    def __init__(self):
        super().__init__('uav_env')
        
        # ROS 2
        
        self.odom_sub = self.create_subscription(Odometry, '/model/crazyflie/odometry', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/crazyflie/gazebo/command/twist', 10)
        #self.timer = self.create_timer(0.1, self.step)
        
        # Gym Environment
        
        self.terminated = False
        self.truncated = False
        
        self.x = np.random.uniform(-2,2)
        self.y = np.random.uniform(-2,2)
        self.z = np.random.uniform(0.5, 2)
        self.agent_location = np.array([self.x, self.y, self.z])
        
        self.tx = np.random.uniform(-2,2)
        self.ty = np.random.uniform(-2,2)
        self.tz = np.random.uniform(0.5, 2)
        self.target_location = np.array([self.tx, self.ty, self.tz])
        
        self.observation_space = spaces.Box(
            low = np.array([-5, -5, -5, -3, -3, -3, -1, -1, -1, 0, 0, 0], dtype = np.float32),
            high = np.array([5, 5, 5, 3, 3, 3, 1, 1, 1, 2, 2, 2], dtype = np.float32),
            dtype = np.float32
        )
        
        self.action_space = gym.spaces.Discrete(27)
          
        self.action_to_pid = {
            0:  (0.00, 0.00, 0.00),
            1:  (0.05, 0.00, 0.00),
            2:  (-0.05, 0.00, 0.00),
            3:  (0.00, 0.01, 0.00),
            4:  (0.00, -0.01, 0.00),
            5:  (0.00, 0.00, 0.02),
            6:  (0.00, 0.00, -0.02),
            
            7:  (0.05, 0.01, 0.00),
            8:  (0.05, -0.01, 0.00),
            9:  (0.05, 0.00, 0.02),
            10: (0.05, 0.00, -0.02),

            11: (-0.05, 0.01, 0.00),
            12: (-0.05, -0.01, 0.00),
            13: (-0.05, 0.00, 0.02),
            14: (-0.05, 0.00, -0.02),

            15: ( 0.00, 0.01, 0.02),
            16: ( 0.00, 0.01, -0.02),
            17: ( 0.00, -0.01, 0.02),
            18: ( 0.00, -0.01, -0.02),

            19: (0.05, 0.01, 0.02),
            20: (0.05, 0.01, -0.02),
            21: (0.05, -0.01, 0.02),
            22: (0.05, -0.01, -0.02),

            23: (-0.05, 0.01, 0.02),
            24: (-0.05, 0.01, -0.02),
            25: (-0.05, -0.01, 0.02),
            26: (-0.05, -0.01, -0.02)
        }
            
        
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        
        self.kp = 1
        self.ki = 1
        self.kd = 1
        
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
    
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        #self.get_logger().info(f"[ODOM] x= {self.x:.2f}, y= {self.y:.2f}, z= {self.z:.2f}")
        self.vx = msg.twist.twist.linear.x
        self.vy = msg.twist.twist.linear.y
        self.vz = msg.twist.twist.linear.z
        
    def get_obs(self):
    
        return np.array([
            self. dx, self.dy, self.dz, 
            self.vx, self.vy, self.vz, 
            self.integral[0], self.integral[1], self.integral[2], 
            self.kp, self.ki, self.kd
        ], dtype = np.float32)
    
    def reset(self):
        self.truncated = False
        self.terminated = False
        self.reward = 0
        self.integral = np.zeros(3)
        self.hover_counter = 0
        self.step_counter = 0
        
        self.kp = 1
        self.ki = 1
        self.kd = 1
        
        self.x = np.random.uniform(-2,2)
        self.y = np.random.uniform(-2,2)
        self.z = np.random.uniform(0.5, 2)
        self.agent_location = np.array([self.x, self.y, self.z])
        
        self.tx = np.random.uniform(-2,2)
        self.ty = np.random.uniform(-2,2)
        self.tz = np.random.uniform(0.5, 2)
        self.target_location = np.array([self.tx, self.ty, self.tz])
        
        subprocess.run([
            "gz", "service",
            "-s", "/world/crazyflie_world/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000",
            "--req",
            f'name: "crazyflie", position: {{x: {self.x}, y: {self.y}, z: {self.z}}}, orientation: {{x: 0, y: 0, z: 0, w: 1}}'
        ])
        
        obs = self.get_obs()
        
        info = 0
        
        return obs, info
        
       
    def choose_action(self):
        action = np.random.randint(0,27)
        return action
    """        
        else:
            return (np.argmax(self.q_values[obs]))  
    """
    def step(self, action):
        
        dkp, dki, dkd = self.action_to_pid[action]
        
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
            time.sleep(self.ts)
            rclpy.spin_once(self, timeout_sec=0.05)
        
            dist = np.linalg.norm(np.array([self.dx, self.dy, self.dz]))
            vel_mag = np.linalg.norm([self.vx, self.vy, self.vz])
        
            if dist <= 0.1:
                self.hover_counter += 1
            else:
                self.hover_counter = 0.0
        
        self.step_counter += 1
        
        if self.step_counter >= 50:
            self.reward += -50
            self.truncated = True    
        elif self.hover_counter >= 2:
            self.reward += 50
            self.terminated = True
        elif self.z < 0.1:
            self.reward += -50.0
            self.terminated = True
            
        elif dist < 0.15 and vel_mag < 0.1:
            self.reward += 5.0
        else: self.reward += (-1.0 * dist) + (-0.1 * vel_mag)
        
        obs = self.get_obs()
        
        info = 0
        
        return obs, self.reward, self.terminated, self.truncated, info
        
def main(args=None):
    rclpy.init(args=args)
    env = uav_env()
    obs, info = env.reset()
    ep_counter = 0
    for _ in range(20):
        while not (env.terminated or env.truncated):
            action = env.choose_action()
            obs, env.reward, env.terminated, env.truncated, info = env.step(action)
            print(f"Step: {env.step_counter} done")
            print(f"Kp: {env.kp}, Ki: {env.ki}, Kd: {env.kd}")
        ep_counter += 1
        print(f"Episode: {ep_counter} done, reward = {env.reward}")
        obs, info = env.reset()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
