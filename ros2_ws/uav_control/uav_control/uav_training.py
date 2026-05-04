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
from collections import defaultdict
import csv
class uav_training(Node, gym.Env):
    def __init__(self):
        super().__init__('uav_training')
        
        # ROS 2
        
        self.odom_sub = self.create_subscription(Odometry, '/model/crazyflie/odometry', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/crazyflie/gazebo/command/twist', 10)
        #self.timer = self.create_timer(0.1, self.step)
        
        # Gym Environment
        
        self.terminated = False
        self.truncated = False
        
        self.x = np.random.randint(-2, 3)
        self.y = np.random.randint(-2, 3)
        self.z = np.random.randint(1, 3)
        self.agent_location = np.array([self.x, self.y, self.z])
        
        self.tx = np.random.randint(-2, 3)
        self.ty = np.random.randint(-2, 3)
        self.tz = np.random.randint(1, 3)
        self.target_location = np.array([self.tx, self.ty, self.tz])
        
        self.observation_space = spaces.Box(
            low = np.array([-2, -2, 1, -2, -2, 1], dtype = np.float32),
            high = np.array([2, 2, 2, 2, 2, 2], dtype = np.float32),
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
        
        self.q_values = defaultdict(lambda: np.zeros(self.action_space.n))
        
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
            self.x, self.y, self.z, 
            self.tx, self.ty, self.tz
        ], dtype = np.float32)
    
    def reset(self):
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
        
        self.x = np.random.randint(-2, 3)
        self.y = np.random.randint(-2, 3)
        self.z = np.random.randint(1, 3)
        self.agent_location = np.array([self.x, self.y, self.z])
        
        self.tx = np.random.randint(-2, 3)
        self.ty = np.random.randint(-2, 3)
        self.tz = np.random.randint(1, 3)
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
        self.obs = self.get_obs()
        
        info = {}
        
        return self.obs, info
        
       
    def choose_action(self, epsilon, state):
        if np.random.random() < epsilon:
            return self.action_space.sample()       
        else:
            return int(np.argmax(self.q_values[state]))  
    
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
            self.reward = -50
            self.ep_reward += self.reward
            self.truncated = True    
        elif self.hover_counter >= 5:
            self.reward = 50
            self.ep_reward += self.reward
            self.terminated = True
        elif self.z < 0.1:
            self.reward = -50
            self.ep_reward += self.reward
            self.terminated = True
            
        elif dist < 0.15 and vel_mag < 0.1:
            self.reward = 5.0
            self.ep_reward += self.reward
        else: 
            self.reward = (-1.0 * dist) + (-0.1 * vel_mag)
            self.ep_reward += self.reward
        
        self.obs = self.get_obs()
        
        info = {}
        
        return self.obs, self.reward, self.terminated, self.truncated, info
        
    def update(self, next_obs, action, state):
        next_state = tuple(np.round(next_obs, 1))
        future_q_value = (not self.terminated) * np.max(self.q_values[next_state])
        
        target = self.reward + self.discount_factor * future_q_value
        
        temporal_difference = target - self.q_values[state][action]
        
        self.q_values[state][action] = (self.q_values[state][action] + self.lr * temporal_difference)
        
        self.training_error.append(temporal_difference)
        
        
def main(args=None):
    rclpy.init(args=args)
    env = uav_training()
    env.obs, info = env.reset()
    ep_counter = 0
    start_epsilon = 1.0
    n_episodes = 20
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1
    epsilon = start_epsilon
    for ep in range(100):
        while not (env.terminated or env.truncated):
            state = tuple(np.round(env.obs, 1))
            action = env.choose_action(epsilon, state)
            next_obs, env.reward, env.terminated, env.truncated, info = env.step(action)
            env.update(next_obs,action, state)
            env.obs = next_obs
            
        ep_counter += 1
        if ep_counter % 10 == 0:
            print(f"Episode: {ep_counter} done, episode reward = {env.ep_reward}")
        
        # log info
        success = int(env.hover_counter >= 5)
        env.episode_rewards.append(env.ep_reward)
        env.episode_steps.append(env.step_counter)
        env.episode_success.append(success)
        env.episode_kp.append(env.kp)
        env.episode_ki.append(env.ki)
        env.episode_kd.append(env.kd)
        
        epsilon = max(final_epsilon, epsilon - epsilon_decay)
        env.obs, info = env.reset()
    
    with open("q_table_100.csv", "w", newline="") as g:
        writer0 = csv.writer(g)
        writer0.writerow(["state"] + [f"a{i}" for i in range(env.action_space.n)])
    
        for state, action_values in env.q_values.items():
            writer0.writerow([state] + list(action_values))
    
        
    with open("training_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "success", "kp", "ki", "kd"])
    
        for i in range(len(env.episode_rewards)):
            writer.writerow([
                i + 1,
                env.episode_rewards[i],
                env.episode_steps[i],
                env.episode_success[i],
                env.episode_kp[i],
                env.episode_ki[i],
                env.episode_kd[i]
            ])
    print("Training results saved to training_results.csv")
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
