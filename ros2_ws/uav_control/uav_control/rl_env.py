#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
import numpy as np
import time
from std_srvs.srv import Empty
import subprocess
class rl_env(Node):
    def __init__(self):
        super().__init__('rl_env')
        self.odom_sub = self.create_subscription(Odometry, '/model/crazyflie/odometry', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/crazyflie/gazebo/command/twist', 10)
        self.timer = self.create_timer(0.1, self.step)
        
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        
        self.tx = 1.0
        self.ty = 1.0
        self.tz = 1.0
        
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        
        """ correct parameters
        self.kpz = 0.5
        self.kpx = 0.5
        self.kpy = 0.5
        
        self.kix = 0.1
        self.kiy = 0.1
        self.kiz = 0.1
        
        self.kdx = 0.4
        self.kdy = 0.4
        self.kdz = 0.4
        """
        """
        self.kp = np.random.uniform(-2,2)
        self.ki = np.random.uniform(-2,2)
        self.kd = np.random.uniform(-2,2)
        """
        
        self.kp = 1.02085068001765
        self.ki = 0.16202208152066655
        self.kd = -0.43773086798694694
        
        """
        self.kp = 0.5
        self.ki = 0.1
        self.kd = 0.4
        """
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        
        self.integral = np.zeros(3)
        self.ts = 0.1
        
        self.uav_pos = np.array([0,0,0])
        self.targ_pos = np.array([0,0,0])
        self.pos_distance = np.array([0,0,0])
        
        self.step_counter = 0
        
        self.ep_counter = 0
        
        self.reward = 0 
        
        self.arrival_counter = 0
        
        self.best_parameters = np.zeros(3)
        
        self.best_reward = 0
        
        self.episodes = 10
        
        # For comparison
        
        self.step_average = 0
    
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        #self.get_logger().info(f"[ODOM] x= {self.x:.2f}, y= {self.y:.2f}, z= {self.z:.2f}")
        self.vx = msg.twist.twist.linear.x
        self.vy = msg.twist.twist.linear.y
        self.vz = msg.twist.twist.linear.z
    
    
    def reset(self):
        self.step_counter = 0
        self.arrival_counter = 0
        self.reward = 0
        self.integral = np.zeros(3)
        """
        self.kp = np.random.uniform(-2,2)
        self.ki = np.random.uniform(-2,2)
        self.kd = np.random.uniform(-2,2)
        """
        
        self.x = 0.0
        self.y = 0.0
        self.z = 0.5
        
        self.kp = 1.02085068001765
        self.ki = 0.16202208152066655
        self.kd = -0.43773086798694694
        
        """
        self.kp = 0.5
        self.ki = 0.1
        self.kd = 0.4
        """
        """
        subprocess.run([
            "gz", "service",
            "-s", "/world/crazyflie_world/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000",
            "--req",
            'name: "crazyflie", position: {x: 0, y: 0, z: 0.5}, orientation: {x: 0, y: 0, z: 0, w: 1}'
        ])
        """
        subprocess.run([
            "gz", "service",
            "-s", "/world/crazyflie_world/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000",
            "--req",
            f'name: "crazyflie", position: {{x: {self.x}, y: {self.y}, z: {self.z}}}, orientation: {{x: 0, y: 0, z: 0, w: 1}}'
        ])
        
        
    """    
    def choose_action(self):
        if np.random.random() < self.epsilon:
            self.kp = np.random.uniform(-2,2)
            self.ki = np.random.uniform(-2,2)
            self.kd = np.random.uniform(-2,2)
            return self.kp, self.ki, self.kd
            
        else:
            return (np.argmax(self.q_values[obs]))  
    """
    def step(self):
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
        dist = np.linalg.norm(np.array([self.dx, self.dy, self.dz]))
        self.step_counter += 0.1
        
        if dist <= 0.1:
            self.arrival_counter += 0.1
            self.reward += 0.5
            if self.arrival_counter >= 2:
                self.ep_counter += 1
                self.reward += 5
                if self.reward > self.best_reward:
                    self.best_parameters = (self.kp, self.ki, self.kd)
                    self.best_reward = self.reward
                print(f"Reward: {self.reward}, Parameters: kp = {self.kp}, ki = {self.ki}, kd = {self.kd}")
                print(f"Episode {self.ep_counter} completed")
                self.step_average += self.step_counter
                if self.ep_counter >= self.episodes:
                    print("Episodes are over")
                    #print(f"Best reward: {self.best_reward}, best parameters: {self.best_parameters[0]}, {self.best_parameters[1]}, {self.best_parameters[2]}")
                    self.step_average = self.step_average / self.episodes
                    print(f"Average time to complete = {self.step_average} seconds")
                    rclpy.shutdown()
                else:
                    self.reset()
        
        else:
            self.reward += dist * -0.01
        
        if self.step_counter >= 7:
            self.ep_counter += 1
            self.reward += -1
            if self.reward > self.best_reward:
                    self.best_parameters = (self.kp, self.ki, self.kd)
                    self.best_reward = self.reward
            print(f"Reward: {self.reward}, Parameters: kp = {self.kp}, ki = {self.ki}, kd = {self.kd}")
            print(f"Episode {self.ep_counter} completed")
            if self.ep_counter >= self.episodes:
                print("Episodes are over")
                print(f"Best reward: {self.best_reward}, best parameters: {self.best_parameters[0], self.best_parameters[1], self.best_parameters[2]}")
                rclpy.shutdown()
            else:
                self.reset()
        
        
            

def main(args=None):
    rclpy.init(args=args)
    node = rl_env()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
# 100 Episodes - Best reward: 14.778853426287947, best parameters: (1.02085068001765, 0.16202208152066655, -0.43773086798694694)

# RL params - Average time to complete = 4.639999999999995 seconds
# Conventional Params - 
