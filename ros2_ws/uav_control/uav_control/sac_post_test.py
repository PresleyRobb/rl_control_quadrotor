#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
import numpy as np
import time
from std_srvs.srv import Empty
import subprocess
import matplotlib.pyplot as plt
class sac_post_test(Node):
    def __init__(self):
        super().__init__('sac_post_test')
        self.odom_sub = self.create_subscription(Odometry, '/model/crazyflie/odometry', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/crazyflie/gazebo/command/twist', 10)
        self.timer = self.create_timer(0.1, self.step)
        
        self.done = False
        self.odom_ready = False
        self.done1 = False
        
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        
        self.x_traj = []
        self.y_traj = []
        self.z_traj = []
        
        self.tx = np.random.uniform(-2, 2)
        self.ty = np.random.uniform(-2, 2)
        self.tz = np.random.uniform(1, 2)
        
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        
        self.dist = 0.0
        
        self.kp = 1.1891385767283094
        self.ki = 0.14405628987410016
        self.kd = 0.35098835803190775
        
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
    
    def plot_trajectory(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Drone path
        ax.plot(self.x_traj, self.y_traj, self.z_traj, label="Flight Path")

        # Start point
        ax.scatter(self.x_traj[0], self.y_traj[0], self.z_traj[0], label="Start")

        # Target point
        ax.scatter(self.tx, self.ty, self.tz, label="Target")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Drone Flight Path (PID Mean Gains)")
        ax.legend()

        plt.show()

    def step(self):
        if self.done1 == False:
            self.dx = 0.0 - self.x
            self.dy = 0.0 - self.y
            self.dz = 1.0 - self.z
       
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
            self.x_traj.append(self.x)
            self.y_traj.append(self.y)
            self.z_traj.append(self.z)
            self.dist = np.linalg.norm(np.array([self.dx, self.dy, self.dz]))
            if self.dist <= 0.01:
                self.done1 = True
        
        elif self.done1 == True:
            self.integral = np.zeros(3)
        
        
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
            self.x_traj.append(self.x)
            self.y_traj.append(self.y)
            self.z_traj.append(self.z)
            self.dist = np.linalg.norm(np.array([self.dx, self.dy, self.dz]))
       	
       	    if self.dist <= 0.01:
       	        print(f"Goal reached")
       	        self.done = True
       	        self.plot_trajectory()
       	    
       	        rclpy.shutdown()        

def main(args=None):
    rclpy.init(args=args)
    node = sac_post_test()
    """
    subprocess.run([
            "gz", "service",
            "-s", "/world/crazyflie_world/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000",
            "--req",
            f'name: "crazyflie", position: {{x: {node.x}, y: {node.y}, z: {node.z}}}, orientation: {{x: 0, y: 0, z: 0, w: 1}}'
        ])
    """
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
