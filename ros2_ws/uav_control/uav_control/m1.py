#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
import numpy as np
import time
class m1(Node):
    def __init__(self):
        super().__init__('m1')
        self.odom_sub = self.create_subscription(Odometry, '/model/crazyflie/odometry', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/crazyflie/gazebo/command/twist', 10)
        self.target_sub = self.create_subscription(Point, '/drone/goal_pose', self.goal_callback, 10)
        self.timer = self.create_timer(0.1, self.move)
        
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        
        self.tx = 0.0
        self.ty = 0.0
        self.tz = 0.0
        
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
        rng = np.random.default_rng()
        
        self.kpz = rng.random()
        self.kpx = rng.random()
        self.kpy = rng.random()
        
        self.kix = rng.random()
        self.kiy = rng.random()
        self.kiz = rng.random()
        
        self.kdx = rng.random()
        self.kdy = rng.random()
        self.kdz = rng.random()
        
        print(f"kpx = {self.kpx}")
        print(f"kpy = {self.kpy}")
        print(f"kpz = {self.kpz}")
        print(f"kix = {self.kix}")
        print(f"kiy = {self.kiy}")
        print(f"kiz = {self.kiz}")
        print(f"kdx = {self.kdx}")
        print(f"kdy = {self.kdy}")
        print(f"kdz = {self.kdz}")
        
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        
        self.integral = np.array ([0.0,0.0,0.0])
        self.ts = 0.1
        
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        self.get_logger().info(f"[ODOM] x= {self.x:.2f}, y= {self.y:.2f}, z= {self.z:.2f}")
        self.vx = msg.twist.twist.linear.x
        self.vy = msg.twist.twist.linear.y
        self.vz = msg.twist.twist.linear.z
        
        
    def goal_callback(self, msg):
        self.tx = msg.x
        self.ty = msg.y
        self.tz = msg.z
        self.get_logger().info(f"New target recieved: x= {self.tx:.2f}, y = {self.ty:.2f}, z = {self.tz:.2f}")
            
    def move(self):
        self.dx = self.tx - self.x
        self.dy = self.ty - self.y
        self.dz = self.tz - self.z
       
        self.integral += np.array([self.dx,self.dy,self.dz])*self.ts
        self.integral = np.clip(self.integral, -0.5, 0.5)
        
        out_x = (self.kpx * self.dx) + (self.kix * self.integral[0]) - (self.kdx * self.vx)
        out_y = (self.kpy * self.dy) + (self.kiy * self.integral[1]) - (self.kdy * self.vy)
        out_z = (self.kpz * self.dz) + (self.kiz * self.integral[2]) - (self.kdz * self.vz)
            
        cmd = Twist()
        cmd.linear.x = np.clip(out_x, -1.0, 1.0)
        cmd.linear.y = np.clip(out_y, -1.0, 1.0)
        cmd.linear.z = np.clip(out_z, -1.0, 1.0)
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = m1()
    time.sleep(3)
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
