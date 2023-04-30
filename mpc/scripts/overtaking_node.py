#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class OverTaking(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('overtaking_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: create subscribers and publishers
        self.subscription = self.create_subscription(LaserScan, '/scan',self.scan_callback, 10)
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # TODO: set PID gains
        self.kp = 12.0
        self.kd = .6
        self.ki =0.00
        self.L=2.0

        # TODO: store history
        self.integral = 0.0
        self.prev_error = 0.0
        self.error = 0.0

        # TODO: store any necessary values you think you'll need

        self.angle_min = -2.35
        self.angle_incr = 0.004352

    def get_range(self, range_data, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR
        #TODO: implement
        Returns:
            range: range measurement in meters at the given angle

        """

        # The line below is: angle_in_lidar = 135deg - angle
        #angle_in_lidar = -1*self.angle_min - angle
        angle_in_lidar = -1*self.angle_min + angle
        index = int(angle_in_lidar/self.angle_incr)
        return range_data[index]
    
    def overtaking_check(self, msg, location, path, waypoint):
        rangetocheck=1
        if (self.get_dist(waypoint,location)< rangetocheck):
            pathgood=self.close_to_obstacle(waypoint,location,msg)
            if (pathgood== False):
                path= self.select_new_path(path)
    
 
                

    def get_dist(self, waypoint_location,car_location_in_world):
        return (((car_location_in_world[0]-waypoint_location[0])**2 + (car_location_in_world[1]-waypoint_location[1])**2 )**.5)

    def Select_new_path(path):
        if(path==2):
            return 1
        else:
            return 2
        
    #helper function checks to see if a given waypoint is close to a lidar detected obstacle
    def close_to_obstacle(self,waypoint_location, self_location, msg):
        range_we_care_about=1
        max_acceptable_dist_from_obstacle= .2
        rangedata=msg.ranges
        angle1rad=(60*np.pi)/180
        #the heading angle of the car
        heading=0
        angles = np.linspace(-angle1rad, angle1rad, num=10)
        for angle in angles:
            theta= (3.14/2)-angle-heading
            dist= self.get_range(rangedata,angle)
            if (dist< range_we_care_about):
                obstacle_location_in_world_frame= (self_location[0]+np.cos(theta)*dist, self_location[1]+np.sin(theta)*dist)
                if (self.get_dist(waypoint_location, obstacle_location_in_world_frame)<max_acceptable_dist_from_obstacle):
                    return False
        return True




    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        #how far we want to be from wall
        dist=1.0 #placeholder
        #range data
        rangedata=msg.ranges
        # TODO: replace with error calculated by get_error()
        error = self.get_error(rangedata, dist) 
        # we probably actually shouldnt do this  next part according to writeup idk i found v inside the pid 
        #see writeup for velocity based on steering angle
        # velocity =  # TODO: calculate desired car velocity based on error
        velocity=.6
        self.L=.5
        self.pid_control(error, velocity) # TODO: actuate the car with PID


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
