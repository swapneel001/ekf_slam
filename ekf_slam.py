#!/usr/bin/env python3
import math
import json
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo
from prob_rob_msgs.msg import Point2DArrayStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from tf2_ros import Buffer, TransformListener, TransformException

heartbeat_period = 0.1


COLOR_TO_ID = {
    'red': 1,
    'green': 2,
    'yellow': 3,
    'magenta': 4,
    'cyan': 5}


class EkfSlam(Node):

    def __init__(self):
        super().__init__('ekf_slam')
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

        self.state = np.zeros((3, 1))  # Robot pose: x, y, theta, initially
        self.Cov = np.eye(3) * 0.01  # Initial covariance matrix
        self.I = np.eye(len(self.state)) #identity matrix for landmarks but dimensions are dynamic so we initialise later
        self.landmark_registry = {}  # landmark_id: landmark id and color 



        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Camera transform (base_link -> camera_rgb_frame)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.T_base_to_camera = None
        self.cam_offset = np.zeros(2)   # [tx, ty] camera position in base frame
        self.cam_yaw = 0.0              # yaw offset camera wrt base
        self.tf_timer = self.create_timer(1.0, self.get_camera_transform)

        self.camera_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_callback,
            10
        )

    def heartbeat(self):
        self.log.info('heartbeat')

    def q2yaw(self, quat):
        """Convert quaternion to yaw angle"""
        roll, pitch, yaw = euler_from_quaternion(
            [quat.x, quat.y, quat.z, quat.w]
        )
        return yaw

    def get_camera_transform(self):
        """Get base_link -> camera_rgb_frame transform"""
        if self.T_base_to_camera is not None:
            return

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                'base_link',          # target frame
                'camera_rgb_frame',   # source frame
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except TransformException as ex:
            self.log.warn(f'Could not get base_link -> camera_rgb_frame transform yet: {ex}')
            return

        self.T_base_to_camera = tf_msg
        trans = tf_msg.transform.translation
        rot = tf_msg.transform.rotation

        self.cam_offset = np.array([trans.x, trans.y])
        self.cam_yaw = self.q2yaw(rot)
        self.tf_timer.cancel()

    def camera_callback(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.log.info(f'Camera intrinsics received: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

    def get_landmark_index(self,landmark_id):
        ''' Returns the index of the landmark in the state vector.
            If the landmark is not found, returns -1.
        '''
        if landmark_id in self.landmark_registry:
            return self.landmark_registry[landmark_id]['index']
        else:
            return -1
        
    def initialize_landmark(self, landmark_id, range_meas, bearing_meas, var_range, var_bearing):
        '''Initializes a new landmark from first observation,
            return landmark index j'''

        # Base robot pose
        x = self.state[0, 0]
        y = self.state[1, 0]
        theta = self.state[2, 0]

        # Camera pose in map frame: base pose + rotated offset
        tx, ty = self.cam_offset[0], self.cam_offset[1]
        x_cam = math.cos(theta) * tx - math.sin(theta) * ty + x
        y_cam = math.sin(theta) * tx + math.cos(theta) * ty + y
        theta_cam = theta + self.cam_yaw

        # Convert range-bearing measurement from camera frame to global coordinates
        mx = x_cam + range_meas * math.cos(theta_cam + bearing_meas)
        my = y_cam + range_meas * math.sin(theta_cam + bearing_meas)

        # Jacobians for landmark initialization
        G_pose = np.array([
            [1, 0, -tx * math.sin(theta) - ty * math.cos(theta) - range_meas * math.sin(theta_cam + bearing_meas)],
            [0, 1,  tx * math.cos(theta) - ty * math.sin(theta) + range_meas * math.cos(theta_cam + bearing_meas)]
        ])

        G_meas = np.array([
            [math.cos(theta_cam + bearing_meas), -range_meas * math.sin(theta_cam + bearing_meas)],
            [math.sin(theta_cam + bearing_meas),  range_meas * math.cos(theta_cam + bearing_meas)]
        ])

        # Measurement noise
        Q = np.array([
            [var_range, 0],
            [0, var_bearing]
        ])

        # Current state dimension (before adding landmark)
        n = len(self.state)

        # New landmark will be at index n (appended to state vector)
        landmark_index = n

        self.state = np.vstack([self.state, np.array([[mx], [my]])])

        # Expand covariance matrix
        P_robot = self.Cov[0:3, 0:3]
        P_mm = G_pose @ P_robot @ G_pose.T + G_meas @ Q @ G_meas.T
        P_rm = P_robot @ G_pose.T

        # New covariance matrix
        new_Cov = np.zeros((n + 2, n + 2))
        new_Cov[0:n, 0:n] = self.Cov
        new_Cov[n:n+2, n:n+2] = P_mm
        new_Cov[0:3, n:n+2] = P_rm
        new_Cov[n:n+2, 0:3] = P_rm.T

        self.Cov = new_Cov

        # Register landmark in registry
        self.landmark_registry[landmark_id] = {
            'index': landmark_index,
            'id': landmark_id
        }

        self.log.info(f'Initialized landmark {landmark_id} at ({mx:.2f}, {my:.2f}), index={landmark_index}')

        return landmark_index


    


    def measurement_update(self, landmark_id, meas_range, meas_bearing):
        """
        Standard EKF range-bearing update to a known landmark.
        landmark_id: landmark index
        meas_range:  measured distance d_m
        meas_bearing: measured bearing theta_m
        """
        # if we don't yet know the camera transform, we cannot fuse properly
        if self.T_base_to_camera is None:
            return
        landmark_index = self.get_landmark_index(landmark_id)
        if landmark_index == -1:
            # Landmark not initialized yet
            landmark_index = self.initialize_landmark(landmark_id, meas_range, meas_bearing)
            return
        
        # landmark position in state vector
        mx = self.state[landmark_index, 0]
        my = self.state[landmark_index + 1, 0]

        # base state
        x = self.state[0, 0]
        y = self.state[1, 0]
        theta = self.state[2, 0]

        # camera pose in map frame: base pose + rotated offset
        tx, ty = self.cam_offset[0], self.cam_offset[1]
        x_cam = math.cos(theta) * tx - math.sin(theta) * ty + x
        y_cam = math.sin(theta) * tx + math.cos(theta) * ty + y
        theta_cam = theta + self.cam_yaw

        # expected measurement from camera frame
        dx = mx - x_cam
        dy = my - y_cam
        q = dx ** 2 + dy ** 2
        if q < 1e-9:
            return

        z1 = math.sqrt(q)
        z2 = self.unwrap(math.atan2(dy, dx) - theta_cam)

        # derivative of camera position wrt theta (how offset rotates)
        n_x = -math.sin(theta) * tx - math.cos(theta) * ty
        n_y =  math.cos(theta) * tx - math.sin(theta) * ty

        # Jacobian H with camera offset
        self.H[0, 0] = -dx / z1
        self.H[0, 1] = -dy / z1
        self.H[0, 2] = (-dx * n_x - dy * n_y) / z1

        self.H[1, 0] =  dy / q
        self.H[1, 1] = -dx / q
        self.H[1, 2] = (dy * n_x - dx * n_y) / q - 1.0

        # Innovation dz = z_meas - h(x)
        dz = np.array([
            [meas_range - z1],
            [self.unwrap(meas_bearing - z2)]
        ])

        # S = H P H^T + R (2x2)
        S = self.H @ self.Cov @ self.H.T + self.Q
        # K = P H^T S^{-1} (3x2)
        K = self.Cov @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ dz
        self.state[2, 0] = self.unwrap(self.state[2, 0])

        self.Cov = (self.I - K @ self.H) @ self.Cov

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    ekf_slam = EkfSlam()
    ekf_slam.spin()
    ekf_slam.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
