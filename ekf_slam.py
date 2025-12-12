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

from tf2_ros import Buffer, TransformListener, TransformException,TransformStamped
from tf2_ros.transform_broadcaster import TransformBroadcaster

heartbeat_period = 0.1


COLOR_TO_ID = {
    'red': 1,
    'green': 2,
    'yellow': 3,
    'magenta': 4,
    'cyan': 5}

ID_TO_COLOR = {v: k for k, v in COLOR_TO_ID.items()}



class EkfSlam(Node):

    '''
    EKF SLAM node for landmark-based SLAM using range-bearing measurements.
    Subscribes to:
    - /odom: Odometry messages for robot motion
    - /vision_<color>/corners: Point2DArrayStamped messages for landmark observations
    - /camera/camera_info: CameraInfo messages for camera intrinsics
    
    Publishes:
    - /ekf_pose: Odometry messages for estimated robot pose
    - /tf: Transforms for landmarks in the map frame
    '''

    def __init__(self):
        super().__init__('ekf_slam')
        self.log = self.get_logger()

        self.state = np.zeros((3, 1))  # Robot pose: x, y, theta, initially
        self.Cov = np.eye(3) * 0.01  # Initial covariance matrix
        self.I = np.eye(len(self.state)) #identity matrix for landmarks but dimensions are dynamic so we initialise later
        self.landmark_registry = {}  # landmark_id: landmark id and color 

        self.landmark_height = 0.5
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


        #odom params 
        self.system_time = None #rlcpy time of current state
        self.initialized = False    #true after first measurement timestamp is set
        self.last_vel = None   #v,w from odom 

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )



        #vision subscription

        self.corner_subs = {}
        for color in COLOR_TO_ID.keys():
            topic = f'/vision_{color}/corners'
            sub = self.create_subscription(
                Point2DArrayStamped,
                topic,
                lambda msg, c=color: self.corner_callback(msg, c),
                10
            )
            self.corner_subs[color] = sub
            self.log.info(f'Subscribed to corner topic: {topic}')

        # EKF matrices
        self.G = np.eye(3)  # Motion model Jacobian
        self.V = np.zeros((3, 2))  # Control noise Jacobian


        #base variances for measurement model - obtained from Lab 4/5
        #these variances are valid for the reliable measurement range, outside that range we inflate by factor 3
        self.var_d_base = 0.1467
        self.var_theta_base = 0.000546

        # Process noise from odometry twist covariance (v, w)
        self.M = np.array([[0.01, 0.0],
                           [0.0,      0.01]])

        #camera information
        self.camera_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_callback,
            10
        )

        #publishers for robot pose - as ekf pose, and landmarks - as tf arrays
        self.odom_pub = self.create_publisher(Odometry, "/ekf_slam_pose", 10)
        self.landmark_pub = TransformBroadcaster(self)


    #CALLBACKS BELOW
    def camera_callback(self, msg):
        '''
        Callback for camera intrinsics
        '''
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.log.info(f'Camera intrinsics received: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

    def odom_callback(self, msg):
        '''
        Callback for odometry messages
        Triggers EKF SLAM prediction step from odometry
        '''

        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.last_vel = (v, w)

        if not self.initialized:
            return

        timestamp = msg.header.stamp
        dt = self.seconds(timestamp) - self.seconds(self.system_time)
        if dt <= 0.0:
            self.log.warn('Non-positive time difference in odom callback, skipping prediction')
            return
        
        self.system_time = timestamp
        self.prediction(v, w, dt)
        self.publish_ekf_pose(timestamp)

    def corner_callback(self, msg, color):

        '''
        Callback for landmark corner detections
        Triggers EKF SLAM measurement update from landmark corner detections
        Uses range and bearing derived from corner pixel locations
        '''
        # Need camera intrinsics first
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            self.log.warn('No camera intrinsics yet, skipping measurement')
            return

        points = msg.points
        num_points = len(points)
        if num_points == 0:
            return

        xs = [p.x for p in points]
        ys = [p.y for p in points]

        min_y = min(ys)
        max_y = max(ys)
        height_pix = max_y - min_y

        # Treat differently for rectangular vs cylindrical perception of landmarks based on number of corners
        if num_points < 8:
            return #landmark is not fully visible/is too far to get a reliable measurement
        else:
            x_sym = sum(xs) / float(num_points)
            shape = 'cyl'

        # Bearing in camera frame (we approximate as base frame bearing for this lab)
        theta_m = math.atan((self.cx - x_sym) / self.fx)

        cos_th = math.cos(theta_m)

        if height_pix <= 0 or abs(cos_th) < 1e-3:
            # too close to 90°, avoid numerical blow-up

            return

        d_m = self.landmark_height * self.fy / (height_pix * cos_th)

        var_d, var_theta = self.measurement_variance(d_m, theta_m)
        landmark_id = COLOR_TO_ID[color]
        timestamp = msg.header.stamp

        if not self.initialized:
            self.initialized = True
            self.system_time = timestamp
            self.log.info("EKF initialized with first measurement timestamp (state left at prior).")
            return

        dt = self.seconds(timestamp) - self.seconds(self.system_time)

        if dt < -0.2:        
            # very late-arriving sample, discard
            #added tolerance for minor clock sync issues
            return
        
        if dt <= 0.0:
            self.measurement_update(landmark_id, d_m, theta_m, var_d, var_theta)
            self.publish_ekf_pose(self.system_time)
            self.publish_landmarks(self.system_time)
            return
        
        if self.last_vel is not None:
            v, w = self.last_vel
            self.prediction(v, w, dt)

        self.measurement_update(landmark_id, d_m, theta_m, var_d, var_theta)
        self.system_time = timestamp
        self.publish_ekf_pose(timestamp)
        self.publish_landmarks(timestamp)



    
    
    # SLAM SPECIFIC METHODS BELOW - Landmark initialization, measurement update, prediction 

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

        self.log.info(f'Landmark color {ID_TO_COLOR[landmark_id]} assigned ID {landmark_id}')

        return landmark_index


    def measurement_update(self, landmark_id, meas_range, meas_bearing, var_range, var_bearing):
        """
        Updated method from EKF Localization to suit SLAM Problem
        EKF range-bearing update to a landmark given landmark ID
        Carries out measurement update step of EKF SLAM, 
        then computes Kalman Gain and carries out innovation step of state and covariance
        """
        # if we don't yet know the camera transform, we cannot fuse properly
        if self.T_base_to_camera is None:
            return
        landmark_index = self.get_landmark_index(landmark_id)
        if landmark_index == -1:
            # Landmark not initialized yet
            landmark_index = self.initialize_landmark(landmark_id, meas_range, meas_bearing, var_range, var_bearing)
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

        #expected measurement
        delta_x = mx - x_cam
        delta_y = my - y_cam
        q = delta_x**2 + delta_y**2
        expected_range = math.sqrt(q)
        expected_bearing = math.atan2(delta_y, delta_x) - theta_cam
        expected_bearing = (expected_bearing + math.pi) % (2 * math.pi) - math.pi  # normalize to [-pi, pi]


        # derivative of camera position wrt theta (how offset rotates)
        n_x = -math.sin(theta) * tx - math.cos(theta) * ty
        n_y =  math.cos(theta) * tx - math.sin(theta) * ty

        #sparse H matrix 
        #use index slicing to replace usage of sifting matrix F - same logically
        H = np.zeros((2, len(self.state)))
        
        H[0,0] = -delta_x / expected_range
        H[0,1] = -delta_y / expected_range
        H[0,2] = -(n_x * delta_x + n_y * delta_y) / expected_range

        H[0,landmark_index] = delta_x / expected_range
        H[0,landmark_index + 1] = delta_y / expected_range

        H[1,0] = delta_y / q
        H[1,1] = -delta_x / q
        H[1,2] = (n_x * delta_y - n_y * delta_x) / q - 1.0

        H[1,landmark_index] = -delta_y / q
        H[1,landmark_index + 1] = delta_x / q

        # Measurement noise
        Q = np.array([
            [var_range, 0],
            [0, var_bearing]
        ])

        I = np.eye(len(self.state))

        # Kalman Gain
        K = self.Cov@H.T @ np.linalg.inv(H @ self.Cov @ H.T + Q)

        # Measurement residual
        z = np.array([[meas_range], [meas_bearing]])
        z_hat = np.array([[expected_range], [expected_bearing]])
        y_residual = z - z_hat
        y_residual[1, 0] = (y_residual[1, 0] + math.pi) % (2 * math.pi) - math.pi  # normalize to [-pi, pi]

        # State update
        self.state = self.state + K @ y_residual

        # Covariance update
        self.Cov = (I - K @ H) @ self.Cov

    
    def prediction(self, v, w, dt):
        
        '''
        EKF SLAM prediction step from odometry
        Carries out prediction step of EKF SLAM using motion model
        Updates robot pose and covariance accordingly
        '''
        theta = self.state[2, 0]

        # small angular velocity → linear motion model
        if abs(w) < 0.01:
            # Jacobians
            self.G[0, 2] = -v * dt * math.sin(theta)
            self.G[1, 2] =  v * dt * math.cos(theta)

            self.V[:, :] = 0.0
            self.V[0, 0] = dt * math.cos(theta)
            self.V[1, 0] = dt * math.sin(theta)
            self.V[2, 1] = dt

            # State update (linear)
            self.state[0, 0] += v * dt * math.cos(theta)
            self.state[1, 0] += v * dt * math.sin(theta)
            self.state[2, 0] = self.unwrap(theta + w * dt)

        else:
            # Arc motion model
            self.G[0, 2] = -v / w * math.cos(theta) + v / w * math.cos(theta + w * dt)
            self.G[1, 2] = -v / w * math.sin(theta) + v / w * math.sin(theta + w * dt)

            self.V[:, :] = 0.0
            self.V[0, 0] = (-math.sin(theta) + math.sin(theta + w * dt)) / w
            self.V[0, 1] = (
                v * (math.sin(theta) - math.sin(theta + w * dt)) / (w ** 2)
                + v * math.cos(theta + w * dt) * dt / w
            )
            self.V[1, 0] = (math.cos(theta) - math.cos(theta + w * dt)) / w
            self.V[1, 1] = (
                -v * (math.cos(theta) - math.cos(theta + w * dt)) / (w ** 2)
                + v * math.sin(theta + w * dt) * dt / w
            )
            self.V[2, 1] = dt

            self.state[0, 0] += -v / w * math.sin(theta) + v / w * math.sin(theta + w * dt)
            self.state[1, 0] +=  v / w * math.cos(theta) - v / w * math.cos(theta + w * dt)
            self.state[2, 0] = self.unwrap(theta + w * dt)
        # Covariance update - landmarks remain unchanged
        #index slicing effectively replaces the use of sifting matrix F
        P_rr = self.Cov[0:3, 0:3].copy()
        P_rm = self.Cov[0:3, 3:].copy()
        P_mr = self.Cov[3:, 0:3].copy()

        self.Cov[0:3, 0:3] = self.G @ P_rr @ self.G.T + self.V @ self.M @ self.V.T

        self.Cov[0:3, 3:] = self.G @ P_rm

        self.Cov[3:, 0:3] = P_mr @ self.G.T


    #LANDMARK AND POSE PUBLISHING 
    def publish_landmarks(self,timestamp):

        '''
        Publishes the current EKF SLAM computed landmark positions as TF transforms.
        '''
        for landmark_id, info in self.landmark_registry.items():
            index = info['index']
            mx = self.state[index, 0]
            my = self.state[index + 1, 0]
            
            #publishing as tf
            self.log.info(f'Publishing landmark {landmark_id} at ({mx:.2f}, {my:.2f})')

            t = TransformStamped()
            t.header.stamp = timestamp
            t.header.frame_id = "map"
            t.child_frame_id = f"landmark_{ID_TO_COLOR[landmark_id]}"
            t.transform.translation.x = float(mx)
            t.transform.translation.y = float(my)
            t.transform.translation.z = 0.0
            t.transform.rotation = self.to_quaternion(0.0)
            self.landmark_pub.sendTransform(t)
    
            
    def publish_ekf_pose(self, timestamp):

        '''Publishes the current EKF computed robot pose as an Odometry message.'''
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "base_footprint"
        #self.log.debug(f'Publishing EKF pose at time {self.seconds(timestamp):.2f}s: x={self.state[0,0]:.2f}, y={self.state[1,0]:.2f}, theta={self.state[2,0]:.2f}')

        odom_msg.pose.pose.position.x = float(self.state[0, 0])
        odom_msg.pose.pose.position.y = float(self.state[1, 0])
        odom_msg.pose.pose.orientation = self.to_quaternion(float(self.state[2, 0]))

        odom_msg.pose.covariance[0]  = float(self.Cov[0, 0])  # xx
        odom_msg.pose.covariance[1]  = float(self.Cov[0, 1])  # xy
        odom_msg.pose.covariance[5]  = float(self.Cov[0, 2])  # x-yaw
        odom_msg.pose.covariance[6]  = float(self.Cov[1, 0])  # yx
        odom_msg.pose.covariance[7]  = float(self.Cov[1, 1])  # yy
        odom_msg.pose.covariance[11] = float(self.Cov[1, 2])  # y-yaw
        odom_msg.pose.covariance[30] = float(self.Cov[2, 0])  # yaw-x
        odom_msg.pose.covariance[31] = float(self.Cov[2, 1])  # yaw-y
        odom_msg.pose.covariance[35] = float(self.Cov[2, 2])  # yaw-yaw

        self.odom_pub.publish(odom_msg)
    #ALL HELPER FUNCTIONS BELOW

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

    def measurement_variance(self, d, theta):
        """
        Linear variance model for range, constant for bearing.
        Based on empirical measurement error analysis.
        """
        # Range variance: linear with distance (R² = 0.82)
        var_d = 0.0026 * d - 0.006
        var_d = max(var_d, 0.0001)  # floor for close range
        
        # Bearing variance: constant
        var_theta = 0.0005
        
        # Inflate outside reliable region (1m <= d <= 10m, |θ| <= 0.6)
        if d < 1.0 or d > 10.0 or abs(theta) > 0.6:
            var_d *= 3.0
            var_theta *= 3.0
            
        return var_d, var_theta

    def get_landmark_index(self,landmark_id):
        ''' Returns the index of the landmark in the state vector.
            If the landmark is not found, returns -1.
        '''
        if landmark_id in self.landmark_registry:
            return self.landmark_registry[landmark_id]['index']
        else:
            return -1

    def seconds(self, timestamp):
        '''Convert ROS2 Time to float seconds.'''
        return timestamp.sec + timestamp.nanosec * 1e-9
    
    @staticmethod
    def to_quaternion(theta):
        """Convert theta angle to quaternion."""
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, theta)
        q = Quaternion()
        q.x = qx
        q.y = qy
        q.z = qz
        q.w = qw
        return q
    @staticmethod
    def unwrap(angle):
        """Wrap angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


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
