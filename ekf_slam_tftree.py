#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

import numpy as np

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from tf_transformations import euler_from_quaternion, quaternion_from_euler

heartbeat_period = 1.0 / 30.0  # ~30 Hz


class EkfSlamTfTree(Node):

    def __init__(self):
        super().__init__('ekf_slam_tftree')
        self.log = self.get_logger()

        self.odom_msg = None      # latest /odom
        self.ekf_msg = None       # latest /ekf_pose
        self.last_map_odom_tf = None  # TransformStamped to broadcast

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.ekf_sub = self.create_subscription(
            Odometry,
            '/ekf_slam_pose',
            self.ekf_callback,
            10
        )

        # Timer for 30 Hz broadcasting
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

    def heartbeat(self):
        """Broadcast the last computed map->odom at ~30 Hz."""
        if self.last_map_odom_tf is None:
            return

        # Refresh timestamp
        self.last_map_odom_tf.header.stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(self.last_map_odom_tf)

    def odom_callback(self, msg: Odometry):
        self.odom_msg = msg

    def ekf_callback(self, msg: Odometry):
        """EKF pose updated: recompute map->odom if possible."""
        self.ekf_msg = msg
        self.update_map_odom()


    def pose_to_xytheta(self, pose):
        x = pose.position.x
        y = pose.position.y
        q = pose.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return x, y, yaw

    def xytheta_to_mat(self, x, y, theta):
        c = math.cos(theta)
        s = math.sin(theta)
        T = np.array([
            [c, -s, x],
            [s,  c, y],
            [0.0, 0.0, 1.0]
        ], dtype=float)
        return T

    def mat_to_xytheta(self, T):
        x = T[0, 2]
        y = T[1, 2]
        theta = math.atan2(T[1, 0], T[0, 0])  # from rotation part
        return x, y, theta


    def update_map_odom(self):
        """Compute map->odom = map->base * (odom->base)^-1"""
        if self.odom_msg is None or self.ekf_msg is None:
            return

        # map -> base from EKF
        xm, ym, th_m = self.pose_to_xytheta(self.ekf_msg.pose.pose)
        T_map_base = self.xytheta_to_mat(xm, ym, th_m)

        # odom -> base from /odom
        xo, yo, th_o = self.pose_to_xytheta(self.odom_msg.pose.pose)
        T_odom_base = self.xytheta_to_mat(xo, yo, th_o)

        # map -> odom = map -> base * (odom -> base)^-1
        T_odom_base_inv = np.linalg.inv(T_odom_base)
        T_map_odom = T_map_base @ T_odom_base_inv

        x_mo, y_mo, th_mo = self.mat_to_xytheta(T_map_odom)

        # Build TransformStamped for map -> odom
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"

        t.transform.translation.x = float(x_mo)
        t.transform.translation.y = float(y_mo)
        t.transform.translation.z = 0.0

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, th_mo)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.last_map_odom_tf = t

    def spin(self):
        rclpy.spin(self)


def main(args=None):
    rclpy.init()
    node = EkfSlamTfTree()
    node.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
