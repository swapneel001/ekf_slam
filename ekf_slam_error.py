import rclpy
from rclpy.node import Node
import math
import json
import os

from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from message_filters import Subscriber, ApproximateTimeSynchronizer

from tf2_ros import Buffer, TransformListener, TransformException
from ament_index_python.packages import get_package_share_directory

map_path = 'src/prob_rob_labs_ros_2/prob_rob_labs/config/landmarks_map.json'

def quat_to_yaw(q: Quaternion) -> float:
    x, y, z, w = q.x, q.y, q.z, q.w
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: float) -> float:
    a = (angle + math.pi) % (2.0 * math.pi)
    if a < 0.0:
        a += 2.0 * math.pi
    return a - math.pi


heartbeat_period = 0.1


class EkfSlamError(Node):

    def __init__(self):
        super().__init__('ekf_slam_error')

        # Robot pose error subscribers
        self.gt_sub = Subscriber(self, PoseStamped, '/tb3/ground_truth/pose')
        self.ekf_sub = Subscriber(self, Odometry, '/ekf_slam_pose')

        self.map_path = map_path

        self.sync = ApproximateTimeSynchronizer(
            [self.gt_sub, self.ekf_sub],
            queue_size=50,
            slop=0.05,
            allow_headerless=False
        )
        self.sync.registerCallback(self.synced_cb)
        self.pos_err_pub = self.create_publisher(Float64, '/ekf_error/position_abs', 10)
        self.yaw_err_pub = self.create_publisher(Float64, '/ekf_error/yaw_abs', 10)

        # TF listener for landmark positions
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Load ground truth landmark positions from config
        self.landmark_gt = self.load_landmark_ground_truth()

        # Create publishers for each landmark error
        self.landmark_err_pubs = {}
        for color in self.landmark_gt.keys():
            topic = f'/ekf_error/landmark_{color}_pos'
            self.landmark_err_pubs[color] = self.create_publisher(Float64, topic, 10)

        # Timer to periodically check landmark errors
        self.landmark_timer = self.create_timer(0.1, self.landmark_error_callback)

        self.get_logger().info('ekf_error_plotter running.')
        self.get_logger().info(f'Tracking {len(self.landmark_gt)} landmarks for error computation.')

    def synced_cb(self, gt_pose_msg: PoseStamped, ekf_odom_msg: Odometry):
        gx = gt_pose_msg.pose.position.x
        gy = gt_pose_msg.pose.position.y
        ex = ekf_odom_msg.pose.pose.position.x
        ey = ekf_odom_msg.pose.pose.position.y

        pos_err = math.hypot(ex - gx, ey - gy)
        yaw_gt = quat_to_yaw(gt_pose_msg.pose.orientation)
        yaw_est = quat_to_yaw(ekf_odom_msg.pose.pose.orientation)
        yaw_diff = wrap_to_pi(yaw_est - yaw_gt)
        yaw_err_abs = abs(yaw_diff)

        self.pos_err_pub.publish(Float64(data=pos_err))
        self.yaw_err_pub.publish(Float64(data=yaw_err_abs))

    def load_landmark_ground_truth(self):
        """Load ground truth landmark positions from landmarks_map.json"""
        landmark_gt = {}
        with open(self.map_path, 'r') as f:
            data = json.load(f)
        for lm in data['landmarks']:
            color = lm['color']
            pos = lm['position']
            landmark_gt[color] = (pos['x'], pos['y'])
        self.get_logger().info(f'Loaded {len(landmark_gt)} ground truth landmarks from {self.map_path}')

        return landmark_gt

    def landmark_error_callback(self):
        """Compute and publish landmark position errors by looking up TF transforms"""
        for color, (gt_x, gt_y) in self.landmark_gt.items():
            frame_name = f"landmark_{color}"
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    'map',
                    frame_name,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05)
                )
                est_x = tf_msg.transform.translation.x
                est_y = tf_msg.transform.translation.y
                pos_err = math.hypot(est_x - gt_x, est_y - gt_y)
                self.landmark_err_pubs[color].publish(Float64(data=pos_err))
            except TransformException:
                # Landmark not yet published, skip
                pass

    def heartbeat(self):
        self.get_logger().info('heartbeat')

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    ekf_error_plotter = EkfSlamError()
    ekf_error_plotter.spin()
    ekf_error_plotter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
