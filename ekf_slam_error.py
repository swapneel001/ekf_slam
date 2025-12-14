import rclpy
from rclpy.node import Node
import math

from geometry_msgs.msg import PoseStamped, Quaternion, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from gazebo_msgs.msg import LinkStates
from message_filters import Subscriber, ApproximateTimeSynchronizer

from tf2_ros import Buffer, TransformListener, TransformException

from visualization_msgs.msg import Marker, MarkerArray

# Map landmark numbers to colors
LANDMARK_NUM_TO_COLOR = {
    1: 'red',
    2: 'green',
    3: 'yellow',
    4: 'magenta',
    5: 'cyan'
}


def quat_to_yaw(q: Quaternion) -> float:
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: float) -> float:
    a = (angle + math.pi) % (2.0 * math.pi)
    if a < 0.0:
        a += 2.0 * math.pi
    return a - math.pi


class EkfSlamError(Node):

    def __init__(self):
        super().__init__('ekf_slam_error')

        # Robot pose error subscribers
        self.gt_sub = Subscriber(self, PoseStamped, '/tb3/ground_truth/pose')
        self.ekf_sub = Subscriber(self, Odometry, '/ekf_slam_pose')

        self.sync = ApproximateTimeSynchronizer(
            [self.gt_sub, self.ekf_sub],
            queue_size=50,
            slop=0.05,
            allow_headerless=False
        )
        self.sync.registerCallback(self.synced_cb)
        self.pos_err_pub = self.create_publisher(Float64, '/ekf_error/position_abs', 10)
        self.yaw_err_pub = self.create_publisher(Float64, '/ekf_error/yaw_abs', 10)

        # TF listener for estimated landmark positions
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Ground truth landmark positions from Gazebo (updated dynamically)
        self.landmark_gt = {}  # color -> (x, y)

        # Subscribe to Gazebo link states for landmark ground truth
        self.link_states_sub = self.create_subscription(
            LinkStates,
            '/gazebo/link_states',
            self.link_states_callback,
            10
        )

        # Create publishers for each landmark error
        self.landmark_err_pubs = {}
        for color in LANDMARK_NUM_TO_COLOR.values():
            topic = f'/ekf_error/landmark_{color}_pos'
            self.landmark_err_pubs[color] = self.create_publisher(Float64, topic, 10)

        # Timer to periodically check landmark errors
        self.landmark_timer = self.create_timer(0.1, self.landmark_error_callback)


        #storing points for trajectory overlaying
        self.gt_traj = []
        self.ekf_traj = []

        self.min_distance = 0.1
        self.gt_last_point = None
        self.ekf_last_point = None

        self.ekf_traj_pub = self.create_publisher(Marker, '/ekf_slam/trajectory', 10)
        self.gt_traj_pub = self.create_publisher(Marker, '/ground_truth/trajectory', 10)

        self.get_logger().info('ekf_slam_error running, waiting for Gazebo link states...')

    def link_states_callback(self, msg):
        """Extract landmark positions from Gazebo link states"""
        for i, name in enumerate(msg.name):
            # Look for landmark links (e.g., "landmark_1::link")
            if name.startswith('landmark_') and '::' in name:
                landmark_part = name.split('::')[0]  # "landmark_1"
                try:
                    landmark_num = int(landmark_part.split('_')[1])
                except (IndexError, ValueError):
                    continue

                if landmark_num in LANDMARK_NUM_TO_COLOR:
                    color = LANDMARK_NUM_TO_COLOR[landmark_num]
                    pose = msg.pose[i]
                    self.landmark_gt[color] = (pose.position.x, pose.position.y)

        # Log once when we have all landmarks
        if len(self.landmark_gt) == 5 and not hasattr(self, '_logged_landmarks'):
            self._logged_landmarks = True
            self.get_logger().info(f'Loaded {len(self.landmark_gt)} landmark positions from Gazebo')
            for color, (x, y) in self.landmark_gt.items():
                self.get_logger().info(f'  {color}: ({x:.2f}, {y:.2f})')

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

        if self.gt_last_point is None:
            self.gt_last_point = (gx, gy)
        if self.ekf_last_point is None:
            self.ekf_last_point = (ex, ey)
        
        if math.hypot(gx - self.gt_last_point[0], gy - self.gt_last_point[1]) >= self.min_distance:
            self.gt_traj.append((gx,gy))
            self.gt_last_point = (gx, gy)
        if math.hypot(ex - self.ekf_last_point[0], ey - self.ekf_last_point[1]) >= self.min_distance:
            self.ekf_traj.append((ex,ey))
            self.ekf_last_point = (ex, ey)
        self.pos_err_pub.publish(Float64(data=pos_err))
        self.yaw_err_pub.publish(Float64(data=yaw_err_abs))
        self.traj_publish()

    def traj_publish(self):
        ekf_marker = Marker()
        ekf_marker.header.frame_id = "map"
        ekf_marker.header.stamp = self.get_clock().now().to_msg()
        ekf_marker.ns = "ekf_trajectory"
        ekf_marker.id = 0
        ekf_marker.type = Marker.LINE_STRIP
        ekf_marker.action = Marker.ADD
        ekf_marker.scale.x = 0.02
        ekf_marker.color.a = 1.0
        ekf_marker.color.r = 1.0
        ekf_marker.color.g = 0.0
        ekf_marker.color.b = 0.0

        for point in self.ekf_traj:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.0
            ekf_marker.points.append(p)

        self.ekf_traj_pub.publish(ekf_marker)

        gt_marker = Marker()
        gt_marker.header.frame_id = "map"
        gt_marker.header.stamp = self.get_clock().now().to_msg()
        gt_marker.ns = "gt_trajectory"
        gt_marker.id = 1
        gt_marker.type = Marker.LINE_STRIP
        gt_marker.action = Marker.ADD
        gt_marker.scale.x = 0.02
        gt_marker.color.a = 1.0
        gt_marker.color.r = 0.0
        gt_marker.color.g = 1.0
        gt_marker.color.b = 0.0

        for point in self.gt_traj:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.0
            gt_marker.points.append(p)

        self.gt_traj_pub.publish(gt_marker)
        self.ekf_traj_pub.publish(ekf_marker)




    def landmark_error_callback(self):
        """Compute and publish landmark position errors by looking up TF transforms"""
        if not self.landmark_gt:
            return  # No ground truth yet

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