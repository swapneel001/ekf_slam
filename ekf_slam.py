import rclpy
from rclpy.node import Node


heartbeat_period = 0.1

class EkfSlam(Node):

    def __init__(self):
        super().__init__('ekf_slam')
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

    def heartbeat(self):
        self.log.info('heartbeat')

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
