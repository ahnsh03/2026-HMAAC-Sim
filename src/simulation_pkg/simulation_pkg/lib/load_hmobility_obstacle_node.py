import rclpy
from rclpy.node import Node
from simulation_pkg import basic


class HmobilityObstacleLoader(Node):
    """H-모빌리티 환경(hmobility_sim)에서만 뜨는 고정 위치 장애물 차량.

    최종평가와 가장 유사한 배치. ego 차량과 같은 방식(타이머)으로,
    Gazebo 월드가 뜬 뒤에 스폰한다.
    """

    def __init__(self):
        super().__init__('hmobility_obstacle_loader')
        self.timer = self.create_timer(4.5, self.load_model_callback)

    def load_model_callback(self):
        self.get_logger().info('Loading H-mobility obstacle car')
        basic.load_model("prius_hybrid_ob1", "prius_hybrid_ob1", basic.hmobility_obstacle_car())
        self.timer.cancel()


def main():
    rclpy.init()
    node = HmobilityObstacleLoader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
