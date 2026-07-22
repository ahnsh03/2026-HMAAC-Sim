import os
import tempfile

import rclpy
from rclpy.node import Node
from simulation_pkg import basic


class EgoCarLoader(Node):
    def __init__(self):
        super().__init__('ego_car_loader')

        # driving_sim 은 라이다 없이, mission_sim / hmobility_sim 은 라이다 달고 스폰한다.
        # 라이다는 prius_hybrid/lidar_front.sdf 스니펫을 스폰 직전에 주입한다.
        # (모델 원본은 라이다 없이 한 벌만 유지)
        self.with_lidar = self.declare_parameter('with_lidar', False).value

        # 4초 후에 한 번만 실행되는 타이머 생성
        self.timer = self.create_timer(4.5, self.load_model_callback)

    def load_model_callback(self):
        if self.with_lidar:
            self.get_logger().info('Loading Prius Hybrid (with front lidar)')
            self.spawn_with_lidar()
        else:
            self.get_logger().info('Loading Prius Hybrid')
            basic.load_model("ego_vehicle", "prius_hybrid", basic.driving_ego())
        self.timer.cancel()

    def spawn_with_lidar(self):
        """model.sdf 에 라이다 스니펫을 끼운 임시 파일을 만들어 스폰한다."""
        model_file = basic.get_model("prius_hybrid")            # .../models/prius_hybrid/model.sdf
        snippet_file = os.path.join(os.path.dirname(model_file), "lidar_front.sdf")

        try:
            with open(model_file) as f:
                sdf = f.read()
            with open(snippet_file) as f:
                snippet = f.read()
        except OSError as e:
            self.get_logger().error(f'라이다 주입 실패, 기본 모델로 스폰: {e}')
            basic.load_model("ego_vehicle", "prius_hybrid", basic.driving_ego())
            return

        sdf = sdf.replace('</model>', snippet + '\n</model>', 1)

        patched = os.path.join(tempfile.gettempdir(), 'ego_with_lidar.sdf')
        with open(patched, 'w') as f:
            f.write(sdf)

        x, y, z, roll, pitch, yaw = basic.driving_ego()
        os.system(
            f"ros2 run gazebo_ros spawn_entity.py -file {patched} -entity ego_vehicle "
            f"-x {x} -y {y} -z {z} -R {roll} -P {pitch} -Y {yaw}"
        )


def main():
    rclpy.init()
    ego_car_loader = EgoCarLoader()

    try:
        rclpy.spin(ego_car_loader)
    except KeyboardInterrupt:
        pass
    finally:
        ego_car_loader.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
