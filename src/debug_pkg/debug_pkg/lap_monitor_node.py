"""트랙 완주 판정과 궤적 기록.

Gazebo 의 ground truth 위치를 쓰므로 검증 전용이다. 제어 입력으로 쓰면 안 된다.
출발점에서 EXIT_RADIUS 밖으로 나갔다가 LAP_RADIUS 안으로 돌아오면 한 바퀴로 센다.
"""

import csv
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy

import math

import numpy as np

from std_msgs.msg import Float32MultiArray
from gazebo_msgs.msg import ModelStates
from interfaces_pkg.msg import MotionCommand, PathPlanningResult

#---------------Variable Setting---------------
MODEL_NAME = "ego_vehicle"
SUB_MODEL_STATES_TOPIC = "/gazebo/model_states"
SUB_CONTROL_TOPIC = "topic_control_signal"
SUB_PATH_TOPIC = "path_planning_result"
SUB_CONTROL_DEBUG_TOPIC = "control_debug"

# BEV <-> 실제 거리 환산. motion_planner 와 같은 규약.
PX_PER_M_LAT = 103.6
PX_PER_M_LON = 98.0
BEV_BOTTOM_ROW = 479.0
BOTTOM_ROW_AHEAD = 4.74
CAR_CENTER_X = 320.0
LANE_ERR_S = 5.0        # 차로중심 오차를 재는 전방거리 [m]
STRAIGHT_KAPPA = 0.02   # 이보다 곡률이 작으면 직선으로 보고 통계에 넣는다 [1/m]

LAP_RADIUS = 3.0        # 출발점 복귀로 인정하는 반경 [m]
EXIT_RADIUS = 10.0      # 이 밖으로 나가야 '출발했다'고 본다 [m]
STUCK_SPEED = 0.05      # 이 거리 이하로만 움직이면 정지로 본다 [m]
STUCK_WARN_SEC = 5.0    # 정지 상태가 이만큼 이어지면 경고 [s]

TRAJECTORY_CSV = "/tmp/lap_trajectory.csv"
SAMPLE_PERIOD = 0.2     # 궤적 기록 주기 [s]
#----------------------------------------------


class LapMonitorNode(Node):
    def __init__(self):
        super().__init__('lap_monitor_node')

        self.model_name = self.declare_parameter('model_name', MODEL_NAME).value
        self.lap_radius = self.declare_parameter('lap_radius', LAP_RADIUS).value
        self.exit_radius = self.declare_parameter('exit_radius', EXIT_RADIUS).value
        self.csv_path = self.declare_parameter('trajectory_csv', TRAJECTORY_CSV).value

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.start_xy = None
        self.left_start = False
        self.lap_count = 0
        self.start_time = None
        self.last_sample_time = 0.0
        self.last_xy = None
        self.stuck_since = None
        self.distance = 0.0
        self.steering = 0
        self.speed = 0
        self.lane_err = float('nan')   # 전방 5m 차로 중심의 횡방향 [m]. 양수면 차가 왼쪽
        self.kappa = 0.0               # 경로 곡률 [1/m]
        # 제어기가 실제로 본 값과 낸 값 (control_debug 토픽)
        self.ld = float('nan')         # 전방 주시거리 [m]
        self.lat = float('nan')        # 그 지점의 차로 중심 횡방향 [m]
        self.ctrl_kappa = float('nan') # 제어기가 쓴 곡률 [1/m]
        self.target_steer = float('nan')
        self.steer_cont = float('nan')
        self.bow = float('nan')          # 경로가 휜 정도 [m]
        self.straight = float('nan')     # 직진 락온 여부
        self.lane_err_sum = 0.0
        self.lane_err_bias = 0.0
        self.lane_err_max = 0.0
        self.lane_err_n = 0

        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # 인지가 문제인지 제어가 문제인지 나중에 가릴 수 있도록,
        # 자세 / 인지 결과 / 제어 명령을 같은 시각에 함께 남긴다.
        self.csv_writer.writerow([
            't', 'x', 'y', 'yaw',          # 지상진실 자세
            'distance', 'lane_err',        # 주행거리, 전방 5m 차로중심 횡방향
            'ld', 'lat', 'kappa',          # 제어기가 본 것: 주시거리, 그 지점 횡방향, 곡률
            'target_steer', 'steer_cont',  # 제어기가 원한 조향(연속)
            'bow', 'straight',             # 경로 활 높이[m], 직진 락온
            'steering', 'speed'])          # 실제 발행한 명령

        self.create_subscription(ModelStates, SUB_MODEL_STATES_TOPIC,
                                 self.model_states_callback, qos_profile)
        self.create_subscription(MotionCommand, SUB_CONTROL_TOPIC,
                                 self.control_callback, qos_profile)
        self.create_subscription(PathPlanningResult, SUB_PATH_TOPIC,
                                 self.path_callback, qos_profile)
        self.create_subscription(Float32MultiArray, SUB_CONTROL_DEBUG_TOPIC,
                                 self.control_debug_callback, qos_profile)

        self.get_logger().info(f"랩 모니터 시작. 궤적: {self.csv_path}")

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def control_callback(self, msg: MotionCommand):
        self.steering = msg.steering
        self.speed = msg.left_speed

    def control_debug_callback(self, msg: Float32MultiArray):
        if len(msg.data) >= 5:
            (self.ld, self.lat, self.ctrl_kappa,
             self.target_steer, self.steer_cont) = msg.data[:5]
            if len(msg.data) >= 7:
                self.bow, self.straight = msg.data[5:7]

    def path_callback(self, msg: PathPlanningResult):
        """차량 위치에서 차로 중심이 옆으로 벗어난 양 [m]. 양수면 차가 왼쪽에 있다.

        전방 한 점의 횡방향 값을 그대로 쓰면 안 된다. 곡선에서는 차가 중앙을 완벽히
        따라가도 전방 5m 지점의 차로 중심은 s^2*kappa/2 (반경 10m 면 1.2m) 만큼
        옆에 있기 때문이다. 그래서 경로를 2차식으로 맞춰 곡률 성분을 뺀 값,
        즉 차량 위치(s=0)로 되짚은 값을 쓴다.
        한 프레임 값은 외삽이라 흔들리지만, 한 바퀴 평균을 내면 실제 치우침이 보인다.
        """
        if len(msg.y_points) < 4:
            return
        ys = np.asarray(msg.y_points, dtype=float)
        xs = np.asarray(msg.x_points, dtype=float)
        s = BOTTOM_ROW_AHEAD + (BEV_BOTTOM_ROW - ys) / PX_PER_M_LON
        d = (xs - CAR_CENTER_X) / PX_PER_M_LAT
        a, b, c = np.polyfit(s, d, 2)
        self.kappa = float(2.0 * a)
        # 전방 LANE_ERR_S 지점의 횡방향 값. 곡선에서는 곡률 성분이 섞이므로
        # 평균을 낼 때는 거의 직선인 구간만 쓴다 (아래 model_states_callback).
        self.lane_err = float(a * LANE_ERR_S ** 2 + b * LANE_ERR_S + c)

    def model_states_callback(self, msg: ModelStates):
        if self.model_name not in msg.name:
            return

        pose = msg.pose[msg.name.index(self.model_name)]
        xy = (pose.position.x, pose.position.y)
        # prius 모델의 전방축이 -Y 라 모델 yaw 에서 90도를 빼야 진행방향이다
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z)) - math.pi / 2
        now = self.now_sec()

        if self.start_xy is None:
            self.start_xy = xy
            self.start_time = now
            self.last_xy = xy
            self.get_logger().info(f"출발점 등록: x={xy[0]:.2f}, y={xy[1]:.2f}")

        if now - self.last_sample_time < SAMPLE_PERIOD:
            return
        self.last_sample_time = now

        step = self.dist(xy, self.last_xy)
        self.distance += step
        self.last_xy = xy

        self.csv_writer.writerow([
            f"{now - self.start_time:.2f}", f"{xy[0]:.3f}", f"{xy[1]:.3f}", f"{yaw:.4f}",
            f"{self.distance:.2f}", f"{self.lane_err:.3f}",
            f"{self.ld:.2f}", f"{self.lat:.3f}", f"{self.ctrl_kappa:.4f}",
            f"{self.target_steer:.2f}", f"{self.steer_cont:.2f}",
            f"{self.bow:.3f}", f"{self.straight:.0f}",
            self.steering, self.speed])
        # 직선 구간에서만 통계를 낸다. 곡선에서는 차가 중앙에 있어도 전방 지점의
        # 횡방향 값이 0 이 아니라, 섞으면 의미 없는 숫자가 된다.
        if (self.speed > 0 and self.lane_err == self.lane_err
                and abs(self.kappa) < STRAIGHT_KAPPA):
            self.lane_err_sum += abs(self.lane_err)
            self.lane_err_bias += self.lane_err
            self.lane_err_max = max(self.lane_err_max, abs(self.lane_err))
            self.lane_err_n += 1
        self.csv_file.flush()

        self.check_stuck(step, now)
        self.check_lap(xy, now)

    def check_stuck(self, step, now):
        if step > STUCK_SPEED:
            self.stuck_since = None
            return
        if self.stuck_since is None:
            self.stuck_since = now
        elif now - self.stuck_since > STUCK_WARN_SEC:
            self.get_logger().warn(
                f"{now - self.stuck_since:.0f}초째 정지 상태 (steering={self.steering}, speed={self.speed})",
                throttle_duration_sec=5.0)

    def check_lap(self, xy, now):
        d = self.dist(xy, self.start_xy)

        if not self.left_start:
            if d > self.exit_radius:
                self.left_start = True
                self.get_logger().info(f"출발 확인 (출발점에서 {d:.1f}m)")
            return

        if d < self.lap_radius:
            self.lap_count += 1
            self.left_start = False
            n = self.lane_err_n or 1
            mean_err = self.lane_err_sum / n
            bias = self.lane_err_bias / n
            self.get_logger().info(
                f"랩 {self.lap_count} 완주! 시간 {now - self.start_time:.1f}s, "
                f"주행거리 {self.distance:.1f}m, "
                f"직선구간 차로중심 오차 평균 {mean_err:.2f}m / 최대 {self.lane_err_max:.2f}m / "
                f"치우침 {bias:+.2f}m (양수=차가 왼쪽)")
            self.lane_err_sum = 0.0
            self.lane_err_bias = 0.0
            self.lane_err_max = 0.0
            self.lane_err_n = 0

    @staticmethod
    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def destroy_node(self):
        if not self.csv_file.closed:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LapMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
