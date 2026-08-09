import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String, Bool
from std_msgs.msg import Float32MultiArray
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand

#---------------Variable Setting---------------
SUB_DETECTION_TOPIC_NAME = "detections"
SUB_PATH_TOPIC_NAME = "path_planning_result"
SUB_TRAFFIC_LIGHT_TOPIC_NAME = "yolov8_traffic_light_info"
SUB_LIDAR_OBSTACLE_TOPIC_NAME = "lidar_obstacle_info"
PUB_TOPIC_NAME = "topic_control_signal"

#----------------------------------------------

# 모션 플랜 발행 주기 (초) - 소수점 필요 (int형은 반영되지 않음)
TIMER = 0.1

#---------------BEV <-> 실제 거리 환산---------------
# 경로는 lane_info_extractor 가 만든 640x480 BEV 픽셀 좌표다.
# y=0 이 가장 먼 곳, y=479 가 차량 앞쪽. x=320 이 차량 중심선.
#
# 아래 값들은 model.sdf 의 카메라 파라미터(높이 2.05m, pitch 0.1rad, hfov 1.0856,
# 640x480)와 SRC_MAT 사다리꼴을 지면에 역투영해서 얻은 것이다.
CAR_CENTER_X = 320.0     # 차량 중심선의 BEV x
BEV_BOTTOM_ROW = 479.0
PX_PER_M_LAT = 103.6     # 가로 방향 [BEV 픽셀 / m]
PX_PER_M_LON = 98.0      # 세로 방향 [BEV 픽셀 / m]
BOTTOM_ROW_AHEAD = 4.74  # BEV 맨 아랫줄이 뒷차축에서 떨어진 거리 [m]

WHEELBASE = 2.86         # 축간거리 [m]
MAX_STEER_RAD = 0.6458   # 하드웨어 최대 조향각 (steering=7 에 해당)
STEER_LIMIT = 7
RAD_PER_STEP = MAX_STEER_RAD / STEER_LIMIT

#---------------차로 추종 제어 파라미터---------------
# 부호 규약: 경로가 오른쪽이면 조향 양수 -> simulation_sender 가 STEERING=-1 을
# 곱해 우조향이 된다.

# 전방 주시거리 Ld = K_LD * v + LD_BASE [m, 뒷차축 기준]
# BEV 가 뒷차축 앞 4.74m 부터 보이므로 그보다 짧게는 잡을 수 없다.
# 길게 잡을수록 부드럽지만 코너를 덜 꺾는다.
K_LD = 0.70
LD_BASE = 3.75
LD_MIN = 4.85            # BEV 맨 아랫줄(4.74m)까지 쓸 수 있도록 낮춘다
LD_MAX = 6.40

# 조향은 두 몫으로 나눈다.
#   앞먹임 = atan(축간거리 * 곡률).  곡선을 도는 데 필요한 몫. 곡률을 시간적으로
#            평활하므로, 일정한 곡선에서는 조향도 일정하게 유지된다.
#   되먹임 = 가까운 지점의 '진짜 횡오차'.  그 지점의 차로 중심에서 곡률 때문에
#            생기는 몫(kappa*s^2/2)을 빼면 남는 것이 실제 이탈량이다.
#
# 행별 정확도를 실측해 보니 가까운 행일수록 정확했다 (차를 진짜 차로 중앙에
# 놓고 측정: y470 -0.10m, y390 -0.16m, y260 -0.15m, y100 -0.33m, y20 -0.54m).
# 그래서 되먹임은 가까운 행에서 읽는다.
S_ERR = 5.00             # 횡오차를 읽는 전방거리 [m]. BEV 최근접(4.74m) 바로 위
K_E = 1.10               # 횡오차 이득. Stanley 형태 atan(K_E * e / v)
V_REF_MIN = 1.2          # 위 식에서 속도 하한 [m/s]
E_LIMIT = 1.2            # 횡오차 제한 [m]. 인지가 튀어도 급조향하지 않도록

# 곡률 추정. 경로가 흔들리면 곡률은 더 크게 흔들리므로 세게 눌러준다.
KAPPA_LPF = 0.30         # 프레임 간 저역통과 계수 (29Hz 에서 시정수 약 0.14초).
                         # 너무 느리면 조향은 미리 꺾이는데 바깥보정이 늦게 들어와
                         # 코너 진입에서 안쪽으로 파고든다.
KAPPA_LIMIT = 0.13       # 곡률 상한 [1/m] (반경 7.7m).
                         # GT 경로에서 잰 이 트랙의 최대 곡률이 약 0.10 이다.
                         # 예전 값 0.30 은 반경 3.3m 로, 노이즈를 그대로 통과시켰다.

# 순수추종의 코너 파고듦 보정 계수. 1.0 이면 기하학적 보정량(kappa*Ld^2/8) 그대로.
# 0 = 보정 없음(순수추종 그대로). 원인을 먼저 가리기 위해 기본은 0 에서 시작한다.
K_CUT_COMP = 0.0

# 오차 되먹임의 점진 이득 문턱 [m]. 이 크기 오차에서 이득의 절반이 걸린다.
# 0 이면 점진 이득을 쓰지 않고 그대로 반응한다(사각지대 없음).
E_SOFT = 0.0

# 조향 배수. 1.0 이 기하학적으로 필요한 값 그대로다.
STEER_GAIN = 1.00

STEER_RATE_LIMIT = 2.2   # 한 tick(TIMER) 당 조향 변화량 상한 [step].
                         # 20 step/s. 더 낮추면 S자 연속코너에서 못 따라간다.
QUANT_HYST = 0.0         # 정수 조향을 바꾸는 문턱 [step]. 0 이면 그냥 반올림.
                         # 크게 잡으면 사각지대가 되어 한쪽으로 밀린 채 안 돌아온다.

#---------------속도---------------
# 속도 명령은 0~255 이고 v[m/s] = speed / 51 이다 (MAX_SPEED=5 기준).
CMD_PER_MPS = 51.0
V_MAX = 2.40             # 직선 최고 속도 [m/s]
V_MIN = 1.40             # 코너 최저 속도 [m/s]
ACCEL_LIMIT = 1.6        # 가속 상한 [m/s^2]
DECEL_LIMIT = 2.2        # 감속 상한 [m/s^2]. 코너 앞에서 급하게 줄이지 않도록
SPEED_LPF = 0.5          # 속도 1차 저역통과 계수 (1 이면 평활 없음)

# 감속 기준 1: 횡가속도. v = sqrt(a * R) 이므로 반경 R 인 코너의 상한이 정해진다.
# GT 차로 중앙선에서 곡률을 재보니 (tools/speed_profile.py) 이 트랙의 최소 반경은
# 4.5m, 90분위는 9.3m 다. 횡가속 3.0 기준이면 R=4.5m 에서도 3.7m/s 까지 되므로
# 2.4m/s 로 달리는 한 이 기준으로는 감속할 일이 없다.
# 실제 제약은 횡가속도가 아니라 조향 각도(아래)와 인지 지연이다.
A_LAT_MAX = 3.5          # 허용 횡가속도 [m/s^2]

# 감속 기준 2: 조향 여유. 조향이 포화에 가까우면 더 꺾을 수 없으므로 속도를 줄인다.
# STEER_FREE 이하에서는 아예 감속하지 않는다 (완만한 곡선에서 괜히 느려지지 않게).
# 감속 기준 2: 조향 여유. 최소 반경 4.5m 코너는 조향 atan(2.86/4.5)=0.57rad,
# 즉 6.2칸이 필요해 최대(7칸)에 거의 닿는다. 여기서는 여유가 없으므로 줄인다.
STEER_FREE = 4.5         # 이 조향까지는 전속 [step]
STEER_FULL = 7.0         # 이 조향이면 V_MIN [step]

#---------------인지 공백 대응---------------
# 인지 파이프라인 주기보다 넉넉히 잡는다.
LANE_TIMEOUT = 1.0       # 이 시간 이상 새 경로가 없으면 인지 공백 [s]
COAST_SPEED = 1.10       # 인지 공백 구간 통과 속도 [m/s]
COAST_MAX_SEC = 6.0      # 이 시간까지만 타성 주행을 허용한다 [s].
                         # 인지가 잠깐 끊기는 구간이 몇 미터쯤 되는데, 여기서 멈춰
                         # 서면 시야가 그대로라 영영 회복하지 못한다. 저속으로
                         # 통과할 만큼은 준다 (1.1m/s x 6s = 6.6m).
SPEED_DECAY_TIME = 1.0   # 그 뒤 속도를 0까지 줄이는 데 걸리는 시간 [s]
#----------------------------------------------


def row_to_ahead(row):
    """BEV 행 -> 뒷차축 기준 전방 거리 [m]."""
    return BOTTOM_ROW_AHEAD + (BEV_BOTTOM_ROW - row) / PX_PER_M_LON


def ahead_to_row(ahead_m):
    """뒷차축 기준 전방 거리 [m] -> BEV 행."""
    return BEV_BOTTOM_ROW - (ahead_m - BOTTOM_ROW_AHEAD) * PX_PER_M_LON


class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        # 토픽 이름 설정
        self.sub_detection_topic = self.declare_parameter('sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter('sub_lane_topic', SUB_PATH_TOPIC_NAME).value
        self.sub_traffic_light_topic = self.declare_parameter('sub_traffic_light_topic', SUB_TRAFFIC_LIGHT_TOPIC_NAME).value
        self.sub_lidar_obstacle_topic = self.declare_parameter('sub_lidar_obstacle_topic', SUB_LIDAR_OBSTACLE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value

        self.timer_period = self.declare_parameter('timer', TIMER).value

        # 주행 제어 파라미터 (재빌드 없이 조정할 수 있도록 노출)
        self.drive_enable = self.declare_parameter('drive_enable', True).value
        self.v_max = self.declare_parameter('v_max', V_MAX).value
        self.v_min = self.declare_parameter('v_min', V_MIN).value
        self.k_ld = self.declare_parameter('k_ld', K_LD).value
        self.ld_base = self.declare_parameter('ld_base', LD_BASE).value
        self.k_cut_comp = self.declare_parameter('k_cut_comp', K_CUT_COMP).value
        self.e_soft = self.declare_parameter('e_soft', E_SOFT).value
        self.steer_gain = self.declare_parameter('steer_gain', STEER_GAIN).value
        self.kappa_lpf = self.declare_parameter('kappa_lpf', KAPPA_LPF).value
        self.steer_rate_limit = self.declare_parameter('steer_rate_limit', STEER_RATE_LIMIT).value
        self.lane_timeout = self.declare_parameter('lane_timeout', LANE_TIMEOUT).value

        # QoS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 변수 초기화
        self.detection_data = None
        self.path_x = None       # BEV x 배열 (y 오름차순)
        self.path_y = None
        self.traffic_light_data = None
        self.lidar_data = None

        self.path_stamp = None   # 마지막으로 경로를 받은 시각 [s]
        self.steer = 0.0         # 연속값 조향 [step]. 발행 직전에만 정수로 만든다
        self.speed_mps = 0.0     # 연속값 속도 [m/s]
        self.kappa = 0.0         # 저역통과한 경로 곡률 [1/m]
        self.debug = None        # (주시거리, 횡방향, 곡률, 목표조향)
        self.lat_err = 0.0       # 곡률 성분을 뺀 진짜 횡오차 [m]

        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0

        # 서브스크라이버 설정
        self.detection_sub = self.create_subscription(DetectionArray, self.sub_detection_topic, self.detection_callback, self.qos_profile)
        self.path_sub = self.create_subscription(PathPlanningResult, self.sub_path_topic, self.path_callback, self.qos_profile)
        self.traffic_light_sub = self.create_subscription(String, self.sub_traffic_light_topic, self.traffic_light_callback, self.qos_profile)
        self.lidar_sub = self.create_subscription(Bool, self.sub_lidar_obstacle_topic, self.lidar_callback, self.qos_profile)

        # 퍼블리셔 설정
        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)
        # 진단용: [주시거리 m, 그 지점 횡방향 m, 곡률 1/m, 목표조향 step, 실제조향 step]
        self.debug_pub = self.create_publisher(Float32MultiArray, 'control_debug', self.qos_profile)

        # 타이머 설정
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info(
            f"순수추종 제어: Ld={self.k_ld:.2f}v+{self.ld_base:.2f} [{LD_MIN}, {LD_MAX}]m, "
            f"코너보정={self.k_cut_comp:.2f}, e_soft={self.e_soft:.2f}, "
            f"kappa_lpf={self.kappa_lpf:.2f}, v={self.v_min:.2f}~{self.v_max:.2f} m/s")

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg

    def path_callback(self, msg: PathPlanningResult):
        if len(msg.y_points) < 2:
            return
        # np.interp 는 x 가 오름차순이어야 한다. path_planner 가 y 오름차순으로 준다.
        self.path_y = np.asarray(msg.y_points, dtype=float)
        self.path_x = np.asarray(msg.x_points, dtype=float)
        self.path_stamp = self.now_sec()

    def traffic_light_callback(self, msg: String):
        self.traffic_light_data = msg

    def lidar_callback(self, msg: Bool):
        self.lidar_data = msg

    def path_point(self, ahead_m):
        """전방 ahead_m 근처의 경로점. (실제 전방거리 [m], 횡방향 [m]) 오른쪽이 양수.

        경로가 없는 구간까지 외삽하면 값이 크게 튀므로 측정 구간 안으로 자르고,
        잘린 만큼 전방거리도 다시 계산해 조향각 계산과 어긋나지 않게 한다.
        """
        row = float(np.clip(ahead_to_row(ahead_m), self.path_y.min(), self.path_y.max()))
        x = float(np.interp(row, self.path_y, self.path_x))
        return row_to_ahead(row), (x - CAR_CENTER_X) / PX_PER_M_LAT

    def update_curvature(self):
        """경로의 곡률 [1/m]. 오른쪽으로 휘면 양수.

        측정 구간(4.7~9.6m) 안에서만 2차식으로 맞춘다. 밖으로 외삽하면 값이 크게
        튀어 조향이 발산한다. 곡률은 길의 성질이라 천천히 변하므로, 프레임 간
        저역통과로 노이즈를 충분히 눌러도 늦지 않는다.
        """
        s = row_to_ahead(self.path_y)
        d = (self.path_x - CAR_CENTER_X) / PX_PER_M_LAT
        raw = 2.0 * float(np.polyfit(s, d, 2)[0])
        raw = float(np.clip(raw, -KAPPA_LIMIT, KAPPA_LIMIT))
        self.kappa += self.kappa_lpf * (raw - self.kappa)
        return self.kappa

    def soft_gain(self, err):
        """작은 오차에는 약하게, 커질수록 제 이득으로 반응한다 [m] -> [m].

        차로 중앙 근처에서는 인지 잡음이 그대로 조향으로 넘어와 꿈틀거린다.
        그렇다고 사각지대를 두면 한쪽으로 밀린 채 안 돌아온다. 그래서 이득을
        |e|/(|e|+E_SOFT) 로 부드럽게 키운다. 0.2m 면 약 1/3, 1.0m 면 약 3/4 이 걸린다.
        """
        if self.e_soft <= 0.0:
            return err
        return err * abs(err) / (abs(err) + self.e_soft)

    def follow_lane(self):
        """곡률 앞먹임 + 가까운 지점의 횡오차 되먹임.

        코너에서 조향이 들쭉날쭉하지 않고 곡률에 맞는 값으로 유지되도록,
        조향의 큰 몫은 시간적으로 평활한 곡률에서 만든다.
        """
        kappa = self.update_curvature()

        # 속도가 붙을수록 멀리 본다. 흔들림을 줄이고 코너 진입을 부드럽게 한다.
        lookahead = float(np.clip(self.k_ld * self.speed_mps + self.ld_base, LD_MIN, LD_MAX))
        dist, lat = self.path_point(lookahead)

        # 곡선을 정확히 따라가고 있어도 전방 주시점의 차로 중심은 옆에 있다.
        # 그 몫과 진짜 오차를 나눠, 되먹임에만 점진 이득을 건다.
        lat_curve = kappa * dist * dist / 2.0
        lat_err = lat - lat_curve
        lat_aim = lat_curve + self.soft_gain(lat_err) + self.k_cut_comp * kappa * dist * dist / 8.0

        delta = math.atan2(2.0 * WHEELBASE * lat_aim, dist * dist + lat_aim * lat_aim)
        target_steer = float(np.clip(self.steer_gain * delta / RAD_PER_STEP,
                                     -STEER_LIMIT, STEER_LIMIT))
        e = lat_err
        self.debug = (dist, lat, kappa, target_steer)
        self.lat_err = e

        # 급격한 조향 변화는 차체를 흔들고 타이어를 미끄러뜨린다
        step = float(np.clip(target_steer - self.steer, -self.steer_rate_limit, self.steer_rate_limit))
        self.steer += step

        # 감속은 길의 곡률만 보고 정한다. 조향 명령을 그대로 쓰면 오차가 흔들릴 때마다
        # 속도가 같이 출렁인다.
        demand = abs(math.atan(WHEELBASE * kappa) / RAD_PER_STEP)
        return self.speed_limit(demand)

    def speed_limit(self, demand_steps):
        """전방 상황에서 낼 수 있는 속도 [m/s].

        두 가지로만 줄인다. 횡가속도 한계와 조향 여유.
        완만한 곡선에서 괜히 느려지지 않도록 STEER_FREE 이하에서는 감속하지 않는다.
        """
        # 횡가속도: 조향각 delta 인 정상선회 반경은 R = 축간거리 / tan(delta)
        delta = abs(demand_steps) * RAD_PER_STEP
        if delta > 1e-3:
            radius = WHEELBASE / math.tan(delta)
            v_lat = math.sqrt(A_LAT_MAX * radius)
        else:
            v_lat = self.v_max

        # 조향 여유
        if demand_steps <= STEER_FREE:
            v_steer = self.v_max
        else:
            ratio = min((demand_steps - STEER_FREE) / (STEER_FULL - STEER_FREE), 1.0)
            v_steer = self.v_max - (self.v_max - self.v_min) * ratio

        return float(np.clip(min(self.v_max, v_lat, v_steer), self.v_min, self.v_max))

    def timer_callback(self):
        try:
            self.update_commands()
        except Exception as exc:  # 한 프레임이 이상해도 노드가 죽으면 안 된다
            self.get_logger().error(f"제어 계산 실패, 정지: {exc}", throttle_duration_sec=2.0)
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0

        self.get_logger().info(f"steering: {self.steering_command}, "
                               f"left_speed: {self.left_speed_command}, "
                               f"right_speed: {self.right_speed_command}",
                               throttle_duration_sec=0.5)

        # 모션 명령 메시지 생성 및 퍼블리시
        motion_command_msg = MotionCommand()
        motion_command_msg.steering = int(self.steering_command)
        motion_command_msg.left_speed = int(self.left_speed_command)
        motion_command_msg.right_speed = int(self.right_speed_command)
        self.publisher.publish(motion_command_msg)

        if self.debug is not None:
            dbg = Float32MultiArray()
            dbg.data = [float(v) for v in self.debug] + [float(self.steer)]
            self.debug_pub.publish(dbg)

    def update_commands(self):
        if self.lidar_data is not None and self.lidar_data.data is True:
            # 라이다가 장애물을 감지한 경우
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0
            return

        if (self.traffic_light_data is not None and self.traffic_light_data.data == 'Red'
                and self.detection_data is not None):
            # 빨간색 신호등을 감지한 경우
            for detection in self.detection_data.detections:
                if detection.class_name == 'traffic_light':
                    y_max = int(detection.bbox.center.position.y + detection.bbox.size.y / 2)
                    if y_max < 150:
                        # 신호등 위치에 따른 정지명령 결정
                        self.steering_command = 0
                        self.left_speed_command = 0
                        self.right_speed_command = 0
                        return

        if self.path_x is None:
            # 첫 경로를 받기 전에는 출발하지 않는다
            self.steer = 0.0
            speed = 0.0
        else:
            speed = self.follow_lane()

            # 경로가 끊기면 마지막 조향을 유지한 채 저속 통과 -> 그래도 안 돌아오면 정지
            age = self.now_sec() - self.path_stamp
            if age > COAST_MAX_SEC:
                decay = max(0.0, 1.0 - (age - COAST_MAX_SEC) / SPEED_DECAY_TIME)
                speed = COAST_SPEED * decay
                if decay <= 0.0:
                    self.get_logger().warn(f"경로 끊김 {age:.1f}s - 정지", throttle_duration_sec=2.0)
            elif age > self.lane_timeout:
                speed = min(speed, COAST_SPEED)
                self.get_logger().warn(f"인지 공백 {age:.1f}s - 저속 통과", throttle_duration_sec=2.0)

        if not self.drive_enable:
            speed = 0.0

        # 속도를 부드럽게 따라가게 한다. 가감속을 모두 제한한 뒤 1차 저역통과를 건다.
        # 다만 정지 명령(0)은 안전을 위해 즉시 반영한다.
        if speed <= 0.0:
            self.speed_mps = 0.0
        else:
            limit = (ACCEL_LIMIT if speed > self.speed_mps else DECEL_LIMIT) * self.timer_period
            stepped = self.speed_mps + float(np.clip(speed - self.speed_mps, -limit, limit))
            self.speed_mps += SPEED_LPF * (stepped - self.speed_mps)

        self.steering_command = self.quantize_steer()
        self.left_speed_command = int(round(self.speed_mps * CMD_PER_MPS))
        self.right_speed_command = self.left_speed_command

    def quantize_steer(self):
        """연속 조향을 정수 -7..7 로 만든다. 히스테리시스로 떨림을 막는다.

        한 칸이 5.3도로 성기기 때문에, 연속값이 두 칸 사이에 있으면 매 tick 마다
        칸을 오가며 조향이 눈에 띄게 튄다. 그래서 지금 칸에서 QUANT_HYST 이상
        벗어날 때만 칸을 바꾼다.
        이렇게 하면 ±QUANT_HYST 만큼 사각지대가 생기지만, 횡오차 되먹임(K_E)이
        이득을 채워주므로 직선에서 한쪽으로 밀리지는 않는다.
        """
        if abs(self.steer - self.steering_command) <= QUANT_HYST:
            return int(self.steering_command)
        return int(np.clip(round(self.steer), -STEER_LIMIT, STEER_LIMIT))


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
