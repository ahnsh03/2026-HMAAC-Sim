import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces_pkg.msg import TargetPoint, LaneInfo, DetectionArray, BoundingBox2D, Detection

from .lane_estimator import LaneCenterEstimator, BEV_W, BEV_H, LANE_WIDTH

#---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_TOPIC_NAME = "detections"

# Publish할 토픽 이름
PUB_TOPIC_NAME = "yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "roi_image"  # 추가: ROI 이미지 퍼블리시 토픽

# 화면에 이미지를 처리하는 과정을 띄울것인지 여부: True, 또는 False 중 택1하여 입력
# CPU 여유가 없으면 창이 인지 주기를 크게 떨어뜨리므로 기본값은 False.
# 켜려면 launch 인자 show_lane_debug:=true 또는 파라미터 show_image 로 준다.
SHOW_IMAGE = False

# BEV 변환 기준점. src_mat 은 원본 이미지에서 차선이 놓인 사다리꼴.
SRC_MAT = [[238, 316], [402, 313], [501, 476], [155, 476]]

# 차로 중심을 뽑을 BEV 행. y 가 클수록 차량에 가깝다.
# 세로 480px 이 약 4.9m 에 해당하므로 y=470 이 약 4.8m, y=20 이 약 8.3m 앞이다.
#
# 아래쪽(가까운) 행을 반드시 포함해야 한다. 트랙 좌측의 급코너에서는 길이 카메라
# 화각을 벗어나 먼 행에는 아무것도 안 잡히고, 차선 정보가 화면 맨 아래에만 남는다.
# 예전에는 420 까지만 봐서 그 구간에서 인지가 통째로 비었다.
TARGET_POINT_YS = (20, 100, 180, 260, 330, 390, 435, 470)

# 차로 중심을 오른쪽으로 이 만큼 민다 [BEV 픽셀, 약 103px = 1m]. 양수 = 오른쪽.
# 좌우 쏠림이 남으면 이 값만 조정하면 된다.
# 구조적 원인(앞먹임 시점, 곡률 추정 구간)을 모두 손본 뒤에도 남는 잔차만
# 여기서 상쇄한다. 트랙 텍스처 기준으로 재보니 코너에서 평균 0.24m 안쪽이었다.
CENTER_OFFSET = 25.0

# 직전 프레임 대비 허용하는 중심 이동량 [px].
# 인지가 약 29Hz 라 한 프레임(35ms)에 차로 중심이 움직일 수 있는 양은 크지 않다.
# 2.5m/s 에서 급코너라도 40px 남짓이므로, 그보다 크게 튀면 오검출로 본다.
MAX_FRAME_JUMP = 80
PREV_VALID_SEC = 0.5       # 이 시간이 지난 직전값은 비교 기준으로 쓰지 않는다
MAX_CONSECUTIVE_MISS = 3   # 연속 실패가 이만큼이면 기준을 버리고 다시 잡는다

# 프레임 간 평활 계수. 1.0 이면 평활 없음.
# 29Hz 에서 0.4 면 시정수가 약 85ms 라, 튐은 줄이면서 지연은 거의 안 생긴다.
SMOOTH_ALPHA = 0.4
# 좌우 경계를 둘 다 본 게 아니라 한쪽만 보고 차로 폭을 더해 만든 값은 덜 믿는다.
# S자 구간처럼 바깥 실선이 화각을 벗어나 점선만 남으면 추정이 흔들리는데,
# 이때 더 세게 눌러야 조향이 따라 흔들리지 않는다.
SMOOTH_ALPHA_WEAK = 0.20

# 어떤 행의 추정이 잠깐 비면 직전 값을 이만큼 유지한다 [s].
# 인지가 어려운 구간에서 유효한 행이 2개로 줄면 경로가 직선이 되어 곡률을 잃는데,
# 그게 오히려 차로 이탈을 키운다. 29Hz 이므로 0.3초는 약 9프레임이다.
HOLD_SEC = 0.0

# 한 프레임 안에서 행끼리 얼마나 어긋나도 되는지 [px].
# 차로 중심은 거리에 대해 매끄러운 곡선이라, 한 행만 크게 튀면 그 행이 틀린 것이다.
# 12지점에서 재보니 행별 편향은 ±7px 로 없는데 표준편차가 70~100px 였다.
# 즉 문제는 치우침이 아니라 일관성이라, 프레임 안에서 튀는 행을 걸러낸다.
INFRAME_MAX_RESIDUAL = 60
INFRAME_MIN_KEEP = 3       # 이보다 적게 남으면 걸러내지 않는다

# 이 개수 미만이면 경로를 만들 수 없으므로 아예 발행하지 않는다
# (motion_planner 가 경로 끊김으로 판단해 감속한다)
MIN_VALID_POINTS = 3
#----------------------------------------------


class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value
        self.lane_width = self.declare_parameter('lane_width', LANE_WIDTH).value
        self.center_offset = self.declare_parameter('center_offset', CENTER_OFFSET).value

        self.estimator = LaneCenterEstimator(SRC_MAT, lane_width=self.lane_width,
                                             center_offset=self.center_offset)

        self.cv_bridge = CvBridge()

        # QoS settings
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.subscriber = self.create_subscription(DetectionArray, self.sub_topic, self.yolov8_detections_callback, self.qos_profile)
        self.publisher = self.create_publisher(LaneInfo, self.pub_topic, self.qos_profile)

        # ROI 이미지 퍼블리셔 추가
        self.roi_image_publisher = self.create_publisher(Image, ROI_IMAGE_TOPIC_NAME, self.qos_profile)

        # 직전 프레임에서 채택한 차로 중심 (행 -> x). 튀는 값을 거르는 기준.
        self.prev_centers = {}
        self.prev_stamp = 0.0
        self.miss_count = 0
        self.sources = {}

        self.get_logger().info(
            f"차로 중심 추정: center_line 기준, lane_width={self.lane_width:.0f}px, "
            f"center_offset={self.center_offset:+.0f}px")

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if len(detection_msg.detections) == 0:
            return

        try:
            centers, debug = self.estimator.estimate(
                detection_msg.detections, TARGET_POINT_YS, self.tracked_ref())
            self.sources = debug['sources']
        except Exception as exc:  # 한 프레임이 이상해도 노드가 죽으면 안 된다
            self.get_logger().error(f"차로 추정 실패: {exc}", throttle_duration_sec=2.0)
            return

        self.publish_debug_image(detection_msg, centers)

        centers = self.reject_frame_jumps(centers)
        centers = self.smooth(centers)
        centers = self.hold_missing(centers)

        if len(centers) < MIN_VALID_POINTS:
            self.miss_count += 1
            if self.miss_count >= MAX_CONSECUTIVE_MISS:
                self.prev_centers = {}
            self.get_logger().warn(
                f"유효한 차로 중심 {len(centers)}개 - 경로 발행 생략 "
                f"(중앙선 조각 {debug['n_center_line']}개, 곡선맞춤 {debug['fit']})",
                throttle_duration_sec=2.0)
            return

        self.miss_count = 0
        self.prev_centers = dict(centers)
        self.prev_stamp = self.now_sec()

        target_points = []
        for y in sorted(centers):
            target_point = TargetPoint()
            target_point.target_x = int(round(centers[y]))
            target_point.target_y = int(y)
            target_points.append(target_point)

        lane = LaneInfo()
        lane.slope = self.center_slope_deg(centers)
        lane.target_points = target_points

        self.publisher.publish(lane)

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def tracked_ref(self):
        """직전 프레임에서 추적하던 차로의 위치 (가장 가까운 행의 중심).

        이걸 다음 프레임 탐색의 출발점으로 주면 같은 차로를 계속 따라간다.
        오래된 값은 쓰지 않는다.
        """
        if not self.prev_centers or self.now_sec() - self.prev_stamp > PREV_VALID_SEC:
            return None
        return self.prev_centers[max(self.prev_centers)]

    @staticmethod
    def center_slope_deg(centers):
        """차로 중심선이 기울어진 각도 [deg]. 양수면 전방이 오른쪽으로 휜다."""
        if len(centers) < 2:
            return 0.0
        ys = sorted(centers)
        far, near = ys[0], ys[-1]
        if near == far:
            return 0.0
        # y 는 아래로 증가하므로 (near - far) 가 전방 거리에 해당한다
        return float(np.degrees(np.arctan2(centers[far] - centers[near], near - far)))

    @staticmethod
    def reject_inframe_outliers(centers):
        """한 프레임 안에서 다른 행들과 어긋나는 행을 버린다.

        차로 중심은 거리에 대해 매끄럽다. 행들에 2차식을 맞춰보고 크게 벗어난
        행은 옆 차로나 갈라지는 도로를 물었다고 본다.
        """
        if len(centers) < 4:
            return centers
        ys = np.array(sorted(centers), dtype=float)
        xs = np.array([centers[y] for y in ys], dtype=float)
        deg = 2 if len(ys) >= 5 else 1
        coeff = np.polyfit(ys, xs, deg)
        residual = np.abs(np.polyval(coeff, ys) - xs)
        # 잔차가 큰 것부터 하나씩 떨궈본다
        keep = residual <= INFRAME_MAX_RESIDUAL
        if keep.sum() < INFRAME_MIN_KEEP:
            return centers
        return {int(y): centers[y] for y, k in zip(ys, keep) if k}

    def reject_frame_jumps(self, centers):
        """직전 프레임 대비 물리적으로 불가능한 이동은 오검출로 보고 버린다."""
        if not self.prev_centers or self.now_sec() - self.prev_stamp > PREV_VALID_SEC:
            return centers
        return {y: x for y, x in centers.items()
                if y not in self.prev_centers or abs(x - self.prev_centers[y]) <= MAX_FRAME_JUMP}

    def hold_missing(self, centers):
        """방금 추정하지 못한 행을 직전 값으로 잠깐 메운다.

        곡률을 살리려면 경로에 점이 3개는 있어야 한다. 한두 행이 잠깐 비었다고
        경로를 통째로 버리면 차가 인지 공백 구간에서 멈춰 버리고, 멈추면 시야가
        그대로라 회복하지도 못한다.
        """
        if (HOLD_SEC <= 0.0 or not self.prev_centers
                or self.now_sec() - self.prev_stamp > HOLD_SEC):
            return centers
        out = dict(self.prev_centers)
        out.update(centers)
        return out

    def smooth(self, centers):
        """프레임 간 평활. 한 프레임짜리 흔들림이 그대로 조향으로 가지 않게 한다.

        마스크 경계는 프레임마다 몇십 픽셀씩 흔들리는데, 그대로 쓰면 조향이 튄다.
        직전 값이 오래됐거나 없으면 평활하지 않고 새 값을 그대로 받는다.
        """
        if not self.prev_centers or self.now_sec() - self.prev_stamp > PREV_VALID_SEC:
            return centers
        out = {}
        for y, x in centers.items():
            prev = self.prev_centers.get(y)
            if prev is None:
                out[y] = x
                continue
            # 근거에 '+' 가 있으면 좌우 경계를 모두 본 것이라 더 믿을 만하다
            two_sided = '+' in str(self.sources.get(y, ''))
            alpha = SMOOTH_ALPHA if two_sided else SMOOTH_ALPHA_WEAK
            out[y] = prev + alpha * (x - prev)
        return out

    def publish_debug_image(self, detection_msg, centers):
        """차로 마스크와 추정한 중심을 한 장에 그려 roi_image 로 내보낸다."""
        try:
            img = self.estimator.debug_image(detection_msg.detections)
            for y, x in centers.items():
                cv2.circle(img, (int(round(x)), int(y)), 5, 255, -1)
            cv2.line(img, (BEV_W // 2, 0), (BEV_W // 2, BEV_H), 128, 1)

            if self.show_image:
                cv2.imshow('lane_bev', img)
                cv2.waitKey(1)

            self.roi_image_publisher.publish(self.cv_bridge.cv2_to_imgmsg(img, encoding="mono8"))
        except Exception as exc:
            self.get_logger().error(f"디버그 이미지 실패: {exc}", throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
