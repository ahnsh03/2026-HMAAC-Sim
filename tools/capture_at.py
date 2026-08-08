#!/usr/bin/env python3
"""특정 구간을 지날 때만 카메라 원본과 차선 BEV 를 저장한다.

반복해서 같은 코너에서 이탈할 때, 그 지점에서 인지가 무엇을 보는지 확인하는 용도.

    python3 tools/capture_at.py XMIN XMAX YMIN YMAX [출력디렉터리]
"""

import os
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy

from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import Image
from interfaces_pkg.msg import DetectionArray
from camera_perception_pkg.lib import camera_perception_func_lib as CPFL

XMIN, XMAX, YMIN, YMAX = (float(v) for v in sys.argv[1:5])
OUT_DIR = sys.argv[5] if len(sys.argv) > 5 else "/tmp/capture"
SRC_MAT = [[238, 316], [402, 313], [501, 476], [155, 476]]


class RegionCapture(Node):
    def __init__(self):
        super().__init__('region_capture')
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        os.makedirs(OUT_DIR, exist_ok=True)
        self.bridge = CvBridge()
        self.xy = None
        self.frame = None
        self.count = 0

        self.create_subscription(ModelStates, '/gazebo/model_states', self.pose_cb, qos)
        self.create_subscription(Image, '/camera/image_raw', self.image_cb, qos)
        self.create_subscription(DetectionArray, 'detections', self.detection_cb, qos)
        self.get_logger().info(f"구간 x[{XMIN},{XMAX}] y[{YMIN},{YMAX}] 감시 중 -> {OUT_DIR}")

    def pose_cb(self, msg: ModelStates):
        if 'ego_vehicle' not in msg.name:
            return
        p = msg.pose[msg.name.index('ego_vehicle')].position
        self.xy = (p.x, p.y)

    def image_cb(self, msg: Image):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def in_region(self):
        return (self.xy is not None
                and XMIN <= self.xy[0] <= XMAX and YMIN <= self.xy[1] <= YMAX)

    def detection_cb(self, msg: DetectionArray):
        if not self.in_region() or self.frame is None:
            return

        tag = f"{self.count:03d}_x{self.xy[0]:+.1f}_y{self.xy[1]:+.1f}"
        cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_cam.png"), self.frame)

        if len(msg.detections) > 0:
            edge = CPFL.draw_edges(msg, cls_name='lane2', color=255)
            h, w = edge.shape[0], edge.shape[1]
            dst = [[round(w * 0.3), 0], [round(w * 0.7), 0], [round(w * 0.7), h], [round(w * 0.3), h]]
            bev = cv2.convertScaleAbs(CPFL.bird_convert(edge, srcmat=SRC_MAT, dstmat=dst))
            cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_bev.png"), bev)
            rows = []
            for y in (20, 100, 180, 260, 340):
                band = bev[y:y + 10, :]
                cols = np.where(band.max(axis=0) > 0)[0]
                rows.append(f"y{y}:{len(np.split(cols, np.where(np.diff(cols) > 10)[0] + 1)) if len(cols) else 0}")
            self.get_logger().info(f"{tag} 차선덩어리 {' '.join(rows)}")
        else:
            self.get_logger().info(f"{tag} detections 없음")

        self.count += 1


def main():
    rclpy.init()
    node = RegionCapture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
