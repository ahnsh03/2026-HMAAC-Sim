#!/usr/bin/env python3
"""차선 인지 디버깅용. /roi_image 한 장을 받아 PNG 로 저장하고 행별 차선 픽셀 위치를 출력한다.

    python3 tools/dump_roi.py [출력경로]
"""

import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/roi.png"


class RoiDumper(Node):
    def __init__(self):
        super().__init__('roi_dumper')
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.bridge = CvBridge()
        self.done = False
        self.create_subscription(Image, 'roi_image', self.callback, qos)

    def callback(self, msg: Image):
        if self.done:
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        h, w = img.shape
        print(f"roi shape: {h} x {w}")

        for y in range(5, h, 25):
            band = img[max(0, y - 5):y + 5, :]
            cols = np.where(band.max(axis=0) > 0)[0]
            if len(cols) == 0:
                print(f"  y={y:3d}: (차선 픽셀 없음)")
                continue
            groups, start = [], cols[0]
            for prev, cur in zip(cols, cols[1:]):
                if cur - prev > 10:
                    groups.append((start, prev))
                    start = cur
            groups.append((start, cols[-1]))
            centers = [int((a + b) / 2) for a, b in groups]
            print(f"  y={y:3d}: 덩어리 {len(groups)}개, 중심 x={centers}")

        cv2.imwrite(OUT_PATH, img)
        print(f"saved: {OUT_PATH}")
        self.done = True


def main():
    rclpy.init()
    node = RoiDumper()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
