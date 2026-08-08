#!/usr/bin/env python3
"""BEV 전체(480행)를 저장하고 행별 차선 픽셀을 출력한다.

ROI 를 어디서 자를지(cutting_idx), 어느 행에서 차선 중심을 뽑을지 정하는 데 쓴다.

    python3 tools/dump_bev.py [출력경로]
"""

import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy

from interfaces_pkg.msg import DetectionArray
from camera_perception_pkg.lib import camera_perception_func_lib as CPFL

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bev.png"
SRC_MAT = [[238, 316], [402, 313], [501, 476], [155, 476]]


class BevDumper(Node):
    def __init__(self):
        super().__init__('bev_dumper')
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.done = False
        self.create_subscription(DetectionArray, 'detections', self.callback, qos)

    def callback(self, msg: DetectionArray):
        if self.done or len(msg.detections) == 0:
            return

        edge = CPFL.draw_edges(msg, cls_name='lane2', color=255)
        h, w = edge.shape[0], edge.shape[1]
        dst_mat = [[round(w * 0.3), 0], [round(w * 0.7), 0], [round(w * 0.7), h], [round(w * 0.3), h]]
        bev = cv2.convertScaleAbs(CPFL.bird_convert(edge, srcmat=SRC_MAT, dstmat=dst_mat))

        print(f"bev shape: {bev.shape[0]} x {bev.shape[1]}  (행 번호가 작을수록 먼 거리)")
        for y in range(0, bev.shape[0], 20):
            band = bev[y:y + 10, :]
            cols = np.where(band.max(axis=0) > 0)[0]
            if len(cols) == 0:
                print(f"  row={y:3d}: -")
                continue
            groups, start = [], cols[0]
            for prev, cur in zip(cols, cols[1:]):
                if cur - prev > 10:
                    groups.append((start, prev))
                    start = cur
            groups.append((start, cols[-1]))
            centers = [int((a + b) / 2) for a, b in groups]
            print(f"  row={y:3d}: 덩어리 {len(groups)}개, 중심 x={centers}")

        cv2.imwrite(OUT_PATH, bev)
        print(f"saved: {OUT_PATH}")
        self.done = True


def main():
    rclpy.init()
    node = BevDumper()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
