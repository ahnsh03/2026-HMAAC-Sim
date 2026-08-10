#!/usr/bin/env python3
"""정지 상태에서 인지 잡음을 분해해 잰다.

차를 세워두면 장면이 고정되므로, 프레임마다 값이 달라지는 것은 전부 인지 잡음이다.
어디서 얼마나 생기는지 층별로 본다.

    바깥 실선 위치 (lane2 의 오른쪽 끝, 행별)
    중앙선 위치   (center_line 곡선 맞춤 결과, 행별)
    차로 중심     (두 경계를 합친 결과, 행별)

행마다 따로 재면 각 행이 독립적으로 흔들리는데, 차로 경계는 거리에 대해
매끄러운 곡선이므로 행들을 묶어 맞추면 잡음이 줄어야 한다. 그 효과도 함께 잰다.

    python3 tools/perception_noise.py [x y 진행방향deg] [프레임수]
"""

import math
import os
import sys
import time

import numpy as np
import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "camera_perception_pkg"))
from camera_perception_pkg.lane_estimator import LaneCenterEstimator  # noqa: E402
from interfaces_pkg.msg import DetectionArray  # noqa: E402

SRC_MAT = [[238, 316], [402, 313], [501, 476], [155, 476]]
ROWS = (20, 100, 180, 260, 330, 390, 435, 470)
PX_PER_M = 103.6


class Probe(Node):
    def __init__(self):
        super().__init__('perception_noise')
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.det = None
        self.est = LaneCenterEstimator(SRC_MAT)
        self.create_subscription(DetectionArray, 'detections',
                                 lambda m: setattr(self, 'det', m), qos)
        self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.cli.wait_for_service(timeout_sec=10.0)

    def place(self, x, y, heading_deg):
        yaw = math.radians(heading_deg) + math.pi / 2   # prius 전방축이 -Y
        st = EntityState()
        st.name = 'ego_vehicle'
        # 안착 높이. 매번 위에서 떨어뜨리면 차가 튀어 카메라가 흔들린다.
        st.pose.position.x, st.pose.position.y, st.pose.position.z = x, y, 0.0117
        st.pose.orientation.z = math.sin(yaw / 2)
        st.pose.orientation.w = math.cos(yaw / 2)
        req = SetEntityState.Request()
        req.state = st
        rclpy.spin_until_future_complete(self, self.cli.call_async(req), timeout_sec=5.0)


def main():
    x, y, hd = (float(v) for v in (sys.argv[1:4] if len(sys.argv) > 3
                                   else (0.0, -22.73, 0.0)))
    n_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 80

    rclpy.init()
    n = Probe()
    n.place(x, y, hd)
    time.sleep(1.0)
    # 주의: 반드시 drive_enable:=false 로 띄운 시뮬에서 돌려야 한다.
    # 제어기가 살아 있으면 차가 움직여서 실제 이동이 잡음으로 섞인다.

    centers = {r: [] for r in ROWS}
    fitted = []
    seen = 0
    last = None
    t0 = time.time()
    while seen < n_frames and time.time() - t0 < 60:
        rclpy.spin_once(n, timeout_sec=0.2)
        if n.det is None or n.det is last or not n.det.detections:
            continue
        last = n.det
        seen += 1
        c, _ = n.est.estimate(n.det.detections, ROWS)
        for r in ROWS:
            centers[r].append(c.get(r, np.nan))
        # 행들을 2차식으로 묶어 맞췄을 때의 값
        ys = np.array([r for r in ROWS if r in c], dtype=float)
        xs = np.array([c[r] for r in ROWS if r in c], dtype=float)
        if len(ys) >= 4:
            fit = np.polyfit(ys, xs, 2)
            fitted.append([float(np.polyval(fit, r)) for r in ROWS])
        else:
            fitted.append([np.nan] * len(ROWS))

    print("정지 상태 (%.2f, %.2f) 진행 %.0f도, 프레임 %d개" % (x, y, hd, seen))
    print("\n%-6s %10s %10s %10s %10s" % ("행", "유효율", "표준편차", "곡선맞춤후", "감소율"))
    fitted = np.array(fitted, dtype=float)
    raw_all, fit_all = [], []
    for i, r in enumerate(ROWS):
        v = np.array(centers[r], dtype=float)
        ok = np.isfinite(v)
        f = fitted[:, i]
        fok = np.isfinite(f)
        if ok.sum() < 5:
            print("%-6d %9.0f%% %10s %10s %10s" % (r, 100 * ok.mean(), "-", "-", "-"))
            continue
        sd, sdf = v[ok].std(), (f[fok].std() if fok.sum() >= 5 else np.nan)
        raw_all.append(sd)
        if np.isfinite(sdf):
            fit_all.append(sdf)
        print("%-6d %9.0f%% %8.1f px %8.1f px %9.0f%%"
              % (r, 100 * ok.mean(), sd, sdf, 100 * (1 - sdf / max(sd, 1e-6))))

    if raw_all:
        print("\n평균 표준편차 %.1f px (%.3f m) -> 곡선맞춤 후 %.1f px (%.3f m)"
              % (np.mean(raw_all), np.mean(raw_all) / PX_PER_M,
                 np.mean(fit_all), np.mean(fit_all) / PX_PER_M))
        print("행별 독립 추출 대신 행들을 묶어 맞추면 잡음이 %.0f%% 준다."
              % (100 * (1 - np.mean(fit_all) / max(np.mean(raw_all), 1e-6))))

    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
