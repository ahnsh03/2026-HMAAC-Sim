#!/usr/bin/env python3
"""주행 로그를 인지 오차와 제어 오차로 나눠서 본다.

기준(지상진실)은 트랙 텍스처에서 만든다. 텍스처는 53.83 x 40.473 m 지면에
붙어 있고 (track.world, 모델 yaw 1.57), 잔디 경계에서 HALF_LANE 안쪽이
자차 차로(바깥 차로)의 중앙이다. 이 기준선은 tools/visualize_lane.py 로
그려서 눈으로 확인했다.

두 오차를 이렇게 나눈다.

  인지 오차 = (인지가 말한 차로 중심) - (진짜 차로 중심)
      제어기가 본 전방 주시점을 차량 자세로 월드 좌표에 놓고, 그 지점의
      진짜 차로 중앙과 비교한다. 제어와 무관하게 인지만의 정확도다.

  제어 오차 = (차량 위치) - (인지가 말한 차로 중심)
      인지가 시킨 대로 따라갔는지를 본다.

  총 오차   = (차량 위치) - (진짜 차로 중심)  = 제어 오차 + 인지 오차

부호는 모두 '차량 오른쪽이 양수'다. 자차는 반시계로 돌므로 오른쪽이 바깥쪽.

    python3 tools/diagnose.py [로그.csv]
"""

import csv
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "src/simulation_pkg/models/race_track/materials/textures/basic_track_2026.jpg")
SIZE_LX, SIZE_LY = 53.83, 40.473   # 지면 크기 [m]
HALF_LANE = 1.70                   # 잔디 경계에서 자차 차로 중앙까지 [m]
CAR_HALF_WIDTH = 0.88              # 차폭 1.7526m 의 절반
TOUCH = HALF_LANE - CAR_HALF_WIDTH  # 이만큼 벗어나면 차체가 차선에 닿는다


import lane_gt


def load_track():
    return lane_gt.build()


def deviation(wx, wy, gt):
    """진짜 차로 중앙에서 벗어난 양 [m]. 양수면 안쪽(점선 쪽).

    차로 폭을 상수로 가정하지 않고, 바깥 실선과 점선 중앙선까지의 거리를
    직접 재서 그 중점을 차로 중앙으로 삼는다.
    """
    dev, _, _ = lane_gt.evaluate(wx, wy, gt)
    return dev


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lap_trajectory.csv"
    gt = load_track()

    rows = [r for r in csv.DictReader(open(path)) if int(r["speed"]) > 0]
    if not rows:
        print("주행 구간이 없다"); return

    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    yaw = np.array([float(r["yaw"]) for r in rows])
    ld = np.array([float(r["ld"]) for r in rows])
    lat = np.array([float(r["lat"]) for r in rows])

    # 총 오차: 차량 위치가 진짜 차로 중앙에서 벗어난 양.
    # dist_m 은 잔디까지의 거리라 '안쪽이 양수'다. 차량 오른쪽(바깥) 기준으로 뒤집는다.
    total = -deviation(x, y, gt)

    # 인지가 지목한 전방 주시점을 월드로 옮긴다.
    # 차량 오른쪽 방향은 진행방향을 -90도 돌린 것: (sin yaw, -cos yaw)
    ok = np.isfinite(ld) & np.isfinite(lat)
    px = x + ld * np.cos(yaw) + lat * np.sin(yaw)
    py = y + ld * np.sin(yaw) - lat * np.cos(yaw)
    # 그 지점이 진짜 차로 중앙에서 벗어난 양 = 인지 오차
    percep = np.full(len(x), np.nan)
    percep[ok] = -deviation(px[ok], py[ok], gt)

    # 제어 오차 = 총 오차 - 인지 오차
    control = total - percep

    def stat(name, v):
        v = v[np.isfinite(v)]
        if not len(v):
            print("  %-10s 표본 없음" % name); return
        print("  %-10s 평균 %+6.2f m   평균|.| %5.2f m   90분위|.| %5.2f m"
              % (name, v.mean(), np.abs(v).mean(), np.percentile(np.abs(v), 90)))

    print("표본 %d개, 주행거리 %.1f m" % (len(rows), float(rows[-1]["distance"])))
    print("\n부호 규약: 양수 = 차로 중앙보다 바깥쪽(잔디 쪽), 음수 = 안쪽(중앙선 쪽)")
    print("\n[전체]")
    stat("총 오차", total)
    stat("인지 오차", percep)
    stat("제어 오차", control)

    # 직선/코너 구분
    kap = np.zeros(len(x))
    for i in range(2, len(x) - 2):
        a, b, c = np.array([x[i-2], y[i-2]]), np.array([x[i], y[i]]), np.array([x[i+2], y[i+2]])
        ab, cb = a - b, c - b
        cross = ab[0] * cb[1] - ab[1] * cb[0]
        na, nc, nac = np.linalg.norm(ab), np.linalg.norm(cb), np.linalg.norm(a - c)
        kap[i] = 0.0 if na * nc * nac < 1e-9 else abs(2 * cross / (na * nc * nac))
    straight = kap < 0.02
    for label, sel in (("직선", straight), ("코너", ~straight)):
        print("\n[%s 구간 %d개]" % (label, sel.sum()))
        stat("총 오차", total[sel])
        stat("인지 오차", percep[sel])
        stat("제어 오차", control[sel])

    # 곡률 추정 품질. 직선에서는 0 이어야 한다.
    # 곡률은 앞먹임 atan(축간거리*곡률) 로 조향에 그대로 들어가므로,
    # 여기서의 편향은 상시 조향 편향이 되고 산포는 조향 흔들림이 된다.
    kap_est = np.array([float(r["kappa"]) if r["kappa"] != "nan" else np.nan for r in rows])
    ks = kap_est[straight]
    ks = ks[np.isfinite(ks)]
    if len(ks):
        print("\n[곡률 추정 품질] 직선 구간 %d개 (참값 0)" % len(ks))
        print("  평균 %+.4f 1/m  ->  상시 조향 %+.2f step"
              % (ks.mean(), math.atan(2.86 * ks.mean()) / (0.6458 / 7)))
        print("  표준편차 %.4f 1/m  ->  조향 흔들림 ±%.2f step"
              % (ks.std(), math.atan(2.86 * ks.std()) / (0.6458 / 7)))
        print("  상한 도달 %.1f%%  (|곡률| > 0.10, 이 트랙 실제 최대 곡률)"
              % (100 * (np.abs(ks) > 0.10).mean()))

    _, d_out, d_in = lane_gt.evaluate(x, y, gt)
    print("\n점선 밟음 %.1f%%  (차체가 점선에 닿음, 차폭 %.2fm 반영)"
          % (100 * (d_in < lane_gt.CAR_HALF_WIDTH).mean(), 2 * lane_gt.CAR_HALF_WIDTH))
    print("실선 밟음 %.1f%%  (차체가 바깥 실선에 닿음)"
          % (100 * (d_out < lane_gt.CAR_HALF_WIDTH).mean()))

    bad = np.argsort(-np.abs(total))[:8]
    print("\n총오차 큰 지점:")
    for i in bad:
        print("  t=%7s x=%7.2f y=%7.2f  총 %+5.2f = 제어 %+5.2f + 인지 %+5.2f  "
              "(조향 목표%5s 실제%3s, 속도%4s)"
              % (rows[i]["t"], x[i], y[i], total[i], control[i], percep[i],
                 rows[i]["target_steer"], rows[i]["steering"], rows[i]["speed"]))


if __name__ == "__main__":
    main()
