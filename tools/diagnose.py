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
import os
import sys

import cv2
import numpy as np

TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "src/simulation_pkg/models/race_track/materials/textures/basic_track_2026.jpg")
SIZE_LX, SIZE_LY = 53.83, 40.473   # 지면 크기 [m]
HALF_LANE = 1.70                   # 잔디 경계에서 자차 차로 중앙까지 [m]
CAR_HALF_WIDTH = 0.88              # 차폭 1.7526m 의 절반
TOUCH = HALF_LANE - CAR_HALF_WIDTH  # 이만큼 벗어나면 차체가 차선에 닿는다


def load_track():
    img = cv2.imread(os.path.normpath(TEX))
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (35, 60, 40), (90, 255, 255))
    dist_m = cv2.distanceTransform((grass == 0).astype(np.uint8), cv2.DIST_L2, 5) \
        * ((SIZE_LX / w + SIZE_LY / h) / 2)
    return grass, dist_m, w, h


def to_px(wx, wy, w, h):
    """월드 -> 텍스처 픽셀. 지면 로컬 (lx, ly) = (wy, -wx), v 는 뒤집혀 있다."""
    u = (np.asarray(wy) + SIZE_LX / 2) / SIZE_LX
    v = 1 - (-np.asarray(wx) + SIZE_LY / 2) / SIZE_LY
    return (np.clip(u * (w - 1), 0, w - 1)).astype(int), (np.clip(v * (h - 1), 0, h - 1)).astype(int)


def deviation(wx, wy, grass, dist_m, w, h):
    """진짜 차로 중앙에서 벗어난 양 [m]. 양수면 안쪽(중앙선 쪽)."""
    cu, cv_ = to_px(wx, wy, w, h)
    return dist_m[cv_, cu] - HALF_LANE


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lap_trajectory.csv"
    grass, dist_m, w, h = load_track()

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
    total = -deviation(x, y, grass, dist_m, w, h)

    # 인지가 지목한 전방 주시점을 월드로 옮긴다.
    # 차량 오른쪽 방향은 진행방향을 -90도 돌린 것: (sin yaw, -cos yaw)
    ok = np.isfinite(ld) & np.isfinite(lat)
    px = x + ld * np.cos(yaw) + lat * np.sin(yaw)
    py = y + ld * np.sin(yaw) - lat * np.cos(yaw)
    # 그 지점이 진짜 차로 중앙에서 벗어난 양 = 인지 오차
    percep = np.full(len(x), np.nan)
    percep[ok] = -deviation(px[ok], py[ok], grass, dist_m, w, h)

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

    touching = np.abs(total) > TOUCH
    print("\n차선 밟음(|총오차| > %.2f m) %.1f%%,  차선 넘음(> %.2f m) %.1f%%"
          % (TOUCH, 100 * touching.mean(), HALF_LANE, 100 * (np.abs(total) > HALF_LANE).mean()))

    bad = np.argsort(-np.abs(total))[:8]
    print("\n총오차 큰 지점:")
    for i in bad:
        print("  t=%7s x=%7.2f y=%7.2f  총 %+5.2f = 제어 %+5.2f + 인지 %+5.2f  "
              "(조향 목표%5s 실제%3s, 속도%4s)"
              % (rows[i]["t"], x[i], y[i], total[i], control[i], percep[i],
                 rows[i]["target_steer"], rows[i]["steering"], rows[i]["speed"]))


if __name__ == "__main__":
    main()
