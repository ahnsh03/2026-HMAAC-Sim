#!/usr/bin/env python3
"""트랙 텍스처에서 자차 차로의 양쪽 경계까지의 거리장을 만든다.

이전에는 '잔디 경계에서 1.70m 안쪽'을 차로 중앙으로 삼았는데, 차로 폭을 상수로
가정하는 셈이라 폭이 3.03~3.35m 로 변하는 구간에서 헐거웠다.

여기서는 두 경계를 직접 찾는다.
  바깥 경계 = 잔디와 맞닿은 실선
  안쪽 경계 = 점선 중앙선 (끊긴 조각들을 이어 붙여 선으로 만든다)

차로 중앙은 두 경계에서 같은 거리에 있는 곳이므로, 차로 폭 상수가 필요 없다.
  이탈량 = (바깥까지 거리 - 안쪽까지 거리) / 2
         양수면 안쪽(점선 쪽)으로 치우친 것.
"""

import os

import cv2
import numpy as np

TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "src/simulation_pkg/models/race_track/materials/textures/basic_track_2026.jpg")
SIZE_LX, SIZE_LY = 53.83, 40.473
CAR_HALF_WIDTH = 0.88          # 차폭 1.7526m 의 절반
# 차체를 한 점으로 보면 코너에서 뒷바퀴가 안쪽을 파고드는 것(off-tracking)을
# 놓친다. model_states 가 주는 위치는 앞뒤 축의 중간이므로, 거기서 앞뒤로
# 축간거리의 절반씩 떨어진 두 축의 좌우 끝, 즉 네 모서리를 모두 본다.
WHEELBASE = 2.86
AXLE_OFFSETS = (+WHEELBASE / 2, -WHEELBASE / 2)   # 앞축, 뒷축 (진행방향 기준)


def build():
    """(dist_out, dist_in, m_per_px, w, h, grass) 를 만든다. 거리는 미터."""
    img = cv2.imread(os.path.normpath(TEX))
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (35, 60, 40), (90, 255, 255))
    m_per_px = (SIZE_LX / w + SIZE_LY / h) / 2

    # 바깥 경계: 잔디까지의 거리
    dist_out = cv2.distanceTransform((grass == 0).astype(np.uint8), cv2.DIST_L2, 5) * m_per_px

    # 링(도로) 위의 흰 표시만 남긴다. 주차장(연회색)과 그 표시는 제외해야 하므로,
    # 잔디에서 너무 멀리 떨어진 곳은 버린다. 링 폭이 6~7m 이므로 8m 로 자른다.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ring = (grass == 0) & (dist_out < 8.0)
    white = ((gray > 150) & ring).astype(np.uint8)

    # 바깥 실선은 잔디 바로 옆이므로 제외하면 점선 중앙선만 남는다
    dashed = (white > 0) & (dist_out > 0.9)
    dashed = dashed.astype(np.uint8)

    # 끊긴 조각을 이어 붙인다. 점선 간격이 약 1.5m 이므로 그보다 큰 커널로 닫는다
    k = int(round(2.2 / m_per_px))
    dashed = cv2.morphologyEx(dashed, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # 링 밖으로 번진 부분 제거
    dashed = (dashed > 0) & ring

    dist_in = cv2.distanceTransform((dashed == 0).astype(np.uint8), cv2.DIST_L2, 5) * m_per_px
    return dist_out, dist_in, m_per_px, w, h, grass


def to_px(wx, wy, w, h):
    u = (np.asarray(wy) + SIZE_LX / 2) / SIZE_LX
    v = 1 - (-np.asarray(wx) + SIZE_LY / 2) / SIZE_LY
    return (np.clip(u * (w - 1), 0, w - 1)).astype(int), (np.clip(v * (h - 1), 0, h - 1)).astype(int)


def evaluate(wx, wy, gt=None):
    """(이탈량, 바깥까지, 안쪽까지) [m]. 이탈량 양수면 안쪽(점선 쪽) 치우침.

    한 점(차체 중앙)만 본다. 차체 전체를 보려면 body_clearance 를 쓴다.
    """
    dist_out, dist_in, _, w, h, _ = gt if gt else build()
    cu, cv_ = to_px(wx, wy, w, h)
    d_out = dist_out[cv_, cu]
    d_in = dist_in[cv_, cu]
    return (d_out - d_in) / 2.0, d_out, d_in


def body_clearance(wx, wy, yaw, gt=None):
    """차체 네 모서리 기준 여유 (안쪽까지, 바깥까지) [m].

    코너에서는 뒷축이 앞축보다 안쪽을 지나므로, 차체 중앙 한 점만 보면
    뒷바퀴가 점선을 밟는 것을 놓친다. 앞축/뒷축의 좌우 끝 네 점 중 가장
    가까운 값을 돌려준다. 0 이하면 그 선을 밟고 있다는 뜻이다.
    """
    dist_out, dist_in, _, w, h, _ = gt if gt else build()
    wx = np.asarray(wx, dtype=float); wy = np.asarray(wy, dtype=float)
    yaw = np.asarray(yaw, dtype=float)
    fx, fy = np.cos(yaw), np.sin(yaw)          # 진행방향
    rx, ry = np.sin(yaw), -np.cos(yaw)         # 차량 오른쪽
    c_in = np.full(wx.shape, np.inf)
    c_out = np.full(wx.shape, np.inf)
    for a in AXLE_OFFSETS:
        for lateral in (+CAR_HALF_WIDTH, -CAR_HALF_WIDTH):
            px = wx + a * fx + lateral * rx
            py = wy + a * fy + lateral * ry
            cu, cv_ = to_px(px, py, w, h)
            c_in = np.minimum(c_in, dist_in[cv_, cu])
            c_out = np.minimum(c_out, dist_out[cv_, cu])
    return c_in, c_out


if __name__ == "__main__":
    import csv
    import sys
    gt = build()
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lap_trajectory.csv"
    rows = [r for r in csv.DictReader(open(path)) if int(r["speed"]) > 0]
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    dev, d_out, d_in = evaluate(x, y, gt)
    lane_w = d_out + d_in
    print("차로 폭 실측: 중앙값 %.2f m (범위 %.2f~%.2f)"
          % (np.median(lane_w), np.percentile(lane_w, 5), np.percentile(lane_w, 95)))
    print("이탈량: 평균 %+.2f m, 평균|.| %.2f m, 90분위|.| %.2f m"
          % (dev.mean(), np.abs(dev).mean(), np.percentile(np.abs(dev), 90)))
    yaw = np.array([float(r["yaw"]) for r in rows])
    c_in, c_out = body_clearance(x, y, yaw, gt)
    print("점선 밟음 %.1f%% (차체 네 모서리 기준)" % (100 * (c_in <= 0).mean()))
    print("실선 밟음 %.1f%% (차체 네 모서리 기준)" % (100 * (c_out <= 0).mean()))
    print("점선까지 여유: 중앙값 %.2f m, 5분위 %.2f m" % (np.median(c_in), np.percentile(c_in, 5)))
