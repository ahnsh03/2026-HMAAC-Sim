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
# 점선 중앙선이 놓인 거리 [m, 잔디 경계 기준]. 차로 폭이 3.03~3.35m 이므로
# 중앙선은 바깥 실선에서 그만큼 안쪽이다. 이 밴드로 제한하면 안쪽 실선과
# 주차장 표시가 중앙선으로 잘못 잡히지 않는다.
CENTER_LINE_BAND = (2.4, 4.4)

# 트랙 텍스처 단면을 직접 재서 얻은 차선 기하 [m].
#   잔디 경계  -24.39
#   바깥 실선  -24.28 ~ -24.16  (두께 0.12)
#   점선 중앙선 -21.30 ~ -21.18  (두께 0.12)
# dist_out 은 잔디까지, dist_in 은 점선의 가까운 면까지 잰다. 그래서
#   - 두 거리가 같아지는 지점은 진짜 차로 중앙보다 CENTER_BIAS 만큼 바깥이다
#   - 바깥 실선을 밟는 것은 잔디에 닿기 전, dist_out < OUTER_LINE_REACH 일 때다
GRASS_TO_LINE = 0.11           # 잔디 경계에서 바깥 실선 바깥 면까지
LINE_WIDTH = 0.12              # 차선 두께
OUTER_LINE_REACH = GRASS_TO_LINE + LINE_WIDTH   # 잔디에서 실선 안쪽 면까지
CENTER_BIAS = 0.115            # (dist_out - dist_in)/2 가 0 인 지점의 바깥쪽 치우침
DASH_MAX_AREA = 300            # 점선 한 칸의 픽셀 넓이 상한. 횡단보도 등 큰 표시를 뺀다
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

    # 링(도로) 위의 흰 표시만 남긴다.
    # 밝기 분포를 재보면 도로 아스팔트 72, 주차장 바닥 118, 차선 223 이상이라
    # 200 으로 자르면 주차장 바닥은 확실히 빠진다.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ring = (grass == 0) & (dist_out < 8.0)
    white = (gray > 200) & ring

    # 점선 중앙선은 바깥 실선에서 차로 폭(약 3.2m)만큼 안쪽에 있다.
    # 그 거리 밴드로 제한하면 바깥 실선, 안쪽 실선, 주차장 표시가 모두 빠진다.
    dashed = (white & (dist_out > CENTER_LINE_BAND[0]) & (dist_out < CENTER_LINE_BAND[1]))
    dashed = dashed.astype(np.uint8)

    # 횡단보도 줄무늬가 밴드에 걸치는 곳이 한 군데 있다. 점선 한 칸은 약 1.5m x 0.15m
    # (약 37 px) 이므로, 그보다 훨씬 큰 덩어리는 점선이 아니다.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dashed, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > DASH_MAX_AREA:
            dashed[labels == i] = 0

    # 끊긴 조각을 이어 붙인다. 커널이 크면 주변 표시까지 뭉개므로 점선 간격만큼만.
    k = int(round(1.2 / m_per_px)) | 1
    dashed = cv2.morphologyEx(dashed, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
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
    return (d_out - d_in) / 2.0 - CENTER_BIAS, d_out, d_in


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
    # 바깥쪽은 잔디가 아니라 실선 안쪽 면이 경계다
    return c_in, c_out - OUTER_LINE_REACH


if __name__ == "__main__":
    import csv
    import sys
    gt = build()
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lap_trajectory.csv"
    rows = [r for r in csv.DictReader(open(path)) if int(r["speed"]) > 0]
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    dev, d_out, d_in = evaluate(x, y, gt)
    # 선 중심 사이 간격 = (잔디~점선 가까운면) - 잔디여백 - 선두께/2 + 선두께/2
    lane_w = d_out + d_in - GRASS_TO_LINE - LINE_WIDTH / 2 + LINE_WIDTH / 2
    print("차로 폭(차선 중심 간격) 실측: 중앙값 %.2f m (범위 %.2f~%.2f)"
          % (np.median(lane_w), np.percentile(lane_w, 5), np.percentile(lane_w, 95)))
    print("이탈량: 평균 %+.2f m, 평균|.| %.2f m, 90분위|.| %.2f m"
          % (dev.mean(), np.abs(dev).mean(), np.percentile(np.abs(dev), 90)))
    yaw = np.array([float(r["yaw"]) for r in rows])
    c_in, c_out = body_clearance(x, y, yaw, gt)
    print("점선 밟음 %.1f%% (차체 네 모서리 기준)" % (100 * (c_in <= 0).mean()))
    print("실선 밟음 %.1f%% (차체 네 모서리 기준)" % (100 * (c_out <= 0).mean()))
    print("점선까지 여유: 중앙값 %.2f m, 5분위 %.2f m" % (np.median(c_in), np.percentile(c_in, 5)))
