#!/usr/bin/env python3
"""정지·정렬 상태에서 행별 차로 중심 추정을 지상진실과 견준다.

왜 이게 필요한가
----------------
직진 구간에서 인지가 만든 경로의 활 높이가 0.327m 였다. 곧은 길인데 휘어 있다.
호모그래피는 직선을 직선으로 보내므로 BEV 사다리꼴(SRC_MAT)이 아무리 틀려도
곧은 차선이 휘지는 않는다. 그러므로 휨은 그 위쪽 어딘가에서 생긴다.

    (a) YOLO 마스크의 경계 자체가 원본 영상에서 이미 휘어 있다
    (b) 행마다 다른 근거(좌우 양쪽 / 실선만 / 중앙선만)를 섞어 쓰면서
        차로 폭 상수의 오차가 행별 단차로 나타난다

이 도구는 둘을 갈라 본다. 원본 영상에서의 직선성과 BEV 행별 오차를 함께 잰다.

측정 조건
---------
차를 GT 차로 중앙에 놓고 도로 방향과 정확히 정렬시킨다. 그러면 곧은 구간에서는
모든 행의 차로 중심이 같은 열에 있어야 하므로, 행별 편차가 곧 인지 오차다.
축척이나 세로 대응을 몰라도 되는 것이 이 배치의 장점이다.

    사용법 (반드시 drive_enable:=false 로 띄운 시뮬에서)
      python3 tools/lane_row_gt.py --scan             측정하기 좋은 지점 목록
      python3 tools/lane_row_gt.py X Y [프레임수]     그 지점에서 측정
"""

import math
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src", "camera_perception_pkg"))

import lane_gt  # noqa: E402
from camera_perception_pkg.lane_estimator import (BEV_H, BEV_W,  # noqa: E402
                                                  LaneCenterEstimator,
                                                  region_edges)
from interfaces_pkg.msg import DetectionArray  # noqa: E402

SRC_MAT = [[238, 316], [402, 313], [501, 476], [155, 476]]
ROWS = (20, 100, 180, 260, 330, 390, 435, 470)

# motion_planner_node 와 같은 BEV 기하
CAR_CENTER_X = 320.0
BEV_BOTTOM_ROW = 479.0
PX_PER_M_LAT = 103.6
PX_PER_M_LON = 98.0
BOTTOM_ROW_AHEAD = 4.74     # BEV 맨 아랫줄이 뒷차축에서 떨어진 거리 [m]
WHEELBASE = 2.86

CAM_W, CAM_H = 640, 480


def row_to_ahead(row):
    return BOTTOM_ROW_AHEAD + (BEV_BOTTOM_ROW - row) / PX_PER_M_LON


# ---------------------------------------------------------------- GT 기하

def dev_at(x, y, gt):
    """차로 중앙에서 벗어난 양 [m]. 양수면 안쪽(점선 쪽)."""
    return float(lane_gt.evaluate(np.array([x]), np.array([y]), gt)[0][0])


def dout_at(x, y, gt):
    """잔디 경계까지의 거리 [m]. 연속인 실선이라 거리장이 매끄럽다."""
    return float(lane_gt.evaluate(np.array([x]), np.array([y]), gt)[1][0])


def dout_grad(x, y, gt, h=0.08):
    gx = (dout_at(x + h, y, gt) - dout_at(x - h, y, gt)) / (2 * h)
    gy = (dout_at(x, y + h, gt) - dout_at(x, y - h, gt)) / (2 * h)
    return gx, gy


def snap_to_iso(x, y, gt, target, iters=8):
    """잔디 경계에서 target [m] 떨어진 등고선으로 옮긴다.

    차로 중앙(dev=0)으로 바로 스냅하지 않는 이유는, dev 가 점선 중앙선까지의
    거리를 쓰기 때문이다. 점선은 끊겨 있어 거리장이 점선 주기(약 3m)로
    출렁이고, 그대로 따라가면 곧은 길에서도 궤적이 ±0.25m 지그재그가 된다.
    """
    for _ in range(iters):
        e = dout_at(x, y, gt) - target
        gx, gy = dout_grad(x, y, gt)
        n2 = gx * gx + gy * gy
        if n2 < 1e-6:
            break
        x -= e * gx / n2
        y -= e * gy / n2
    return x, y


def tangent_at(x, y, gt, hint):
    """도로 진행 방향 단위벡터. hint 와 같은 쪽을 고른다."""
    gx, gy = dout_grad(x, y, gt)
    n = math.hypot(gx, gy)
    if n < 1e-9:
        return hint
    tx, ty = -gy / n, gx / n
    if tx * hint[0] + ty * hint[1] < 0:
        tx, ty = -tx, -ty
    return tx, ty


DS = 0.20          # 중앙선 추적 간격 [m]
SMOOTH_M = 2.0     # 추적 중간 평활 길이 [m]
FIT_HALF_M = 2.5   # 곡률·방향을 잴 때 쓰는 국소 맞춤 반경 [m]
ISO_TARGET = 1.70  # 잔디 경계에서 자차 차로 중앙까지의 대략 거리 [m]
DEV_SMOOTH_M = 6.0  # 점선 기반 보정을 평활하는 길이 [m]. 점선 주기보다 길게


def _cyclic_smooth(P, window_m):
    k = int(window_m / DS) | 1
    ker = np.ones(k) / k
    Q = P.copy()
    for j in (0, 1):
        Q[:, j] = np.convolve(np.r_[P[-k:, j], P[:, j], P[:k, j]], ker, 'same')[k:-k]
    return Q


def trace_loop(gt, start=(-5.09, -22.69), hint=(1.0, 0.0), max_m=260.0):
    """차로 중앙선을 한 바퀴 따라가 닫힌 곡선으로 만든다.

    두 단계로 만든다.
      1) 잔디 경계에서 일정 거리인 등고선을 따라간다. 연속인 실선이라 매끄럽다.
      2) 그 위에서 두 경계 기준 이탈량을 재고, 점선 주기보다 길게 평활한 뒤
         그만큼 옆으로 민다. 차로 폭이 3.03~3.35m 로 변하는 것을 이렇게 반영한다.

    등고선만 쓰면 차로 폭 변화를 놓치고, 점선 거리장을 그대로 따라가면 점선
    주기로 출렁인다. 둘을 나눠 쓰면 양쪽 문제를 모두 피한다.
    """
    x, y = snap_to_iso(start[0], start[1], gt, ISO_TARGET)
    t = tangent_at(x, y, gt, hint)
    pts = [(x, y)]
    for i in range(int(max_m / DS)):
        x, y = snap_to_iso(x + DS * t[0], y + DS * t[1], gt, ISO_TARGET)
        t = tangent_at(x, y, gt, t)
        if i > int(20.0 / DS) and math.hypot(x - pts[0][0], y - pts[0][1]) < DS:
            break
        pts.append((x, y))

    P = _cyclic_smooth(np.array(pts), SMOOTH_M)

    # 두 경계 기준 이탈량으로 보정한다. 양수면 안쪽(점선 쪽)으로 치우친 것이므로
    # 잔디 쪽(= dist_out 이 줄어드는 방향의 반대)으로 그만큼 되민다.
    dev = np.array([dev_at(px, py, gt) for px, py in P])
    k = int(DEV_SMOOTH_M / DS) | 1
    ker = np.ones(k) / k
    dev = np.convolve(np.r_[dev[-k:], dev, dev[:k]], ker, 'same')[k:-k]
    for i, (px, py) in enumerate(P):
        gx, gy = dout_grad(px, py, gt)
        n = math.hypot(gx, gy)
        if n < 1e-9:
            continue
        P[i, 0] -= dev[i] * gx / n
        P[i, 1] -= dev[i] * gy / n
    return _cyclic_smooth(P, 1.0)


def _local_fit(P, i, half_m=FIT_HALF_M):
    """i 번째 점 주변을 국소 좌표계에서 2차식으로 맞춘다. (진행각, 곡률)."""
    n = len(P)
    h = int(half_m / DS)
    idx = [(i + k) % n for k in range(-h, h + 1)]
    seg = P[idx]
    # 대략적 진행 방향으로 회전시켜 두면 lat = f(lon) 이 함수가 된다
    th0 = math.atan2(seg[-1, 1] - seg[0, 1], seg[-1, 0] - seg[0, 0])
    c, s = math.cos(th0), math.sin(th0)
    d = seg - P[i]
    lon = d[:, 0] * c + d[:, 1] * s
    lat = -d[:, 0] * s + d[:, 1] * c
    a2, a1, _ = np.polyfit(lon, lat, 2)
    # lat' = a1 이므로 진행각은 th0 + atan(a1), 곡률은 2*a2 / (1+a1^2)^1.5
    heading = th0 + math.atan(a1)
    kappa = 2.0 * a2 / (1.0 + a1 * a1) ** 1.5
    return heading, float(kappa)


def loop_curvature(P):
    return np.array([_local_fit(P, i)[1] for i in range(len(P))])


def nearest_index(P, x, y):
    return int(np.argmin((P[:, 0] - x) ** 2 + (P[:, 1] - y) ** 2))


def loop_heading(P, i):
    return _local_fit(P, i)[0]


def expected_lateral(P, i0, origin, heading, ahead_m, lon_range=(1.0, 12.0)):
    """차량 좌표계에서 전방 ahead_m 인 곳의 차로 중심 횡방향 [m]. 오른쪽이 양수.

    점 두 개를 잇는 보간 대신 구간 전체를 2차식으로 맞춰 읽는다.
    추적점 하나하나에는 텍스처 픽셀만큼(약 2.5cm) 잡음이 있어, 보간하면
    그 잡음이 그대로 기댓값에 실린다.
    """
    n = len(P)
    idx = [(i0 + k) % n for k in range(-10, int(16.0 / DS))]
    seg = P[idx]
    fx, fy = math.cos(heading), math.sin(heading)
    rx, ry = math.sin(heading), -math.cos(heading)      # 차량 오른쪽
    d = seg - np.asarray(origin)
    lon = d[:, 0] * fx + d[:, 1] * fy
    lat = d[:, 0] * rx + d[:, 1] * ry
    sel = (lon >= lon_range[0]) & (lon <= lon_range[1])
    if sel.sum() < 10 or not (lon[sel].min() <= ahead_m <= lon[sel].max()):
        return float('nan')
    return float(np.polyval(np.polyfit(lon[sel], lat[sel], 2), ahead_m))


# ---------------------------------------------------------------- 원본 영상 직선성

def polygons_by_class(detections):
    out = {}
    for det in detections:
        mask = getattr(det, 'mask', None)
        if mask is None or not mask.data:
            continue
        out.setdefault(det.class_name, []).append(
            np.array([(p.x, p.y) for p in mask.data], dtype=np.float32))
    return out


def line_bow(xs, ys):
    """점들을 직선에 맞췄을 때의 (RMS 잔차, 최대 활 높이). 단위는 입력과 같다."""
    if len(xs) < 6 or float(np.ptp(ys)) < 20:
        return float('nan'), float('nan')
    coeff = np.polyfit(ys, xs, 1)
    res = xs - np.polyval(coeff, ys)
    return float(np.sqrt(np.mean(res ** 2))), float(np.max(np.abs(res)))


def mask_right_edge(polys, width, height, y_lo, y_hi):
    """폴리곤을 채운 마스크의 행별 오른쪽 끝. 이미지 가장자리에 붙은 행은 뺀다."""
    img = np.zeros((height, width), dtype=np.uint8)
    for poly in polys:
        pts = poly.astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(img, [pts], 255)
    ys, xs = [], []
    for row in range(y_lo, y_hi):
        cols = np.where(img[row] > 0)[0]
        if len(cols) == 0 or cols[-1] >= width - 2:
            continue
        ys.append(row)
        xs.append(float(cols[-1]))
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------- 측정

class Probe(Node):
    def __init__(self):
        super().__init__('lane_row_gt')
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.det = None
        self.est = LaneCenterEstimator(SRC_MAT)
        self.create_subscription(DetectionArray, 'detections',
                                 lambda m: setattr(self, 'det', m), qos)
        self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        if not self.cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("/gazebo/set_entity_state 없음. 시뮬이 떠 있는지 확인")

    def place(self, x, y, heading):
        """진행방향 heading [rad] 으로 세운다. prius 모델 전방축이 -Y 라 90도 더한다."""
        st = EntityState()
        st.name = 'ego_vehicle'
        st.pose.position.x, st.pose.position.y, st.pose.position.z = x, y, 0.0117
        psi = heading + math.pi / 2
        st.pose.orientation.z = math.sin(psi / 2)
        st.pose.orientation.w = math.cos(psi / 2)
        req = SetEntityState.Request()
        req.state = st
        rclpy.spin_until_future_complete(self, self.cli.call_async(req), timeout_sec=5.0)

    def collect(self, n_frames, timeout=40.0):
        frames, last, t0 = [], None, time.time()
        while len(frames) < n_frames and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.det is None or self.det is last or not self.det.detections:
                continue
            last = self.det
            frames.append(self.det)
        return frames


def measure(node, P, i0, gt, n_frames):
    x, y = P[i0]
    heading = loop_heading(P, i0)
    node.place(x, y, heading)
    time.sleep(1.2)                     # 서스펜션이 가라앉을 때까지
    node.place(x, y, heading)           # 떨어지며 틀어진 자세를 다시 잡는다
    time.sleep(0.8)
    frames = node.collect(n_frames)
    if not frames:
        print("detections 를 받지 못했다. 시뮬과 인지 노드를 확인하라.")
        return None

    # 차량 위치는 앞뒤 축의 중간이므로, BEV 전방거리의 기준인 뒷차축은 뒤로 반칸이다
    rear = (x - math.cos(heading) * WHEELBASE / 2,
            y - math.sin(heading) * WHEELBASE / 2)

    per_row = {r: [] for r in ROWS}
    per_src = {r: {} for r in ROWS}
    per_width = {r: [] for r in ROWS}
    bev_bow, cam_bow_lane, cam_bow_cl = [], [], []

    for det in frames:
        centers, dbg = node.est.estimate(det.detections, ROWS)
        for r in ROWS:
            if r in centers:
                per_row[r].append(centers[r])
                s = dbg['sources'].get(r, '?')
                per_src[r][s] = per_src[r].get(s, 0) + 1

        # 좌우 경계를 실제로 둘 다 봤을 때의 차로 폭. 한쪽만 보이는 행에서
        # 상수 LANE_WIDTH 를 더해 만든 값이 맞는지 견주는 기준이 된다.
        lane2_bev = node.est._last[2] if node.est._last else None
        if lane2_bev is not None:
            for r in ROWS:
                left, right = region_edges(lane2_bev, r, None, node.est.tilt)
                if left is not None and right is not None and right < BEV_W - 3:
                    per_width[r].append(right - left)

        # BEV 에서 행별 중심이 직선에서 벗어난 정도
        ys = np.array([r for r in ROWS if r in centers], dtype=float)
        xs = np.array([centers[r] for r in ROWS if r in centers], dtype=float)
        if len(ys) >= 5:
            _, bow = line_bow(xs, ys)
            bev_bow.append(bow / PX_PER_M_LAT)

        # 원본 영상에서의 직선성 (BEV 이전 단계)
        polys = polygons_by_class(det.detections)
        if 'lane2' in polys:
            ex, ey = mask_right_edge(polys['lane2'], CAM_W, CAM_H, 320, 470)
            _, bow = line_bow(ex, ey)
            if np.isfinite(bow):
                cam_bow_lane.append(bow)
        if 'center_line' in polys:
            allp = np.vstack(polys['center_line'])
            _, bow = line_bow(allp[:, 0], allp[:, 1])
            if np.isfinite(bow):
                cam_bow_cl.append(bow)

    kap = float(loop_curvature(P)[i0])
    print("\n지점 (%.2f, %.2f) 진행 %.1f도, GT 곡률 %+.4f 1/m (반경 %.0f m), 프레임 %d개"
          % (x, y, math.degrees(heading), kap,
             1 / max(abs(kap), 1e-6), len(frames)))
    print("차로 중앙에서 %.3f m (0 에 가까울수록 정확한 측정)" % dev_at(x, y, gt))

    print("\n%-5s %6s %6s %8s %7s %8s %8s   %s"
          % ("행", "전방m", "유효율", "추정횡m", "±", "GT횡m", "오차m", "근거 / 실측차로폭"))
    errs = []
    for r in ROWS:
        vals = np.array(per_row[r], dtype=float)
        ahead = row_to_ahead(r)
        gt_lat = expected_lateral(P, i0, rear, heading, ahead)
        rate = 100 * len(vals) / len(frames)
        if len(vals) < 3:
            print("%-5d %6.2f %5.0f%% %8s %7s %8.3f %8s   -"
                  % (r, ahead, rate, "-", "-", gt_lat, "-"))
            continue
        est_lat = float(np.mean(vals) - CAR_CENTER_X) / PX_PER_M_LAT
        sd = float(np.std(vals)) / PX_PER_M_LAT
        err = est_lat - gt_lat
        errs.append((ahead, err))
        src = max(per_src[r].items(), key=lambda kv: kv[1])
        wid = per_width[r]
        wtxt = ("%.0fpx %.0f%%" % (np.mean(wid), 100 * len(wid) / len(frames))) if wid else "-"
        print("%-5d %6.2f %5.0f%% %8.3f %7.3f %8.3f %+8.3f   %-14s %s"
              % (r, ahead, rate, est_lat, sd, gt_lat, err,
                 "%s %.0f%%" % (src[0], 100 * src[1] / max(len(vals), 1)), wtxt))

    if len(errs) >= 4:
        a = np.array([e[0] for e in errs])
        v = np.array([e[1] for e in errs])
        slope, intercept = np.polyfit(a, v, 1)
        resid = v - np.polyval([slope, intercept], a)
        print("\n오차 분해: 상수 %+.3f m, 거리비례 %+.3f m/m, 남는 휨 %.3f m"
              % (intercept, slope, float(np.sqrt(np.mean(resid ** 2)))))

    def show(name, arr, unit):
        if arr:
            print("%-28s %.3f %s (프레임 %d개)" % (name, float(np.mean(arr)), unit, len(arr)))
        else:
            print("%-28s -" % name)

    print()
    show("원본영상 실선 경계 활", cam_bow_lane, "px")
    show("원본영상 중앙선 활", cam_bow_cl, "px")
    show("BEV 행별 중심 활", bev_bow, "m")
    return errs


# ---------------------------------------------------------------- 지점 고르기

def perceived_kappa(centers, near_span=2.6):
    """제어기가 이 인지 결과에서 뽑을 곡률 [1/m]. motion_planner 와 같은 방식.

    오른쪽으로 휘면 양수라 GT 곡률(왼쪽이 양수)과는 부호가 반대다.
    비교할 수 있도록 부호를 뒤집어 돌려준다.
    """
    if len(centers) < 4:
        return float('nan')
    s = np.array([row_to_ahead(r) for r in sorted(centers)])
    d = np.array([(centers[r] - CAR_CENTER_X) / PX_PER_M_LAT for r in sorted(centers)])
    if float(s.max() - s.min()) < 1.8:
        return float('nan')
    near = s <= (s.min() + near_span)
    if near.sum() >= 4:
        s, d = s[near], d[near]
    if len(s) < 3:
        return float('nan')
    return -2.0 * float(np.polyfit(s, d, 2)[0])


def profile_bow(centers):
    """행별 중심이 직선에서 벗어난 정도 [m]."""
    if len(centers) < 4:
        return float('nan')
    ys = np.array(sorted(centers), dtype=float)
    xs = np.array([centers[r] for r in ys], dtype=float)
    _, bow = line_bow(xs, ys)
    return bow / PX_PER_M_LAT


INFRAME_MAX_RESIDUAL = 60
INFRAME_MIN_KEEP = 3


def reject_inframe_outliers(centers):
    """한 프레임 안에서 다른 행들과 어긋나는 행만 버린다.

    행을 아예 목록에서 빼면 그 행밖에 안 보이는 급코너에서 인지가 통째로 비어
    차가 트랙을 벗어난다. 튈 때만 버리면 정보를 굶기지 않는다.
    """
    if len(centers) < 4:
        return centers
    ys = np.array(sorted(centers), dtype=float)
    xs = np.array([centers[y] for y in ys], dtype=float)
    deg = 2 if len(ys) >= 5 else 1
    residual = np.abs(np.polyval(np.polyfit(ys, xs, deg), ys) - xs)
    keep = residual <= INFRAME_MAX_RESIDUAL
    if keep.sum() < INFRAME_MIN_KEEP:
        return centers
    return {int(y): centers[y] for y, k in zip(ys, keep) if k}


# 비교할 설정들. 값은 lane_estimator 의 모듈 상수를 잠시 바꿔 끼운다.
VARIANTS = [
    ("원래",            dict(rows=ROWS, RUN_MIN_WIDTH=40.0)),
    ("470 제외",        dict(rows=tuple(r for r in ROWS if r != 470), RUN_MIN_WIDTH=250.0)),
    ("행이상치제거",     dict(rows=ROWS, RUN_MIN_WIDTH=40.0, inframe=True)),
    ("이상치+조각150",   dict(rows=ROWS, RUN_MIN_WIDTH=150.0, inframe=True)),
    ("이상치+조각250",   dict(rows=ROWS, RUN_MIN_WIDTH=250.0, inframe=True)),
]


def apply_variant(cfg):
    """lane_estimator 모듈 상수를 바꾸고, 되돌릴 값을 남긴다."""
    import camera_perception_pkg.lane_estimator as le
    saved = {}
    for key, val in cfg.items():
        if key in ('rows', 'inframe'):
            continue
        saved[key] = getattr(le, key)
        setattr(le, key, val)
    return saved


def restore_variant(saved):
    import camera_perception_pkg.lane_estimator as le
    for key, val in saved.items():
        setattr(le, key, val)


def compare(node, P, gt, spots, n_frames):
    """여러 지점에서 프레임을 한 번만 모으고, 설정만 바꿔가며 견준다."""
    collected = []
    for x0, y0, label in spots:
        i0 = nearest_index(P, x0, y0)
        x, y = P[i0]
        heading = loop_heading(P, i0)
        node.place(x, y, heading)
        time.sleep(1.2)
        node.place(x, y, heading)
        time.sleep(0.8)
        frames = node.collect(n_frames)
        rear = (x - math.cos(heading) * WHEELBASE / 2,
                y - math.sin(heading) * WHEELBASE / 2)
        kap_gt = float(loop_curvature(P)[i0])
        collected.append((label, i0, rear, heading, kap_gt, frames))
        print("%-10s (%6.2f,%6.2f) GT곡률 %+.4f  프레임 %d개"
              % (label, x, y, kap_gt, len(frames)))

    print("\n지점별 성적 (곡률오차 = 인지가 낸 곡률 - GT곡률, 부호까지 본다)")
    header = "%-12s" % "설정"
    for label, *_ in collected:
        header += " | %-22s" % label
    print(header)
    print("-" * len(header))

    for name, cfg in VARIANTS:
        saved = apply_variant(cfg)
        rows = cfg['rows']
        line = "%-12s" % name
        for label, i0, rear, heading, kap_gt, frames in collected:
            kaps, bows, errs, n_rows, blank = [], [], [], [], 0
            for det in frames:
                centers, _ = node.est.estimate(det.detections, rows)
                if cfg.get('inframe'):
                    centers = reject_inframe_outliers(centers)
                n_rows.append(len(centers))
                if len(centers) < 3:
                    blank += 1
                k = perceived_kappa(centers)
                if np.isfinite(k):
                    kaps.append(k)
                b = profile_bow(centers)
                if np.isfinite(b):
                    bows.append(b)
                for r, cx in centers.items():
                    gl = expected_lateral(P, i0, rear, heading, row_to_ahead(r))
                    if np.isfinite(gl):
                        errs.append(abs((cx - CAR_CENTER_X) / PX_PER_M_LAT - gl))
            line += " | k%+.3f 활%.2f 오차%.2f 행%.1f 공백%.0f%%" % (
                (np.mean(kaps) - kap_gt) if kaps else float('nan'),
                np.mean(bows) if bows else float('nan'),
                np.mean(errs) if errs else float('nan'),
                np.mean(n_rows) if n_rows else 0.0,
                100.0 * blank / max(len(frames), 1))
        restore_variant(saved)
        print(line)


def scan(P):
    """곧은 곳과 굽은 곳을 뽑아 어디서 측정할지 고른다."""
    kap = loop_curvature(P)
    print("차로 중앙선 %d점 (%.2fm 간격, 한 바퀴 %.1f m)" % (len(P), DS, DS * len(P)))
    print("\n%-22s %10s %9s  %s" % ("위치 (x, y)", "곡률1/m", "진행deg", "구분"))
    step = int(2.0 / DS)
    for i in range(0, len(P), step):
        k = kap[i]
        kind = "직진" if abs(k) < 0.012 else ("완만" if abs(k) < 0.045 else "코너")
        print("(%7.2f, %7.2f)      %+9.4f %9.1f  %s"
              % (P[i, 0], P[i, 1], k, math.degrees(loop_heading(P, i)), kind))


def main():
    gt = lane_gt.build()
    # 진행 방향은 GT 에서 읽는다. 사람이 눈대중으로 준 각도를 쓰면 GT 쪽에
    # 인위적 기울기가 섞여 측정 자체가 무의미해진다.
    # 스폰 지점에서 +x 로 출발하는 것이 실제 주행 방향이다 (반시계 폐곡선).
    P = trace_loop(gt)

    if len(sys.argv) > 1 and sys.argv[1] == "--scan":
        scan(P)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        # 뒤의 두 곳은 수정본이 실제로 이탈한 지점이다. 인지가 비지 않는지 본다.
        spots = [(17.35, -6.36, "직선(우측)"),
                 (0.56, 23.14, "직선(상단)"),
                 (-15.13, 12.50, "좌코너"),
                 (-13.22, 7.01, "우완만"),
                 (10.22, 21.34, "이탈지점A"),
                 (-17.77, -11.96, "이탈지점B")]
        rclpy.init()
        node = Probe()
        try:
            compare(node, P, gt, spots, n_frames)
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return

    if len(sys.argv) < 3:
        print(__doc__)
        return

    x0, y0 = float(sys.argv[1]), float(sys.argv[2])
    n_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    i0 = nearest_index(P, x0, y0)

    rclpy.init()
    node = Probe()
    try:
        measure(node, P, i0, gt, n_frames)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
