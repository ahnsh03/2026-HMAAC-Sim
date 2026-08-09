#!/usr/bin/env python3
"""GT 차로 중앙선에서 곡률을 재고, 구간별 적정 속도와 달성 가능한 랩타임을 계산한다.

방법은 레이싱라인 문헌의 표준적인 forward-backward pass 다
(Velenis & Tsiotras 2005, "Optimal velocity profile generation" 계열).

  1) 각 지점의 곡률 kappa 로 정상선회 속도 상한을 구한다
         v_curve = sqrt(a_lat / |kappa|)
     여기에 조향 한계도 함께 건다. 조향 delta 로 낼 수 있는 최소 반경은
     R = 축간거리 / tan(delta) 이므로, |kappa| > tan(delta_max)/L 인 길은
     아예 돌 수 없다.
  2) 뒤로 한 번 훑어 제동 한계를 반영한다 (코너 앞에서 미리 줄이도록).
  3) 앞으로 한 번 훑어 구동 한계를 반영한다 (코너 탈출 후 서서히 올리도록).

이렇게 나온 프로파일이 물리적으로 가능한 최속이고, 랩타임의 하한이 된다.
실제로는 인지 지연이 있으므로 여기에 여유를 두고 쓴다.
"""

import os
import sys

import cv2
import numpy as np

TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "src/simulation_pkg/models/race_track/materials/textures/basic_track_2026.jpg")
SIZE_LX, SIZE_LY = 53.83, 40.473
HALF_LANE = 1.70          # 잔디 경계에서 자차 차로 중앙까지 [m]

# 차량 제원 (models/prius_hybrid/model.sdf)
WHEELBASE = 2.86          # 축간거리 [m]
MAX_STEER = 0.6458        # 최대 조향각 [rad] (steering=7)
MAX_SPEED = 5.0           # 플러그인 상한 [m/s] (config.py MAX_SPEED)

# 주행 여유
A_LAT = 3.0               # 허용 횡가속도 [m/s^2]
A_ACC = 1.6               # 가속 [m/s^2]
A_DEC = 2.2               # 감속 [m/s^2]
STEER_MARGIN = 0.75       # 조향 여유. 최대 조향의 이만큼만 정상선회에 쓴다


def centerline(traj_csv="/tmp/lap_trajectory.csv"):
    """GT 자차 차로 중앙선을 순서 있는 폐곡선으로 뽑는다.

    거리변환의 등고선을 그대로 따라가려 하면 순서를 잡기 어렵다.
    대신 이미 주행한 궤적(순서가 있다)을 GT 위로 밀어 옮긴다.
    거리변환의 기울기는 잔디에서 멀어지는 방향이므로, 각 점을
    (HALF_LANE - 현재거리) 만큼 그 방향으로 옮기면 차로 중앙에 놓인다.
    """
    import csv
    img = cv2.imread(os.path.normpath(TEX))
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (35, 60, 40), (90, 255, 255))
    m_per_px = (SIZE_LX / w + SIZE_LY / h) / 2
    dist_m = cv2.distanceTransform((grass == 0).astype(np.uint8), cv2.DIST_L2, 5) * m_per_px
    gy, gx = np.gradient(dist_m)          # 픽셀 단위 기울기

    rows = [r for r in csv.DictReader(open(traj_csv)) if int(r["speed"]) > 0]
    P = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    # 한 바퀴만 쓴다: 출발점 근처로 돌아온 첫 지점까지
    start = P[0]
    far = np.where(np.linalg.norm(P - start, axis=1) > 10.0)[0]
    if len(far):
        back = np.where(np.linalg.norm(P[far[0]:] - start, axis=1) < 3.0)[0]
        if len(back):
            P = P[:far[0] + back[0] + 1]

    def to_px(wx, wy):
        u = (wy + SIZE_LX / 2) / SIZE_LX * (w - 1)
        v = (1 - (-wx + SIZE_LY / 2) / SIZE_LY) * (h - 1)
        return np.clip(u, 0, w - 1).astype(int), np.clip(v, 0, h - 1).astype(int)

    out = P.copy()
    for _ in range(4):                    # 몇 번 반복하면 등고선에 수렴한다
        cu, cv_ = to_px(out[:, 0], out[:, 1])
        d = dist_m[cv_, cu]
        gux, guy = gx[cv_, cu], gy[cv_, cu]
        n = np.hypot(gux, guy)
        ok = n > 1e-6
        # 픽셀 기울기 -> 월드 방향. u 는 +wy, v 는 -wx 방향이다.
        dirx = np.where(ok, -(-guy / np.maximum(n, 1e-9)), 0.0)
        diry = np.where(ok, gux / np.maximum(n, 1e-9), 0.0)
        step = (HALF_LANE - d)
        out[:, 0] += step * dirx
        out[:, 1] += step * diry
    return out


def resample(P, step=0.25):
    """등간격으로 다시 뽑고 살짝 평활한다."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = np.arange(0, d[-1], step)
    Q = np.stack([np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1])], axis=1)
    k = 21
    ker = np.ones(k) / k
    for i in (0, 1):                      # 폐곡선이므로 순환 평활
        Q[:, i] = np.convolve(np.r_[Q[-k:, i], Q[:, i], Q[:k, i]], ker, 'same')[k:-k]
    return Q, s


def curvature(Q):
    dx = np.gradient(Q[:, 0]); dy = np.gradient(Q[:, 1])
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5
    return np.where(denom > 1e-9, (dx * ddy - dy * ddx) / np.maximum(denom, 1e-9), 0.0)


def profile(kap, ds, v_cap):
    """정상선회 상한 -> 뒤로 훑어 제동 -> 앞으로 훑어 구동."""
    k = np.maximum(np.abs(kap), 1e-6)
    v = np.minimum(np.sqrt(A_LAT / k), v_cap)
    n = len(v)
    for _ in range(2):                    # 폐곡선이라 두 바퀴 돌려 수렴시킨다
        for i in range(n - 2, -1, -1):    # 제동
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * A_DEC * ds))
        v[-1] = min(v[-1], np.sqrt(v[0] ** 2 + 2 * A_DEC * ds))
        for i in range(1, n):             # 구동
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * A_ACC * ds))
        v[0] = min(v[0], np.sqrt(v[-1] ** 2 + 2 * A_ACC * ds))
    return v


def main():
    v_cap = float(sys.argv[1]) if len(sys.argv) > 1 else MAX_SPEED
    P = centerline()
    Q, _ = resample(P)
    ds = 0.25
    kap = curvature(Q)
    R = 1.0 / np.maximum(np.abs(kap), 1e-6)

    # 조향으로 낼 수 있는 최소 반경
    r_min = WHEELBASE / np.tan(MAX_STEER * STEER_MARGIN)
    print("트랙 길이 %.1f m (자차 차로 중앙선)" % (len(Q) * ds))
    print("곡률: 최대 %.3f 1/m (반경 %.1f m), 90분위 %.3f (반경 %.1f m)"
          % (np.abs(kap).max(), R.min(), np.percentile(np.abs(kap), 90),
             1 / np.percentile(np.abs(kap), 90)))
    print("조향 %.0f%% 로 낼 수 있는 최소 반경 %.1f m -> %s"
          % (STEER_MARGIN * 100, r_min,
             "여유 있음" if R.min() > r_min else "부족! 최대조향 필요"))

    v = profile(kap, ds, v_cap)
    t = np.sum(ds / v)
    print("\n[속도 상한 %.1f m/s, 횡가속 %.1f, 가속 %.1f, 감속 %.1f]" % (v_cap, A_LAT, A_ACC, A_DEC))
    print("  속도 범위 %.2f ~ %.2f m/s (평균 %.2f)" % (v.min(), v.max(), len(Q) * ds / t))
    print("  이론 랩타임 %.1f 초" % t)

    # 곡률이 큰 구간을 코너로 묶어 보고한다
    corner = np.abs(kap) > 0.03
    groups, cur = [], None
    for i, c in enumerate(corner):
        if c and cur is None:
            cur = i
        elif not c and cur is not None:
            groups.append((cur, i)); cur = None
    if cur is not None:
        groups.append((cur, len(corner)))
    print("\n코너별 (곡률>0.03, 즉 반경 33m 이하):")
    print("  %-22s %8s %8s %8s" % ("위치 (x, y)", "길이m", "최소R", "권장v"))
    for a, b in groups:
        if (b - a) * ds < 3.0:
            continue
        mid = (a + b) // 2
        print("  (%6.1f,%6.1f)         %6.1f  %7.1f  %6.2f"
              % (Q[mid, 0], Q[mid, 1], (b - a) * ds, R[a:b].min(), v[a:b].min()))


if __name__ == "__main__":
    main()
