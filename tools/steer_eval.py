#!/usr/bin/env python3
"""GT 경로의 곡률로 '이상적인 조향값'을 만들고 실제 조향과 비교한다.

GT 는 제어에 쓰지 않는다. 평가에만 쓴다.

트랙 텍스처에서 자차 차로 중앙선을 뽑고, 차량 위치에서 가장 가까운 점의
곡률을 구하면, 그 자리에서 내야 할 조향은 정상선회식으로 정해진다.

    이상 조향 [step] = atan(축간거리 * 곡률) / (최대조향 / 7)

이걸 실제 조향과 견주면 세 가지가 한 번에 보인다.
  - 치우침: 평균적으로 덜 꺾는지 더 꺾는지
  - 시점  : 이상값보다 앞서 꺾는지 늦게 꺾는지 (상호상관의 지연)
  - 안정성: 이상값은 매끄러운데 실제가 오르내리면 '했다 풀었다' 하는 것

    python3 tools/steer_eval.py [로그.csv]
"""

import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_gt

WHEELBASE = 2.86
MAX_STEER = 0.6458
RAD_PER_STEP = MAX_STEER / 7.0


def gt_centerline(rows, gt):
    """주행 궤적을 GT 차로 중앙으로 밀어 옮겨 순서 있는 중앙선을 얻는다."""
    dist_out, dist_in, _, w, h, _ = gt
    gy, gx = np.gradient(dist_out)
    P = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    out = P.copy()
    for _ in range(8):
        cu, cv = lane_gt.to_px(out[:, 0], out[:, 1], w, h)
        step = (dist_in[cv, cu] - dist_out[cv, cu]) / 2.0     # 차로 중앙 쪽으로
        gux, guy = gx[cv, cu], gy[cv, cu]
        n = np.hypot(gux, guy)
        ok = n > 1e-6
        out[ok, 0] += step[ok] * (guy[ok] / n[ok])
        out[ok, 1] += step[ok] * (gux[ok] / n[ok])
    return out


def smooth_curvature(P, win=41):
    """중앙선의 부호 있는 곡률 [1/m]. 오른쪽으로 휘면 양수.

    표본이 0.2초(약 0.45m) 간격이라, 곡률을 그대로 미분하면 궤적 노이즈가
    그대로 증폭된다. 길의 곡률은 천천히 변하므로 넉넉히 평활한다.
    win=41 이면 약 18m 구간을 본다.
    """
    k = np.ones(win) / win
    Q = P.copy()
    for i in (0, 1):
        Q[:, i] = np.convolve(np.r_[np.full(win, P[0, i]), P[:, i], np.full(win, P[-1, i])],
                              k, 'same')[win:-win]
    dx, dy = np.gradient(Q[:, 0]), np.gradient(Q[:, 1])
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5
    return np.where(denom > 1e-9, (dx * ddy - dy * ddx) / np.maximum(denom, 1e-9), 0.0)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lap_trajectory.csv"
    gt = lane_gt.build()
    rows = [r for r in csv.DictReader(open(path)) if int(r["speed"]) > 0]
    if len(rows) < 50:
        print("주행 구간이 부족하다"); return

    P = gt_centerline(rows, gt)
    kap = smooth_curvature(P)

    # 진행방향에 대한 부호를 맞춘다. 곡률 부호는 좌표계 회전 방향을 따르므로,
    # 조향 부호 규약(오른쪽이 양수)에 맞추려면 진행방향 기준으로 다시 본다.
    ideal = np.arctan(WHEELBASE * -kap) / RAD_PER_STEP

    steer = np.array([float(r["steering"]) for r in rows])
    target = np.array([float(r["target_steer"]) if r["target_steer"] != "nan" else np.nan
                       for r in rows])

    ok = np.isfinite(ideal) & np.isfinite(steer)
    err = steer[ok] - ideal[ok]
    print("표본 %d개" % ok.sum())
    print("\n[조향 정확도] 이상 조향 대비")
    print("  평균 오차 %+.2f step (%+.1f 도)   %s"
          % (err.mean(), err.mean() * math.degrees(RAD_PER_STEP),
             "덜 꺾음" if abs(err.mean()) > 0.15 and err.mean() * np.sign(ideal[ok].mean()) < 0 else ""))
    print("  평균|오차| %.2f step, 90분위 %.2f step" % (np.abs(err).mean(), np.percentile(np.abs(err), 90)))

    # 시점: 실제 조향을 앞뒤로 밀어보며 이상값과 가장 잘 맞는 지연을 찾는다
    best = None
    for lag in range(-15, 16):                      # 표본 0.2초 간격
        a = steer[ok]; b = ideal[ok]
        if lag > 0:
            a2, b2 = a[lag:], b[:-lag]
        elif lag < 0:
            a2, b2 = a[:lag], b[-lag:]
        else:
            a2, b2 = a, b
        if len(a2) < 50:
            continue
        c = np.corrcoef(a2, b2)[0, 1]
        if best is None or c > best[1]:
            best = (lag, c)
    print("\n[조향 시점] 지연 %+.1f 초에서 상관 최대 (%.2f)"
          % (best[0] * 0.2, best[1]))
    print("  %s" % ("음수 = 이상값보다 먼저 꺾는다 (코너 진입 전 파고듦)"
                    if best[0] < 0 else "양수 = 이상값보다 늦게 꺾는다"))

    # 안정성: 이상값은 매끄러운데 실제가 오르내리는 정도
    d_ideal = np.abs(np.diff(ideal[ok]))
    d_steer = np.abs(np.diff(steer[ok]))
    print("\n[조향 안정성] 표본 간 변화량")
    print("  이상 조향 평균 %.3f step, 실제 조향 평균 %.3f step (%.1f 배)"
          % (d_ideal.mean(), d_steer.mean(), d_steer.mean() / max(d_ideal.mean(), 1e-6)))
    # 방향이 뒤집히는 횟수 = '했다 풀었다'
    def reversals(v):
        d = np.diff(v)
        s = np.sign(d[np.abs(d) > 0.5])
        return int(np.sum(s[1:] * s[:-1] < 0))
    print("  방향 반전: 이상 %d회, 실제 %d회" % (reversals(ideal[ok]), reversals(steer[ok])))

    if np.isfinite(target).any():
        t = target[ok & np.isfinite(target)]
        i2 = ideal[ok & np.isfinite(target)]
        print("\n[연속 조향값(target_steer) 기준]")
        print("  평균 오차 %+.2f step, 평균|오차| %.2f step"
              % ((t - i2).mean(), np.abs(t - i2).mean()))
        print("  표본 간 변화량 평균 %.3f step (실제 정수 조향 %.3f)"
              % (np.abs(np.diff(t)).mean(), d_steer.mean()))


if __name__ == "__main__":
    main()
