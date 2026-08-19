#!/usr/bin/env python3
"""주행 로그의 곡률 추정을 지상진실 곡률과 구간별로 견준다.

직진에서만 보면 "편향이 줄었다"는 결론이 나오지만, 그 편향이 코너 조향을
공급하고 있었다면 직진만 고쳐서는 차가 코너에서 바깥으로 밀린다.
그래서 곡률 크기별로 나눠 본다.

부호: 제어기 곡률은 오른쪽이 양수, GT 곡률은 왼쪽이 양수라 뒤집어 맞춘다.

    python3 tools/kappa_eval.py 로그1.csv [로그2.csv ...]
"""

import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_gt          # noqa: E402
import lane_row_gt      # noqa: E402

WHEELBASE = 2.86
RAD_PER_STEP = 0.6458 / 7
# 조향을 명령해도 그만큼 돌지 않는다. 실측한 요레이트 이득.
ACTUATOR_GAIN = 0.71


def analyse(path, P, kap_gt_all):
    rows = [r for r in csv.DictReader(open(path)) if int(r["speed"]) > 0]
    if not rows:
        print("%-14s 주행 구간 없음" % os.path.basename(path))
        return

    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    kap_ctrl = np.array([float(r["kappa"]) for r in rows])

    # 제어기는 전방 주시점 부근의 곡률을 쓰므로, GT 도 그만큼 앞에서 읽어야 한다.
    # 차량 위치의 곡률과 견주면 코너 진입/탈출이 통째로 엉뚱한 구간으로 분류된다.
    ld = np.array([float(r["ld"]) for r in rows])
    ld = np.where(np.isfinite(ld), ld, 5.0)
    n = len(P)
    idx = [(int(np.argmin((P[:, 0] - px) ** 2 + (P[:, 1] - py) ** 2))
            + int(round(l / lane_row_gt.DS))) % n
           for px, py, l in zip(x, y, ld)]
    kap_gt = -kap_gt_all[idx]

    ok = np.isfinite(kap_ctrl)
    kap_ctrl, kap_gt = kap_ctrl[ok], kap_gt[ok]

    print("\n%s  (표본 %d개)" % (os.path.basename(path), len(kap_ctrl)))
    print("  %-14s %6s %10s %10s %10s %10s"
          % ("구간", "표본", "GT곡률", "추정곡률", "부족분", "조향부족"))
    bands = [("직진 |k|<.02", np.abs(kap_gt) < 0.02),
             ("완만 .02~.06", (np.abs(kap_gt) >= 0.02) & (np.abs(kap_gt) < 0.06)),
             ("코너 |k|>=.06", np.abs(kap_gt) >= 0.06)]
    for label, sel in bands:
        if sel.sum() < 5:
            print("  %-14s %6d %10s" % (label, sel.sum(), "-"))
            continue
        g, c = kap_gt[sel].mean(), kap_ctrl[sel].mean()
        # 곡률 부족이 조향에서 몇 칸에 해당하는지
        lack = (math.atan(WHEELBASE * abs(g)) - math.atan(WHEELBASE * abs(c))) / RAD_PER_STEP
        print("  %-14s %6d %+10.4f %+10.4f %+10.4f %+9.2f칸"
              % (label, sel.sum(), g, c, c - g, lack))

    corner = np.abs(kap_gt) >= 0.06
    if corner.sum() >= 5:
        ratio = np.abs(kap_ctrl[corner]).mean() / max(np.abs(kap_gt[corner]).mean(), 1e-9)
        print("  코너에서 필요한 곡률의 %.0f%% 만 내고 있다" % (100 * ratio))

    # 조향 명령이 실제로 얼마나 나갔는지. 곡률이 맞아도 조향이 모자라면 밀린다.
    # 필요 조향은 정상선회 atan(축간거리*곡률) 을 액추에이터 이득으로 나눈 값이다.
    steer = np.array([float(r["steer_cont"]) for r in rows])[ok]
    need = np.array([math.atan(WHEELBASE * abs(k)) / RAD_PER_STEP / ACTUATOR_GAIN
                     for k in kap_gt])
    print("  %-14s %10s %10s %10s" % ("구간", "필요조향", "실제조향", "비율"))
    for label, sel in bands:
        if sel.sum() < 5:
            continue
        n, a = need[sel].mean(), np.abs(steer[sel]).mean()
        print("  %-14s %9.2f칸 %9.2f칸 %9.0f%%" % (label, n, a, 100 * a / max(n, 1e-9)))


def main():
    gt = lane_gt.build()
    P = lane_row_gt.trace_loop(gt)
    kap_gt_all = lane_row_gt.loop_curvature(P)
    for path in sys.argv[1:]:
        analyse(path, P, kap_gt_all)


if __name__ == "__main__":
    main()
