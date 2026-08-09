#!/usr/bin/env python3
"""기준 차선과 주행 궤적을 트랙 그림 위에 그려 저장한다.

숫자만 보면 어디서 어떻게 벗어났는지 알 수 없다. 눈으로 확인하는 용도다.
기준은 `lane_gt` 와 같은 것을 쓰므로 `tools/lane_gt.py` 의 수치와 일치한다.

    노랑   바깥쪽 실선 (잔디 경계)
    마젠타 점선 중앙선
    파랑   자차 차로 중앙 (두 경계에서 같은 거리)
    초록→빨강  주행 궤적. 차체가 차선에 닿는 정도로 색이 변한다
    빨간 원+화살표  스폰 지점과 진행 방향

    python3 tools/gt_viz.py [로그.csv] [출력.png]
"""

import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_gt

OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "lane_reference.png")
SCALE = 3                                  # 보기 좋게 확대
SPAWN = (-5.092991, -22.694952)            # 012_deploy_lib.driving_ego()


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lap_trajectory.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else OUT_DEFAULT

    gt = lane_gt.build()
    dist_out, dist_in, m_per_px, w, h, grass = gt
    img = cv2.imread(os.path.normpath(lane_gt.TEX))
    big = cv2.resize(img, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)

    def dots(mask, color, rad):
        ys, xs = np.where(mask)
        for x, y in zip(xs, ys):
            cv2.circle(big, (int(x) * SCALE, int(y) * SCALE), rad, color, -1)

    on_road = grass == 0
    dots((dist_out > 0) & (dist_out < 0.12), (0, 220, 255), 1)          # 바깥 실선
    dots((dist_in < 0.10) & on_road, (255, 0, 255), 2)                  # 점선 중앙선
    dots((np.abs(dist_out - dist_in) < 0.06) & on_road, (255, 80, 0), 2)  # 차로 중앙

    rows = [r for r in csv.DictReader(open(log)) if int(r["speed"]) > 0]
    if rows:
        x = np.array([float(r["x"]) for r in rows])
        y = np.array([float(r["y"]) for r in rows])
        yaw = np.array([float(r["yaw"]) for r in rows])
        c_in, c_out = lane_gt.body_clearance(x, y, yaw, gt)
        clearance = np.minimum(c_in, c_out)          # 가장 가까운 차선까지의 여유
        cu, cv_ = lane_gt.to_px(x, y, w, h)
        for i in range(len(x)):
            # 여유 0.5m 이상이면 초록, 0 이면 빨강
            a = 1.0 - float(np.clip(clearance[i] / 0.5, 0.0, 1.0))
            cv2.circle(big, (int(cu[i]) * SCALE, int(cv_[i]) * SCALE), 3,
                       (0, int(255 * (1 - a)), int(255 * a)), -1)
        touching = (clearance <= 0).mean()
        print("궤적 표본 %d개, 차선 밟음 %.1f%%, 차로 중앙 이탈 평균 %.2f m"
              % (len(x), 100 * touching, np.abs(lane_gt.evaluate(x, y, gt)[0]).mean()))

    # 스폰 지점과 진행 방향
    su, sv = lane_gt.to_px(np.array([SPAWN[0]]), np.array([SPAWN[1]]), w, h)
    sx, sy = int(su[0]) * SCALE, int(sv[0]) * SCALE
    du, dv = lane_gt.to_px(np.array([SPAWN[0] + 4.0]), np.array([SPAWN[1]]), w, h)
    cv2.circle(big, (sx, sy), 14, (255, 255, 255), 3)
    cv2.circle(big, (sx, sy), 5, (0, 0, 255), -1)
    cv2.arrowedLine(big, (sx, sy), (int(du[0]) * SCALE, int(dv[0]) * SCALE),
                    (0, 0, 255), 3, tipLength=0.3)

    for i, (text, color) in enumerate([
            ("yellow = outer solid line", (0, 220, 255)),
            ("magenta = dashed center line", (255, 0, 255)),
            ("blue = ego lane center (reference)", (255, 80, 0)),
            ("green->red = driven path, clearance 0.5m -> 0m", (0, 200, 0)),
            ("red circle = spawn point and heading", (0, 0, 255))]):
        pos = (12, 28 + 24 * i)
        cv2.putText(big, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(big, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, big)
    print("저장:", out)


if __name__ == "__main__":
    main()
