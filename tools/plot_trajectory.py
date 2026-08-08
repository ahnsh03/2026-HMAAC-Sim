#!/usr/bin/env python3
"""lap_monitor_node 가 남긴 궤적 CSV 를 그림으로 그린다.

    python3 tools/plot_trajectory.py [csv] [png]

조향 크기를 색으로 표시해 어느 코너에서 크게 꺾었는지 함께 본다.
"""

import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lap_trajectory.csv"
PNG_PATH = sys.argv[2] if len(sys.argv) > 2 else "/tmp/lap_trajectory.png"


def main():
    t, x, y, steering, speed = [], [], [], [], []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            t.append(float(row['t']))
            x.append(float(row['x']))
            y.append(float(row['y']))
            steering.append(int(row['steering']))
            speed.append(int(row['speed']))

    fig, (ax_map, ax_ctrl) = plt.subplots(1, 2, figsize=(14, 6))

    sc = ax_map.scatter(x, y, c=[abs(s) for s in steering], cmap='viridis', s=6)
    ax_map.plot(x[0], y[0], 'r*', markersize=16, label='start')
    ax_map.set_aspect('equal')
    ax_map.set_xlabel('x [m]')
    ax_map.set_ylabel('y [m]')
    ax_map.set_title('trajectory (color = |steering|)')
    ax_map.legend()
    ax_map.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax_map)

    ax_ctrl.plot(t, steering, label='steering')
    ax_ctrl.plot(t, [s / 10.0 for s in speed], label='speed / 10')
    ax_ctrl.set_xlabel('t [s]')
    ax_ctrl.set_title('control history')
    ax_ctrl.legend()
    ax_ctrl.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=110)
    print(f"saved: {PNG_PATH}  (samples={len(t)}, duration={t[-1]:.1f}s)")


if __name__ == '__main__':
    main()
