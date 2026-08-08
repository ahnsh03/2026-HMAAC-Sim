#!/usr/bin/env bash
# 인지·제어 파이프라인 각 단계의 실제 발행 주기를 잰다.
# 이 값이 낼 수 있는 주행 속도의 상한을 결정한다.

WS="${WS:-/root/hmobility_ws}"
WINDOW="${WINDOW:-8}"

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "${WS}/install/local_setup.bash"

for topic in /camera/image_raw /detections /yolov8_lane_info /path_planning_result /topic_control_signal /cmd_vel; do
    printf '%-26s ' "${topic}"
    timeout "${WINDOW}" ros2 topic hz "${topic}" 2>/dev/null \
        | grep -m1 'average rate' \
        || echo "no data"
done
