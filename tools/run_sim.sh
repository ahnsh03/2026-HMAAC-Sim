#!/usr/bin/env bash
# 컨테이너(hmobility-sim) 안에서 driving_sim 을 깨끗하게 재실행한다.
#
#   ./tools/run_sim.sh                          # 포그라운드 실행
#   ./tools/run_sim.sh -d                       # 백그라운드 실행 (로그: /tmp/driving_sim.log)
#   ./tools/run_sim.sh -d drive_enable:=false   # 런치 인자 전달
#   LAUNCH=mission_sim ./tools/run_sim.sh
#
# ROS 의 setup.bash 는 미정의 변수를 참조하므로 `set -u` 를 쓰지 않는다.

WS="${WS:-/root/hmobility_ws}"
LAUNCH="${LAUNCH:-driving_sim}"
LOG="${LOG:-/tmp/driving_sim.log}"
DETACH=0
if [ "${1:-}" = "-d" ]; then
    DETACH=1
    shift
fi
LAUNCH_ARGS=("$@")

# 이전 실행 잔재 정리. pkill 패턴이 자기 자신을 잡지 않도록 launch 는 이름으로 지운다.
# 노드가 살아남으면 같은 토픽에 두 번 발행하거나 lap_trajectory.csv 를 서로 덮어써서
# 결과가 뒤섞이므로, 죽을 때까지 확인한다.
killall -9 gazebo gzserver gzclient rqt 2>/dev/null
pkill -9 -f "ros2 launch simulation_pkg" 2>/dev/null
for _ in 1 2 3 4 5; do
    pkill -9 -f "hmobility_ws/install/.*/lib/.*_node" 2>/dev/null
    sleep 0.5
    pgrep -f "hmobility_ws/install/.*/lib/.*_node" > /dev/null || break
done
if pgrep -f "hmobility_ws/install/.*/lib/.*_node" > /dev/null; then
    echo "경고: 이전 노드가 남아 있다" >&2
    pgrep -af "hmobility_ws/install/.*/lib/.*_node" >&2
fi
sleep 1

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "${WS}/install/local_setup.bash"

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}"
mkdir -p "${XDG_RUNTIME_DIR}"

# 모델은 전부 ~/.gazebo/models 에 있다. 온라인 DB 를 켜두면 gzserver 가 이걸 받으러
# 30초를 블로킹하고, 그 사이 factory 로 스폰된 ego_vehicle 의 센서 초기화가 타임아웃되어
# /camera/image_raw 가 영영 나오지 않는다.
export GAZEBO_MODEL_DATABASE_URI=""

cd "${WS}" || exit 1

if [ "${DETACH}" -eq 1 ]; then
    nohup ros2 launch simulation_pkg "${LAUNCH}.launch.py" "${LAUNCH_ARGS[@]}" > "${LOG}" 2>&1 &
    echo "launched ${LAUNCH} ${LAUNCH_ARGS[*]} (pid $!), log: ${LOG}"
else
    exec ros2 launch simulation_pkg "${LAUNCH}.launch.py" "${LAUNCH_ARGS[@]}"
fi
