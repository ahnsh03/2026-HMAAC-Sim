#!/usr/bin/env bash
# N 랩을 달리고 지상진실 기준으로 채점한다.
#
#   ./tools/run_laps.sh [랩수] [이름]
#
# 랩 사이 편차가 커서 (같은 설정에서 밟음 6.5~13.5%) 한 번의 결과로 판단하면
# 안 된다. 같은 설정을 두 번 이상 돌려 견주는 것을 전제로 만들었다.

WS="${WS:-/root/hmobility_ws}"
LAPS="${1:-3}"
NAME="${2:-run}"
TIMEOUT="${TIMEOUT:-400}"
LOG=/tmp/driving_sim.log
OUT="/tmp/laps_${NAME}.csv"

"${WS}/tools/run_sim.sh" -d > /dev/null

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "${WS}/install/local_setup.bash"

echo "[${NAME}] ${LAPS}랩 주행 시작 (최대 ${TIMEOUT}초)"
start=$(date +%s)
while true; do
    if grep -q "랩 ${LAPS} 완주" "${LOG}" 2>/dev/null; then
        echo "[${NAME}] ${LAPS}랩 완주"
        break
    fi
    if [ $(($(date +%s) - start)) -gt "${TIMEOUT}" ]; then
        echo "[${NAME}] 시간 초과. 그때까지의 주행으로 채점한다"
        break
    fi
    sleep 3
done

sleep 1
cp /tmp/lap_trajectory.csv "${OUT}" 2>/dev/null
killall -9 gazebo gzserver gzclient 2>/dev/null
sleep 1

grep -E "완주|정지 상태" "${LOG}" | tail -5
echo
cd "${WS}" && python3 tools/lane_gt.py "${OUT}"
echo
python3 tools/diagnose.py "${OUT}" 2>/dev/null | sed -n '/곡률 추정 품질/,/상한 도달/p'
