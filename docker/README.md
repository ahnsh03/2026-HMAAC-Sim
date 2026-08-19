# Docker (ROS 2 Humble + Gazebo Classic)

호스트가 Ubuntu 26.04 (WSL2) 처럼 Humble 공식 지원 대상이 아닐 때, 이 레포를
컨테이너 안에서 빌드·실행하기 위한 환경이다. `install.sh` 를 호스트에서 돌리는 대신
같은 내용을 이미지에 굳혀 둔 것이라 **`install.sh` 를 컨테이너에서 다시 돌릴 필요는 없다.**

| 서비스 | 이미지 | 용도 |
|--------|--------|------|
| `hmobility-sim` | `hmobility-sim:humble` | 기본. CPU torch |
| `hmobility-sim-gpu` | `hmobility-sim:humble-cuda` | YOLO 추론이 병목일 때. CUDA torch + `gpus: all` |

두 컨테이너는 **같은 워크스페이스(`..`)를 마운트**하므로 동시에 쓰지 말 것 — `install/` 이 충돌한다.

## 기동

```bash
xhost +local:docker
cd docker
docker compose build hmobility-sim      # 첫 1회
docker compose up -d hmobility-sim
docker exec -it hmobility-sim bash
```

레포 루트가 컨테이너의 `/root/hmobility_ws` 로 마운트된다. 편집은 호스트에서, 빌드·실행은 컨테이너에서 한다.

## 빌드·실행 (컨테이너 안)

```bash
export AMENT_PREFIX_PATH=''
export CMAKE_PREFIX_PATH=''
source /opt/ros/humble/setup.bash

mkdir -p ~/.gazebo/models
cp -r /root/hmobility_ws/src/simulation_pkg/models/* ~/.gazebo/models/
rm -rf ~/.gazebo/models/etc

sudo rosdep init 2>/dev/null || true
rosdep update
rosdep install -i --from-path src --rosdistro humble -y

colcon build --packages-select interfaces_pkg --allow-overriding interfaces_pkg && source install/local_setup.bash
colcon build --symlink-install --packages-select camera_perception_pkg --allow-overriding camera_perception_pkg && source install/local_setup.bash
colcon build --symlink-install --packages-select decision_making_pkg --allow-overriding decision_making_pkg && source install/local_setup.bash
colcon build --symlink-install --packages-select debug_pkg --allow-overriding debug_pkg && source install/local_setup.bash
colcon build --symlink-install --packages-select simulation_pkg --allow-overriding simulation_pkg && source install/local_setup.bash
colcon build --symlink-install --packages-select lidar_perception_pkg --allow-overriding lidar_perception_pkg && source install/local_setup.bash
colcon build --packages-select plugin_pkg && source install/local_setup.bash

killall -9 gzserver gzclient 2>/dev/null || true
ros2 launch simulation_pkg driving_sim.launch.py
```

이미지 안에 `qqq`(gzserver 강제 종료), `MOVE`/`STOP`(`/go` 서비스 호출) 이 alias 로 들어 있다.

## 종료

```bash
docker compose down
xhost -local:docker
```

## 실차 실습 레포와 함께 쓸 때

실차 레포([ahnsh03/2026-HMAAC](https://github.com/ahnsh03/2026-HMAAC))는 **자기 레포 안에 자기 Docker 환경**을 갖고 있다.
패키지 이름(`camera_perception_pkg` 등)이 겹치므로 **한 컨테이너에 두 레포를 섞어 마운트하지 말 것.**
`network_mode: host` 라 동시 기동 시 토픽이 충돌하니 `ROS_DOMAIN_ID` 를 나눈다 — 시뮬 `0`, 실차 `1` (각 compose 기본값).
