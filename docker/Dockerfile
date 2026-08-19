# ROS 2 Humble + Gazebo Classic (Ubuntu 22.04) — 시뮬 레포 전용 개발 이미지
#
# 호스트가 Ubuntu 26.04 (WSL2) 라 Humble 을 네이티브로 깔 수 없어 컨테이너로 돌린다.
# apt 목록은 src/*/package.xml 의 선언과 install.sh 를 따른다.
#
#   cd docker && docker compose build hmobility-sim && docker compose up -d hmobility-sim
FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV QT_X11_NO_MITSHM=1
ENV ROS_DOMAIN_ID=0

# rviz2 / urdf / robot_state_publisher / image_transport / tf2_ros 는 desktop-full 에 포함.
# 여기서 채우는 것은 desktop-full 에 없는 Gazebo Classic 연동과 ros2_control 스택이다.
#
# 주의: package.xml 에 ros_gz_bridge / ros_gz_sim (Gazebo Ignition 계열) 도 선언돼 있으나
# 실제 실행 환경에는 설치돼 있지 않았고 launch 에서도 쓰이지 않는다. Gazebo Classic 으로만
# 돌리므로 여기에도 넣지 않는다. GZ 로 옮길 때 함께 추가할 것.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    nano \
    vim \
    tmux \
    psmisc \
    locales \
    python3-pip \
    python3-dev \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros \
    ros-humble-gazebo-msgs \
    ros-humble-gazebo-plugins \
    ros-humble-gazebo-ros2-control \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-ackermann-msgs \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    mesa-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# install.sh 의 pip 목록. 버전은 실제로 돌던 컨테이너에서 확인한 값으로 고정한다.
#  - apt 의 sympy(distutils) 와 충돌하므로 pip 쪽을 --ignore-installed 로 덮는다.
#  - CUDA torch(~수 GB) 대신 CPU torch 고정. GPU 추론이 필요하면 Dockerfile.cuda 를 쓴다.
#  - colcon/ament 는 setuptools 58.x 를 요구한다 (최신 setuptools 는 strip_trailing_zero 오류).
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --ignore-installed sympy mpmath && \
    pip3 install --no-cache-dir \
      "torch==2.13.0+cpu" "torchvision==0.28.0+cpu" \
      --index-url https://download.pytorch.org/whl/cpu && \
    printf '%s\n' 'torch==2.13.0+cpu' 'torchvision==0.28.0+cpu' > /tmp/torch-constraints.txt && \
    pip3 install --no-cache-dir \
      -c /tmp/torch-constraints.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      opencv-python \
      pyserial \
      "ultralytics==8.2.69" \
      pynput \
      "transformers==4.44.2" \
      huggingface_hub && \
    pip3 install --no-cache-dir "setuptools==58.2.0" "packaging==23.2" && \
    rm -f /tmp/torch-constraints.txt

RUN mkdir -p /root/.gazebo/models

# compose 가 레포 루트를 여기에 마운트한다
WORKDIR /root/hmobility_ws

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "[ -f /root/hmobility_ws/install/local_setup.bash ] && source /root/hmobility_ws/install/local_setup.bash" >> /root/.bashrc && \
    echo "export ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-0}" >> /root/.bashrc && \
    echo "alias MOVE='ros2 service call /go std_srvs/srv/SetBool \"{data: true}\"'" >> /root/.bashrc && \
    echo "alias STOP='ros2 service call /go std_srvs/srv/SetBool \"{data: false}\"'" >> /root/.bashrc && \
    printf '%s\n' \
      'qqq() {' \
      '  PIDS=$(ps aux | grep "[g]zserver" | awk "{print \$2}")' \
      '  for pid in $PIDS; do kill -9 $pid; done' \
      '}' >> /root/.bashrc

CMD ["bash"]
