#!/usr/bin/env python3

import os
import re
import subprocess
import tempfile
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_prefix,
    get_package_share_directory,
)

def generate_launch_description():

    subprocess.run(['killall', 'gzserver'])
    subprocess.run(['killall', 'gzclient'])

    package_dir=get_package_share_directory('simulation_pkg')
    world_file = os.path.join(package_dir, 'worlds', 'track.world')

    # track.world 의 초기 시점은 top-down (driving_sim / mission_sim 용) 이다.
    # hmobility 는 ego 차량과 신호등을 비스듬히 보는 시점을 쓰므로,
    # world 파일을 복제하지 않고 카메라 pose 한 줄만 바꾼 사본을 만들어 띄운다.
    # (트랙을 수정할 곳이 track.world 한 곳으로 유지된다)
    # 카메라 블록을 통째로 갈아끼운다. track.world 쪽 주석이 바뀌어도
    # 안 깨지고, 투영 방식(top-down 은 ortho, 여기는 perspective)까지 함께 바꾼다.
    HMOBILITY_CAMERA = (
        "<camera name='user_camera'>"
        "<pose>-26 -36 10 0 0.319 0.545</pose>"
        "<view_controller>orbit</view_controller>"
        "<projection_type>perspective</projection_type>"
        "</camera>"
    )
    try:
        with open(world_file) as f:
            world_xml = f.read()
        world_xml, n = re.subn(
            r"<camera name='user_camera'>.*?</camera>",
            HMOBILITY_CAMERA,
            world_xml, count=1, flags=re.S)
        if n == 0:
            print('[hmobility_sim] 경고: user_camera 를 찾지 못해 기본 시점을 씁니다')

        patched = os.path.join(tempfile.gettempdir(), 'hmobility_track.world')
        with open(patched, 'w') as f:
            f.write(world_xml)
        world_file = patched
    except OSError as e:
        print(f'[hmobility_sim] 시점 변경 실패, 기본 world 사용: {e}')

    # 신호등 색상 제어 플러그인(.so) 위치를 Gazebo 에 직접 알려준다.
    # 셸 source 시점에 따라 LD_LIBRARY_PATH 에 안 잡히는 경우가 있어서,
    # launch 가 매번 확실하게 넣어준다.
    #
    # 플러그인을 빌드하기 전에 열어둔 터미널이면 ament index 에 아직 없다.
    # 그 경우엔 이 launch 파일 위치에서 워크스페이스 install 경로를 역산한다.
    # (install/simulation_pkg/share/simulation_pkg -> install 까지 세 단계 위)
    try:
        plugin_dir = os.path.join(get_package_prefix('plugin_pkg'), 'lib')
    except PackageNotFoundError:
        install_dir = os.path.abspath(os.path.join(package_dir, '..', '..', '..'))
        plugin_dir = os.path.join(install_dir, 'plugin_pkg', 'lib')

    if not os.path.isdir(plugin_dir):
        print(f"[hmobility_sim] 경고: 신호등 플러그인을 찾을 수 없습니다 -> {plugin_dir}")
        print("[hmobility_sim]   colcon build --packages-select plugin_pkg 를 먼저 실행하세요")

    return LaunchDescription([
        SetEnvironmentVariable(
            'GAZEBO_PLUGIN_PATH',
            plugin_dir + ':' + os.environ.get('GAZEBO_PLUGIN_PATH', '')),

        ExecuteProcess(
            cmd=['gazebo', '--verbose', world_file, '-s', 'libgazebo_ros_factory.so'],
            output='screen'),

        ExecuteProcess(
            cmd=['rqt'],
            output='screen'),

        # 신호등 스폰은 traffic_light_gui_node 가 직접 담당한다 (색을 바꿀 때마다
        # 다시 스폰해야 하므로). 그래서 별도 로더 노드는 실행하지 않는다.

        Node(
            package='simulation_pkg',
            executable='load_ego_car_node',
            parameters=[{'with_lidar': True}],
            output='screen'
        ),

        Node(
            package='simulation_pkg',
            executable='traffic_light_gui_node',
            output='screen'
        ),

        # 최종평가와 가장 유사한 고정 장애물 차량 (hmobility 전용)
        Node(
            package='simulation_pkg',
            executable='load_hmobility_obstacle_node',
            output='screen'
        ),

        Node(
            package='camera_perception_pkg',
            executable='yolov8_node',
            output='screen'
        ),

        Node(
            package='debug_pkg',
            executable='yolov8_visualizer_node',
            output='screen'
        ),

        Node(
            package='debug_pkg',
            executable='path_visualizer_node',
            output='screen'
        ),

        Node(
            package='camera_perception_pkg',
            executable='lane_info_extractor_node',
            output='screen'
        ),

        Node(
            package='camera_perception_pkg',
            executable='traffic_light_detector_node',
            output='screen'
        ),

        Node(
            package='decision_making_pkg',
            executable='path_planner_node',
            output='screen'
        ),

        Node(
            package='decision_making_pkg',
            executable='motion_planner_node',
            output='screen'
        ),

        Node(
            package='simulation_pkg',
            executable='sim_simulation_sender_node',
            output='screen'
        ),

    ])
