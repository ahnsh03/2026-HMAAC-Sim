#!/usr/bin/env python3

import os
import subprocess
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

    package_dir = get_package_share_directory('simulation_pkg')
    world_file = os.path.join(package_dir, 'worlds', 'track.world')

    # 신호등 색상 제어 플러그인(.so) 위치를 Gazebo 에 직접 알려준다.
    # 플러그인을 빌드하기 전에 열어둔 터미널이면 ament index 에 아직 없으므로,
    # 그 경우엔 이 launch 파일 위치에서 워크스페이스 install 경로를 역산한다.
    try:
        plugin_dir = os.path.join(get_package_prefix('plugin_pkg'), 'lib')
    except PackageNotFoundError:
        install_dir = os.path.abspath(os.path.join(package_dir, '..', '..', '..'))
        plugin_dir = os.path.join(install_dir, 'plugin_pkg', 'lib')

    if not os.path.isdir(plugin_dir):
        print(f"[mission_sim] 경고: 신호등 플러그인을 찾을 수 없습니다 -> {plugin_dir}")
        print("[mission_sim]   colcon build --packages-select plugin_pkg 를 먼저 실행하세요")

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
        
        # Node(
        #     package='simulation_pkg', 
        #     executable='video_recording_node',
        #     output='screen'
        # ),
        
        # 신호등 스폰 + 색상 제어 GUI (hmobility_sim 과 동일, 위치만 mission 용)
        Node(
            package='simulation_pkg',
            executable='traffic_light_gui_node',
            parameters=[{'placement': 'mission'}],
            output='screen',
        ),

        Node(
            package='simulation_pkg',
            executable='load_ego_car_node',
            parameters=[{'with_lidar': True}],
            output='screen',
        ),

        Node(
            package='simulation_pkg',
            executable='load_obstable_car_node',
            output='screen',
        ),

        # 
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
