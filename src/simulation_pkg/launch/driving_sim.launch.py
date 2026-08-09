#!/usr/bin/env python3

import os
import subprocess
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def arg(name, value_type):
    """런치 인자를 노드 파라미터 타입으로 넘긴다.

    LaunchConfiguration 은 문자열이라 그대로 주면 double/int 파라미터 선언과
    타입이 어긋나 노드가 뜨지 않는다.
    """
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def generate_launch_description():
        
    subprocess.run(['killall', 'gzserver'])
    subprocess.run(['killall', 'gzclient'])
    
    package_dir=get_package_share_directory('simulation_pkg')
    world_file = os.path.join(package_dir, 'worlds', 'track.world')

    # 인지 파이프라인이 CPU 를 다 쓰기 때문에 시각화는 기본으로 끈다.
    show_lane_debug = LaunchConfiguration('show_lane_debug')
    debug_viz = LaunchConfiguration('debug_viz')
    use_rqt = LaunchConfiguration('rqt')
    drive_enable = LaunchConfiguration('drive_enable')
    lap_monitor = LaunchConfiguration('lap_monitor')


    return LaunchDescription([
        DeclareLaunchArgument('show_lane_debug', default_value='false',
                              description='lane_info_extractor 의 BEV 창 표시'),
        DeclareLaunchArgument('debug_viz', default_value='false',
                              description='yolov8_visualizer / path_visualizer 실행'),
        DeclareLaunchArgument('rqt', default_value='false',
                              description='rqt 실행'),
        DeclareLaunchArgument('drive_enable', default_value='true',
                              description='false 면 조향만 계산하고 속도 0 (인지 검증용)'),
        DeclareLaunchArgument('lap_monitor', default_value='true',
                              description='랩 판정/궤적 기록 노드 실행'),
        DeclareLaunchArgument('imgsz', default_value='640',
                              description='YOLO 추론 해상도. CPU 라면 480 이 1.6배 빠르다'),
        DeclareLaunchArgument('conf', default_value='0.5',
                              description='YOLO confidence 임계값'),
        DeclareLaunchArgument('center_offset', default_value='0.0',
                              description='차로 중심 보정 [BEV 픽셀, 약 103px=1m]. 양수면 오른쪽'),
        DeclareLaunchArgument('v_max', default_value='2.4',
                              description='직선 최고 속도 [m/s]'),
        DeclareLaunchArgument('v_min', default_value='1.4',
                              description='코너 최저 속도 [m/s]'),
        DeclareLaunchArgument('steer_gain', default_value='1.00',
                              description='조향 배수. 1.0 이 기하학적으로 필요한 값'),
        DeclareLaunchArgument('k_cut_comp', default_value='0.0',
                              description='코너 파고듦 보정 배수'),
        DeclareLaunchArgument('e_soft', default_value='0.0',
                              description='오차 되먹임 점진이득 문턱 [m]. 작을수록 민감'),
        DeclareLaunchArgument('kappa_lpf', default_value='0.30',
                              description='곡률 저역통과 계수. 클수록 빠르고 노이지'),
        DeclareLaunchArgument('steer_rate_limit', default_value='2.2',
                              description='tick 당 조향 변화 상한 [step]'),
        DeclareLaunchArgument('lane_width', default_value='328.0',
                              description='차로 폭 [BEV 픽셀]'),

        ExecuteProcess(
            cmd=['gazebo', '--verbose', world_file, '-s', 'libgazebo_ros_factory.so'],
            output='screen'),
            
        ExecuteProcess(
            cmd=['rqt'], 
            output='screen',
            condition=IfCondition(use_rqt)),
        
        # Node(
        #     package='simulation_pkg', 
        #     executable='video_recording_node',
        #     output='screen'
        # ),
        
        Node(
            package='simulation_pkg',
            executable='load_ego_car_node',
            output='screen'
        ),

        Node(
            package='camera_perception_pkg',
            executable='yolov8_node',
            output='screen',
            parameters=[{'imgsz': arg('imgsz', int), 'threshold': arg('conf', float)}]
        ),
        
        Node(
            package='debug_pkg',
            executable='yolov8_visualizer_node',
            output='screen',
            condition=IfCondition(debug_viz)
        ),   

        Node(
            package='debug_pkg',
            executable='path_visualizer_node',
            output='screen',
            condition=IfCondition(debug_viz)
        ),
        
        Node(
            package='camera_perception_pkg', 
            executable='lane_info_extractor_node',
            output='screen',
            parameters=[{'show_image': arg('show_lane_debug', bool),
                         'center_offset': arg('center_offset', float),
                         'lane_width': arg('lane_width', float)}]
        ),

        Node(
            package='decision_making_pkg', 
            executable='path_planner_node',
            output='screen'
        ),
       
        Node(
            package='decision_making_pkg', 
            executable='motion_planner_node',
            output='screen',
            parameters=[{'drive_enable': arg('drive_enable', bool),
                         'v_max': arg('v_max', float),
                         'v_min': arg('v_min', float),
                         'steer_gain': arg('steer_gain', float),
                         'k_cut_comp': arg('k_cut_comp', float),
                         'e_soft': arg('e_soft', float),
                         'kappa_lpf': arg('kappa_lpf', float),
                         'steer_rate_limit': arg('steer_rate_limit', float)}]
        ),
       
        Node(
            package='simulation_pkg', 
            executable='sim_simulation_sender_node',
            output='screen'
        ),

        Node(
            package='debug_pkg',
            executable='lap_monitor_node',
            output='screen',
            condition=IfCondition(lap_monitor)
        ),
                     
    ])
