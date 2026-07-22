cmake_minimum_required(VERSION 3.8)
project(plugin_pkg)

# 패키지 이름은 plugin_pkg 이지만, 빌드되는 플러그인 라이브러리 이름은
# traffic_light_plugin 으로 유지한다 (model.sdf 가 libtraffic_light_plugin.so 를 참조).

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

link_directories(${gazebo_dev_LIBRARY_DIRS})

add_library(traffic_light_plugin SHARED
  src/traffic_light_plugin.cpp
)

target_include_directories(traffic_light_plugin PUBLIC
  ${GAZEBO_INCLUDE_DIRS}
)

ament_target_dependencies(traffic_light_plugin
  gazebo_dev
  gazebo_ros
  rclcpp
  std_msgs
)

target_link_libraries(traffic_light_plugin
  ${GAZEBO_LIBRARIES}
)

install(TARGETS traffic_light_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_package()
