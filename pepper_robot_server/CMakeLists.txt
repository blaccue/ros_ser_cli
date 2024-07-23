cmake_minimum_required(VERSION 3.0.2)
project(pepper_robot_server)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  cv_bridge
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/pepper_robot_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
