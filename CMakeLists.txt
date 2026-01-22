cmake_minimum_required(VERSION 3.0.2)
project(gaze_pros_control)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
)
catkin_python_setup()
## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Glass_Gaze.msg
  Gait_Info.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES gaze_pros_control
 CATKIN_DEPENDS geometry_msgs roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

catkin_install_python(PROGRAMS
  scripts/env_perception/gaze_publisher.py
  scripts/env_perception/fpp_node.py

  scripts/imu_related/pose_publisher_serial.py
  scripts/imu_related/pose_publisher.py
  scripts/imu_related/imu_serial.py
  scripts/imu_related/imu_.py
  scripts/imu_related/multi_imu_rec.py
  scripts/imu_related/Xsens_MTI_630/mtnode.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
