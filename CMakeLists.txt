cmake_minimum_required(VERSION 2.8.3)
project(pb_vi_ekf)

# default to release build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_FLAGS "-std=c++14 -mavx2 -mfma -fopenmp")

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(GTest)
find_package(catkin COMPONENTS
  rosbag
  roscpp
  nav_msgs
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
)

if (NOT TARGET common_cpp)
  add_subdirectory(lib/common_cpp)
  include_directories(lib/common_cpp/include)
endif()

if (NOT TARGET geometry)
  add_subdirectory(lib/geometry)
  include_directories(lib/geometry/include)
endif()

if (NOT TARGET feature_tracker)
  add_subdirectory(lib/feature-tracker)
  include_directories(lib/feature-tracker/include)
endif()

include_directories(include/pb_vi_ekf
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  )

add_library(pb_vi_ekf
    src/ekf.cpp
)
target_link_libraries(pb_vi_ekf 
  ${YAML_CPP_LIBRARIES}
  common_cpp
  geometry
  )

if (${GTEST_FOUND})
  add_definitions(-DPBVIEKF_DIR="${CMAKE_CURRENT_LIST_DIR}")
  include_directories(include ${GTEST_INCLUDE_DIRS})
  add_executable(jacobians_test
    src/test/jacobians_test.cpp
    )
  target_link_libraries(jacobians_test
    pb_vi_ekf
    common_cpp
    geometry
    gtest_main
    gtest
    ${GTEST_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    )
endif()

## BUILD THE ROS executables if catkin is present
if (${catkin_FOUND})
  message("Found catkin - building ROS parser")
  add_executable(rosbag_parser 
    src/rosbag_parser.cpp
    )
  target_link_libraries(rosbag_parser
    common_cpp
    geometry
    feature_tracker
    pb_vi_ekf
    ${catkin_LIBRARIES}
    )
else()
  message("No catkin found - not building ROSbag parser")
endif()
