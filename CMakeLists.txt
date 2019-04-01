cmake_minimum_required(VERSION 2.8.3)
project(pb_vi_ekf)

# default to release build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_FLAGS "-std=c++14 -mavx2 -mfma -fopenmp")

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

add_subdirectory(lib/common_cpp)
add_subdirectory(lib/geometry)

include_directories(include
  lib/common_cpp/include
  lib/geometry/include
  ${EIGEN3_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  )
