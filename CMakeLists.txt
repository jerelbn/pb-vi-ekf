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

if (NOT TARGET common_cpp)
  add_subdirectory(lib/common_cpp)
  include_directories(lib/common_cpp/include)
endif()

if (NOT TARGET geometry)
  add_subdirectory(lib/geometry)
  include_directories(lib/geometry/include)
endif()

include_directories(include/pb_vi_ekf
  ${EIGEN3_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
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
