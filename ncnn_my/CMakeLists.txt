project(NCNN_DEMO)
cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/hupeng/code_c/github/ncnn/build/install/lib/cmake/ncnn/")
find_package(ncnn   REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB SRC src/*.cpp)

add_executable(run ${SRC})
target_link_libraries(run ncnn ${OpenCV_LIBS})