project(NCNN_DEMO)
cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/hupeng/code_c/github/ncnn/build/install/lib/cmake/ncnn/")
find_package(ncnn   REQUIRED)
find_package(OpenCV REQUIRED)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(yolov5s src/yolov5s.cpp)
target_link_libraries(yolov5s ncnn ${OpenCV_LIBS})