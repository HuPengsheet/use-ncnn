project(NCNN_DEMO)
cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/hp/code/github/ncnn/build/install/lib/cmake/ncnn/")

find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)

include_directories("/home/hp/code/github/ncnn/build/install/include/ncnn/" ${OpenCV_INCLUDE_DIRS})

link_directories("/home/hp/code/github/ncnn/build/install/lib/")
if(ncnn_FOUND)
    message(STATUS "ncnn moudle find!!")
else()
    message(WARING "ncnn moudle NOT find!!")
endif()


#target_include_directories( ${OpenCV_INCLUDE_DIRS})
add_executable(run src/main.cpp)
target_link_libraries(run ncnn ${OpenCV_LIBS})