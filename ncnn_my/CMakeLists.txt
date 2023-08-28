project(NCNN_DEMO)
cmake_minimum_required(VERSION 2.8.12)
 
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/hupeng/code_c/github/ncnn/build/install/lib/cmake/ncnn/")
find_package(ncnn REQUIRED )
if(ncnn_FOUND)
    message("ncnn moudle find!!")
endif()