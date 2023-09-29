project(NCNN_DEMO)
cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/hupeng/code/demo/ncnn/build/install")

find_package(OpenCV REQUIRED)
find_package(ncnn)
if(ncnn_FOUND)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

    add_executable(resnet18 src/resnet18.cpp)
    target_link_libraries(resnet18 ncnn ${OpenCV_LIBS})
    
    add_executable(yolov5s src/yolov5s.cpp)
    target_link_libraries(yolov5s ncnn ${OpenCV_LIBS})
    
    add_executable(resnet18_pnnx src/resnet18_pnnx.cpp)
    target_link_libraries(resnet18_pnnx ncnn ${OpenCV_LIBS})
    
    add_executable(yolov5s_pnnx src/yolov5s_pnnx.cpp)
    target_link_libraries(yolov5s_pnnx ncnn ${OpenCV_LIBS})

else()
    message(WARNING "not find ncnn ,please check CMAKE_PREFIX_PATH ,and change to you ncnn install dir")
endif()   




