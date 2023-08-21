project(protobuf_cmake)
cmake_minimum_required(VERSION 2.8.12)


find_package(PkgConfig REQUIRED)
pkg_check_modules(PROTOBUF REQUIRED protobuf)
find_package(Protobuf REQUIRED)


protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS proto_file/onnx.proto)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_executable(run src/main.cpp ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(run ${PROTOBUF_LDFLAGS})