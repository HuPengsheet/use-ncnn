g++ -o run main.cpp onnx.pb.cc  `pkg-config --cflags --libs protobuf`
./run