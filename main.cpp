
#include "onnx.pb.h"

#include <algorithm>
#include <float.h>
#include <fstream>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <iostream>
#include <limits.h>
#include <limits>
#include <set>
#include <stdio.h>

using namespace std;

static bool read_proto_from_binary(const char* filepath, onnx::ModelProto* message)
{
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }
    
    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);
    bool success = message->ParseFromCodedStream(&codedstr);
    fs.close();
    return success;
}

int main(){
    const char * path="model.onnx";
    onnx::ModelProto model;

    // load
    bool s1 = read_proto_from_binary(path, &model);
    if (!s1)
    {
        fprintf(stderr, "read_proto_from_binary failed\n");
        return -1;
    }
    else{
        cout<<"onnx读入成功"<<endl;;
    }


    const onnx::GraphProto& graph = model.graph();
    
    onnx::GraphProto* mutable_graph = model.mutable_graph();
    
    int ver =model.ir_version();
    cout<<ver<<endl;

    
    cout<<"程序运行结束"<<endl;
    return 0;
}