
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
        cout<<"onnx read success"<<endl;;
    }

    //const &，引用类型，加了const修饰，不改变graph值
    const onnx::GraphProto& graph = model.graph();
    cout<<"input:  ";
    for(int i=0;i<graph.input_size();i++){
        cout<<graph.input(i).name()<<" ";
    }
    cout<<endl;

    //mutable_graph(),返回的是指针
    //GraphProto* 
    onnx::GraphProto* mutable_graph = model.mutable_graph();
    


    cout<<"***************************************"<<endl;
    cout<<"print (initializer)TensorProto :"<<endl;
    std::map<std::string, onnx::TensorProto> weights;

    for (int j = 0; j < graph.initializer_size(); j++)
    {
        const onnx::TensorProto& initializer = graph.initializer(j);

        cout<<"name     : "<<initializer.name()<<endl;
        cout<<"datatype : "<<initializer.data_type()<<endl;
        for (int i=0;i<initializer.dims_size();i++){
            cout<<initializer.dims(i)<<" ";
        }
        cout<<endl;

        cout<<"***************************************"<<endl;
        weights[initializer.name()] = initializer;
    }    
    cout<<"***************************************"<<endl;


    

    
    cout<<"打印每个节点的名字"<<endl;
    int node_size = graph.node_size();
    cout<<"node_size= "<<node_size<<endl;
    //遍历图中的每个节点
    cout<<"***************************************"<<endl;
    for(int i=0;i<node_size;i++){

        const onnx::NodeProto&  _node =graph.node(i);

        cout<<"输入节点 :";
        for(int i=0;i<_node.input_size();i++){
            cout<<_node.input(i)<<" ";
        }
        cout <<endl;

        cout<<"输出节点 :";
        for(int i=0;i<_node.output_size();i++){
            cout<<_node.output(i)<<" ";
        }
        cout <<endl;

        cout<<"node name    : "<<_node.name()<<endl;
        cout<<"node op_type : "<<_node.op_type()<<endl;
        cout<<"node domain  : "<<_node.domain()<<endl;
        cout<<"***************************************"<<endl;

    }


    
    cout<<"程序运行结束"<<endl;
    return 0;
}