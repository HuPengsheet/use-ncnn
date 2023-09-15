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

static void print_mode_info(const onnx::ModelProto& model)
{
    cout<<"ir_version:"<<model.ir_version()<<endl;
    cout<<"producer_name:"<<model.producer_name()<<endl;
    cout<<"producer_version"<<model.producer_version()<<endl;
    cout<<"domain:"<<model.domain()<<endl;
    cout<<"model_version:"<<model.model_version()<<endl;
}



static void print_graph_info(const onnx::GraphProto& graph)
{
    cout<<"the name of graph :"<<graph.name()<<endl;
    cout<<"input:  ";
    for(int i=0;i<graph.input_size();i++){
        cout<<graph.input(i).name()<<" ";
    }
    cout<<endl;
    cout<<"output:  ";
    for(int i=0;i<graph.output_size();i++){
        cout<<graph.output(i).name()<<" ";
    }
    cout<<endl;
    cout<<"the length of node :"<<graph.node_size()<<endl;

    
    
    int node_size = graph.node_size();
    //遍历图中的每个节点
    cout<<"***************************************"<<endl;
    for(int i=0;i<node_size;i++){
        const onnx::NodeProto&  _node =graph.node(i);

        cout<<"node name    : "<<_node.name()<<endl;
        cout<<"node op_type : "<<_node.op_type()<<endl;
        cout<<"node domain  : "<<_node.domain()<<endl;

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

        cout<<"***************************************"<<endl;
    }
}

static void print_initializer_info(const onnx::GraphProto graph)
{
    
    cout<<"print (initializer)TensorProto :"<<endl;
    cout<<"***************************************"<<endl;
    std::map<std::string, onnx::TensorProto> weights;

    for (int j = 0; j < graph.initializer_size(); j++)
    {
        const onnx::TensorProto& initializer = graph.initializer(j);

        cout<<"name     : "<<initializer.name()<<endl;
        cout<<"datatype : "<<initializer.data_type()<<endl;
        cout<<"dims : ";
        for (int i=0;i<initializer.dims_size();i++){
            cout<<initializer.dims(i)<<" ";
        }
        cout<<endl;

        cout<<"***************************************"<<endl;
        weights[initializer.name()] = initializer;
    }    
    
}

static void print_tensor_data(const onnx::GraphProto graph){
    cout<<"TensorData"<<endl;
    cout<<"***************************************"<<endl;
    for (int j = 0; j < graph.initializer_size(); j++)
    {
        onnx::TensorProto t=graph.initializer(j);
        cout<<"datatype : "<<t.data_type()<<endl;
       //cout<<"float_data_size : "<<t.float_data_size()<<endl;
       cout<<"has_raw_data : "<<t.has_raw_data()<<endl;

        const std::string& raw_data = t.raw_data();
        int size = (int)raw_data.size() / 4;
        cout<<"data size:"<<size<<endl;

    }

        
}

int main(){

    const char * path="/home/hupeng/code_c/github/ONNX_Parse/onnx_file/model.onnx";
    onnx::ModelProto model;

    // load model
    bool s1 = read_proto_from_binary(path, &model);
    if (!s1)
    {
        cout<<"read_proto_from_binary failed"<<endl;
        return -1;
    }
    else{
        cout<<"onnx read success"<<endl;;
    }
    const onnx::GraphProto& graph = model.graph();

    //print_mode_info(model);
    //print_graph_info(graph);
   // print_initializer_info(graph);
    print_tensor_data(graph);

    





    

    
 


    
    cout<<"程序运行结束"<<endl;
    return 0;
}