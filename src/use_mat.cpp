#include "layer.h"
#include "net.h"
#include<iostream>


int main(){

    ncnn::Mat input(10);

    std::cout<<"w:"<<input.w<<std::endl;
    std::cout<<"h:"<<input.h<<std::endl;
    std::cout<<"c:"<<input.c<<std::endl;
    std::cout<<"program done!!!"<<std::endl;
    return 0;
}