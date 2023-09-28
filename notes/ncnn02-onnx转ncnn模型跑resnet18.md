# onnx转ncnn模型跑resnet18

​	ncnn提供了多种模型转换工具，可以快速的将caffe、onnx等模型一键转换为ncnn的格式，在源代码编译后这些工具存放在`you_dir/ncnn/build/tools`目录下。本次使用`onnx2ncnn`工具，把在pytorch导出的resnet18的onnx文件，转换为ncnn格式并完成推理。

## 项目结构

```
.
├── bin
│   ├── onnx2ncnn
├── CMakeLists.txt
├── image
│   ├── dog.jpg
│   └── bus.jpg
├── model_param
├── python
│   ├── export_res18.py
└── src
    ├── resnet18.cpp
```

​	src目录下存放我们的源代码，python目录下存放python的脚本，model_param存放模型文件，image目录下存放推理用的图片，bin目录下存放可执行文件和项目生成的可执行文件。**其中onnx2ncnn，是在`you_dir/ncnn/build/tools/onnx/onnx2ncnn`拷贝过来的**。

## 在python中导出resnet18的onnx

```python
import torch
import torchvision.models as models
import torch.onnx as onnx

# 加载预训练的ResNet-18模型
resnet = models.resnet18(pretrained=True)

# 将模型设置为评估模式
resnet.eval()

# 创建一个示例输入张量
dummy_input = torch.randn(1, 3, 224, 224)

# 使用torch.onnx.export函数导出模型为ONNX格式
onnx_file_path = "../model_param/resnet18.onnx"
onnx.export(resnet, dummy_input, onnx_file_path)

print("ResNet-18模型已成功导出为ONNX格式：", onnx_file_path)
```

​	上面是`export_res18.py`的代码，代码也很简单，就是给一个示例输入，然后跑一遍模型，相对应的图结构就会保存下来。运行代码后会在`model_param/`下生成一个resnet18.onnx的模型文件。

## 模型转换

​	ncnn官方提供了模型转换工具，来将导出的onnx模型转换为ncnn支持的格式，所有模型转换的源代码都在`ncnn/tools`目录下，在编译后也同样会在`build/tools/`下生成对应的可执行程序。我们将`you_dir/ncnn/build/tools/onnx/onnx2ncnn`复制到我们的bin目录中来。

​	我们在项目根目录下执行：

```shell
bin/onnx2ncnn model_param/resnet18.onnx model_param/resnet18.param model_param/resnet18.bin
```

​	即使用onnx2ncnn工具，将resnet18.onnx转换为resnet18.param和resnet18.bin，其中resnet18.param为模型的参数信息（记录的是计算图的结构），resnet18.bin里存放的是模型的所有具体的参数。

​	我们可以看一看resnet18.param的参数

```
7767517
58 66
Input            input.1                  0 1 input.1
Convolution      Conv_0                   1 1 input.1 input.4 0=64 1=7 11=7 2=1 12=1 3=2 13=2 4=3 14=3 15=3 16=3 5=1 6=9408
ReLU             Relu_1                   1 1 input.4 onnx::MaxPool_125
Pooling          MaxPool_2                1 1 onnx::MaxPool_125 input.8 0=0 1=3 11=3 2=2 12=2 3=1 13=1 14=1 15=1 5=1
Split            splitncnn_0              1 2 input.8 input.8_splitncnn_0 input.8_splitncnn_1
Convolution      Conv_3                   1 1 input.8_splitncnn_1 input.16 0=64 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=36864
ReLU             Relu_4                   1 1 input.16 onnx::Conv_129
Convolution      Conv_5                   1 1 onnx::Conv_129 onnx::Add_198 0=64 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=36864
BinaryOp         Add_6                    2 1 onnx::Add_198 input.8_splitncnn_0 onnx::Relu_132 0=0
ReLU             Relu_7                   1 1 onnx::Relu_132 input.24
Split            splitncnn_1              1 2 input.24 input.24_splitncnn_0 input.24_splitncnn_1
Convolution      Conv_8                   1 1 input.24_splitncnn_1 input.32 0=64 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=36864
ReLU             Relu_9                   1 1 input.32 onnx::Conv_136
Convolution      Conv_10                  1 1 onnx::Conv_136 onnx::Add_204 0=64 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=36864
BinaryOp         Add_11                   2 1 onnx::Add_204 input.24_splitncnn_0 onnx::Relu_139 0=0
ReLU             Relu_12                  1 1 onnx::Relu_139 input.40
Split            splitncnn_2              1 2 input.40 input.40_splitncnn_0 input.40_splitncnn_1
Convolution      Conv_13                  1 1 input.40_splitncnn_1 input.48 0=128 1=3 11=3 2=1 12=1 3=2 13=2 4=1 14=1 15=1 16=1 5=1 6=73728
ReLU             Relu_14                  1 1 input.48 onnx::Conv_143
Convolution      Conv_15                  1 1 onnx::Conv_143 onnx::Add_210 0=128 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=147456
Convolution      Conv_16                  1 1 input.40_splitncnn_0 onnx::Add_213 0=128 1=1 11=1 2=1 12=1 3=2 13=2 4=0 14=0 15=0 16=0 5=1 6=8192
BinaryOp         Add_17                   2 1 onnx::Add_210 onnx::Add_213 onnx::Relu_148 0=0
ReLU             Relu_18                  1 1 onnx::Relu_148 input.60
Split            splitncnn_3              1 2 input.60 input.60_splitncnn_0 input.60_splitncnn_1
Convolution      Conv_19                  1 1 input.60_splitncnn_1 input.68 0=128 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=147456
ReLU             Relu_20                  1 1 input.68 onnx::Conv_152
Convolution      Conv_21                  1 1 onnx::Conv_152 onnx::Add_219 0=128 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=147456
BinaryOp         Add_22                   2 1 onnx::Add_219 input.60_splitncnn_0 onnx::Relu_155 0=0
ReLU             Relu_23                  1 1 onnx::Relu_155 input.76
Split            splitncnn_4              1 2 input.76 input.76_splitncnn_0 input.76_splitncnn_1
Convolution      Conv_24                  1 1 input.76_splitncnn_1 input.84 0=256 1=3 11=3 2=1 12=1 3=2 13=2 4=1 14=1 15=1 16=1 5=1 6=294912
ReLU             Relu_25                  1 1 input.84 onnx::Conv_159
Convolution      Conv_26                  1 1 onnx::Conv_159 onnx::Add_225 0=256 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=589824
Convolution      Conv_27                  1 1 input.76_splitncnn_0 onnx::Add_228 0=256 1=1 11=1 2=1 12=1 3=2 13=2 4=0 14=0 15=0 16=0 5=1 6=32768
BinaryOp         Add_28                   2 1 onnx::Add_225 onnx::Add_228 onnx::Relu_164 0=0
ReLU             Relu_29                  1 1 onnx::Relu_164 input.96
Split            splitncnn_5              1 2 input.96 input.96_splitncnn_0 input.96_splitncnn_1
Convolution      Conv_30                  1 1 input.96_splitncnn_1 input.104 0=256 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=589824
ReLU             Relu_31                  1 1 input.104 onnx::Conv_168
Convolution      Conv_32                  1 1 onnx::Conv_168 onnx::Add_234 0=256 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=589824
BinaryOp         Add_33                   2 1 onnx::Add_234 input.96_splitncnn_0 onnx::Relu_171 0=0
ReLU             Relu_34                  1 1 onnx::Relu_171 input.112
Split            splitncnn_6              1 2 input.112 input.112_splitncnn_0 input.112_splitncnn_1
Convolution      Conv_35                  1 1 input.112_splitncnn_1 input.120 0=512 1=3 11=3 2=1 12=1 3=2 13=2 4=1 14=1 15=1 16=1 5=1 6=1179648
ReLU             Relu_36                  1 1 input.120 onnx::Conv_175
Convolution      Conv_37                  1 1 onnx::Conv_175 onnx::Add_240 0=512 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=2359296
Convolution      Conv_38                  1 1 input.112_splitncnn_0 onnx::Add_243 0=512 1=1 11=1 2=1 12=1 3=2 13=2 4=0 14=0 15=0 16=0 5=1 6=131072
BinaryOp         Add_39                   2 1 onnx::Add_240 onnx::Add_243 onnx::Relu_180 0=0
ReLU             Relu_40                  1 1 onnx::Relu_180 input.132
Split            splitncnn_7              1 2 input.132 input.132_splitncnn_0 input.132_splitncnn_1
Convolution      Conv_41                  1 1 input.132_splitncnn_1 input.140 0=512 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=2359296
ReLU             Relu_42                  1 1 input.140 onnx::Conv_184
Convolution      Conv_43                  1 1 onnx::Conv_184 onnx::Add_249 0=512 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=2359296
BinaryOp         Add_44                   2 1 onnx::Add_249 input.132_splitncnn_0 onnx::Relu_187 0=0
ReLU             Relu_45                  1 1 onnx::Relu_187 input.148
Pooling          GlobalAveragePool_46     1 1 input.148 onnx::Flatten_189 0=1 4=1
Flatten          Flatten_47               1 1 onnx::Flatten_189 onnx::Gemm_190
InnerProduct     Gemm_48                  1 1 onnx::Gemm_190 191 0=1000 1=1 2=512000
```

​	简单的和大家分析一下怎么看这个参数。首先`7767517`是一个magic数，表明这是ncnn的格式。`58 66`分别是layer和blob的个数。可能很多初学者分不清layer和blob的区别，我在这里为大家简单介绍一下

![image-20230924124443271](../image/1.png)

​	我们使用netron打开resnet18.param可以看到resnet18的结构，其中像Convolution，ReLU，Pooling，Split，BinaryOp都是一个算子也就是layer。blob可以看作是中间数据的存储，以Split算子为例，它有1个输入2个输出，则一共有3个blob，像Convolution和Relu等算子它输入输出都是1个blob。所以一般情况下blob数看到回比layer数多的多。

​	回到参数当中来：

`Convolution      Conv_0   1 1 input.1 input.4 0=64 1=7 11=7 2=1 12=1 3=2 13=2 4=3 14=3 15=3 16=3 5=1 6=9408`

`Convolution`：layer类型

`Conv_0`：layer的名字

`input.1`：输入blob的名字

`input.4`：输出blob的名字

`0=64 1=7 11=7 2=1 12=1 3=2 13=2 4=3 14=3 15=3 16=3 5=1 6=9408`：layer的参数信息

​	其实对ncnn的使用者来说，我们主要需要关注的是整个模型的输入输出。对于当前这个网络来说，整个网络的输入blob名字是`input.1`，输出blob是`191`，这些信息我们在写推理代码的时候回用到。

## 推理代码编写

```c++
#include "net.h"
#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>


static int detect_resnet18(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net resnet18;

    resnet18.opt.use_vulkan_compute = true;

    //分别加载模型的参数和数据
    if (resnet18.load_param("model_param/resnet18.param"))
        exit(-1);
    if (resnet18.load_model("model_param/resnet18.bin"))
        exit(-1);
	//opencv读取图片是BGR格式，我们需要转换为RGB格式
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, 224, 224);
    
    //图像归一标准化，以R通道为例（x/225-0.485）/0.229，化简后可以得到下面的式子
    //需要注意的式substract_mean_normalize里的方差其实是方差的倒数，这样在算的时候就可以将除法转换为乘法计算
    //所以norm_vals里用的是1除
    const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = resnet18.create_extractor();
	
    //把图像数据放入input.1这个blob里
    ex.input("input.1", in);

    ncnn::Mat out;
    //提取出推理结果，推理结果存放在191这个blob里
    ex.extract("191", out);

    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
	
    //使用opencv读取图片
    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> cls_scores;
    detect_resnet18(m, cls_scores);
	
    //打印得分前三的类别
    print_topk(cls_scores, 3);

    return 0;
}

```

​	推理代码主要参照ncnn/examples/squeezenet.cpp编写，这里把代码需要注意的地方给大家解释一下 

```c++
ncnn::Net resnet18;
resnet18.opt.use_vulkan_compute = true;
//分别加载模型的参数和数据
if (resnet18.load_param("model_param/resnet18.param"))
    exit(-1);
if (resnet18.load_model(""))
    exit(-1);
//opencv读取图片是BGR格式，我们需要转换为RGB格式
ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, 224, 224);
```

​	创建一个Net，再加载模型的参数和数据，`model_param/resnet18.param`和`model_param/resnet18.bin`就是我们使用onnx2ncnn工具将resnet18.onnx转换出来的。**Opencv读取图片默认是BGR格式，我们需要转换成RGB格式**。

```c++
const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
in.substract_mean_normalize(mean_vals, norm_vals);
```

​	imagenet图片三通道的均值和标准差分别是mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]。以R通道为例，原始图片的像素值是从0到255，所以像素值归一化即像x/255，减去均值再除以方差就是（x/255-0.485）/0.299，把255乘下去也就是（x-0.485×255）/255×0.299。如果把归一化和标准化一起处理的话，等价均值就是0.485×255，等价方差就是255×0.299。但由于substract_mean_normalize里的方差实际是方差的倒数，这样可以把除法转换为乘法来计算加快效率，所以这里norm_vals用的是方差的倒数。

```c++
ncnn::Extractor ex = resnet18.create_extractor();
//把图像数据放入input.1这个blob里
ex.input("input.1", in);
ncnn::Mat out;
//提取出推理结果，推理结果存放在191这个blob里
ex.extract("191", out);
```

​	创建提取类，把输入放在input.1这个blob里面，再提取191这个blob的值放在out里。我们前面提到过整个网络的输入输出就是input.1和191。最后我们就可以使用print_topk打印出分数前三高的类别。

## CMakeLists.txt编写

```cmake
project(NCNN_DEMO)
cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_BUILD_TYPE Debug)


set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/hupeng/code/github/ncnn/build/install")
find_package(ncnn   REQUIRED)
find_package(OpenCV REQUIRED)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(resnet18 src/resnet18.cpp)
target_link_libraries(resnet18 ncnn ${OpenCV_LIBS})
```

​	`set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/hupeng/code/github/ncnn/build/install")`是设置了ncnn库的搜索路径，因为我在编译安装ncnn的时候不是安装在系统目录下，所有这里要知名一下，`set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)`设置了一下可执行文件的输出路径是在bin目录下，

`target_link_libraries(resnet18 ncnn ${OpenCV_LIBS})`链接到ncnn和opencv库。
