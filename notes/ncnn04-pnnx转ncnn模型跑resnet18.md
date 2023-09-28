## pnnx转ncnn模型跑resnet18

​	pnnx是ncnn的作者nihui自己开发的神经网络中间层格式，去掉了onnx这个中间商，可以直接从pytorch到ncnn。关于pnnx的介绍大家可以去看作者的讲解：https://zhuanlan.zhihu.com/p/427620428。

## 编译pnnx工具

​	1.pnnx有预编译的工具包，可以在https://github.com/pnnx/pnnx/releases/download/20230915/pnnx-20230915-ubuntu.zip上下载

```shell
wget https://github.com/pnnx/pnnx/releases/download/20230915/pnnx-20230915-ubuntu.zip
```

​	2.本教程选择从源码自己编译，来生成pnnx工具

```shell
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.10.0+cpu.zip 
cd you_dir/ncnn/tools/pnnx
mkdir build 
cd build
cmake .. -DTorchVision_INSTALL_DIR=/home/hupeng/code/github/libtorch
make -j8
```

​	有一些地方需要大家注意，第一个就是我们需要下载好libtorch的包并解压出来，pnnx编译时候会用到的libtorch的包。然后我们进入到`you_dir/ncnn/tools/pnnx`下，在建的build目录下编译。其次`TorchVision_INSTALL_DIR`就是我们解压出的libtorch包的路径，需要改成你自己的路径。并且需要有安装了torch的python环境下编译。编译过程会有点久，大家耐心等待。编译后会在生成`build/src/pnnx`的可执行文件。

## 在python中导出resnet18的torch.script

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

jit_model = torch.jit.trace(model, dummy_input）
jit_model.save('model_param/res18.pth') 

```

​	上面是`export_res18_pnnx.py`的代码，代码也很简单，就是给一个示例输入，然后跑一遍模型，相对应的图结构就会保存下来。运行代码后会在`model_param/res18.pth`下生成一个resnet18.onnx的模型文件。

## 模型转换

​	我们将编译生成的pnnx复制到我们的bin目录中来，我们在项目根目录下执行：

```shell
bin/pnnx model_param/res18.pth inputshape=[1,3,224,224]
```

​	即使用pnnx工具，将res18.pth转换为ncnn的格式，会在model_param下生成多个文件，我们需要的是model_param/res18.ncnn.param和model_param/res18.ncnn.bin。

​	我们可以看一看res18.ncnn.param的参数

```
7767517
50 58
Input                    in0                      0 1 in0
Convolution              convrelu_0               1 1 in0 1 0=64 1=7 11=7 12=1 13=2 14=3 2=1 3=2 4=3 5=1 6=9408 9=1
Pooling                  maxpool2d_22             1 1 1 2 0=0 1=3 11=3 12=2 13=1 2=2 3=1 5=1
Split                    splitncnn_0              1 2 2 3 4
Convolution              convrelu_1               1 1 4 5 0=64 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=36864 9=1
Convolution              conv_3                   1 1 5 6 0=64 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=36864
BinaryOp                 add_0                    2 1 6 3 7 0=0
ReLU                     relu_25                  1 1 7 8
Split                    splitncnn_1              1 2 8 9 10
Convolution              convrelu_2               1 1 10 11 0=64 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=36864 9=1
Convolution              conv_5                   1 1 11 12 0=64 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=36864
BinaryOp                 add_1                    2 1 12 9 13 0=0
ReLU                     relu_27                  1 1 13 14
Split                    splitncnn_2              1 2 14 15 16
Convolution              convrelu_3               1 1 16 17 0=128 1=3 11=3 12=1 13=2 14=1 2=1 3=2 4=1 5=1 6=73728 9=1
Convolution              conv_8                   1 1 17 18 0=128 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=147456
Convolution              conv_7                   1 1 15 19 0=128 1=1 11=1 12=1 13=2 14=0 2=1 3=2 4=0 5=1 6=8192
BinaryOp                 add_2                    2 1 18 19 20 0=0
ReLU                     relu_29                  1 1 20 21
Split                    splitncnn_3              1 2 21 22 23
Convolution              convrelu_4               1 1 23 24 0=128 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=147456 9=1
Convolution              conv_10                  1 1 24 25 0=128 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=147456
BinaryOp                 add_3                    2 1 25 22 26 0=0
ReLU                     relu_31                  1 1 26 27
Split                    splitncnn_4              1 2 27 28 29
Convolution              convrelu_5               1 1 29 30 0=256 1=3 11=3 12=1 13=2 14=1 2=1 3=2 4=1 5=1 6=294912 9=1
Convolution              conv_13                  1 1 30 31 0=256 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=589824
Convolution              conv_12                  1 1 28 32 0=256 1=1 11=1 12=1 13=2 14=0 2=1 3=2 4=0 5=1 6=32768
BinaryOp                 add_4                    2 1 31 32 33 0=0
ReLU                     relu_33                  1 1 33 34
Split                    splitncnn_5              1 2 34 35 36
Convolution              convrelu_6               1 1 36 37 0=256 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=589824 9=1
Convolution              conv_15                  1 1 37 38 0=256 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=589824
BinaryOp                 add_5                    2 1 38 35 39 0=0
ReLU                     relu_35                  1 1 39 40
Split                    splitncnn_6              1 2 40 41 42
Convolution              convrelu_7               1 1 42 43 0=512 1=3 11=3 12=1 13=2 14=1 2=1 3=2 4=1 5=1 6=1179648 9=1
Convolution              conv_18                  1 1 43 44 0=512 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=2359296
Convolution              conv_17                  1 1 41 45 0=512 1=1 11=1 12=1 13=2 14=0 2=1 3=2 4=0 5=1 6=131072
BinaryOp                 add_6                    2 1 44 45 46 0=0
ReLU                     relu_37                  1 1 46 47
Split                    splitncnn_7              1 2 47 48 49
Convolution              convrelu_8               1 1 49 50 0=512 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=2359296 9=1
Convolution              conv_20                  1 1 50 51 0=512 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=2359296
BinaryOp                 add_7                    2 1 51 48 52 0=0
ReLU                     relu_39                  1 1 52 53
Pooling                  gap_0                    1 1 53 54 0=1 4=1
Reshape                  reshape_40               1 1 54 55 0=1 1=1 2=-1
Flatten                  flatten_41               1 1 55 56
InnerProduct             linear_21                1 1 56 out0 0=1000 1=1 2=512000
```

​	我们需要记住输入blob是in0，输出blob是out0。

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
    if (resnet18.load_param("model_param/res18.ncnn.param"))
        exit(-1);
    if (resnet18.load_model("model_param/res18.ncnn.bin"))
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
	
    //把图像数据放入in0这个blob里
    ex.input("in0", in);

    ncnn::Mat out;
    //提取出推理结果，推理结果存放在out0这个blob里
    ex.extract("out0", out);

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

​	代码逻辑和第二节课的一样，只需要改一下模型加载的权重名字，和输出输出的blob名字就可以。编译运行后就可以执行

```shell
bin/resnet18_pnnx image/dog.jpg 
```

