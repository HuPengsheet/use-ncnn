# Layer类

```c++
//src/layer.h

class NCNN_EXPORT Layer
{
public:
    
    Layer();
    virtual ~Layer();
   
    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);
    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);
public:
    bool one_blob_only;
    bool support_inplace;
    bool support_vulkan;
    bool support_packing;
    bool support_bf16_storage;
    bool support_fp16_storage;
    bool support_int8_storage;
    bool support_image_storage;
    bool support_tensor_storage;
    bool support_reserved_00;
    bool support_reserved_0;
    bool support_reserved_1;
    bool support_reserved_2;
    bool support_reserved_3;
    bool support_reserved_4;
    bool support_reserved_5;
    bool support_reserved_6;
    bool support_reserved_7;
    bool support_reserved_8;
    bool support_reserved_9;
    int featmask;
public:
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
    std::vector<int> bottoms;
    std::vector<int> tops;
    std::vector<Mat> bottom_shapes;
    std::vector<Mat> top_shapes;
};
```

​	上面是Layer类主要的定义（为了简洁，只放了重要的代码）。Layer类是一个虚类，在实现具体算子的时候会继承这个类，并重写虚函数，实现多态的效果。

## Layer类的主要属性

```c++
std::vector<int> bottoms;
std::vector<int> tops;
std::vector<Mat> bottom_shapes;
std::vector<Mat> top_shapes;
```

​	bottoms和tops分别是输入和输出的blob序号，blob可以理解成算子的输入输出。以Conv算子为例的话，Conv有一个输入blob和一个输出blob，bottoms记录了Conv输入的数据存放的序号，tops记录的Conv输出数据blob要存的地方的序号，Conv算子计算时从输入blob取数据，计算完毕后把数据存放到输出blob中。以Add算子为例的话，它就有两个输入Blob和一个输出Blob。

​	类似的，bottom_shapes和top_shapes记录的是blob的尺寸。

```c++
bool one_blob_only;
bool support_inplace;
bool support_vulkan;
bool support_packing;
bool support_bf16_storage;
bool support_fp16_storage;
bool support_int8_storage;
bool support_image_storage;
bool support_tensor_storage;
```

​	这些布尔变量，主要就是算子的类型`one_blob_only`即这个算子是否输入输出只有一个blob，`support_inplace`是指这个算子是否支持就地推理，比如一些激活函数算子就支持就地推理。其余的一些变量也是类似的，这里就不在赘述。

## Layer类的主要方法

```c++
virtual int load_param(const ParamDict& pd);
virtual int load_model(const ModelBin& mb);
```

​	`load_param`加载算子的参数，如Conv算子会有卷积核大小，步距等参数。`load_model`加载算子权重，如Conv算子有weight和bias权重，如果该算子没有权重，那只要不重写这个函数就可以。

```c++
virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
```

​	算子的forward函数一共有四个。算子`support_inplace`为true则调用的是`forward_inplace`函数，support_inplace为false调用的是`forward`函数。`one_blob_only`为true则调用的是函数参数为`std::vector<Mat>& bottom_blobs`，为false则调用的是函数参数为`Mat& bottom_top_blob`。

​	画个表

## Layer类的工厂模式

​	在代码编译完成后，会生成`build/src/layer_declaration.h`文件，`layer_declaration.h`里就是所有支持算子的集合。以卷积算子为例

```c++
//build/src/layer_declaration.h

#include "layer/convolution.h"
#include "layer/x86/convolution_x86.h"
namespace ncnn {
class Convolution_final : virtual public Convolution, virtual public Convolution_x86
{
public:
    virtual int create_pipeline(const Option& opt) {
        { int ret = Convolution::create_pipeline(opt); if (ret) return ret; }
        { int ret = Convolution_x86::create_pipeline(opt); if (ret) return ret; }
        return 0;
    }
    virtual int destroy_pipeline(const Option& opt) {
        { int ret = Convolution_x86::destroy_pipeline(opt); if (ret) return ret; }
        { int ret = Convolution::destroy_pipeline(opt); if (ret) return ret; }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Convolution_final)
} // namespace ncnn
```

​	卷积算子最后的实现是Convolution_final类，分别继承了Convolution类和Convolution_x86类。DEFINE_LAYER_CREATOR是一个宏，它定义在layer.h里面

```c++
//src/layer.h

#define DEFINE_LAYER_CREATOR(name)                          \
    ::ncnn::Layer* name##_layer_creator(void* /*userdata*/) \
    {                                                       \
        return new name;                                    \
    }
```

​	调用`DEFINE_LAYER_CREATOR(Convolution_final)`，实际上就是定义了Convolution_final_layer_creator这个函数，这个函数会new一个Convolution_final对象出来，从而实现工厂模式。

```c++
//build/src/layer_registry.h

static const layer_registry_entry layer_registry[] = {
#if NCNN_STRING
{"AbsVal", AbsVal_final_layer_creator},
#else
{AbsVal_final_layer_creator},
#endif
#if NCNN_STRING
{"ArgMax", 0},
#else
{0},
#endif
#if NCNN_STRING
{"BatchNorm", BatchNorm_final_layer_creator},
#else
{BatchNorm_final_layer_creator},
#endif
#if NCNN_STRING
{"Bias", Bias_final_layer_creator},
#else
{Bias_final_layer_creator},
#endif
#if NCNN_STRING
{"BNLL", BNLL_final_layer_creator},
#else
{BNLL_final_layer_creator},
#endif
----------------------------仅列举部分代码
```

​	在`/build/src/layer_registry.h`中定义了`layer_registry`这个数组，每个元素都是对应的算子名字和生产它们的工厂，到时候根据名字取到对应的工厂函数，调用工厂函数就可以产生对应的算子对象。

# Blob类

```c++
class NCNN_EXPORT Blob
{
public:
    // empty
    Blob();

public:
#if NCNN_STRING
    // blob name
    std::string name;
#endif // NCNN_STRING
    // layer index which produce this blob as output
    int producer;
    // layer index which need this blob as input
    int consumer;
    // shape hint
    Mat shape;
};
```

​	Blob类就很简单了，producer是产生这个blob的算子索引，consumer是用到这个blob的算子索引。shape就是尺寸。

# Net类的设计

```c++
class NetPrivate;
class NCNN_EXPORT Net
{
public:
    Net();
    virtual ~Net();
public:
    Option opt;
    
	int load_param(const char* protopath);
	int load_param(FILE* fp);
    int load_param(const DataReader& dr);
    
    int load_param_bin(const char* protopath);
    int load_param_bin(FILE* fp);
    int load_param_bin(const DataReader& dr);

    void clear();

    // construct an Extractor from network
    Extractor create_extractor() const;

    // get input/output indexes/names
    const std::vector<int>& input_indexes() const;
    const std::vector<int>& output_indexes() const;
    const std::vector<Blob>& blobs() const;
    const std::vector<Layer*>& layers() const;

    std::vector<Blob>& mutable_blobs();
    std::vector<Layer*>& mutable_layers();

protected:
    friend class Extractor;
    virtual Layer* create_custom_layer(int index);
    virtual Layer* create_overwrite_builtin_layer(int typeindex);

private:
    Net(const Net&);
    Net& operator=(const Net&);

private:
    NetPrivate* const d;
};


class NetPrivate
{
public:
    NetPrivate(Option& _opt);
    Option& opt;
    friend class Extractor;
    
    int forward_layer(int layer_index, std::vector<Mat>& blob_mats, const Option& opt) const;
    int convert_layout(Mat& bottom_blob, const Layer* layer, const Option& opt) const;
    int do_forward_layer(const Layer* layer, std::vector<Mat>& blob_mats, const Option& opt) const;

    void update_input_output_indexes();
    void update_input_output_names();


    std::vector<Blob> blobs;
    std::vector<Layer*> layers;

    std::vector<int> input_blob_indexes;
    std::vector<int> output_blob_indexes;
    std::vector<const char*> input_blob_names;
    std::vector<const char*> output_blob_names;


    std::vector<custom_layer_registry_entry> custom_layer_registry;
    std::vector<overwrite_builtin_layer_registry_entry> overwrite_builtin_layer_registry;

    PoolAllocator* local_blob_allocator;
    PoolAllocator* local_workspace_allocator;
};
```

​	上面就是Net类的主要定义（进行了部分精简）。总体上分成了两个部分，分别是`Net`类和`NetPrivate`类，下面我们对两个类都分别介绍。

## Net类

### 主要的类方法

```c++
int load_param(const char* protopath);
int load_param(FILE* fp);
int load_param(const DataReader& dr);

int load_model(const char* protopath);
int load_model(FILE* fp);
int load_model(const DataReader& dr);
```

load_param和load_model就是加载网络的模型结果和权重，例如我们在ncnn的example的demo里可以看到如下的代码

```c++
ncnn::Net yolov5;
if (yolov5.load_param("yolov5s_6.2.param"))
    exit(-1);
if (yolov5.load_model("yolov5s_6.2.bin"))
    exit(-1);
```

​	如上就是调用这两个函数，加载模型参数和权重，具体的细节，我们会在后面做进一步的介绍。其他的函数用的不多，我们不做进一步的介绍。

## NetPrivate类

​	Net类里面有一个NetPrivate的指针，会在Net构造函数里倍new出来。

### 主要的类属性值

```c++
std::vector<Blob> blobs;
std::vector<Layer*> layers;

std::vector<int> input_blob_indexes;
std::vector<int> output_blob_indexes;
std::vector<const char*> input_blob_names;
std::vector<const char*> output_blob_names;
```

​	`blobs`里存放的是网络里所有的blobs，到时候通过索引对其访问。同样`layers`里是网络里所有的Layer，到时候通过索引对其访问。因为Layer是一个虚类，所以这里是`Layer *`，这样才可以实现多态。

​	input_blob_indexes和output_blob_indexes是记录的是整个网络输入输出的blob索引，对应的input_blob_names和output_blob_names存储的是名字。

```c++
PoolAllocator* local_blob_allocator;
PoolAllocator* local_workspace_allocator;
```

​	这两个值，就是我们第一节里提到的内存管理相关内容。

### 主要的类方法

```c++
int forward_layer(int layer_index, std::vector<Mat>& blob_mats, const Option& opt) const;
int do_forward_layer(const Layer* layer, std::vector<Mat>& blob_mats, const Option& opt) const;
```

​	`forward_layer`是整个网络的forward，而`do_forward_layer`是某个算子的do_forward_layer。

# Optional类

```c++
class NCNN_EXPORT Option
{
public:
    // default option
    Option();

public:
    // light mode
    // intermediate blob will be recycled when enabled
    // enabled by default
    bool lightmode;

    // thread count
    // default value is the one returned by get_cpu_count()
    int num_threads;

    // blob memory allocator
    Allocator* blob_allocator;

    // workspace memory allocator
    Allocator* workspace_allocator;

    // the time openmp threads busy-wait for more work before going to sleep
    // default value is 20ms to keep the cores enabled
    // without too much extra power consumption afterwards
    int openmp_blocktime;

    // enable winograd convolution optimization
    // improve convolution 3x3 stride1 performance, may consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_winograd_convolution;

    // enable sgemm convolution optimization
    // improve convolution 1x1 stride1 performance, may consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_sgemm_convolution;

    // enable quantized int8 inference
    // use low-precision int8 path for quantized model
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_int8_inference;

    // enable vulkan compute
    bool use_vulkan_compute;

    // enable bf16 data type for storage
    // improve most operator performance on all arm devices, may consume more memory
    bool use_bf16_storage;

    // enable options for gpu inference
    bool use_fp16_packed;
    bool use_fp16_storage;
    bool use_fp16_arithmetic;
    bool use_int8_packed;
    bool use_int8_storage;
    bool use_int8_arithmetic;

    // enable simd-friendly packed memory layout
    // improve all operator performance on all arm devices, will consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_packing_layout;

    bool use_shader_pack8;

    // subgroup option
    bool use_subgroup_basic;
    bool use_subgroup_vote;
    bool use_subgroup_ballot;
    bool use_subgroup_shuffle;

    // turn on for adreno
    bool use_image_storage;
    bool use_tensor_storage;

    bool use_reserved_0;

    // enable DAZ(Denormals-Are-Zero) and FTZ(Flush-To-Zero)
    // default value is 3
    // 0 = DAZ OFF, FTZ OFF
    // 1 = DAZ ON , FTZ OFF
    // 2 = DAZ OFF, FTZ ON
    // 3 = DAZ ON,  FTZ ON
    int flush_denormals;

    bool use_local_pool_allocator;

    // enable local memory optimization for gpu inference
    bool use_shader_local_memory;

    // enable cooperative matrix optimization for gpu inference
    bool use_cooperative_matrix;

    // more fine-grained control of winograd convolution
    bool use_winograd23_convolution;
    bool use_winograd43_convolution;
    bool use_winograd63_convolution;

    // this option is turned on for A53/A55 automatically
    // but you can force this on/off if you wish
    bool use_a53_a55_optimized_kernel;

    bool use_reserved_7;
    bool use_reserved_8;
    bool use_reserved_9;
    bool use_reserved_10;
    bool use_reserved_11;
};
```

​	Optional类里是网络运行的各种参数，以`num_threads`为例，num_threads是使用openmp加速时多线程的线程个数。	

