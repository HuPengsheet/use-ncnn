# ncnn的模型推理过程

```c++
ncnn::Net squeezenet;

squeezenet.opt.use_vulkan_compute = true;

if (squeezenet.load_param("squeezenet_v1.1.param"))
    exit(-1);
if (squeezenet.load_model("squeezenet_v1.1.bin"))
    exit(-1);

ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);

const float mean_vals[3] = {104.f, 117.f, 123.f};
in.substract_mean_normalize(mean_vals, 0);

ncnn::Extractor ex = squeezenet.create_extractor();

ex.input("data", in);

ncnn::Mat out;
ex.extract("prob", out);
```

​	我们以这段代码为例，向大家讲解网络推理的过程。

```	
7767517
75 83
Input            data             0 1 data 0=227 1=227 2=3
Convolution      conv1            1 1 data conv1 0=64 1=3 2=1 3=2 4=0 5=1 6=1728
ReLU             relu_conv1       1 1 conv1 conv1_relu_conv1 0=0.000000
Pooling          pool1            1 1 conv1_relu_conv1 pool1 0=0 1=3 2=2 3=0 4=0
Convolution      fire2/squeeze1x1 1 1 pool1 fire2/squeeze1x1 0=16 1=1 2=1 3=1 4=0 5=1 6=1024
ReLU             fire2/relu_squeeze1x1 1 1 fire2/squeeze1x1 fire2/squeeze1x1_fire2/relu_squeeze1x1 0=0.000000
Split            splitncnn_0      1 2 fire2/squeeze1x1_fire2/relu_squeeze1x1 fire2/squeeze1x1_fire2/relu_squeeze1x1_splitncnn_0 fire2/squeeze1x1_fire2/relu_squeeze1x1_splitncnn_1
Convolution      fire2/expand1x1  1 1 fire2/squeeze1x1_fire2/relu_squeeze1x1_splitncnn_1 fire2/expand1x1 0=64 1=1 2=1 3=1 4=0 5=1 6=1024
ReLU             fire2/relu_expand1x1 1 1 fire2/expand1x1 fire2/expand1x1_fire2/relu_expand1x1 0=0.000000
Convolution      fire2/expand3x3  1 1 fire2/squeeze1x1_fire2/relu_squeeze1x1_splitncnn_0 fire2/expand3x3 0=64 1=3 2=1 3=1 4=1 5=1 6=9216
ReLU             fire2/relu_expand3x3 1 1 fire2/expand3x3 fire2/expand3x3_fire2/relu_expand3x3 0=0.000000
Concat           fire2/concat     2 1 fire2/expand1x1_fire2/relu_expand1x1 fire2/expand3x3_fire2/relu_expand3x3 fire2/concat 0=0
Convolution      fire3/squeeze1x1 1 1 fire2/concat fire3/squeeze1x1 0=16 1=1 2=1 3=1 4=0 5=1 6=2048
ReLU             fire3/relu_squeeze1x1 1 1 fire3/squeeze1x1 fire3/squeeze1x1_fire3/relu_squeeze1x1 0=0.000000
Split            splitncnn_1      1 2 fire3/squeeze1x1_fire3/relu_squeeze1x1 fire3/squeeze1x1_fire3/relu_squeeze1x1_splitncnn_0 fire3/squeeze1x1_fire3/relu_squeeze1x1_splitncnn_1
Convolution      fire3/expand1x1  1 1 fire3/squeeze1x1_fire3/relu_squeeze1x1_splitncnn_1 fire3/expand1x1 0=64 1=1 2=1 3=1 4=0 5=1 6=1024
ReLU             fire3/relu_expand1x1 1 1 fire3/expand1x1 fire3/expand1x1_fire3/relu_expand1x1 0=0.000000
Convolution      fire3/expand3x3  1 1 fire3/squeeze1x1_fire3/relu_squeeze1x1_splitncnn_0 fire3/expand3x3 0=64 1=3 2=1 3=1 4=1 5=1 6=9216
ReLU             fire3/relu_expand3x3 1 1 fire3/expand3x3 fire3/expand3x3_fire3/relu_expand3x3 0=0.000000
Concat           fire3/concat     2 1 fire3/expand1x1_fire3/relu_expand1x1 fire3/expand3x3_fire3/relu_expand3x3 fire3/concat 0=0
Pooling          pool3            1 1 fire3/concat pool3 0=0 1=3 2=2 3=0 4=0
Convolution      fire4/squeeze1x1 1 1 pool3 fire4/squeeze1x1 0=32 1=1 2=1 3=1 4=0 5=1 6=4096
ReLU             fire4/relu_squeeze1x1 1 1 fire4/squeeze1x1 fire4/squeeze1x1_fire4/relu_squeeze1x1 0=0.000000
Split            splitncnn_2      1 2 fire4/squeeze1x1_fire4/relu_squeeze1x1 fire4/squeeze1x1_fire4/relu_squeeze1x1_splitncnn_0 fire4/squeeze1x1_fire4/relu_squeeze1x1_splitncnn_1
Convolution      fire4/expand1x1  1 1 fire4/squeeze1x1_fire4/relu_squeeze1x1_splitncnn_1 fire4/expand1x1 0=128 1=1 2=1 3=1 4=0 5=1 6=4096
ReLU             fire4/relu_expand1x1 1 1 fire4/expand1x1 fire4/expand1x1_fire4/relu_expand1x1 0=0.000000
Convolution      fire4/expand3x3  1 1 fire4/squeeze1x1_fire4/relu_squeeze1x1_splitncnn_0 fire4/expand3x3 0=128 1=3 2=1 3=1 4=1 5=1 6=36864
ReLU             fire4/relu_expand3x3 1 1 fire4/expand3x3 fire4/expand3x3_fire4/relu_expand3x3 0=0.000000
Concat           fire4/concat     2 1 fire4/expand1x1_fire4/relu_expand1x1 fire4/expand3x3_fire4/relu_expand3x3 fire4/concat 0=0
Convolution      fire5/squeeze1x1 1 1 fire4/concat fire5/squeeze1x1 0=32 1=1 2=1 3=1 4=0 5=1 6=8192
ReLU             fire5/relu_squeeze1x1 1 1 fire5/squeeze1x1 fire5/squeeze1x1_fire5/relu_squeeze1x1 0=0.000000
Split            splitncnn_3      1 2 fire5/squeeze1x1_fire5/relu_squeeze1x1 fire5/squeeze1x1_fire5/relu_squeeze1x1_splitncnn_0 fire5/squeeze1x1_fire5/relu_squeeze1x1_splitncnn_1
Convolution      fire5/expand1x1  1 1 fire5/squeeze1x1_fire5/relu_squeeze1x1_splitncnn_1 fire5/expand1x1 0=128 1=1 2=1 3=1 4=0 5=1 6=4096
ReLU             fire5/relu_expand1x1 1 1 fire5/expand1x1 fire5/expand1x1_fire5/relu_expand1x1 0=0.000000
Convolution      fire5/expand3x3  1 1 fire5/squeeze1x1_fire5/relu_squeeze1x1_splitncnn_0 fire5/expand3x3 0=128 1=3 2=1 3=1 4=1 5=1 6=36864
ReLU             fire5/relu_expand3x3 1 1 fire5/expand3x3 fire5/expand3x3_fire5/relu_expand3x3 0=0.000000
Concat           fire5/concat     2 1 fire5/expand1x1_fire5/relu_expand1x1 fire5/expand3x3_fire5/relu_expand3x3 fire5/concat 0=0
Pooling          pool5            1 1 fire5/concat pool5 0=0 1=3 2=2 3=0 4=0
Convolution      fire6/squeeze1x1 1 1 pool5 fire6/squeeze1x1 0=48 1=1 2=1 3=1 4=0 5=1 6=12288
ReLU             fire6/relu_squeeze1x1 1 1 fire6/squeeze1x1 fire6/squeeze1x1_fire6/relu_squeeze1x1 0=0.000000
Split            splitncnn_4      1 2 fire6/squeeze1x1_fire6/relu_squeeze1x1 fire6/squeeze1x1_fire6/relu_squeeze1x1_splitncnn_0 fire6/squeeze1x1_fire6/relu_squeeze1x1_splitncnn_1
Convolution      fire6/expand1x1  1 1 fire6/squeeze1x1_fire6/relu_squeeze1x1_splitncnn_1 fire6/expand1x1 0=192 1=1 2=1 3=1 4=0 5=1 6=9216
ReLU             fire6/relu_expand1x1 1 1 fire6/expand1x1 fire6/expand1x1_fire6/relu_expand1x1 0=0.000000
Convolution      fire6/expand3x3  1 1 fire6/squeeze1x1_fire6/relu_squeeze1x1_splitncnn_0 fire6/expand3x3 0=192 1=3 2=1 3=1 4=1 5=1 6=82944
ReLU             fire6/relu_expand3x3 1 1 fire6/expand3x3 fire6/expand3x3_fire6/relu_expand3x3 0=0.000000
Concat           fire6/concat     2 1 fire6/expand1x1_fire6/relu_expand1x1 fire6/expand3x3_fire6/relu_expand3x3 fire6/concat 0=0
Convolution      fire7/squeeze1x1 1 1 fire6/concat fire7/squeeze1x1 0=48 1=1 2=1 3=1 4=0 5=1 6=18432
ReLU             fire7/relu_squeeze1x1 1 1 fire7/squeeze1x1 fire7/squeeze1x1_fire7/relu_squeeze1x1 0=0.000000
Split            splitncnn_5      1 2 fire7/squeeze1x1_fire7/relu_squeeze1x1 fire7/squeeze1x1_fire7/relu_squeeze1x1_splitncnn_0 fire7/squeeze1x1_fire7/relu_squeeze1x1_splitncnn_1
Convolution      fire7/expand1x1  1 1 fire7/squeeze1x1_fire7/relu_squeeze1x1_splitncnn_1 fire7/expand1x1 0=192 1=1 2=1 3=1 4=0 5=1 6=9216
ReLU             fire7/relu_expand1x1 1 1 fire7/expand1x1 fire7/expand1x1_fire7/relu_expand1x1 0=0.000000
Convolution      fire7/expand3x3  1 1 fire7/squeeze1x1_fire7/relu_squeeze1x1_splitncnn_0 fire7/expand3x3 0=192 1=3 2=1 3=1 4=1 5=1 6=82944
ReLU             fire7/relu_expand3x3 1 1 fire7/expand3x3 fire7/expand3x3_fire7/relu_expand3x3 0=0.000000
Concat           fire7/concat     2 1 fire7/expand1x1_fire7/relu_expand1x1 fire7/expand3x3_fire7/relu_expand3x3 fire7/concat 0=0
Convolution      fire8/squeeze1x1 1 1 fire7/concat fire8/squeeze1x1 0=64 1=1 2=1 3=1 4=0 5=1 6=24576
ReLU             fire8/relu_squeeze1x1 1 1 fire8/squeeze1x1 fire8/squeeze1x1_fire8/relu_squeeze1x1 0=0.000000
Split            splitncnn_6      1 2 fire8/squeeze1x1_fire8/relu_squeeze1x1 fire8/squeeze1x1_fire8/relu_squeeze1x1_splitncnn_0 fire8/squeeze1x1_fire8/relu_squeeze1x1_splitncnn_1
Convolution      fire8/expand1x1  1 1 fire8/squeeze1x1_fire8/relu_squeeze1x1_splitncnn_1 fire8/expand1x1 0=256 1=1 2=1 3=1 4=0 5=1 6=16384
ReLU             fire8/relu_expand1x1 1 1 fire8/expand1x1 fire8/expand1x1_fire8/relu_expand1x1 0=0.000000
Convolution      fire8/expand3x3  1 1 fire8/squeeze1x1_fire8/relu_squeeze1x1_splitncnn_0 fire8/expand3x3 0=256 1=3 2=1 3=1 4=1 5=1 6=147456
ReLU             fire8/relu_expand3x3 1 1 fire8/expand3x3 fire8/expand3x3_fire8/relu_expand3x3 0=0.000000
Concat           fire8/concat     2 1 fire8/expand1x1_fire8/relu_expand1x1 fire8/expand3x3_fire8/relu_expand3x3 fire8/concat 0=0
Convolution      fire9/squeeze1x1 1 1 fire8/concat fire9/squeeze1x1 0=64 1=1 2=1 3=1 4=0 5=1 6=32768
ReLU             fire9/relu_squeeze1x1 1 1 fire9/squeeze1x1 fire9/squeeze1x1_fire9/relu_squeeze1x1 0=0.000000
Split            splitncnn_7      1 2 fire9/squeeze1x1_fire9/relu_squeeze1x1 fire9/squeeze1x1_fire9/relu_squeeze1x1_splitncnn_0 fire9/squeeze1x1_fire9/relu_squeeze1x1_splitncnn_1
Convolution      fire9/expand1x1  1 1 fire9/squeeze1x1_fire9/relu_squeeze1x1_splitncnn_1 fire9/expand1x1 0=256 1=1 2=1 3=1 4=0 5=1 6=16384
ReLU             fire9/relu_expand1x1 1 1 fire9/expand1x1 fire9/expand1x1_fire9/relu_expand1x1 0=0.000000
Convolution      fire9/expand3x3  1 1 fire9/squeeze1x1_fire9/relu_squeeze1x1_splitncnn_0 fire9/expand3x3 0=256 1=3 2=1 3=1 4=1 5=1 6=147456
ReLU             fire9/relu_expand3x3 1 1 fire9/expand3x3 fire9/expand3x3_fire9/relu_expand3x3 0=0.000000
Concat           fire9/concat     2 1 fire9/expand1x1_fire9/relu_expand1x1 fire9/expand3x3_fire9/relu_expand3x3 fire9/concat 0=0
Dropout          drop9            1 1 fire9/concat fire9/concat_drop9
Convolution      conv10           1 1 fire9/concat_drop9 conv10 0=1000 1=1 2=1 3=1 4=1 5=1 6=512000
ReLU             relu_conv10      1 1 conv10 conv10_relu_conv10 0=0.000000
Pooling          pool10           1 1 conv10_relu_conv10 pool10 0=1 1=0 2=1 3=0 4=1
Softmax          prob             1 1 pool10 prob 0=0
```

## 模型的加载和初始化

```c++
ncnn::Net squeezenet;

squeezenet.opt.use_vulkan_compute = true;

if (squeezenet.load_param("squeezenet_v1.1.param"))
    exit(-1);
if (squeezenet.load_model("squeezenet_v1.1.bin"))
    exit(-1);
```

​	模型的加载和初始化，这个在第四节就讲过了，这里就不在赘述。

## 图像的加载和归一标准化

```c++
ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);

const float mean_vals[3] = {104.f, 117.f, 123.f};
in.substract_mean_normalize(mean_vals, 0);
```

​	把opencv读取的图片转换为，ncnn的Mat，然后进行归一化和标准化的操作。

## 创建Extractor类

​	`ncnn::Extractor ex = squeezenet.create_extractor()`创建Extractor类，Extractor类是Net的类的友元类，我们看看他的定义	

```c++
class ExtractorPrivate;
class NCNN_EXPORT Extractor
{
public:
    virtual ~Extractor();

    // copy
    Extractor(const Extractor&);

    // assign
    Extractor& operator=(const Extractor&);


    void clear();

    void set_light_mode(bool enable);
    void set_num_threads(int num_threads);
    
    void set_blob_allocator(Allocator* allocator);
    void set_workspace_allocator(Allocator* allocator);

    int input(const char* blob_name, const Mat& in);
    int extract(const char* blob_name, Mat& feat, int type = 0);
    int extract(int blob_index, Mat& feat, int type = 0);


protected:
    friend Extractor Net::create_extractor() const;
    Extractor(const Net* net, size_t blob_count);

private:
    ExtractorPrivate* const d;
};

class ExtractorPrivate
{
public:
    ExtractorPrivate(const Net* _net)
        : net(_net)
    {
    }
    const Net* net;
    std::vector<Mat> blob_mats;
    Option opt;
};
```

​	两个重要的函数`input`和`extract`。`input`函数就是把Mat数据存放到`ExtractorPrivate`里的`blob_mats`中，对应的`extract`就是提取出`blob_mats`中某个Mat的值。

​	当创建`Extractor`类时，`ExtractorPrivate`里的net也就被赋值为我们正在使用的这个Net，也就是squeezenet。

## 输入数据

```c++
ex.input("data", in);
```

看看input函数的实现

```c++
int Extractor::input(int blob_index, const Mat& in)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    d->blob_mats[blob_index] = in;

    return 0;
}
```

​	其实非常简单，就是`blob_mats`中对应索引的Mat赋值。

## 提取数据，网络前向推理

```c++
ncnn::Mat out;
ex.extract("prob", out);
```

我们继续看extract函数的实现

```c++
int Extractor::extract(int blob_index, Mat& feat, int type)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    int old_blocktime = get_kmp_blocktime();
    set_kmp_blocktime(d->opt.openmp_blocktime);

    int old_flush_denormals = get_flush_denormals();
    set_flush_denormals(d->opt.flush_denormals);

    int ret = 0;

    if (d->blob_mats[blob_index].dims == 0)
    {
        int layer_index = d->net->blobs()[blob_index].producer;

        // use local allocator
        if (d->opt.use_local_pool_allocator)
        {
            if (!d->opt.blob_allocator)
            {
                d->opt.blob_allocator = d->net->d->local_blob_allocator;
            }
            if (!d->opt.workspace_allocator)
            {
                d->opt.workspace_allocator = d->net->d->local_workspace_allocator;
            }
        }
		ret = d->net->d->forward_layer(layer_index, d->blob_mats, d->opt);
    }

    feat = d->blob_mats[blob_index];

    if (d->opt.use_packing_layout && (type == 0) && feat.elempack != 1)
    {
        Mat bottom_blob_unpacked;
        convert_packing(feat, bottom_blob_unpacked, 1, d->opt);
        feat = bottom_blob_unpacked;
    }

    if (feat.elembits() == 8 && (type == 0))
    {
        Mat feat_fp32;
        cast_int8_to_float32(feat, feat_fp32, d->opt);
        feat = feat_fp32;
    }


    if (d->opt.use_local_pool_allocator && feat.allocator == d->net->d->local_blob_allocator)
    {
        feat = feat.clone();
    }

    set_kmp_blocktime(old_blocktime);
    set_flush_denormals(old_flush_denormals);

    return ret;
}

```

​	上面代码有点长我们先看这段：

```c++
if (d->blob_mats[blob_index].dims == 0)
{
    int layer_index = d->net->blobs()[blob_index].producer;

    // use local allocator
    if (d->opt.use_local_pool_allocator)
    {
        if (!d->opt.blob_allocator)
        {
            d->opt.blob_allocator = d->net->d->local_blob_allocator;
        }
        if (!d->opt.workspace_allocator)
        {
            d->opt.workspace_allocator = d->net->d->local_workspace_allocator;
        }
    }
    ret = d->net->d->forward_layer(layer_index, d->blob_mats, d->opt);
}
```

​	如果我们要提取的blob的Mat为空，则说明生成这个blob的算子还没有被执行，因此我们需要执行这个算子，也就是执行`ret = d->net->d->forward_layer(layer_index, d->blob_mats, d->opt)`。接下来我们看forward_layer

```c++
int NetPrivate::forward_layer(int layer_index, std::vector<Mat>& blob_mats, const Option& opt) const
{
    const Layer* layer = layers[layer_index];

    for (size_t i = 0; i < layer->bottoms.size(); i++)
    {
        int bottom_blob_index = layer->bottoms[i];

        if (blob_mats[bottom_blob_index].dims == 0)
        {
            int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, opt);
            if (ret != 0)
                return ret;
        }
    }

    int ret = 0;
    if (layer->featmask)
    {
        ret = do_forward_layer(layer, blob_mats, get_masked_option(opt, layer->featmask));
    }
    else
    {
        ret = do_forward_layer(layer, blob_mats, opt);
    }

    if (ret != 0)
        return ret;

    return 0;
}
```

​	forward_layer是一个递归函数，我们要执行一个算子，则需要判断它的输入blob的Mat是否为空，为空则需要递归的去执行生产该blob的算子，直到某一个算子的输入blob的Mat不为空，则递归终止，这个算子满足执行。我们在前面人工`ex.input("data", in)`人工放了一个blob的Mat，执行的肯定就是用的输入数据的那个算子，其实也就是conv1算子。`do_forward_layer`就是在递归完成后执行每一个算子，下面是do_forward_layer的代码。

```c++
int NetPrivate::do_forward_layer(const Layer* layer, std::vector<Mat>& blob_mats, const Option& opt) const
{
    if (layer->one_blob_only)
    {
        int bottom_blob_index = layer->bottoms[0];
        int top_blob_index = layer->tops[0];

        Mat& bottom_blob_ref = blob_mats[bottom_blob_index];
        Mat bottom_blob;

        if (opt.lightmode)
        {
            // deep copy for inplace forward if data is shared
            if (layer->support_inplace && *bottom_blob_ref.refcount != 1)
            {
                bottom_blob = bottom_blob_ref.clone(opt.blob_allocator);
            }
        }
        if (bottom_blob.dims == 0)
        {
            bottom_blob = bottom_blob_ref;
        }

        convert_layout(bottom_blob, layer, opt);

        // forward
        if (opt.lightmode && layer->support_inplace)
        {
            Mat& bottom_top_blob = bottom_blob;
            int ret = layer->forward_inplace(bottom_top_blob, opt);
            if (ret != 0)
                return ret;

            // store top blob
            blob_mats[top_blob_index] = bottom_top_blob;
        }
        else
        {
            Mat top_blob;
            int ret = layer->forward(bottom_blob, top_blob, opt);
            if (ret != 0)
                return ret;

            // store top blob
            blob_mats[top_blob_index] = top_blob;
        }

        if (opt.lightmode)
        {
            // delete after taken in light mode
            blob_mats[bottom_blob_index].release();
        }
    }
    else
    {
        std::vector<Mat> bottom_blobs(layer->bottoms.size());
        for (size_t i = 0; i < layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            Mat& bottom_blob_ref = blob_mats[bottom_blob_index];
            bottom_blobs[i].release();

            if (opt.lightmode)
            {
                // deep copy for inplace forward if data is shared
                if (layer->support_inplace && *bottom_blob_ref.refcount != 1)
                {
                    bottom_blobs[i] = bottom_blob_ref.clone(opt.blob_allocator);
                }
            }
            if (bottom_blobs[i].dims == 0)
            {
                bottom_blobs[i] = bottom_blob_ref;
            }

            convert_layout(bottom_blobs[i], layer, opt);
        }

        // forward
        if (opt.lightmode && layer->support_inplace)
        {
            std::vector<Mat>& bottom_top_blobs = bottom_blobs;
            int ret = layer->forward_inplace(bottom_top_blobs, opt);
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i = 0; i < layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats[top_blob_index] = bottom_top_blobs[i];
            }
        }
        else
        {
            std::vector<Mat> top_blobs(layer->tops.size());
            int ret = layer->forward(bottom_blobs, top_blobs, opt);
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i = 0; i < layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats[top_blob_index] = top_blobs[i];
            }
        }

        if (opt.lightmode)
        {
            for (size_t i = 0; i < layer->bottoms.size(); i++)
            {
                int bottom_blob_index = layer->bottoms[i];

                // delete after taken in light mode
                blob_mats[bottom_blob_index].release();
            }
        }
    }

    return 0;
}
```

​	会根据每个算子不同的类型，具体就是opt.lightmode，layer->one_blob_only，layer->support_inplace，去执行不同参数的layer->forward函数。

最后我们在回到extract函数中

```c++
feat = d->blob_mats[blob_index];
```

​	把网络已经推理好的结果提取出来，再进行一些后处理，就完成了整个的推理过程。