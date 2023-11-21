# ncnn的模型加载

```c++
ncnn::Net squeezenet;
if (squeezenet.load_param("squeezenet_v1.1.param"))
    exit(-1);
if (squeezenet.load_model("squeezenet_v1.1.bin"))
    exit(-1)
```

​	以上面几行代码为例，详细为大家讲讲load_param和load_model函数里都发生了哪些东西。

## 模型param文件的加载

### 打开文件初始化DataReaderFromStdio

```c++
int Net::load_param(const char* protopath)
{
    FILE* fp = fopen(protopath, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", protopath);
        return -1;
    }

    int ret = load_param(fp);
    fclose(fp);
    return ret;
}

int Net::load_param(FILE* fp)
{
    DataReaderFromStdio dr(fp);
    return load_param(dr);
}
```

​	调用参数类型为const char*的load_param，获得对应文件的，文件描述符。再调用参数为FILE*的load_param，初始化一个DataReaderFromStdio类，DataReaderFromStdio这个类我们不做详细介绍，它的主要作用就是读取一个个读取文本内容。

对应打开的param文件如下	

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

### 初始化Layer

​	最后调用参数为const DataReader&的load_param

```c++
int Net::load_param(const DataReader& dr)
{
#define SCAN_VALUE(fmt, v)                \
    if (dr.scan(fmt, &v) != 1)            \
    {                                     \
        NCNN_LOGE("parse " #v " failed"); \
        return -1;                        \
    }

    int magic = 0;
    SCAN_VALUE("%d", magic)
    if (magic != 7767517)
    {
        NCNN_LOGE("param is too old, please regenerate");
        return -1;
    }

    // parse
    int layer_count = 0;
    int blob_count = 0;
    SCAN_VALUE("%d", layer_count)
    SCAN_VALUE("%d", blob_count)
    if (layer_count <= 0 || blob_count <= 0)
    {
        NCNN_LOGE("invalid layer_count or blob_count");
        return -1;
    }

    d->layers.resize((size_t)layer_count);
    d->blobs.resize((size_t)blob_count);

    ParamDict pd;

    int blob_index = 0;
    for (int i = 0; i < layer_count; i++)
    {
        char layer_type[256];
        char layer_name[256];
        int bottom_count = 0;
        int top_count = 0;
        SCAN_VALUE("%255s", layer_type)
        SCAN_VALUE("%255s", layer_name)
        SCAN_VALUE("%d", bottom_count)
        SCAN_VALUE("%d", top_count)

        Layer* layer = create_overwrite_builtin_layer(layer_type);
        if (!layer)
        {
            layer = create_layer(layer_type);
        }
        if (!layer)
        {
            layer = create_custom_layer(layer_type);
        }
        if (!layer)
        {
            NCNN_LOGE("layer %s not exists or registered", layer_type);
            clear();
            return -1;
        }


        layer->type = std::string(layer_type);
        layer->name = std::string(layer_name);
        //         NCNN_LOGE("new layer %d %s", i, layer_name);

        layer->bottoms.resize(bottom_count);

        for (int j = 0; j < bottom_count; j++)
        {
            char bottom_name[256];
            SCAN_VALUE("%255s", bottom_name)

            int bottom_blob_index = find_blob_index_by_name(bottom_name);
            if (bottom_blob_index == -1)
            {
                Blob& blob = d->blobs[blob_index];

                bottom_blob_index = blob_index;

                blob.name = std::string(bottom_name);
                //                 NCNN_LOGE("new blob %s", bottom_name);

                blob_index++;
            }

            Blob& blob = d->blobs[bottom_blob_index];

            blob.consumer = i;

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j = 0; j < top_count; j++)
        {
            Blob& blob = d->blobs[blob_index];

            char blob_name[256];
            SCAN_VALUE("%255s", blob_name)

            blob.name = std::string(blob_name);
            //             NCNN_LOGE("new blob %s", blob_name);

            blob.producer = i;

            layer->tops[j] = blob_index;

            blob_index++;
        }

        // layer specific params
        int pdlr = pd.load_param(dr);
        if (pdlr != 0)
        {
            NCNN_LOGE("ParamDict load_param %d %s failed", i, layer->name.c_str());
            continue;
        }

        if (layer->support_int8_storage)
        {
            // no int8 gpu support yet
            opt.use_vulkan_compute = false;
        }

        // pull out top shape hints
        Mat shape_hints = pd.get(30, Mat());
        if (!shape_hints.empty())
        {
            const int* psh = shape_hints;
            for (int j = 0; j < top_count; j++)
            {
                Blob& blob = d->blobs[layer->tops[j]];

                int dims = psh[0];
                if (dims == 1)
                {
                    blob.shape = Mat(psh[1], (void*)0, 4u, 1);
                }
                if (dims == 2)
                {
                    blob.shape = Mat(psh[1], psh[2], (void*)0, 4u, 1);
                }
                if (dims == 3)
                {
                    blob.shape = Mat(psh[1], psh[2], psh[3], (void*)0, 4u, 1);
                }

                psh += 4;
            }
        }

        // set bottom and top shape hints
        layer->bottom_shapes.resize(bottom_count);
        for (int j = 0; j < bottom_count; j++)
        {
            layer->bottom_shapes[j] = d->blobs[layer->bottoms[j]].shape;
        }

        layer->top_shapes.resize(top_count);
        for (int j = 0; j < top_count; j++)
        {
            layer->top_shapes[j] = d->blobs[layer->tops[j]].shape;
        }

        // pull out layer specific feature disabled set
        layer->featmask = pd.get(31, 0);

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            NCNN_LOGE("layer load_param %d %s failed", i, layer->name.c_str());
            continue;
        }

        d->layers[i] = layer;
    }

    d->update_input_output_indexes();
    d->update_input_output_names();


    return 0;
}

```

​	

#### 第一步：检验magic number和读取layer数和blob数。

```c++
int magic = 0;
SCAN_VALUE("%d", magic)
if (magic != 7767517)
{
    NCNN_LOGE("param is too old, please regenerate");
    return -1;
}

// parse
int layer_count = 0;
int blob_count = 0;
SCAN_VALUE("%d", layer_count)
SCAN_VALUE("%d", blob_count)
```

​	读取的就是这两行的内容，magic=7767517，layer_count=75， blob_count=83。

```
7767517
75 83
```



#### 第二步：初始化每一个layer

以param文件里的这行为例：`Convolution      conv1            1 1 data conv1 0=64 1=3 2=1 3=2 4=0 5=1 6=1728`

```c++
char layer_type[256];
char layer_name[256];
int bottom_count = 0;
int top_count = 0;
SCAN_VALUE("%255s", layer_type)
SCAN_VALUE("%255s", layer_name)
SCAN_VALUE("%d", bottom_count)
SCAN_VALUE("%d", top_count)

Layer* layer = create_overwrite_builtin_layer(layer_type);
if (!layer)
{
    layer = create_layer(layer_type);
}
if (!layer)
{
    layer = create_custom_layer(layer_type);
}
if (!layer)
{
    NCNN_LOGE("layer %s not exists or registered", layer_type);
    clear();
    return -1;
}
```

​	分别读入算子类型，算子名称，算子输入blob个数，输出blob个数到`layer_type`，`layer_name`，`bottom_count`和`top_count`中。接下来就是执行`create_overwrite_builtin_layer`函数，create_overwrite_builtin_layer的作用是我们可以自己自己实现算子，然后把ncnn原本的算子覆盖掉，然后在初始化算子的时候，就可以初始化我们自己实现的算子了，但是大部分情况下我们也不会去干这种事情，所以这里layer为0。执行`layer = create_layer(layer_type)`这行函数,这行代码，会根据我们前面讲到的算子工厂，来new一个对应的算子。

​	`create_custom_layer`是给我们自定义算子使用。

​	如果没有符合的算子，这保存并退出。

```c++
layer->bottoms.resize(bottom_count);
for (int j = 0; j < bottom_count; j++)
{
    char bottom_name[256];
    SCAN_VALUE("%255s", bottom_name)

    int bottom_blob_index = find_blob_index_by_name(bottom_name);
    if (bottom_blob_index == -1)
    {
        Blob& blob = d->blobs[blob_index];

        bottom_blob_index = blob_index;

        blob.name = std::string(bottom_name);
        //                 NCNN_LOGE("new blob %s", bottom_name);

        blob_index++;
    }

    Blob& blob = d->blobs[bottom_blob_index];

    blob.consumer = i;

    layer->bottoms[j] = bottom_blob_index;
}

layer->tops.resize(top_count);
for (int j = 0; j < top_count; j++)
{
    Blob& blob = d->blobs[blob_index];

    char blob_name[256];
    SCAN_VALUE("%255s", blob_name)

    blob.name = std::string(blob_name);
    //             NCNN_LOGE("new blob %s", blob_name);

    blob.producer = i;

    layer->tops[j] = blob_index;

    blob_index++;
}


```

​	上面的代码就是，初始化算子的输出输出blob是哪个，并进一步初始化blob的生产算子和消费算子是哪一个。

```c++
Mat shape_hints = pd.get(30, Mat());
if (!shape_hints.empty())
{
    const int* psh = shape_hints;
    for (int j = 0; j < top_count; j++)
    {
        Blob& blob = d->blobs[layer->tops[j]];

        int dims = psh[0];
        if (dims == 1)
        {
            blob.shape = Mat(psh[1], (void*)0, 4u, 1);
        }
        if (dims == 2)
        {
            blob.shape = Mat(psh[1], psh[2], (void*)0, 4u, 1);
        }
        if (dims == 3)
        {
            blob.shape = Mat(psh[1], psh[2], psh[3], (void*)0, 4u, 1);
        }

        psh += 4;
    }
}

layer->bottom_shapes.resize(bottom_count);
for (int j = 0; j < bottom_count; j++)
{
    layer->bottom_shapes[j] = d->blobs[layer->bottoms[j]].shape;
}

layer->top_shapes.resize(top_count);
for (int j = 0; j < top_count; j++)
{
    layer->top_shapes[j] = d->blobs[layer->tops[j]].shape;
}

// pull out layer specific feature disabled set
layer->featmask = pd.get(31, 0);
```

​	初始化blob的尺寸和算子的尺寸。

```c++
int lr = layer->load_param(pd);
```

​	加载每个算子各自的参数。对应的就是0=64 1=3 2=1 3=2 4=0 5=1 6=1728。









## 模型bin文件的加载

```c++
int Net::load_model(FILE* fp)
{
    DataReaderFromStdio dr(fp);
    return load_model(dr);
}

int Net::load_model(const char* modelpath)
{
    FILE* fp = fopen(modelpath, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", modelpath);
        return -1;
    }

    int ret = load_model(fp)和
    fclose(fp);
    return ret;
}
```

​	和上面类似的，调用两次load_model，参数类型分别为const char*和FILE*的。

```c++
int Net::load_model(const DataReader& dr)
{
    if (d->layers.empty())
    {
        NCNN_LOGE("network graph not ready");
        return -1;
    }

    int layer_count = (int)d->layers.size();

    // load file
    int ret = 0;


    ModelBinFromDataReader mb(dr);
    for (int i = 0; i < layer_count; i++)
    {
        Layer* layer = d->layers[i];

        //Here we found inconsistent content in the parameter file.
        if (!layer)
        {
            NCNN_LOGE("load_model error at layer %d, parameter file has inconsistent content.", i);
            ret = -1;
            break;
        }

        int lret = layer->load_model(mb);
        if (lret != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("layer load_model %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("layer load_model %d failed", i);
#endif
            ret = -1;
            break;
        }

        if (layer->support_int8_storage)
        {
            // no int8 gpu support yet
            opt.use_vulkan_compute = false;
        }

        Option opt1 = get_masked_option(opt, layer->featmask);
        int cret = layer->create_pipeline(opt1);
        if (cret != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("layer create_pipeline %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("layer create_pipeline %d failed", i);
#endif
            ret = -1;
            break;
        }
    }

    if (opt.use_local_pool_allocator)
    {
        if (opt.blob_allocator == 0)
        {
            if (!d->local_blob_allocator)
            {
                d->local_blob_allocator = new PoolAllocator;
                d->local_blob_allocator->set_size_compare_ratio(0.f);
            }
        }
        if (opt.workspace_allocator == 0)
        {
            if (!d->local_workspace_allocator)
            {
                d->local_workspace_allocator = new PoolAllocator;
                d->local_workspace_allocator->set_size_compare_ratio(0.f);
            }
        }
    }

    return ret;
}

```

​	其实上面的代码很简单，就是遍历每一个算子调用`int lret = layer->load_model(mb)`，让每一个算子自己需要加载对应的权重，然后就是初始化一些参数。
