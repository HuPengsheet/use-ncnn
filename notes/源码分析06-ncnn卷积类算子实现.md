# ncnn卷积类算子实现

​	像常见的Conv，最大池化，平均池化等算子都是卷积类算子，这里以二维卷积算子为例，给大家分析ncnn中朴素版卷积的实现。

## Convolution算子的定义

```c++
class Convolution : public Layer
{
public:
    Convolution();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int create_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const;
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, int kernel_w, int kernel_h, const Option& opt) const;

#if NCNN_INT8
    int forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif

public:
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_left; // -233=SAME_UPPER -234=SAME_LOWER
    int pad_right;
    int pad_top;
    int pad_bottom;
    float pad_value;
    int bias_term;

    int weight_data_size;

    int int8_scale_term;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    int dynamic_weight;

    // model
    Mat weight_data;
    Mat bias_data;

#if NCNN_INT8
    Mat weight_data_int8_scales;
    Mat bottom_blob_int8_scales;
    Mat top_blob_int8_scales;
#endif
};

} 
```

​	总体来看，Convolution继承自虚类Layer，然后重写了对应的虚函数，并额外增添了一些属性值，例如kernel_w卷积核的宽，kernel_h卷积核的高等。weight_data和bias_data保存的是卷积对应的权重。

## Convolution的构造函数

```c++
Convolution::Convolution()
{
    one_blob_only = true;
    support_inplace = false;
}
```

​	很简单，就是表明这个算子仅一个blob输入输出，不支持就地推理。

## load_param加载卷积相关参数

```c++
int Convolution::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    int8_scale_term = pd.get(8, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    dynamic_weight = pd.get(19, 0);

    if (dynamic_weight)
    {
        one_blob_only = false;
    }

    if (int8_scale_term)
    {
#if NCNN_INT8
        support_int8_storage = true;
#else
        NCNN_LOGE("please build ncnn with NCNN_INT8 enabled for int8 inference");
        return -1;
#endif
    }

    return 0;
}
```

```
Convolution      conv1            1 1 data conv1 0=64 1=3 2=1 3=2 4=0 5=1 6=1728
```

​	以上面为例，就是把param里0，1，2对应的参数，赋值给对应Convolution的属性值。

## load_model加载卷积相关参数

```c++
int Convolution::load_model(const ModelBin& mb)
{
    if (dynamic_weight)
        return 0;

    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}
```

​	类似的，就是读取bin文件的数据，用来初始化weight_data和bias_data。

## forward卷积计算

```c++
int Convolution::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8(bottom_blob, top_blob, opt);
    }
#endif

    // flattened blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        int num_input = weight_data_size / num_output;
        if (bottom_blob.w * bottom_blob.elempack == num_input)
        {
            // call InnerProduct
            ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::InnerProduct);

            // set param
            ncnn::ParamDict pd;
            pd.set(0, num_output);
            pd.set(1, bias_term);
            pd.set(2, weight_data_size);
            pd.set(8, int8_scale_term);
            pd.set(9, activation_type);
            pd.set(10, activation_params);

            op->load_param(pd);

            // set weights
            ncnn::Mat weights[4];
            weights[0] = weight_data;
            weights[1] = bias_data;

#if NCNN_INT8
            if (int8_scale_term)
            {
                weights[2] = weight_data_int8_scales;
                weights[3] = bottom_blob_int8_scales;
            }
#endif

            op->load_model(ModelBinFromMatArray(weights));

            op->create_pipeline(opt);

            // forward
            op->forward(bottom_blob, top_blob, opt);

            op->destroy_pipeline(opt);

            delete op;

            return 0;
        }
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const int w = bottom_blob_bordered.w;
    const int h = bottom_blob_bordered.h;
    const size_t elemsize = bottom_blob_bordered.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int ret = convolution(bottom_blob_bordered, top_blob, weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    return 0;
}

static int convolution(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_term)
                    sum = bias_data[p];

                const float* kptr = (const float*)weight_data + maxk * inch * p;

                for (int q = 0; q < inch; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        float val = sptr[space_ofs[k]]; // 20.72
                        float wt = kptr[k];
                        sum += val * wt; // 41.45
                    }

                    kptr += maxk;
                }

                outptr[j] = activation_ss(sum, activation_type, activation_params);
            }

            outptr += outw;
        }
    }

    return 0;
}
```

​	forward里进行一些参数的计算，实际调用的是convolution函数。

```c++
std::vector<int> _space_ofs(maxk);
int* space_ofs = &_space_ofs[0];
{
    int p1 = 0;
    int p2 = 0;
    int gap = w * dilation_h - kernel_w * dilation_w;
    for (int i = 0; i < kernel_h; i++)
    {
        for (int j = 0; j < kernel_w; j++)
        {
            space_ofs[p1] = p2;
            p1++;
            p2 += dilation_w;
        }
        p2 += gap;
    }
}
```

​	上面的代码是在计算卷积核的偏移。如果3*3的卷积核，在输入特征图尺寸为w=10，h=10的情况下的话，对应的偏移为，[0,1,2,10,11,12,20,21,22]。

```c++
for (int p = 0; p < outch; p++)      //遍历输出特征图的每一个channel
{
    float* outptr = top_blob.channel(p);

    for (int i = 0; i < outh; i++)   //遍历每个channel的h
    {
        for (int j = 0; j < outw; j++)  //遍历每个channel的w
        {
            float sum = 0.f;

            if (bias_term)              //如果有bias，加上去
                sum = bias_data[p];

            const float* kptr = (const float*)weight_data + maxk * inch * p;   //取卷积核的数据

            for (int q = 0; q < inch; q++)    //遍历输入特征图遍历每个channel
            {
                const Mat m = bottom_blob.channel(q);
                const float* sptr = m.row(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++) // 29.23   //遍历卷积核
                {
                    float val = sptr[space_ofs[k]]; // 20.72
                    float wt = kptr[k];
                    sum += val * wt; // 41.45
                }

                kptr += maxk;
            }

            outptr[j] = activation_ss(sum, activation_type, activation_params);  //顺便计算激活函数
        }

        outptr += outw;
    }
}
```

​	直接把解释写在每行代码里。