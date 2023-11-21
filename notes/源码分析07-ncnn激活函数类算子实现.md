## ncnn激活函数类算子实现

​	ncnn里实现了多种激活函数，这里以Relu激活函数为例，为大家讲解。

## Relu类定义

```c++
class ReLU : public Layer
{
public:
    ReLU();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    float slope;
};
```

​	代码很简洁，就重写了两个虚函数，额外定义了一个slope参数

## Relu的构造函数

```c++
ReLU::ReLU()
{
    one_blob_only = true;
    support_inplace = true;
}
```

​	很简单，就是表明这个算子仅一个blob输入输出，支持就地推理。

## Relu的参数加载与算子计算

```
int ReLU::load_param(const ParamDict& pd)
{
    slope = pd.get(0, 0.f);

    return 0;
}

int ReLU::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = 0;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
        }
    }

    return 0;
}
```

​	`load_param`获取一个参数。

​	`forward_inplace`函数也非常简洁，就是遍历输入Mat的每一个值，如果这个值小于0，则赋值为0。如果用到了slope则，小于0的值乘slope即可。

