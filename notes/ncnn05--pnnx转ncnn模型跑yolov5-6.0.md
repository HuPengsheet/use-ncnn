# pnnx转ncnn模型跑resnet18

## 模型导出

```shell
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v6.0  #切换到v6.0分支
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt #下载6.0的权重
python export.py --weights yolov5s.pt --img 640 --batch 1 --train
cp yolov5s.torchscript.pt ../use-ncnn/model_param/
```

​	大家需要提前配置好虚拟环境，安装好队友的pytorch的环境，这里对这个过程不在赘述。`python export.py --weights yolov5s.pt --img 640 --batch 1 --train`加--tracin参数是去掉模型的后处理，避免NMS等操作也一起导出，拖累网络的推理速度。然后将生成的torchscript放到我们的model_param文件夹下。

## 模型转换

​	我们将编译生成的pnnx复制到我们的bin目录中来，我们在项目根目录下执行：

```shell
bin/pnnx model_param/yolov5s.torchscript.ptinputshape=[1,3,640,640]
```

​	即使用pnnx工具，将yolov5s.torchscript.pt转换为ncnn的格式，会在model_param下生成多个文件，我们需要的是model_param/yolov5s.torchscript.ncnn.param和model_param/yolov5s.torchscript.ncnn.bin。	

​	yolov5s.torchscript.ncnn.param的内容如下

```
7767517
167 191
Input                    in0                      0 1 in0
Convolution              conv_57                  1 1 in0 1 0=32 1=6 11=6 12=1 13=2 14=2 2=1 3=2 4=2 5=1 6=3456
Swish                    silu_0                   1 1 1 2
Convolution              conv_58                  1 1 2 3 0=64 1=3 11=3 12=1 13=2 14=1 2=1 3=2 4=1 5=1 6=18432
Swish                    silu_1                   1 1 3 4
Split                    splitncnn_0              1 2 4 5 6
Convolution              conv_59                  1 1 6 7 0=32 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=2048
Swish                    silu_2                   1 1 7 8
Split                    splitncnn_1              1 2 8 9 10
Convolution              conv_60                  1 1 10 11 0=32 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=1024
Swish                    silu_3                   1 1 11 12
Convolution              conv_61                  1 1 12 13 0=32 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=9216
Swish                    silu_4                   1 1 13 14
BinaryOp                 add_0                    2 1 9 14 15 0=0
Convolution              conv_62                  1 1 5 16 0=32 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=2048
Swish                    silu_5                   1 1 16 17
Concat                   cat_0                    2 1 15 17 18 0=0
Convolution              conv_63                  1 1 18 19 0=64 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=4096
Swish                    silu_6                   1 1 19 20
Convolution              conv_64                  1 1 20 21 0=128 1=3 11=3 12=1 13=2 14=1 2=1 3=2 4=1 5=1 6=73728
Swish                    silu_7                   1 1 21 22
Split                    splitncnn_2              1 2 22 23 24
Convolution              conv_65                  1 1 24 25 0=64 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=8192
Swish                    silu_8                   1 1 25 26
Split                    splitncnn_3              1 2 26 27 28
Convolution              conv_66                  1 1 28 29 0=64 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=4096
Swish                    silu_9                   1 1 29 30
Convolution              conv_67                  1 1 30 31 0=64 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=36864
Swish                    silu_10                  1 1 31 32
BinaryOp                 add_1                    2 1 27 32 33 0=0
Split                    splitncnn_4              1 2 33 34 35
Convolution              conv_68                  1 1 35 36 0=64 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=4096
Swish                    silu_11                  1 1 36 37
Convolution              conv_69                  1 1 37 38 0=64 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=36864
Swish                    silu_12                  1 1 38 39
BinaryOp                 add_2                    2 1 34 39 40 0=0
Convolution              conv_70                  1 1 23 41 0=64 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=8192
Swish                    silu_13                  1 1 41 42
Concat                   cat_1                    2 1 40 42 43 0=0
Convolution              conv_71                  1 1 43 44 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384
Swish                    silu_14                  1 1 44 45
Split                    splitncnn_5              1 2 45 46 47
Convolution              conv_72                  1 1 47 48 0=256 1=3 11=3 12=1 13=2 14=1 2=1 3=2 4=1 5=1 6=294912
Swish                    silu_15                  1 1 48 49
Split                    splitncnn_6              1 2 49 50 51
Convolution              conv_73                  1 1 51 52 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=32768
Swish                    silu_16                  1 1 52 53
Split                    splitncnn_7              1 2 53 54 55
Convolution              conv_74                  1 1 55 56 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384
Swish                    silu_17                  1 1 56 57
Convolution              conv_75                  1 1 57 58 0=128 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=147456
Swish                    silu_18                  1 1 58 59
BinaryOp                 add_3                    2 1 54 59 60 0=0
Split                    splitncnn_8              1 2 60 61 62
Convolution              conv_76                  1 1 62 63 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384
Swish                    silu_19                  1 1 63 64
Convolution              conv_77                  1 1 64 65 0=128 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=147456
Swish                    silu_20                  1 1 65 66
BinaryOp                 add_4                    2 1 61 66 67 0=0
Split                    splitncnn_9              1 2 67 68 69
Convolution              conv_78                  1 1 69 70 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384
Swish                    silu_21                  1 1 70 71
Convolution              conv_79                  1 1 71 72 0=128 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=147456
Swish                    silu_22                  1 1 72 73
BinaryOp                 add_5                    2 1 68 73 74 0=0
Convolution              conv_80                  1 1 50 75 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=32768
Swish                    silu_23                  1 1 75 76
Concat                   cat_2                    2 1 74 76 77 0=0
Convolution              conv_81                  1 1 77 78 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=65536
Swish                    silu_24                  1 1 78 79
Split                    splitncnn_10             1 2 79 80 81
Convolution              conv_82                  1 1 81 82 0=512 1=3 11=3 12=1 13=2 14=1 2=1 3=2 4=1 5=1 6=1179648
Swish                    silu_25                  1 1 82 83
Split                    splitncnn_11             1 2 83 84 85
Convolution              conv_83                  1 1 85 86 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=131072
Swish                    silu_26                  1 1 86 87
Split                    splitncnn_12             1 2 87 88 89
Convolution              conv_84                  1 1 89 90 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=65536
Swish                    silu_27                  1 1 90 91
Convolution              conv_85                  1 1 91 92 0=256 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=589824
Swish                    silu_28                  1 1 92 93
BinaryOp                 add_6                    2 1 88 93 94 0=0
Convolution              conv_86                  1 1 84 95 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=131072
Swish                    silu_29                  1 1 95 96
Concat                   cat_3                    2 1 94 96 97 0=0
Convolution              conv_87                  1 1 97 98 0=512 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=262144
Swish                    silu_30                  1 1 98 99
Convolution              conv_88                  1 1 99 100 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=131072
Swish                    silu_31                  1 1 100 101
Split                    splitncnn_13             1 2 101 102 103
Pooling                  maxpool2d_117            1 1 103 104 0=0 1=5 11=5 12=1 13=2 2=1 3=2 5=1
Split                    splitncnn_14             1 2 104 105 106
Pooling                  maxpool2d_118            1 1 106 107 0=0 1=5 11=5 12=1 13=2 2=1 3=2 5=1
Split                    splitncnn_15             1 2 107 108 109
Pooling                  maxpool2d_119            1 1 109 110 0=0 1=5 11=5 12=1 13=2 2=1 3=2 5=1
Concat                   cat_4                    4 1 102 105 108 110 111 0=0
Convolution              conv_89                  1 1 111 112 0=512 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=524288
Swish                    silu_32                  1 1 112 113
Convolution              conv_90                  1 1 113 114 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=131072
Swish                    silu_33                  1 1 114 115
Split                    splitncnn_16             1 2 115 116 117
Interp                   upsample_120             1 1 117 118 0=1 1=2.000000e+00 2=2.000000e+00 6=0
Concat                   cat_5                    2 1 118 80 119 0=0
Split                    splitncnn_17             1 2 119 120 121
Convolution              conv_91                  1 1 121 122 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=65536
Swish                    silu_34                  1 1 122 123
Convolution              conv_92                  1 1 123 124 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384
Swish                    silu_35                  1 1 124 125
Convolution              conv_93                  1 1 125 126 0=128 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=147456
Swish                    silu_37                  1 1 126 127
Convolution              conv_94                  1 1 120 128 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=65536
Swish                    silu_36                  1 1 128 129
Concat                   cat_6                    2 1 127 129 130 0=0
Convolution              conv_95                  1 1 130 131 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=65536
Swish                    silu_38                  1 1 131 132
Convolution              conv_96                  1 1 132 133 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=32768
Swish                    silu_39                  1 1 133 134
Split                    splitncnn_18             1 2 134 135 136
Interp                   upsample_121             1 1 136 137 0=1 1=2.000000e+00 2=2.000000e+00 6=0
Concat                   cat_7                    2 1 137 46 138 0=0
Split                    splitncnn_19             1 2 138 139 140
Convolution              conv_97                  1 1 140 141 0=64 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384
Swish                    silu_40                  1 1 141 142
Convolution              conv_98                  1 1 142 143 0=64 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=4096
Swish                    silu_41                  1 1 143 144
Convolution              conv_99                  1 1 144 145 0=64 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=36864
Swish                    silu_43                  1 1 145 146
Convolution              conv_100                 1 1 139 147 0=64 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384
Swish                    silu_42                  1 1 147 148
Concat                   cat_8                    2 1 146 148 149 0=0
Convolution              conv_101                 1 1 149 150 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384
Swish                    silu_44                  1 1 150 151
Split                    splitncnn_20             1 2 151 152 153
Convolution              conv_102                 1 1 153 154 0=128 1=3 11=3 12=1 13=2 14=1 2=1 3=2 4=1 5=1 6=147456
Swish                    silu_45                  1 1 154 155
Concat                   cat_9                    2 1 155 135 156 0=0
Split                    splitncnn_21             1 2 156 157 158
Convolution              conv_103                 1 1 158 159 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=32768
Swish                    silu_46                  1 1 159 160
Convolution              conv_104                 1 1 160 161 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384
Swish                    silu_47                  1 1 161 162
Convolution              conv_105                 1 1 162 163 0=128 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=147456
Swish                    silu_49                  1 1 163 164
Convolution              conv_106                 1 1 157 165 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=32768
Swish                    silu_48                  1 1 165 166
Concat                   cat_10                   2 1 164 166 167 0=0
Convolution              conv_107                 1 1 167 168 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=65536
Swish                    silu_50                  1 1 168 169
Split                    splitncnn_22             1 2 169 170 171
Convolution              conv_108                 1 1 171 172 0=256 1=3 11=3 12=1 13=2 14=1 2=1 3=2 4=1 5=1 6=589824
Swish                    silu_51                  1 1 172 173
Concat                   cat_11                   2 1 173 116 174 0=0
Split                    splitncnn_23             1 2 174 175 176
Convolution              conv_109                 1 1 176 177 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=131072
Swish                    silu_52                  1 1 177 178
Convolution              conv_110                 1 1 178 179 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=65536
Swish                    silu_53                  1 1 179 180
Convolution              conv_111                 1 1 180 181 0=256 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=589824
Swish                    silu_55                  1 1 181 182
Convolution              conv_112                 1 1 175 183 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=131072
Swish                    silu_54                  1 1 183 184
Concat                   cat_12                   2 1 182 184 185 0=0
Convolution              conv_114                 1 1 152 out0 0=255 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=32640
Convolution              conv_115                 1 1 170 out1 0=255 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=65280
Convolution              conv_113                 1 1 185 188 0=512 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=262144
Swish                    silu_56                  1 1 188 189
Convolution              conv_116                 1 1 189 out2 0=255 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=130560

```

​	yolov5采用特征金字塔，一共用三个地方输出，输出的blob分别是out0、out1和out2,只需要在对应的代码中修改输入输出就可以了。

