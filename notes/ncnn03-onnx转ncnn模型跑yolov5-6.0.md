# onnx转ncnn模型跑yolov5-6.0

## 模型导出

```shell
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v6.0  #切换到v6.0分支
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt #下载6.0的权重
python export.py --weights yolov5s.pt --img 640 --batch 1 --train
python -m onnxsim yolov5s.onnx yolov5s-sim.onnx
```

​	大家需要提前配置好虚拟环境，安装好对应的pytorch的环境，这里对这个过程不在赘述。`python export.py --weights yolov5s.pt --img 640 --batch 1 --train`加--tracin参数是去掉模型的后处理，避免NMS等操作也一起导出，拖累网络的推理速度。然后使用onnxsim工具把网络结构进行算子精简。最后会生成`yolov5s-sim.onnx`的onnx模型文件

## 模型转换

​	ncnn官方提供了模型转换工具，来将导出的onnx模型转换为ncnn支持的格式，所有模型转换的源代码都在`ncnn/tools`目录下，在编译后也同样会在`build/tools/`下生成对应的可执行程序。我们将`you_dir/ncnn/build/tools/onnx/onnx2ncnn`复制到我们的bin目录中来。

​	我们在项目根目录下执行：

```shell
bin/onnx2ncnn model_param/yolov5s-sim.onnx model_param/yolov5s-6.0.param model_param/yolov5s-6.0.bin
```

​	即使用onnx2ncnn工具，将yolov5s-sim.onnx转换为yolov5s-6.0.param和yolov5s-6.0.bin，其中yolov5s-6.0.param为模型的参数信息（记录的是计算图的结构），yolov5s-6.0.bin里存放的是模型的所有具体的参数。下面是yolov5s-6.0.param的内容

```
7767517
173 197
Input            images                   0 1 images
Convolution      Conv_0                   1 1 images 122 0=32 1=6 11=6 2=1 12=1 3=2 13=2 4=2 14=2 15=2 16=2 5=1 6=3456
Swish            Mul_2                    1 1 122 124
Convolution      Conv_3                   1 1 124 125 0=64 1=3 11=3 2=1 12=1 3=2 13=2 4=1 14=1 15=1 16=1 5=1 6=18432
Swish            Mul_5                    1 1 125 127
Split            splitncnn_0              1 2 127 127_splitncnn_0 127_splitncnn_1
Convolution      Conv_6                   1 1 127_splitncnn_1 128 0=32 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=2048
Swish            Mul_8                    1 1 128 130
Split            splitncnn_1              1 2 130 130_splitncnn_0 130_splitncnn_1
Convolution      Conv_9                   1 1 130_splitncnn_1 131 0=32 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=1024
Swish            Mul_11                   1 1 131 133
Convolution      Conv_12                  1 1 133 134 0=32 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=9216
Swish            Mul_14                   1 1 134 136
BinaryOp         Add_15                   2 1 130_splitncnn_0 136 137 0=0
Convolution      Conv_16                  1 1 127_splitncnn_0 138 0=32 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=2048
Swish            Mul_18                   1 1 138 140
Concat           Concat_19                2 1 137 140 141 0=0
Convolution      Conv_20                  1 1 141 142 0=64 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=4096
Swish            Mul_22                   1 1 142 144
Convolution      Conv_23                  1 1 144 145 0=128 1=3 11=3 2=1 12=1 3=2 13=2 4=1 14=1 15=1 16=1 5=1 6=73728
Swish            Mul_25                   1 1 145 147
Split            splitncnn_2              1 2 147 147_splitncnn_0 147_splitncnn_1
Convolution      Conv_26                  1 1 147_splitncnn_1 148 0=64 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=8192
Swish            Mul_28                   1 1 148 150
Split            splitncnn_3              1 2 150 150_splitncnn_0 150_splitncnn_1
Convolution      Conv_29                  1 1 150_splitncnn_1 151 0=64 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=4096
Swish            Mul_31                   1 1 151 153
Convolution      Conv_32                  1 1 153 154 0=64 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=36864
Swish            Mul_34                   1 1 154 156
BinaryOp         Add_35                   2 1 150_splitncnn_0 156 157 0=0
Split            splitncnn_4              1 2 157 157_splitncnn_0 157_splitncnn_1
Convolution      Conv_36                  1 1 157_splitncnn_1 158 0=64 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=4096
Swish            Mul_38                   1 1 158 160
Convolution      Conv_39                  1 1 160 161 0=64 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=36864
Swish            Mul_41                   1 1 161 163
BinaryOp         Add_42                   2 1 157_splitncnn_0 163 164 0=0
Convolution      Conv_43                  1 1 147_splitncnn_0 165 0=64 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=8192
Swish            Mul_45                   1 1 165 167
Concat           Concat_46                2 1 164 167 168 0=0
Convolution      Conv_47                  1 1 168 169 0=128 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=16384
Swish            Mul_49                   1 1 169 171
Split            splitncnn_5              1 2 171 171_splitncnn_0 171_splitncnn_1
Convolution      Conv_50                  1 1 171_splitncnn_1 172 0=256 1=3 11=3 2=1 12=1 3=2 13=2 4=1 14=1 15=1 16=1 5=1 6=294912
Swish            Mul_52                   1 1 172 174
Split            splitncnn_6              1 2 174 174_splitncnn_0 174_splitncnn_1
Convolution      Conv_53                  1 1 174_splitncnn_1 175 0=128 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=32768
Swish            Mul_55                   1 1 175 177
Split            splitncnn_7              1 2 177 177_splitncnn_0 177_splitncnn_1
Convolution      Conv_56                  1 1 177_splitncnn_1 178 0=128 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=16384
Swish            Mul_58                   1 1 178 180
Convolution      Conv_59                  1 1 180 181 0=128 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=147456
Swish            Mul_61                   1 1 181 183
BinaryOp         Add_62                   2 1 177_splitncnn_0 183 184 0=0
Split            splitncnn_8              1 2 184 184_splitncnn_0 184_splitncnn_1
Convolution      Conv_63                  1 1 184_splitncnn_1 185 0=128 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=16384
Swish            Mul_65                   1 1 185 187
Convolution      Conv_66                  1 1 187 188 0=128 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=147456
Swish            Mul_68                   1 1 188 190
BinaryOp         Add_69                   2 1 184_splitncnn_0 190 191 0=0
Split            splitncnn_9              1 2 191 191_splitncnn_0 191_splitncnn_1
Convolution      Conv_70                  1 1 191_splitncnn_1 192 0=128 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=16384
Swish            Mul_72                   1 1 192 194
Convolution      Conv_73                  1 1 194 195 0=128 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=147456
Swish            Mul_75                   1 1 195 197
BinaryOp         Add_76                   2 1 191_splitncnn_0 197 198 0=0
Convolution      Conv_77                  1 1 174_splitncnn_0 199 0=128 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=32768
Swish            Mul_79                   1 1 199 201
Concat           Concat_80                2 1 198 201 202 0=0
Convolution      Conv_81                  1 1 202 203 0=256 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=65536
Swish            Mul_83                   1 1 203 205
Split            splitncnn_10             1 2 205 205_splitncnn_0 205_splitncnn_1
Convolution      Conv_84                  1 1 205_splitncnn_1 206 0=512 1=3 11=3 2=1 12=1 3=2 13=2 4=1 14=1 15=1 16=1 5=1 6=1179648
Swish            Mul_86                   1 1 206 208
Split            splitncnn_11             1 2 208 208_splitncnn_0 208_splitncnn_1
Convolution      Conv_87                  1 1 208_splitncnn_1 209 0=256 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=131072
Swish            Mul_89                   1 1 209 211
Split            splitncnn_12             1 2 211 211_splitncnn_0 211_splitncnn_1
Convolution      Conv_90                  1 1 211_splitncnn_1 212 0=256 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=65536
Swish            Mul_92                   1 1 212 214
Convolution      Conv_93                  1 1 214 215 0=256 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=589824
Swish            Mul_95                   1 1 215 217
BinaryOp         Add_96                   2 1 211_splitncnn_0 217 218 0=0
Convolution      Conv_97                  1 1 208_splitncnn_0 219 0=256 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=131072
Swish            Mul_99                   1 1 219 221
Concat           Concat_100               2 1 218 221 222 0=0
Convolution      Conv_101                 1 1 222 223 0=512 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=262144
Swish            Mul_103                  1 1 223 225
Convolution      Conv_104                 1 1 225 226 0=256 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=131072
Swish            Mul_106                  1 1 226 228
Split            splitncnn_13             1 2 228 228_splitncnn_0 228_splitncnn_1
Pooling          MaxPool_107              1 1 228_splitncnn_1 229 0=0 1=5 11=5 2=1 12=1 3=2 13=2 14=2 15=2 5=1
Split            splitncnn_14             1 2 229 229_splitncnn_0 229_splitncnn_1
Pooling          MaxPool_108              1 1 229_splitncnn_1 230 0=0 1=5 11=5 2=1 12=1 3=2 13=2 14=2 15=2 5=1
Split            splitncnn_15             1 2 230 230_splitncnn_0 230_splitncnn_1
Pooling          MaxPool_109              1 1 230_splitncnn_1 231 0=0 1=5 11=5 2=1 12=1 3=2 13=2 14=2 15=2 5=1
Concat           Concat_110               4 1 228_splitncnn_0 229_splitncnn_0 230_splitncnn_0 231 232 0=0
Convolution      Conv_111                 1 1 232 233 0=512 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=524288
Swish            Mul_113                  1 1 233 235
Convolution      Conv_114                 1 1 235 236 0=256 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=131072
Swish            Mul_116                  1 1 236 238
Split            splitncnn_16             1 2 238 238_splitncnn_0 238_splitncnn_1
Interp           Resize_120               1 1 238_splitncnn_1 243 0=1 1=2.000000e+00 2=2.000000e+00 3=0 4=0 6=0
Concat           Concat_121               2 1 243 205_splitncnn_0 244 0=0
Split            splitncnn_17             1 2 244 244_splitncnn_0 244_splitncnn_1
Convolution      Conv_122                 1 1 244_splitncnn_1 245 0=128 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=65536
Swish            Mul_124                  1 1 245 247
Convolution      Conv_125                 1 1 247 248 0=128 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=16384
Swish            Mul_127                  1 1 248 250
Convolution      Conv_128                 1 1 250 251 0=128 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=147456
Swish            Mul_130                  1 1 251 253
Convolution      Conv_131                 1 1 244_splitncnn_0 254 0=128 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=65536
Swish            Mul_133                  1 1 254 256
Concat           Concat_134               2 1 253 256 257 0=0
Convolution      Conv_135                 1 1 257 258 0=256 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=65536
Swish            Mul_137                  1 1 258 260
Convolution      Conv_138                 1 1 260 261 0=128 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=32768
Swish            Mul_140                  1 1 261 263
Split            splitncnn_18             1 2 263 263_splitncnn_0 263_splitncnn_1
Interp           Resize_144               1 1 263_splitncnn_1 268 0=1 1=2.000000e+00 2=2.000000e+00 3=0 4=0 6=0
Concat           Concat_145               2 1 268 171_splitncnn_0 269 0=0
Split            splitncnn_19             1 2 269 269_splitncnn_0 269_splitncnn_1
Convolution      Conv_146                 1 1 269_splitncnn_1 270 0=64 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=16384
Swish            Mul_148                  1 1 270 272
Convolution      Conv_149                 1 1 272 273 0=64 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=4096
Swish            Mul_151                  1 1 273 275
Convolution      Conv_152                 1 1 275 276 0=64 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=36864
Swish            Mul_154                  1 1 276 278
Convolution      Conv_155                 1 1 269_splitncnn_0 279 0=64 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=16384
Swish            Mul_157                  1 1 279 281
Concat           Concat_158               2 1 278 281 282 0=0
Convolution      Conv_159                 1 1 282 283 0=128 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=16384
Swish            Mul_161                  1 1 283 285
Split            splitncnn_20             1 2 285 285_splitncnn_0 285_splitncnn_1
Convolution      Conv_162                 1 1 285_splitncnn_1 286 0=128 1=3 11=3 2=1 12=1 3=2 13=2 4=1 14=1 15=1 16=1 5=1 6=147456
Swish            Mul_164                  1 1 286 288
Concat           Concat_165               2 1 288 263_splitncnn_0 289 0=0
Split            splitncnn_21             1 2 289 289_splitncnn_0 289_splitncnn_1
Convolution      Conv_166                 1 1 289_splitncnn_1 290 0=128 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=32768
Swish            Mul_168                  1 1 290 292
Convolution      Conv_169                 1 1 292 293 0=128 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=16384
Swish            Mul_171                  1 1 293 295
Convolution      Conv_172                 1 1 295 296 0=128 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=147456
Swish            Mul_174                  1 1 296 298
Convolution      Conv_175                 1 1 289_splitncnn_0 299 0=128 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=32768
Swish            Mul_177                  1 1 299 301
Concat           Concat_178               2 1 298 301 302 0=0
Convolution      Conv_179                 1 1 302 303 0=256 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=65536
Swish            Mul_181                  1 1 303 305
Split            splitncnn_22             1 2 305 305_splitncnn_0 305_splitncnn_1
Convolution      Conv_182                 1 1 305_splitncnn_1 306 0=256 1=3 11=3 2=1 12=1 3=2 13=2 4=1 14=1 15=1 16=1 5=1 6=589824
Swish            Mul_184                  1 1 306 308
Concat           Concat_185               2 1 308 238_splitncnn_0 309 0=0
Split            splitncnn_23             1 2 309 309_splitncnn_0 309_splitncnn_1
Convolution      Conv_186                 1 1 309_splitncnn_1 310 0=256 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=131072
Swish            Mul_188                  1 1 310 312
Convolution      Conv_189                 1 1 312 313 0=256 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=65536
Swish            Mul_191                  1 1 313 315
Convolution      Conv_192                 1 1 315 316 0=256 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=589824
Swish            Mul_194                  1 1 316 318
Convolution      Conv_195                 1 1 309_splitncnn_0 319 0=256 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=131072
Swish            Mul_197                  1 1 319 321
Concat           Concat_198               2 1 318 321 322 0=0
Convolution      Conv_199                 1 1 322 323 0=512 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=262144
Swish            Mul_201                  1 1 323 325
Convolution      Conv_202                 1 1 285_splitncnn_0 326 0=255 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=32640
Reshape          Reshape_225              1 1 326 349 0=-1 1=85 2=3
Permute          Transpose_226            1 1 349 output 0=1
Convolution      Conv_227                 1 1 305_splitncnn_0 351 0=255 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=65280
Reshape          Reshape_250              1 1 351 374 0=-1 1=85 2=3
Permute          Transpose_251            1 1 374 375 0=1
Convolution      Conv_252                 1 1 325 376 0=255 1=1 11=1 2=1 12=1 3=1 13=1 4=0 14=0 15=0 16=0 5=1 6=130560
Reshape          Reshape_275              1 1 376 399 0=-1 1=85 2=3
Permute          Transpose_276            1 1 399 400 0=1
```

​	yolov5采用特征金字塔，一共用三个地方输出，输出的blob分别是output、375和400。**另外我们要将reshape的参数进行修改，变成0=-1，这是为了支持动态尺寸输入**。

## YOLOV5后处理

​	首先假设我们设置的图片输入大小为640，当图片输入尺寸为1080×810×3，我们会把长边缩放到640，然后短边根据长边的缩放比例同步缩放，缩放完后图片的尺寸为640×480×3。最后我们需要将480上取整对齐到64的倍数，也就是512，所以图片的尺寸最后就是640×512×3。

​	图片经过模型的输出后会有三个尺度的输出，分别是8倍下采样，16倍下采样，32倍下采样，对应的就是80×64×[(5+cls)×3]，40×32×[(5+cls)×3]，20×16×[(5+cls)×3]。**(5+cls)×3中5是bbox的坐标和置信度，cls是类别数，3是anchor的数量**。

​	其中输出的bbox的四个坐标(tx,ty,pw,ph)也需要进行进一步的处理：

![image-20230927222718561](../image/222.png)

**bx=(2⋅σ(tx)−0.5)+cx**

**by=(2⋅σ(ty)−0.5)+cy**

**bw=pw⋅(2⋅σ(tw))^2**

**bh=ph⋅(2⋅σ(th)^2**

​	这四个公式都是yolov5代码里写的，最后将bx,by乘以下采样的倍数就对应到图片原始尺寸的坐标，bw和bh乘以anchor的宽和高就是图像原始尺度的宽高。最后将这些框放一起做NMS后处理就可以。这些都会在后面的推理代码中体现。

## 推理代码

```c++

#include "layer.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>
#include <iostream>

#define YOLOV5_V60 1 
void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int z=0; z<m.d; z++)
        {
            for (int y=0; y<m.h; y++)
            {
                for (int x=0; x<m.w; x++)
                {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");
    }
}
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;
	//遍历三个anchor
    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);
		//遍历每一个结果
        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);
                float box_confidence = sigmoid(featptr[4]);
                if (box_confidence >= prob_threshold)
                {
                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++)
                    {
                        float score = featptr[5 + k];
                        if (score > class_score)
                        {
                            class_index = k;
                            class_score = score;
                        }
                    }
                    float confidence = box_confidence * sigmoid(class_score);
                    if (confidence >= prob_threshold)
                    {
                        // yolov5/models/yolo.py Detect forward
                        // y = x[i].sigmoid()
                        // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
						//下面的代码就是前面说的对坐标的处理，
                        //bx,by乘以下采样的倍数就对应到图片原始尺寸的坐标，bw和bh乘以anchor的宽和高就是图像原始尺度的宽高
                        float dx = sigmoid(featptr[0]);
                        float dy = sigmoid(featptr[1]);
                        float dw = sigmoid(featptr[2]);
                        float dh = sigmoid(featptr[3]);

                        float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                        float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;

                        float x0 = pb_cx - pb_w * 0.5f;
                        float y0 = pb_cy - pb_h * 0.5f;
                        float x1 = pb_cx + pb_w * 0.5f;
                        float y1 = pb_cy + pb_h * 0.5f;

                        Object obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = x1 - x0;
                        obj.rect.height = y1 - y0;
                        obj.label = class_index;
                        obj.prob = confidence;

                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}

static int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net yolov5;

    yolov5.opt.use_vulkan_compute = true;
    //加载模型的参数
    if (yolov5.load_param("model_param/yolov5s-6.0.param"))
        exit(-1);
    if (yolov5.load_model("model_param/yolov5s-6.0.bin"))
        exit(-1);

    const int target_size = 640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;
    const int MAX_STRIDE = 64;
    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    //判断长边，然后做对应缩放
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    //把wpad和hpad对齐到MAX_STRIDE
    //就是利用除法的截断原理
    //其次就是因为要上取整所以加了MAX_STRIDE - 1
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    //填充
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    //归一化
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov5.create_extractor();
    std::cout<<in_pad.w<<"  "<<in_pad.h<<std::endl;
    ex.input("images", in_pad);

    std::vector<Object> proposals;

    

    //对8倍下采样的输出进行提取和处理
    {
        ncnn::Mat out;
        ex.extract("output", out);
        
        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    //对16倍下采样的输出进行提取和处理
    {
        ncnn::Mat out;
        ex.extract("375", out);
        

        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }
    
    //对32倍下采样的输出进行提取和处理
    {
        ncnn::Mat out;

        ex.extract("400", out);
        
        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    //nms前对所有bbox根据框的置信度排序
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolov5(m, objects);

    draw_objects(m, objects);

    return 0;
}

```

​	有一个地方需要注意一下，其他的直接把解释下代码里了

```c++
int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
```

​	比如w=640,h=480,MAX_STRIDE=64,那么wpad=（640+64-1）/64×64-640=0。hapd=（480+64-1）/64×64-640=543/64×64-640=8*64-480=32。**注意这里是整数的除法，小数会被截断，为了上取整所以加了MAX_STRIDE - 1**。

## 运行

编译完成后，运行

```shell
bin/yolov5s image/bus.jpg 
```



