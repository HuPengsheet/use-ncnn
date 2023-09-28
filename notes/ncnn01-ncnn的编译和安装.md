## ncnn源码下载

1.直接在https://github.com/Tencent/ncnn地址进行下载。如果下载太慢可以在gitee里下载，地址是https://gitee.com/Tencent/ncnn?_from=gitee_search。

2.建议使用git下载

```shell
git clone https://github.com/Tencent/ncnn.git
cd ncnn 
git submodule update --init
```

## 安装相关的库

```shell
sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libvulkan-dev vulkan-utils libopencv-dev
```

## 编译ncnn

```cmake
option(NCNN_SHARED_LIB "shared library support" OFF)
option(NCNN_ENABLE_LTO "enable link-time optimization" OFF)
option(NCNN_OPENMP "openmp support" ON)
option(NCNN_STDIO "load model from external file" ON)
option(NCNN_STRING "plain and verbose string" ON)
option(NCNN_INSTALL_SDK "install ncnn library and headers" ON)
option(NCNN_SIMPLEOCV "minimal opencv structure emulation" OFF)
option(NCNN_SIMPLEOMP "minimal openmp runtime emulation" OFF)
option(NCNN_SIMPLESTL "minimal cpp stl structure emulation" OFF)
option(NCNN_THREADS "build with threads" ON)
option(NCNN_BENCHMARK "print benchmark information for every layer" OFF)
option(NCNN_C_API "build with C api" ON)
option(NCNN_PLATFORM_API "build with platform api candy" ON)
option(NCNN_PIXEL "convert and resize from/to image pixel" ON)
option(NCNN_PIXEL_ROTATE "rotate image pixel orientation" ON)
option(NCNN_PIXEL_AFFINE "warp affine image pixel" ON)
option(NCNN_PIXEL_DRAWING "draw basic figure and text" ON)
option(NCNN_CMAKE_VERBOSE "print verbose cmake messages" OFF)
option(NCNN_VULKAN "vulkan compute support" OFF)
option(NCNN_SYSTEM_GLSLANG "use system glslang library" OFF)
option(NCNN_RUNTIME_CPU "runtime dispatch cpu routines" ON)
option(NCNN_DISABLE_PIC "disable position-independent code" OFF)
option(NCNN_BUILD_TESTS "build tests" OFF)
option(NCNN_COVERAGE "build for coverage" OFF)
option(NCNN_ASAN "build for address sanitizer" OFF)
option(NCNN_BUILD_BENCHMARK "build benchmark" ON)
option(NCNN_PYTHON "build python api" OFF)
option(NCNN_INT8 "int8 inference" ON)
option(NCNN_BF16 "bf16 inference" ON)
option(NCNN_FORCE_INLINE "force inline some function" ON)
```

​	这部分代码是ncnn的主目录下CMakeLists.txt里的内容，用于控制构建选项，设置NCNN_BENCHMARK=ON可以打印出每个算子的耗时，NCNN_VULKAN=ON可以开启vulkan加速。由于大家可能有不同的编译需求，本教程尽量把ncnn所有能编译的都一起编译了。

```shell
#在ncnn主目录下
mkdir build && cd build
cmake .. -DNCNN_BENCHMARK=ON -DNCNN_VULKAN=ON
make -j8
```

​	成功编译后，我们对编译后的库进行安装，默认安装的根目录是：`your_dir/ncnn/build/install/`,也就是会在`build`目录下新建一个`install`目录来安装。

```shell
sudo make install
```

## 运行demo，验证编译成功

```shell
cd ../examples
../build/examples/squeezenet ../images/256-ncnn.png
```

运行结果：

```
532 = 0.165649
920 = 0.094421
716 = 0.062408
```

到此，代码编译成功

## 关于编译可能遇到的问题

1.如果想设置交叉编译器或者想编译成最小体积等额外的操做可以看：

https://github.com/Tencent/ncnn/wiki/how-to-build

https://github.com/Tencent/ncnn/wiki/build-minimal-library

2.关于编译的时候protobuf的问题可以看：

https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-protobuf-problem.zh

