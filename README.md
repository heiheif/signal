# Signal Processing Library

这是一个用于水声信号处理的C语言库，提供了丰富的信号处理和声学分析功能。

## 功能特性

### 1. 声学模型与仿真
- 声线传播模型
- 抛物方程声场计算
- 被动声纳方程计算

### 2. 通信信号生成
- 基础信号生成
- 连续波信号(CW)
- 线性调频信号(LFM)
- 双曲线调频信号(HFM)
- 相移键控信号(PSK)
- 频移键控信号(FSK)

### 3. 信号处理算法
- 匹配滤波处理
- 线谱分析
- 自适应能量检测
- 自适应谱检测
- 信号特征提取
- 加权中值滤波
- 多项式背景拟合

## 编译要求

- CMake 3.10或更高版本
- C编译器(支持C11标准)
- 数学库(libm)

## 编译步骤

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## 使用方法

1. 包含头文件:
```c
#include "signal_lib.h"
```

2. 链接库文件:
```cmake
target_link_libraries(your_project signal_lib)
```

3. 使用示例:
```c
// 创建一个1秒的1000Hz正弦信号
Signal* sig = generate_basic_signal(1000.0, 8000.0, 1.0);

// 进行频谱分析
Complex* spectrum = malloc(sig->length * sizeof(Complex));
size_t spec_length;
spectrum_analysis(sig, spectrum, &spec_length);

// 清理资源
destroy_signal(sig);
free(spectrum);
```

## API文档

详细的API文档请参考 `src/include/signal_lib.h` 文件中的注释说明。

## 注意事项

1. 所有返回指针的函数在使用完毕后需要手动释放内存
2. 信号处理前请确保采样率满足奈奎斯特采样定理
3. 部分算法可能需要较大的计算资源，请根据实际需求选择合适的参数

## 许可证

MIT License 