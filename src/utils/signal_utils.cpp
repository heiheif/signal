#include "../include/signal_lib.hpp"

// 创建信号结构体并分配内存
SIGNAL_LIB_API Signal* create_signal(size_t length, double fs) {
    if (length == 0 || fs <= 0) {
        return NULL;
    }

    Signal* sig = (Signal*)malloc(sizeof(Signal));
    if (!sig) {
        return NULL;
    }

    sig->data = (double*)malloc(length * sizeof(double));
    if (!sig->data) {
        free(sig);
        return NULL;
    }

    sig->length = length;
    sig->fs = fs;
    memset(sig->data, 0, length * sizeof(double));
    return sig;
}

// 释放信号结构体内存
SIGNAL_LIB_API void destroy_signal(Signal* sig) {
    if (sig) {
        free(sig->data);
        free(sig);
    }
}

// 信号功率计算
SIGNAL_LIB_API double calculate_signal_power(const Signal* sig)
{
    double* signal = sig->data;

    double sumSquared = 0.0;
    for (size_t i = 0; i < sig->length; i++)
    {
        sumSquared += signal[i] * signal[i];
    }
    return sumSquared / sig->length;    
}

// 添加AWGN噪声
SIGNAL_LIB_API Signal* add_AWGN_noise(const Signal* sig, double targetSNRdB)
{
    // 参数校验
    if (!sig || !sig->data || sig->length == 0) {
        return NULL;
    }
    
    // 计算信号功率
    double signalPower = calculate_signal_power(sig);
    
    // 根据SNR计算噪声功率
    double noisePower = signalPower / pow(10.0, targetSNRdB / 10.0);
    
    // 创建新的信号结构体来存储加噪后的信号
    Signal* noisySignal = create_signal(sig->length, sig->fs);
    if (!noisySignal) {
        return NULL;
    }
    
    // 初始化随机数生成器
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    
    // 生成高斯白噪声并添加到信号中
    // 使用简化的Box-Muller变换
    double noiseStd = sqrt(noisePower);
    
    for (size_t i = 0; i < sig->length; i++) {
        // 生成标准正态分布随机数 (简化版Box-Muller)
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        
        // 避免log(0)
        if (u1 == 0.0) u1 = 1e-10;
        
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
        
        // 生成噪声并添加到原信号
        double noise = noiseStd * z;
        noisySignal->data[i] = sig->data[i] + noise;
    }
    
    return noisySignal;
}

SIGNAL_LIB_API Signal* generate_sea_noise(double fs, double duration)
{
    // 参数校验
    if (fs <= 0 || duration <= 0) {
        return NULL;
    }
    
    // 计算信号长度
    size_t length = (size_t)(fs * duration);
    
    // 创建信号结构体
    Signal* seaNoise = create_signal(length, fs);
    if (!seaNoise) {
        return NULL;
    }
    
    // 初始化随机数生成器
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    
    // 生成白噪声
    for (size_t i = 0; i < length; i++) {
        // 使用Box-Muller变换生成高斯白噪声
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        
        if (u1 == 0.0) u1 = 1e-10;
        
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
        seaNoise->data[i] = z;
    }
    
    // 应用简单的低通滤波器，模拟海洋噪声的频谱特性
    // 使用一阶RC低通滤波器: y[n] = alpha * x[n] + (1-alpha) * y[n-1]
    double cutoff_freq = fs * 0.1;  // 截止频率设为采样率的10%
    double alpha = cutoff_freq * 2.0 * PI / fs / (cutoff_freq * 2.0 * PI / fs + 1.0);
    
    for (size_t i = 1; i < length; i++) {
        seaNoise->data[i] = alpha * seaNoise->data[i] + (1.0 - alpha) * seaNoise->data[i-1];
    }
    
    // 添加一些低频成分，模拟海浪声
    double wave_freq = 0.5; // 0.5 Hz的低频成分
    for (size_t i = 0; i < length; i++) {
        double t = (double)i / fs;
        double wave_component = 0.3 * sin(2.0 * PI * wave_freq * t);
        seaNoise->data[i] += wave_component;
    }
    
    // 归一化幅度
    double max_amp = 0.0;
    for (size_t i = 0; i < length; i++) {
        if (fabs(seaNoise->data[i]) > max_amp) {
            max_amp = fabs(seaNoise->data[i]);
        }
    }
    
    if (max_amp > 0.0) {
        for (size_t i = 0; i < length; i++) {
            seaNoise->data[i] /= max_amp;
        }
    }
    
    return seaNoise;
}




// 复数加法 (保持原有实现，简单运算无需优化)
SIGNAL_LIB_API Complex complex_add(Complex a, Complex b) {
    Complex result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

// 复数减法 (保持原有实现，简单运算无需优化)
SIGNAL_LIB_API Complex complex_subtract(Complex a, Complex b) {
    Complex result;
    result.real = a.real - b.real;
    result.imag = a.imag - b.imag;
    return result;
}

// 复数乘法 (使用增强版本提高数值稳定性)
SIGNAL_LIB_API Complex complex_multiply(Complex a, Complex b) {
    // 优先使用增强版本
    return eigen_complex_multiply(a, b);
}

// 复数除法 (使用增强版本避免除零和提高稳定性)
SIGNAL_LIB_API Complex complex_divide(Complex a, Complex b) {
    // 优先使用增强版本
    return eigen_complex_divide(a, b);
}

// 复数模值 (使用增强版本避免溢出/下溢)
SIGNAL_LIB_API double complex_abs(Complex z) {
    // 优先使用增强版本
    return eigen_complex_abs(z);
}

// 复数相位
SIGNAL_LIB_API double complex_phase(Complex z) {
    return atan2(z.imag, z.real);
}

// 汉宁窗函数
SIGNAL_LIB_API void apply_hanning_window(double* data, size_t length) {
    if (!data || length == 0) {
        return;
    }

    for (size_t i = 0; i < length; i++) {
        double x = 2.0 * PI * i / (length - 1);
        data[i] *= 0.5 * (1.0 - cos(x));
    }
}

// 汉明窗函数
SIGNAL_LIB_API void apply_hamming_window(double* data, size_t length) {
    if (!data || length == 0) {
        return;
    }

    for (size_t i = 0; i < length; i++) {
        double x = 2.0 * PI * i / (length - 1);
        data[i] *= 0.54 - 0.46 * cos(x);
    }
}

// FFT递归实现 (使用增强的复数运算)
void fft_recursive(Complex* data, size_t n, int inverse) {
    if (n <= 1) return;

    // 分配内存给偶数和奇数部分
    Complex* even = (Complex*)malloc(n/2 * sizeof(Complex));
    Complex* odd = (Complex*)malloc(n/2 * sizeof(Complex));
    if (!even || !odd) {
        free(even);
        free(odd);
        return;
    }

    // 分离偶数和奇数序列
    for (size_t i = 0; i < n/2; i++) {
        even[i] = data[2*i];
        odd[i] = data[2*i+1];
    }

    // 递归计算FFT
    fft_recursive(even, n/2, inverse);
    fft_recursive(odd, n/2, inverse);

    // 合并结果 (使用增强的复数运算)
    for (size_t k = 0; k < n/2; k++) {
        double angle = (inverse ? 2.0 : -2.0) * PI * k / n;
        Complex w = create_eigen_complex(cos(angle), sin(angle));
        Complex t = complex_multiply(w, odd[k]);  // 已经是增强版本
        data[k] = complex_add(even[k], t);
        data[k + n/2] = complex_subtract(even[k], t);

        // 如果是IFFT，需要除以2
        if (inverse) {
            data[k].real /= 2.0;
            data[k].imag /= 2.0;
            data[k + n/2].real /= 2.0;
            data[k + n/2].imag /= 2.0;
        }
    }

    // 释放内存
    free(even);
    free(odd);
}

// FFT变换 (优先使用增强版本)
SIGNAL_LIB_API void fft(Complex* data, size_t n) {
    // 尝试使用增强版本
    if (enhanced_fft(data, n, 1) != 0) {
        // 回退到原始实现
        fft_recursive(data, n, 0);
    }
}

// IFFT变换 (优先使用增强版本)
SIGNAL_LIB_API void ifft(Complex* data, size_t n) {
    // 尝试使用增强版本
    if (enhanced_fft(data, n, 0) != 0) {
        // 回退到原始实现
        fft_recursive(data, n, 1);
    }
} 



