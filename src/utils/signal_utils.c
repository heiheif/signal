#include "../include/signal_lib.h"

// 创建信号结构体并分配内存
Signal* create_signal(size_t length, double fs) {
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
void destroy_signal(Signal* sig) {
    if (sig) {
        free(sig->data);
        free(sig);
    }
}

// 复数加法
Complex complex_add(Complex a, Complex b) {
    Complex result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

// 复数减法
Complex complex_subtract(Complex a, Complex b) {
    Complex result;
    result.real = a.real - b.real;
    result.imag = a.imag - b.imag;
    return result;
}

// 复数乘法
Complex complex_multiply(Complex a, Complex b) {
    Complex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

// 复数除法
Complex complex_divide(Complex a, Complex b) {
    Complex result;
    double denom = b.real * b.real + b.imag * b.imag;
    if (denom == 0.0) {
        result.real = result.imag = 0.0;
        return result;
    }
    result.real = (a.real * b.real + a.imag * b.imag) / denom;
    result.imag = (a.imag * b.real - a.real * b.imag) / denom;
    return result;
}

// 复数模值
double complex_abs(Complex z) {
    return sqrt(z.real * z.real + z.imag * z.imag);
}

// 复数相位
double complex_phase(Complex z) {
    return atan2(z.imag, z.real);
}

// 汉宁窗函数
void apply_hanning_window(double* data, size_t length) {
    if (!data || length == 0) {
        return;
    }

    for (size_t i = 0; i < length; i++) {
        double x = 2.0 * PI * i / (length - 1);
        data[i] *= 0.5 * (1.0 - cos(x));
    }
}

// 汉明窗函数
void apply_hamming_window(double* data, size_t length) {
    if (!data || length == 0) {
        return;
    }

    for (size_t i = 0; i < length; i++) {
        double x = 2.0 * PI * i / (length - 1);
        data[i] *= 0.54 - 0.46 * cos(x);
    }
}

// FFT递归实现
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

    // 合并结果
    for (size_t k = 0; k < n/2; k++) {
        double angle = (inverse ? 2.0 : -2.0) * PI * k / n;
        Complex w = {cos(angle), sin(angle)};
        Complex t = complex_multiply(w, odd[k]);
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

// FFT变换
void fft(Complex* data, size_t n) {
    fft_recursive(data, n, 0);
}

// IFFT变换
void ifft(Complex* data, size_t n) {
    fft_recursive(data, n, 1);
} 

