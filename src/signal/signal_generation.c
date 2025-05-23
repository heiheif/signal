#include "../include/signal_lib.h"

// 基础信号生成函数实现
Signal* generate_basic_signal(double freq, double fs, double duration) {
    if (freq <= 0 || fs <= 0 || duration <= 0) {
        return NULL;
    }

    // 检查采样点数是否超过最大限制
    size_t length = (size_t)(fs * duration);
    if (length > MAX_SIGNAL_LENGTH) {
        return NULL;
    }

    // 创建信号
    Signal* sig = create_signal(length, fs);
    if (!sig) {
        return NULL;
    }

    // 生成正弦信号
    double dt = 1.0 / fs;
    for (size_t i = 0; i < length; i++) {
        sig->data[i] = sin(2.0 * PI * freq * i * dt);
    }

    return sig;
}

// 频谱分析函数实现
int spectrum_analysis(const Signal* sig, Complex* spectrum, size_t* spec_length) {
    if (!sig || !spectrum || !spec_length || sig->length == 0) {
        return -1;
    }

    // 计算FFT长度
    size_t fft_length = 1;
    while (fft_length < sig->length) {
        fft_length <<= 1;
    }
    *spec_length = fft_length;

    // 将信号数据复制到频谱数组
    for (size_t i = 0; i < sig->length; i++) {
        spectrum[i].real = sig->data[i];
        spectrum[i].imag = 0.0;
    }
    // 补零
    for (size_t i = sig->length; i < fft_length; i++) {
        spectrum[i].real = spectrum[i].imag = 0.0;
    }

    // 执行FFT
    fft_recursive(spectrum, fft_length, 1);

    return 0;
}

// CW信号生成函数实现
Signal* generate_cw(double freq, double fs, double duration, double amplitude, double phase) {
    if (freq <= 0 || fs <= 0 || duration <= 0) {
        return NULL;
    }

    // 检查采样点数是否超过最大限制
    size_t length = (size_t)(fs * duration);
    if (length > MAX_SIGNAL_LENGTH) {
        return NULL;
    }

    // 创建信号
    Signal* sig = create_signal(length, fs);
    if (!sig) {
        return NULL;
    }

    // 生成CW信号
    double dt = 1.0 / fs;
    for (size_t i = 0; i < length; i++) {
        sig->data[i] = amplitude * cos(2.0 * PI * freq * i * dt + phase);
    }

    return sig;
}

// LFM信号生成函数实现
Signal* generate_lfm(double f_start, double f_end, double fs, double duration) {
    if (f_start <= 0 || f_end <= 0 || fs <= 0 || duration <= 0) {
        return NULL;
    }

    double f_max = fmax(f_start, f_end);
    if (f_max > fs / 2) {  // 检查是否满足采样定理
        return NULL;
    }

    // 检查采样点数是否超过最大限制
    size_t length = (size_t)(fs * duration);
    if (length > MAX_SIGNAL_LENGTH) {
        return NULL;
    }

    // 创建信号
    Signal* sig = create_signal(length, fs);
    if (!sig) {
        return NULL;
    }

    // 计算频率斜率
    double k = (f_end - f_start) / duration;
    
    // 生成LFM信号
    double dt = 1.0 / fs;
    for (size_t i = 0; i < length; i++) {
        double t = i * dt;
        double phase = 2.0 * PI * (f_start * t + 0.5 * k * t * t);
        sig->data[i] = cos(phase);
    }

    return sig;
}

// HFM信号生成函数实现
Signal* generate_hfm(double f_start, double f_end, double fs, double duration) {
    if (f_start <= 0 || f_end <= 0 || fs <= 0 || duration <= 0) {
        return NULL;
    }

    double f_max = fmax(f_start, f_end);
    if (f_max > fs / 2) {  // 检查是否满足采样定理
        return NULL;
    }

    // 检查采样点数是否超过最大限制
    size_t length = (size_t)(fs * duration);
    if (length > MAX_SIGNAL_LENGTH) {
        return NULL;
    }

    // 创建信号
    Signal* sig = create_signal(length, fs);
    if (!sig) {
        return NULL;
    }

    // 计算参数
    double k = log(f_end / f_start) / duration;
    
    // 生成HFM信号
    double dt = 1.0 / fs;
    for (size_t i = 0; i < length; i++) {
        double t = i * dt;
        double phase = 2.0 * PI * f_start * (exp(k * t) - 1) / k;
        sig->data[i] = cos(phase);
    }

    return sig;
}

// PSK信号生成函数实现
Signal* generate_psk(double carrier_freq, double fs, int symbol_count, 
                    int samples_per_symbol, const int* symbols) {
    if (carrier_freq <= 0 || fs <= 0 || symbol_count <= 0 || 
        samples_per_symbol <= 0 || !symbols) {
        return NULL;
    }

    if (carrier_freq > fs / 2) {  // 检查是否满足采样定理
        return NULL;
    }

    // 检查采样点数是否超过最大限制
    size_t length = (size_t)(symbol_count * samples_per_symbol);
    if (length > MAX_SIGNAL_LENGTH) {
        return NULL;
    }

    // 创建信号
    Signal* sig = create_signal(length, fs);
    if (!sig) {
        return NULL;
    }

    // 生成PSK信号
    double dt = 1.0 / fs;
    for (int i = 0; i < symbol_count; i++) {
        double phase = PI * symbols[i];  // BPSK调制
        for (int j = 0; j < samples_per_symbol; j++) {
            size_t idx = i * samples_per_symbol + j;
            sig->data[idx] = cos(2.0 * PI * carrier_freq * idx * dt + phase);
        }
    }

    return sig;
}

// FSK信号生成函数实现
Signal* generate_fsk(double* freqs, int freq_count, double fs, 
                    int symbol_count, int samples_per_symbol, const int* symbols) {
    if (!freqs || freq_count <= 0 || fs <= 0 || symbol_count <= 0 || 
        samples_per_symbol <= 0 || !symbols) {
        return NULL;
    }

    // 检查所有频率是否满足采样定理
    for (int i = 0; i < freq_count; i++) {
        if (freqs[i] <= 0 || freqs[i] > fs / 2) {
            return NULL;
        }
    }

    // 检查采样点数是否超过最大限制
    size_t length = (size_t)(symbol_count * samples_per_symbol);
    if (length > MAX_SIGNAL_LENGTH) {
        return NULL;
    }

    // 创建信号
    Signal* sig = create_signal(length, fs);
    if (!sig) {
        return NULL;
    }

    // 生成FSK信号
    double dt = 1.0 / fs;
    for (int i = 0; i < symbol_count; i++) {
        int freq_idx = symbols[i] % freq_count;
        double freq = freqs[freq_idx];
        for (int j = 0; j < samples_per_symbol; j++) {
            size_t idx = i * samples_per_symbol + j;
            sig->data[idx] = cos(2.0 * PI * freq * idx * dt);
        }
    }

    return sig;
} 