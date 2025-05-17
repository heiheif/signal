#include "../include/signal_lib.h"

// 计算时域特征
static void calculate_time_domain_features(const Signal* sig, double* features) {
    if (!sig || !features || sig->length == 0) {
        return;
    }

    // 计算均值
    double mean = 0.0;
    for (size_t i = 0; i < sig->length; i++) {
        mean += sig->data[i];
    }
    mean /= (double)sig->length;
    features[0] = mean;

    // 计算方差
    double variance = 0.0;
    for (size_t i = 0; i < sig->length; i++) {
        variance += pow(sig->data[i] - mean, 2);
    }
    variance /= (double)(sig->length - 1);
    features[1] = variance;

    // 计算标准差
    features[2] = sqrt(variance);

    // 计算偏度
    double skewness = 0.0;
    for (size_t i = 0; i < sig->length; i++) {
        skewness += pow((sig->data[i] - mean) / sqrt(variance), 3);
    }
    skewness *= ((double)sig->length / ((double)(sig->length - 1) * (double)(sig->length - 2)));
    features[3] = skewness;

    // 计算峰度
    double kurtosis = 0.0;
    for (size_t i = 0; i < sig->length; i++) {
        kurtosis += pow((sig->data[i] - mean) / sqrt(variance), 4);
    }
    kurtosis = (kurtosis * ((double)sig->length * ((double)sig->length + 1)) / 
               ((double)(sig->length - 1) * (double)(sig->length - 2) * (double)(sig->length - 3))) - 
               (3.0 * pow((double)(sig->length - 1), 2) / 
               ((double)(sig->length - 2) * (double)(sig->length - 3)));
    features[4] = kurtosis;
}

// 计算频域特征
static void calculate_frequency_domain_features(const Signal* sig, double* features) {
    if (!sig || !features || sig->length == 0) {
        return;
    }

    // 计算FFT长度
    size_t fft_length = 1;
    while (fft_length < sig->length) {
        fft_length <<= 1;
    }

    // 分配内存
    Complex* spectrum = (Complex*)calloc(fft_length, sizeof(Complex));
    if (!spectrum) {
        return;
    }

    // 复制数据
    for (size_t i = 0; i < sig->length; i++) {
        spectrum[i].real = sig->data[i];
        spectrum[i].imag = 0.0;
    }

    // 执行FFT
    fft_recursive(spectrum, fft_length, 1);

    // 计算功率谱
    double* power_spectrum = (double*)malloc(fft_length/2 * sizeof(double));
    if (!power_spectrum) {
        free(spectrum);
        return;
    }

    for (size_t i = 0; i < fft_length/2; i++) {
        power_spectrum[i] = sqrt(spectrum[i].real * spectrum[i].real + 
                               spectrum[i].imag * spectrum[i].imag);
    }

    // 计算频域特征
    double total_power = 0.0;
    double weighted_freq_sum = 0.0;
    double max_power = 0.0;
    size_t max_power_freq = 0;

    for (size_t i = 0; i < fft_length/2; i++) {
        double freq = (double)i * sig->fs / (double)fft_length;
        double power = power_spectrum[i];
        
        total_power += power;
        weighted_freq_sum += freq * power;
        
        if (power > max_power) {
            max_power = power;
            max_power_freq = i;
        }
    }

    // 中心频率
    features[0] = weighted_freq_sum / total_power;
    
    // 主频
    features[1] = (double)max_power_freq * sig->fs / (double)fft_length;
    
    // 频带宽度
    double bandwidth = 0.0;
    for (size_t i = 0; i < fft_length/2; i++) {
        double freq = (double)i * sig->fs / (double)fft_length;
        bandwidth += pow(freq - features[0], 2) * power_spectrum[i] / total_power;
    }
    features[2] = sqrt(bandwidth);

    // 清理内存
    free(spectrum);
    free(power_spectrum);
}

// 提取信号特征
int extract_features(const Signal* sig, double* time_features, double* freq_features) {
    if (!sig || !time_features || !freq_features || sig->length == 0) {
        return -1;
    }

    // 计算时域特征
    calculate_time_domain_features(sig, time_features);

    // 计算频域特征
    calculate_frequency_domain_features(sig, freq_features);

    return 0;
}

// 加权中值滤波器实现
Signal* weighted_median_filter(const Signal* input, const double* weights,
                             size_t window_size) {
    if (!input || !weights || window_size == 0 || 
        window_size > input->length || !(window_size % 2)) {
        return NULL;
    }

    Signal* output = create_signal(input->length, input->fs);
    if (!output) {
        return NULL;
    }

    size_t half_window = window_size / 2;

    // 对每个采样点进行加权中值滤波
    for (size_t i = 0; i < input->length; i++) {
        // 提取窗口内的样本并计算加权值
        double* window_samples = (double*)malloc(window_size * sizeof(double));
        double* weighted_samples = (double*)malloc(window_size * sizeof(double));
        if (!window_samples || !weighted_samples) {
            free(window_samples);
            free(weighted_samples);
            destroy_signal(output);
            return NULL;
        }

        // 填充窗口样本
        size_t k = 0;
        for (size_t j = 0; j < window_size; j++) {
            size_t idx = i + j - half_window;
            if (idx < 0) {
                idx = 0;
            } else if (idx >= input->length) {
                idx = input->length - 1;
            }
            window_samples[k] = input->data[idx];
            weighted_samples[k] = window_samples[k] * weights[j];
            k++;
        }

        // 对加权样本进行排序
        for (size_t m = 0; m < window_size-1; m++) {
            for (size_t n = 0; n < window_size-m-1; n++) {
                if (weighted_samples[n] > weighted_samples[n+1]) {
                    double temp = weighted_samples[n];
                    weighted_samples[n] = weighted_samples[n+1];
                    weighted_samples[n+1] = temp;
                    
                    temp = window_samples[n];
                    window_samples[n] = window_samples[n+1];
                    window_samples[n+1] = temp;
                }
            }
        }

        // 取中值作为输出
        output->data[i] = window_samples[window_size/2];

        free(window_samples);
        free(weighted_samples);
    }

    return output;
}

// 多项式背景拟合
int polynomial_background_fit(const Signal* input, int order,
                            Signal* background, Signal* detrended) {
    if (!input || !background || !detrended || order < 0 || 
        input->length != background->length || 
        input->length != detrended->length) {
        return -1;
    }

    // 构建范德蒙德矩阵和数据向量
    double** vander = (double**)malloc((order + 1) * sizeof(double*));
    double* x = (double*)malloc(input->length * sizeof(double));
    double* y = (double*)malloc(input->length * sizeof(double));
    double* coeffs = (double*)malloc((order + 1) * sizeof(double));

    if (!vander || !x || !y || !coeffs) {
        free(vander);
        free(x);
        free(y);
        free(coeffs);
        return -1;
    }

    for (int i = 0; i <= order; i++) {
        vander[i] = (double*)malloc(input->length * sizeof(double));
        if (!vander[i]) {
            for (int j = 0; j < i; j++) {
                free(vander[j]);
            }
            free(vander);
            free(x);
            free(y);
            free(coeffs);
            return -1;
        }
    }

    // 准备数据
    for (size_t i = 0; i < input->length; i++) {
        x[i] = (double)i / input->length;
        y[i] = input->data[i];
    }

    // 构建范德蒙德矩阵
    for (size_t i = 0; i < input->length; i++) {
        for (int j = 0; j <= order; j++) {
            vander[j][i] = pow(x[i], j);
        }
    }

    // 使用最小二乘法求解
    // 简化版本：直接使用对角线元素进行求解
    double* ata = (double*)calloc((order + 1) * (order + 1), sizeof(double));
    double* atb = (double*)calloc(order + 1, sizeof(double));

    // 计算A^T * A
    for (int i = 0; i <= order; i++) {
        for (int j = 0; j <= order; j++) {
            for (size_t k = 0; k < input->length; k++) {
                ata[i * (order + 1) + j] += vander[i][k] * vander[j][k];
            }
        }
    }

    // 计算A^T * b
    for (int i = 0; i <= order; i++) {
        for (size_t k = 0; k < input->length; k++) {
            atb[i] += vander[i][k] * y[k];
        }
    }

    // 求解系数（简化版本，仅使用对角线元素）
    for (int i = 0; i <= order; i++) {
        coeffs[i] = atb[i] / ata[i * (order + 1) + i];
    }

    // 计算拟合背景
    for (size_t i = 0; i < input->length; i++) {
        background->data[i] = 0.0;
        for (int j = 0; j <= order; j++) {
            background->data[i] += coeffs[j] * pow(x[i], j);
        }
    }

    // 计算去趋势信号
    for (size_t i = 0; i < input->length; i++) {
        detrended->data[i] = input->data[i] - background->data[i];
    }

    // 释放内存
    for (int i = 0; i <= order; i++) {
        free(vander[i]);
    }
    free(vander);
    free(x);
    free(y);
    free(coeffs);
    free(ata);
    free(atb);

    return 0;
} 

