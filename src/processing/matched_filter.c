#include "../include/signal_lib.h"

// 匹配滤波器实现
Signal* matched_filter(const Signal* input, const Signal* reference) {
    if (!input || !reference || input->length == 0 || reference->length == 0) {
        return NULL;
    }

    // 计算FFT长度（需要是2的幂，且大于输入信号和参考信号的总长度）
    size_t total_length = input->length + reference->length - 1;
    size_t fft_length = 1;
    while (fft_length < total_length) {
        fft_length <<= 1;
    }

    // 分配内存
    Complex* input_spectrum = (Complex*)calloc(fft_length, sizeof(Complex));
    Complex* ref_spectrum = (Complex*)calloc(fft_length, sizeof(Complex));
    Complex* output_spectrum = (Complex*)calloc(fft_length, sizeof(Complex));
    if (!input_spectrum || !ref_spectrum || !output_spectrum) {
        free(input_spectrum);
        free(ref_spectrum);
        free(output_spectrum);
        return NULL;
    }

    // 将输入信号转换为复数形式并补零
    for (size_t i = 0; i < input->length; i++) {
        input_spectrum[i].real = input->data[i];
        input_spectrum[i].imag = 0.0;
    }
    for (size_t i = input->length; i < fft_length; i++) {
        input_spectrum[i].real = input_spectrum[i].imag = 0.0;
    }

    // 将参考信号转换为复数形式并补零
    for (size_t i = 0; i < reference->length; i++) {
        ref_spectrum[i].real = reference->data[i];
        ref_spectrum[i].imag = 0.0;
    }
    for (size_t i = reference->length; i < fft_length; i++) {
        ref_spectrum[i].real = ref_spectrum[i].imag = 0.0;
    }

    // 执行FFT
    fft(input_spectrum, fft_length);
    fft(ref_spectrum, fft_length);

    // 在频域进行匹配滤波（复共轭相乘）
    for (size_t i = 0; i < fft_length; i++) {
        Complex conj_ref = {ref_spectrum[i].real, -ref_spectrum[i].imag};  // 复共轭
        output_spectrum[i] = complex_multiply(input_spectrum[i], conj_ref);
    }

    // 执行IFFT
    ifft(output_spectrum, fft_length);

    // 创建输出信号
    Signal* output = create_signal(total_length, input->fs);
    if (!output) {
        free(input_spectrum);
        free(ref_spectrum);
        free(output_spectrum);
        return NULL;
    }

    // 提取实部作为输出
    for (size_t i = 0; i < total_length; i++) {
        output->data[i] = output_spectrum[i].real;
    }

    // 释放内存
    free(input_spectrum);
    free(ref_spectrum);
    free(output_spectrum);

    return output;
}

// 自适应匹配滤波器（考虑多普勒频移）
Signal* adaptive_matched_filter(const Signal* input, const Signal* reference, 
                              double doppler_min, double doppler_max, 
                              double doppler_step) {
    if (!input || !reference || input->length == 0 || reference->length == 0 ||
        input->fs != reference->fs || doppler_min > doppler_max || 
        doppler_step <= 0) {
        return NULL;
    }

    Signal* best_output = NULL;
    double max_peak = 0.0;
    
    // 遍历不同的多普勒频移因子
    for (double doppler = doppler_min; doppler <= doppler_max; doppler += doppler_step) {
        // 创建频移后的参考信号
        Signal* doppler_ref = create_signal(reference->length, reference->fs);
        if (!doppler_ref) {
            continue;
        }

        // 应用多普勒频移
        double dt = 1.0 / reference->fs;
        for (size_t i = 0; i < reference->length; i++) {
            double t = i * dt;
            double phase = 2.0 * PI * doppler * t;
            doppler_ref->data[i] = reference->data[i] * cos(phase);
        }

        // 执行匹配滤波
        Signal* current_output = matched_filter(input, doppler_ref);
        destroy_signal(doppler_ref);

        if (!current_output) {
            continue;
        }

        // 寻找最大峰值
        double current_peak = 0.0;
        for (size_t i = 0; i < current_output->length; i++) {
            if (fabs(current_output->data[i]) > current_peak) {
                current_peak = fabs(current_output->data[i]);
            }
        }

        // 更新最佳匹配结果
        if (current_peak > max_peak) {
            if (best_output) {
                destroy_signal(best_output);
            }
            best_output = current_output;
            max_peak = current_peak;
        } else {
            destroy_signal(current_output);
        }
    }

    return best_output;
}

// CFAR检测器实现
int cfar_detector(const Signal* signal, size_t window_size, size_t guard_size,
                 double pfa, double* threshold, int* detection_result) {
    if (!signal || !threshold || !detection_result || 
        window_size < 2 || guard_size >= window_size/2 ||
        pfa <= 0 || pfa >= 1) {
        return -1;
    }

    size_t half_window = window_size / 2;
    size_t half_guard = guard_size / 2;

    // 初始化检测结果
    memset(detection_result, 0, signal->length * sizeof(int));

    // 计算CFAR因子
    double cfar_factor = -2.0 * log(pfa);

    // 对每个采样点进行CFAR检测
    for (size_t i = half_window; i < signal->length - half_window; i++) {
        double noise_power = 0.0;
        int valid_samples = 0;

        // 计算参考单元的平均噪声功率
        for (size_t j = i - half_window; j <= i + half_window; j++) {
            // 跳过保护单元
            if (j >= i - half_guard && j <= i + half_guard) {
                continue;
            }
            noise_power += signal->data[j] * signal->data[j];
            valid_samples++;
        }

        // 计算自适应阈值并进行检测
        if (valid_samples > 0) {
            noise_power /= valid_samples;
            threshold[i] = sqrt(noise_power * cfar_factor);
            detection_result[i] = (fabs(signal->data[i]) > threshold[i]) ? 1 : 0;
        } else {
            threshold[i] = 0.0;
            detection_result[i] = 0;
        }
    }

    return 0;
} 

