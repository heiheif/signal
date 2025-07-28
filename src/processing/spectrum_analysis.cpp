#include "../include/signal_lib.hpp"

// 线谱分析函数实现
int analyze_line_spectrum(const Signal* sig, double min_peak_height,
                         double min_peak_distance, LineSpectrum* result) {
    if (!sig || !result || sig->length == 0) {
        return -1;
    }

    // 计算FFT长度
    size_t fft_length = 1;
    while (fft_length < sig->length) {
        fft_length <<= 1;
    }

    Complex* spectrum = (Complex*)calloc(fft_length, sizeof(Complex));
    if (!spectrum) {
        return -1;
    }

    // 执行FFT (使用增强版本)
    for (size_t i = 0; i < sig->length; i++) {
        spectrum[i].real = sig->data[i];
        spectrum[i].imag = 0.0;
    }
    
    // 优先使用增强FFT
    if (enhanced_fft(spectrum, fft_length, 1) != 0) {
        // 回退到原始实现
        fft_recursive(spectrum, fft_length, 1);
    }

    // 计算幅度谱
    double* magnitude = (double*)malloc(fft_length/2 * sizeof(double));
    if (!magnitude) {
        free(spectrum);
        return -1;
    }

    for (size_t i = 0; i < fft_length/2; i++) {
        magnitude[i] = complex_abs(spectrum[i]);
    }

    // 寻找峰值
    size_t max_peaks = 100;  // 最大峰值数量限制
    double* peak_freqs = (double*)malloc(max_peaks * sizeof(double));
    double* peak_amps = (double*)malloc(max_peaks * sizeof(double));
    size_t peak_count = 0;

    if (!peak_freqs || !peak_amps) {
        free(spectrum);
        free(magnitude);
        free(peak_freqs);
        free(peak_amps);
        return -1;
    }

    // 峰值检测
    for (size_t i = 1; i < fft_length/2 - 1; i++) {
        if (magnitude[i] > magnitude[i-1] && magnitude[i] > magnitude[i+1] &&
            magnitude[i] > min_peak_height) {
            // 检查是否与已有峰值距离过近
            int valid_peak = 1;
            for (size_t j = 0; j < peak_count; j++) {
                double freq_diff = fabs(i * sig->fs / fft_length - peak_freqs[j]);
                if (freq_diff < min_peak_distance) {
                    valid_peak = 0;
                    break;
                }
            }

            if (valid_peak && peak_count < max_peaks) {
                peak_freqs[peak_count] = i * sig->fs / fft_length;
                peak_amps[peak_count] = magnitude[i];
                peak_count++;
            }
        }
    }

    // 分配结果内存
    result->frequencies = (double*)malloc(peak_count * sizeof(double));
    result->amplitudes = (double*)malloc(peak_count * sizeof(double));
    if (!result->frequencies || !result->amplitudes) {
        free(spectrum);
        free(magnitude);
        free(peak_freqs);
        free(peak_amps);
        return -1;
    }

    // 复制结果
    memcpy(result->frequencies, peak_freqs, peak_count * sizeof(double));
    memcpy(result->amplitudes, peak_amps, peak_count * sizeof(double));
    result->peak_count = peak_count;

    // 释放内存
    free(spectrum);
    free(magnitude);
    free(peak_freqs);
    free(peak_amps);

    return 0;
}

// 自适应能量检测器实现
int adaptive_energy_detect(const Signal* sig, double false_alarm_rate,
                         double* threshold, int* detection_result) {
    if (!sig || !threshold || !detection_result || 
        false_alarm_rate <= 0 || false_alarm_rate >= 1) {
        return -1;
    }

    size_t window_size = 64;  // 滑动窗口大小
    size_t guard_size = 4;    // 保护单元大小

    // 计算信号能量
    double* energy = (double*)malloc(sig->length * sizeof(double));
    if (!energy) {
        return -1;
    }

    for (size_t i = 0; i < sig->length; i++) {
        energy[i] = sig->data[i] * sig->data[i];
    }

    // CFAR检测
    size_t half_window = window_size / 2;
    size_t half_guard = guard_size / 2;
    double cfar_factor = -log(false_alarm_rate);  // CFAR因子

    for (size_t i = 0; i < sig->length; i++) {
        double noise_power = 0.0;
        int valid_samples = 0;

        // 计算参考单元的平均噪声功率
        size_t start_idx = (i > half_window) ? (i - half_window) : 0;
        size_t end_idx = (i + half_window < sig->length) ? (i + half_window) : (sig->length - 1);
        
        for (size_t j = start_idx; j <= end_idx; j++) {
            // 跳过保护单元
            if (j >= i - half_guard && j <= i + half_guard) {
                continue;
            }
            noise_power += energy[j];
            valid_samples++;
        }

        if (valid_samples > 0) {
            noise_power /= valid_samples;
            threshold[i] = noise_power * cfar_factor;
            detection_result[i] = (energy[i] > threshold[i]) ? 1 : 0;
        } else {
            threshold[i] = 0.0;
            detection_result[i] = 0;
        }
    }

    free(energy);
    return 0;
}

// 自适应谱检测器实现
int adaptive_spectrum_detect(const Signal* sig, double false_alarm_rate,
                           double* frequencies, double* thresholds,
                           int* detection_result) {
    if (!sig || !frequencies || !thresholds || !detection_result ||
        false_alarm_rate <= 0 || false_alarm_rate >= 1) {
        return -1;
    }

    // 计算FFT长度
    size_t fft_length = 1;
    while (fft_length < sig->length) {
        fft_length <<= 1;
    }

    Complex* spectrum = (Complex*)calloc(fft_length, sizeof(Complex));
    if (!spectrum) {
        return -1;
    }

    // 执行FFT (使用增强版本)
    for (size_t i = 0; i < sig->length; i++) {
        spectrum[i].real = sig->data[i];
        spectrum[i].imag = 0.0;
    }
    
    // 优先使用增强FFT
    if (enhanced_fft(spectrum, fft_length, 1) != 0) {
        // 回退到原始实现
        fft_recursive(spectrum, fft_length, 1);
    }

    // 计算功率谱
    double* power_spectrum = (double*)malloc(fft_length/2 * sizeof(double));
    if (!power_spectrum) {
        free(spectrum);
        return -1;
    }

    for (size_t i = 0; i < fft_length/2; i++) {
        power_spectrum[i] = complex_abs(spectrum[i]);
        frequencies[i] = i * sig->fs / fft_length;
    }

    // 自适应谱检测
    size_t window_size = 32;  // 频域滑动窗口大小
    size_t half_window = window_size / 2;
    double cfar_factor = -log(false_alarm_rate);

    for (size_t i = 0; i < fft_length/2; i++) {
        double avg_power = 0.0;
        int valid_samples = 0;

        // 计算参考单元的平均功率
        size_t start_idx = (i > half_window) ? (i - half_window) : 0;
        size_t end_idx = (i + half_window < fft_length/2) ? (i + half_window) : (fft_length/2 - 1);

        for (size_t j = start_idx; j <= end_idx; j++) {
            if (j != i) {  // 排除当前单元
                avg_power += power_spectrum[j];
                valid_samples++;
            }
        }

        if (valid_samples > 0) {
            avg_power /= valid_samples;
            thresholds[i] = avg_power * cfar_factor;
            detection_result[i] = (power_spectrum[i] > thresholds[i]) ? 1 : 0;
        } else {
            thresholds[i] = 0.0;
            detection_result[i] = 0;
        }
    }

    free(spectrum);
    free(power_spectrum);
    return 0;
} 

