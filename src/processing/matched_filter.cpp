#include "../include/signal_lib.hpp"

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

    // 执行FFT (使用增强版本)
    if (enhanced_fft(input_spectrum, fft_length, 1) != 0) {
        fft(input_spectrum, fft_length);  // 回退到原始实现
    }
    if (enhanced_fft(ref_spectrum, fft_length, 1) != 0) {
        fft(ref_spectrum, fft_length);  // 回退到原始实现
    }

    // 在频域进行匹配滤波（复共轭相乘）
    for (size_t i = 0; i < fft_length; i++) {
        Complex conj_ref = {ref_spectrum[i].real, -ref_spectrum[i].imag};  // 复共轭
        output_spectrum[i] = complex_multiply(input_spectrum[i], conj_ref);
    }

    // 执行IFFT (使用增强版本)
    if (enhanced_fft(output_spectrum, fft_length, 0) != 0) {
        ifft(output_spectrum, fft_length);  // 回退到原始实现
    }

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

// === 信号处理增强功能实现 ===

/**
 * 脉冲压缩处理
 */
Signal* pulse_compression(const Signal* received_signal, 
                         const Signal* transmitted_signal) {
    if (!received_signal || !transmitted_signal) {
        return NULL;
    }
    
    // 创建输出信号（长度为两信号长度之和减1）
    size_t output_length = received_signal->length + transmitted_signal->length - 1;
    Signal* compressed_signal = create_signal(output_length, received_signal->fs);
    if (!compressed_signal) {
        return NULL;
    }
    
    // 执行匹配滤波（相关运算）
    for (size_t n = 0; n < output_length; n++) {
        double correlation = 0.0;
        
        for (size_t k = 0; k < transmitted_signal->length; k++) {
            int recv_index = (int)n - (int)k;
            if (recv_index >= 0 && recv_index < (int)received_signal->length) {
                // 匹配滤波使用发射信号的共轭
                correlation += received_signal->data[recv_index] * 
                              transmitted_signal->data[transmitted_signal->length - 1 - k];
            }
        }
        
        compressed_signal->data[n] = correlation;
    }
    
    printf("Pulse compression completed: input_len=%zu, template_len=%zu, output_len=%zu\n",
           received_signal->length, transmitted_signal->length, output_length);
    
    return compressed_signal;
}

/**
 * 多普勒处理
 */
double* doppler_processing(const Signal* received_signal,
                          double pulse_width,
                          size_t* doppler_bins) {
    if (!received_signal || pulse_width <= 0 || !doppler_bins) {
        return NULL;
    }
    
    // 计算多普勒处理参数
    size_t samples_per_pulse = (size_t)(pulse_width * received_signal->fs);
    size_t num_pulses = received_signal->length / samples_per_pulse;
    
    if (num_pulses < 2) {
        *doppler_bins = 0;
        return NULL;
    }
    
    *doppler_bins = num_pulses;
    double* doppler_spectrum = (double*)malloc(num_pulses * sizeof(double));
    if (!doppler_spectrum) {
        *doppler_bins = 0;
        return NULL;
    }
    
    // 对每个脉冲进行FFT处理（简化实现）
    for (size_t pulse = 0; pulse < num_pulses; pulse++) {
        double real_sum = 0.0;
        double imag_sum = 0.0;
        
        for (size_t sample = 0; sample < samples_per_pulse; sample++) {
            size_t index = pulse * samples_per_pulse + sample;
            if (index < received_signal->length) {
                double phase = -2.0 * PI * pulse * sample / num_pulses;
                real_sum += received_signal->data[index] * cos(phase);
                imag_sum += received_signal->data[index] * sin(phase);
            }
        }
        
        doppler_spectrum[pulse] = sqrt(real_sum * real_sum + imag_sum * imag_sum);
    }
    
    printf("Doppler processing: %zu pulses, %zu samples/pulse, %zu doppler bins\n",
           num_pulses, samples_per_pulse, *doppler_bins);
    
    return doppler_spectrum;
}

/**
 * 计算检测距离
 */
double calculate_detection_range(const SoundFieldResult* sound_field,
                                double target_strength,
                                double noise_level,
                                double detection_threshold) {
    if (!sound_field || !sound_field->transmission_loss_db) {
        return -1.0;
    }
    
    // 假设声纳方程：SL - 2*TL + TS - NL ≥ DT
    // 其中：SL=源级, TL=传输损失, TS=目标强度, NL=噪声级, DT=检测阈值
    
    double source_level = 220.0;  // 假设声源级为220dB re 1μPa@1m
    double required_snr = detection_threshold;
    
    // 遍历距离，找到满足检测条件的最大距离
    double max_detection_range = -1.0;
    
    for (size_t r = 0; r < sound_field->range_points; r++) {
        double range_m = r * sound_field->range_step_m;
        if (range_m == 0) continue;
        
        // 获取该距离处的传输损失（取深度中间值）
        size_t mid_depth = sound_field->depth_points / 2;
        size_t index = r * sound_field->depth_points + mid_depth;
        
        if (r < sound_field->range_points && mid_depth < sound_field->depth_points) {
            double tl = sound_field->transmission_loss_db[mid_depth][r];
            
            // 计算声纳方程左边
            double sonar_equation = source_level - 2.0 * tl + target_strength - noise_level;
            
            if (sonar_equation >= required_snr) {
                max_detection_range = range_m;
            } else {
                // 一旦不满足条件就停止搜索
                break;
            }
        }
    }
    
    printf("Detection range calculation: TS=%.1fdB, NL=%.1fdB, DT=%.1fdB, Range=%.0fm\n",
           target_strength, noise_level, detection_threshold, max_detection_range);
    
    return max_detection_range;
}

/**
 * 生成线性调频(LFM)信号
 */
Signal* generate_lfm_signal(double duration,
                           double start_freq,
                           double end_freq,
                           double sampling_rate) {
    if (duration <= 0 || sampling_rate <= 0 || start_freq <= 0 || end_freq <= 0) {
        return NULL;
    }
    
    size_t signal_length = (size_t)(duration * sampling_rate);
    Signal* lfm_signal = create_signal(signal_length, sampling_rate);
    if (!lfm_signal) {
        return NULL;
    }
    
    double freq_rate = (end_freq - start_freq) / duration;  // 频率变化率
    
    for (size_t i = 0; i < signal_length; i++) {
        double t = (double)i / sampling_rate;
        
        // LFM信号的瞬时频率
        double instant_freq = start_freq + freq_rate * t;
        
        // LFM信号的相位（频率的积分）
        double phase = 2.0 * PI * (start_freq * t + 0.5 * freq_rate * t * t);
        
        lfm_signal->data[i] = cos(phase);
    }
    
    printf("Generated LFM signal: %.2fs, %.0f-%.0fHz, fs=%.0fHz\n",
           duration, start_freq, end_freq, sampling_rate);
    
    return lfm_signal;
}

/**
 * 生成相位编码信号
 */
Signal* generate_phase_coded_signal(const int* code_sequence,
                                   size_t code_length,
                                   double chip_duration,
                                   double carrier_freq,
                                   double sampling_rate) {
    if (!code_sequence || code_length == 0 || chip_duration <= 0 || 
        carrier_freq <= 0 || sampling_rate <= 0) {
        return NULL;
    }
    
    size_t samples_per_chip = (size_t)(chip_duration * sampling_rate);
    size_t total_samples = code_length * samples_per_chip;
    
    Signal* coded_signal = create_signal(total_samples, sampling_rate);
    if (!coded_signal) {
        return NULL;
    }
    
    for (size_t chip = 0; chip < code_length; chip++) {
        // 根据编码值确定相位（0或π）
        double phase_offset = (code_sequence[chip] > 0) ? 0.0 : PI;
        
        for (size_t sample = 0; sample < samples_per_chip; sample++) {
            size_t index = chip * samples_per_chip + sample;
            if (index < total_samples) {
                double t = (double)index / sampling_rate;
                double phase = 2.0 * PI * carrier_freq * t + phase_offset;
                coded_signal->data[index] = cos(phase);
            }
        }
    }
    
    printf("Generated phase coded signal: %zu chips, %.3fs/chip, fc=%.0fHz\n",
           code_length, chip_duration, carrier_freq);
    
    return coded_signal;
} 

