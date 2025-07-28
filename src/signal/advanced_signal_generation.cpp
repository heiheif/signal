#include "../include/signal_lib.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

// === 船舶噪声仿真实现 ===

/**
 * @brief 生成船舶辐射噪声信号
 * @details 基于船舶类型、尺寸、速度等参数仿真辐射噪声谱特性
 */
SIGNAL_LIB_API Signal* generate_ship_noise(const ShipMotionState* motion_state,
                                          double fs,
                                          double freq_min,
                                          double freq_max,
                                          double source_level_db) {
    if (!motion_state || fs <= 0 || freq_min <= 0 || freq_max <= freq_min || motion_state->duration <= 0) {
        std::cout << "!motion_state || fs <= 0 || freq_min <= 0 || freq_max <= freq_min || motion_state->duration <= 0" << std::endl;
        std::cout << "motion_state: " << motion_state << std::endl;
        std::cout << "fs: " << fs << std::endl;
        std::cout << "freq_min: " << freq_min << std::endl;
        std::cout << "freq_max: " << freq_max << std::endl;
        std::cout << "motion_state->duration: " << motion_state->duration << std::endl;
        return NULL;
    }

    size_t length = (size_t)(fs * motion_state->duration);
    if (length > MAX_SIGNAL_LENGTH) {
        std::cout << "length > MAX_SIGNAL_LENGTH" << std::endl;
        return NULL;
    }

    Signal* sig = create_signal(length, fs);
    if (!sig) {
        std::cout << "!sig" << std::endl;
        return NULL;
    }

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    // 船舶噪声谱建模参数
    double base_freq = 50.0;  // 基础机械频率(Hz)
    double propeller_freq = motion_state->velocity * 10.0 / 3.14159; // 螺旋桨频率估算
    
    // 根据船舶类型调整参数
    double noise_factor = 1.0;
    double spectral_slope = -1.0;  // 频谱斜率
    if (motion_state->type == 0) {  // 水面舰船
        noise_factor = 1.5;
        spectral_slope = -0.8;
    } else {  // 潜艇
        noise_factor = 0.8;
        spectral_slope = -1.2;
    }

    // 速度影响因子
    double velocity_factor = 1.0 + motion_state->velocity / 10.0;
    
    // 生成时域噪声信号
    for (size_t i = 0; i < length; i++) {
        double t = i / fs;
        double noise_sample = 0.0;
        
        // 宽带噪声成分
        noise_sample += normal_dist(gen) * noise_factor;
        
        // 机械噪声线谱成分
        for (int h = 1; h <= 5; h++) {
            double freq = base_freq * h;
            if (freq >= freq_min && freq <= freq_max) {
                double amplitude = pow(h, spectral_slope) * velocity_factor;
                noise_sample += amplitude * sin(2.0 * PI * freq * t + normal_dist(gen) * 0.1);
            }
        }
        
        // 螺旋桨噪声成分
        for (int h = 1; h <= 8; h++) {
            double freq = propeller_freq * h;
            if (freq >= freq_min && freq <= freq_max) {
                double amplitude = 0.5 * pow(h, spectral_slope) * velocity_factor;
                noise_sample += amplitude * sin(2.0 * PI * freq * t + normal_dist(gen) * 0.2);
            }
        }
        
        // 流体噪声(高频成分)
        double flow_noise = normal_dist(gen) * velocity_factor * 0.3;
        noise_sample += flow_noise;
        
        sig->data[i] = noise_sample;
    }

    // 应用频带限制滤波
    // 简化实现：使用频域滤波
    Complex* spectrum = (Complex*)malloc(length * sizeof(Complex));
    if (spectrum) {
        // 转换到频域
        for (size_t i = 0; i < length; i++) {
            spectrum[i].real = sig->data[i];
            spectrum[i].imag = 0.0;
        }
        
        fft(spectrum, length);
        
        // 频域滤波
        for (size_t i = 0; i < length; i++) {
            double freq = i * fs / length;
            if (freq > length / 2) freq = fs - freq;  // 负频率部分
            
            double filter_gain = 1.0;
            if (freq < freq_min || freq > freq_max) {
                filter_gain = 0.01;  // 衰减带外频率
            }
            
            spectrum[i].real *= filter_gain;
            spectrum[i].imag *= filter_gain;
        }
        
        ifft(spectrum, length);
        
        // 转换回时域
        for (size_t i = 0; i < length; i++) {
            sig->data[i] = spectrum[i].real;
        }
        
        free(spectrum);
    }

    // 应用声源级缩放
    double rms = 0.0;
    for (size_t i = 0; i < length; i++) {
        rms += sig->data[i] * sig->data[i];
    }
    rms = sqrt(rms / length);
    
    if (rms > 0) {
        double target_rms = pow(10.0, source_level_db / 20.0);
        double scale_factor = target_rms / rms;
        for (size_t i = 0; i < length; i++) {
            sig->data[i] *= scale_factor;
        }
    }

    return sig;
}

// === OFDM信号生成实现 ===

SIGNAL_LIB_API OFDMParams* create_ofdm_params(double carrier_freq, double bandwidth, 
                                              int num_subcarriers, double cp_ratio) {
    if (carrier_freq <= 0 || bandwidth <= 0 || num_subcarriers <= 0 || 
        cp_ratio < 0 || cp_ratio >= 1.0) {
        return NULL;
    }

    OFDMParams* params = (OFDMParams*)malloc(sizeof(OFDMParams));
    if (!params) {
        return NULL;
    }

    params->carrier_freq = carrier_freq;
    params->bandwidth = bandwidth;
    params->num_subcarriers = num_subcarriers;
    params->symbol_duration = num_subcarriers / bandwidth;  // 符号持续时间
    params->cp_ratio = cp_ratio;
    params->data_bits = NULL;
    params->data_length = 0;

    return params;
}

SIGNAL_LIB_API void destroy_ofdm_params(OFDMParams* params) {
    if (params) {
        if (params->data_bits) {
            free(params->data_bits);
        }
        free(params);
    }
}

SIGNAL_LIB_API Signal* generate_ofdm(const OFDMParams* params, double fs, double duration) {
    if (!params || fs <= 0 || duration <= 0) {
        return NULL;
    }

    if (params->carrier_freq > fs / 2) {
        return NULL;  // 违反采样定理
    }

    size_t total_length = (size_t)(fs * duration);
    if (total_length > MAX_SIGNAL_LENGTH) {
        return NULL;
    }

    Signal* sig = create_signal(total_length, fs);
    if (!sig) {
        return NULL;
    }

    // 计算OFDM参数
    int num_symbols = (int)(duration * params->bandwidth / params->num_subcarriers);
    int samples_per_symbol = (int)(fs * params->symbol_duration);
    int cp_length = (int)(samples_per_symbol * params->cp_ratio);
    int symbol_with_cp_length = samples_per_symbol + cp_length;

    // 生成随机数据比特(如果未提供)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> bit_dist(0, 1);

    size_t output_idx = 0;

    for (int sym = 0; sym < num_symbols && output_idx < total_length; sym++) {
        // 创建频域数据
        Complex* freq_data = (Complex*)calloc(params->num_subcarriers, sizeof(Complex));
        if (!freq_data) {
            break;
        }

        // 生成QPSK调制的子载波数据
        for (int k = 0; k < params->num_subcarriers; k++) {
            if (k == 0 || k == params->num_subcarriers / 2) {
                // 直流和奈奎斯特频率置零
                freq_data[k].real = freq_data[k].imag = 0.0;
            } else {
                // QPSK调制
                int bit1 = bit_dist(gen);
                int bit2 = bit_dist(gen);
                double scale = 1.0 / sqrt(2.0);
                freq_data[k].real = (bit1 ? scale : -scale);
                freq_data[k].imag = (bit2 ? scale : -scale);
            }
        }

        // IFFT变换到时域
        Complex* time_data = (Complex*)malloc(params->num_subcarriers * sizeof(Complex));
        if (time_data) {
            memcpy(time_data, freq_data, params->num_subcarriers * sizeof(Complex));
            ifft(time_data, params->num_subcarriers);

            // 添加循环前缀
            for (int i = 0; i < cp_length && output_idx < total_length; i++) {
                int src_idx = params->num_subcarriers - cp_length + i;
                sig->data[output_idx++] = time_data[src_idx].real;
            }

            // 添加有用符号
            for (int i = 0; i < params->num_subcarriers && output_idx < total_length; i++) {
                sig->data[output_idx++] = time_data[i].real;
            }

            free(time_data);
        }

        free(freq_data);
    }

    // 上变频到载波频率
    if (params->carrier_freq > 0) {
        for (size_t i = 0; i < total_length; i++) {
            double t = i / fs;
            double baseband = sig->data[i];
            sig->data[i] = baseband * cos(2.0 * PI * params->carrier_freq * t);
        }
    }

    return sig;
}

// === DSSS信号生成实现 ===

SIGNAL_LIB_API DSSSParams* create_dsss_params(double carrier_freq, double bit_rate, 
                                              double chip_rate, size_t code_length) {
    if (carrier_freq <= 0 || bit_rate <= 0 || chip_rate <= 0 || code_length == 0) {
        return NULL;
    }

    if (chip_rate < bit_rate) {
        return NULL;  // 码片速率必须大于比特速率
    }

    DSSSParams* params = (DSSSParams*)malloc(sizeof(DSSSParams));
    if (!params) {
        return NULL;
    }

    params->carrier_freq = carrier_freq;
    params->bit_rate = bit_rate;
    params->chip_rate = chip_rate;
    params->code_length = code_length;
    
    // 分配扩频码内存
    params->spreading_code = (int*)malloc(code_length * sizeof(int));
    if (!params->spreading_code) {
        free(params);
        return NULL;
    }

    params->data_bits = NULL;
    params->data_length = 0;

    return params;
}

SIGNAL_LIB_API void destroy_dsss_params(DSSSParams* params) {
    if (params) {
        if (params->spreading_code) {
            free(params->spreading_code);
        }
        if (params->data_bits) {
            free(params->data_bits);
        }
        free(params);
    }
}

SIGNAL_LIB_API int generate_pn_sequence(int* sequence, size_t length, unsigned int seed) {
    if (!sequence || length == 0) {
        return -1;
    }

    // 使用线性反馈移位寄存器(LFSR)生成伪随机序列
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dist(0, 1);

    for (size_t i = 0; i < length; i++) {
        sequence[i] = dist(gen) ? 1 : -1;  // 双极性序列
    }

    return 0;
}

SIGNAL_LIB_API Signal* generate_dsss(const DSSSParams* params, double fs, double duration) {
    if (!params || !params->spreading_code || fs <= 0 || duration <= 0) {
        return NULL;
    }

    if (params->carrier_freq > fs / 2) {
        return NULL;  // 违反采样定理
    }

    size_t total_length = (size_t)(fs * duration);
    if (total_length > MAX_SIGNAL_LENGTH) {
        return NULL;
    }

    Signal* sig = create_signal(total_length, fs);
    if (!sig) {
        return NULL;
    }

    // 计算时间参数
    double bit_duration = 1.0 / params->bit_rate;
    double chip_duration = 1.0 / params->chip_rate;
    int samples_per_chip = (int)(fs * chip_duration);
    int chips_per_bit = (int)(params->chip_rate / params->bit_rate);

    // 生成随机数据比特(如果未提供)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> bit_dist(0, 1);

    size_t output_idx = 0;
    double t = 0.0;

    while (output_idx < total_length) {
        // 生成数据比特
        int data_bit = bit_dist(gen) ? 1 : -1;  // 双极性

        // 对每个比特进行扩频
        for (int chip_idx = 0; chip_idx < chips_per_bit && output_idx < total_length; chip_idx++) {
            int spreading_chip = params->spreading_code[chip_idx % params->code_length];
            int modulated_chip = data_bit * spreading_chip;

            // 生成码片对应的采样点
            for (int sample = 0; sample < samples_per_chip && output_idx < total_length; sample++) {
                t = output_idx / fs;
                // BPSK调制到载波
                sig->data[output_idx] = (double)modulated_chip * 
                                       cos(2.0 * PI * params->carrier_freq * t);
                output_idx++;
            }
        }
    }

    return sig;
}

// === 组合信号生成实现 ===

SIGNAL_LIB_API Signal* concatenate_signals_with_gap(const Signal* signal1, 
                                                   const Signal* signal2,
                                                   double gap_duration) {
    if (!signal1 || !signal2 || gap_duration < 0) {
        return NULL;
    }

    if (fabs(signal1->fs - signal2->fs) > 1e-6) {
        return NULL;  // 采样率必须相同
    }

    size_t gap_samples = (size_t)(signal1->fs * gap_duration);
    size_t total_length = signal1->length + gap_samples + signal2->length;

    if (total_length > MAX_SIGNAL_LENGTH) {
        return NULL;
    }

    Signal* result = create_signal(total_length, signal1->fs);
    if (!result) {
        return NULL;
    }

    size_t idx = 0;

    // 复制第一个信号
    memcpy(result->data, signal1->data, signal1->length * sizeof(double));
    idx += signal1->length;

    // 添加空白
    for (size_t i = 0; i < gap_samples; i++) {
        result->data[idx++] = 0.0;
    }

    // 复制第二个信号
    memcpy(result->data + idx, signal2->data, signal2->length * sizeof(double));

    return result;
}

SIGNAL_LIB_API Signal* generate_composite_signal(CompositeSignalType signal_type,
                                                double fs,
                                                double duration,
                                                double freq_min,
                                                double freq_max) {
    if (fs <= 0 || duration <= 0 || freq_min <= 0 || freq_max <= freq_min) {
        return NULL;
    }

    double gap_duration = 1.0;  // 1秒间隔
    double cw_duration = 1.0;   // CW持续时间
    double lfm_duration = 1.0;  // LFM持续时间

    Signal* result = NULL;

    switch (signal_type) {
        case SIGNAL_SHIP_NOISE: {
            ShipMotionState motion_state = {0};
            motion_state.type = 0;  // 水面舰船
            motion_state.duration = duration;
            motion_state.velocity = 5.0;
            motion_state.depth = 10.0;
            motion_state.length = 100.0;
            motion_state.displacement = 5000.0;
            motion_state.engine_power = 10000.0;
            
            result = generate_ship_noise(&motion_state, fs, freq_min, freq_max, 15.0);
            break;
        }

        case SIGNAL_CW:
            result = generate_cw(freq_min, fs, duration, 1.0, 0.0);
            break;

        case SIGNAL_LFM:
            result = generate_lfm(freq_min, freq_max, fs, duration);
            break;

        case SIGNAL_HFM:
            result = generate_hfm(freq_min, freq_max, fs, duration);
            break;

        case SIGNAL_OFDM_COMPOSITE: {
            // 创建组合信号：CW + 空白 + LFM上扫 + 空白 + OFDM + 空白 + LFM下扫
            Signal* cw_sig = generate_cw(freq_min, fs, cw_duration, 1.0, 0.0);
            Signal* lfm_up = generate_lfm(freq_min, freq_max, fs, lfm_duration);
            
            // 创建OFDM信号
            double carrier_freq = (freq_min + freq_max) / 2.0;
            double bandwidth = freq_max - freq_min;
            OFDMParams* ofdm_params = create_ofdm_params(carrier_freq, bandwidth, 1024, 0.25);
            Signal* ofdm_sig = NULL;
            if (ofdm_params) {
                ofdm_sig = generate_ofdm(ofdm_params, fs, duration - 3 * gap_duration - 2 * lfm_duration - cw_duration);
                destroy_ofdm_params(ofdm_params);
            }
            
            Signal* lfm_down = generate_lfm(freq_max, freq_min, fs, lfm_duration);

            // 组合信号
            if (cw_sig && lfm_up && ofdm_sig && lfm_down) {
                Signal* temp1 = concatenate_signals_with_gap(cw_sig, lfm_up, gap_duration);
                Signal* temp2 = concatenate_signals_with_gap(temp1, ofdm_sig, gap_duration);
                result = concatenate_signals_with_gap(temp2, lfm_down, gap_duration);
                
                if (temp1) destroy_signal(temp1);
                if (temp2) destroy_signal(temp2);
            }

            if (cw_sig) destroy_signal(cw_sig);
            if (lfm_up) destroy_signal(lfm_up);
            if (ofdm_sig) destroy_signal(ofdm_sig);
            if (lfm_down) destroy_signal(lfm_down);
            break;
        }

        case SIGNAL_DSSS_COMPOSITE: {
            // 创建组合信号：CW + 空白 + LFM上扫 + 空白 + DSSS + 空白 + LFM下扫
            Signal* cw_sig = generate_cw(freq_min, fs, cw_duration, 1.0, 0.0);
            Signal* lfm_up = generate_lfm(freq_min, freq_max, fs, lfm_duration);
            
            // 创建DSSS信号
            double carrier_freq = (freq_min + freq_max) / 2.0;
            DSSSParams* dsss_params = create_dsss_params(carrier_freq, 10000.0, 20000.0, 127);
            Signal* dsss_sig = NULL;
            if (dsss_params) {
                // 生成扩频码
                generate_pn_sequence(dsss_params->spreading_code, dsss_params->code_length, 12345);
                dsss_sig = generate_dsss(dsss_params, fs, duration - 3 * gap_duration - 2 * lfm_duration - cw_duration);
                destroy_dsss_params(dsss_params);
            }
            
            Signal* lfm_down = generate_lfm(freq_max, freq_min, fs, lfm_duration);

            // 组合信号
            if (cw_sig && lfm_up && dsss_sig && lfm_down) {
                Signal* temp1 = concatenate_signals_with_gap(cw_sig, lfm_up, gap_duration);
                Signal* temp2 = concatenate_signals_with_gap(temp1, dsss_sig, gap_duration);
                result = concatenate_signals_with_gap(temp2, lfm_down, gap_duration);
                
                if (temp1) destroy_signal(temp1);
                if (temp2) destroy_signal(temp2);
            }

            if (cw_sig) destroy_signal(cw_sig);
            if (lfm_up) destroy_signal(lfm_up);
            if (dsss_sig) destroy_signal(dsss_sig);
            if (lfm_down) destroy_signal(lfm_down);
            break;
        }

        default:
            // 对于FSK和PSK组合信号，使用现有的generate_fsk和generate_psk函数
            break;
    }

    return result;
} 