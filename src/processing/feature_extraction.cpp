#include "../include/signal_lib.hpp"

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

    // 执行FFT (使用增强版本)
    if (enhanced_fft(spectrum, fft_length, 1) != 0) {
        // 回退到原始实现
        fft_recursive(spectrum, fft_length, 1);
    }

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

// 计算时频特征
static void calculate_time_freq_features(const Signal* sig, double* features) {
    if (!sig || !features || sig->length == 0) {
        return;
    }

    // 分配内存用于短时傅里叶变换
    size_t window_size = 256;
    if (window_size > sig->length) {
        window_size = sig->length;
    }
    
    size_t hop_size = window_size / 2;
    size_t num_frames = 1 + (sig->length - window_size) / hop_size;
    
    if (num_frames < 1) {
        num_frames = 1;
    }
    
    // 分配内存
    double* window = (double*)malloc(window_size * sizeof(double));
    Complex** stft = (Complex**)malloc(num_frames * sizeof(Complex*));
    
    if (!window || !stft) {
        free(window);
        free(stft);
        return;
    }
    
    for (size_t i = 0; i < num_frames; i++) {
        stft[i] = (Complex*)calloc(window_size, sizeof(Complex));
        if (!stft[i]) {
            for (size_t j = 0; j < i; j++) {
                free(stft[j]);
            }
            free(stft);
            free(window);
            return;
        }
    }
    
    // 计算汉宁窗
    for (size_t i = 0; i < window_size; i++) {
        window[i] = 0.5 * (1 - cos(2 * PI * i / (window_size - 1)));
    }
    
    // 执行短时傅里叶变换
    for (size_t frame = 0; frame < num_frames; frame++) {
        size_t offset = frame * hop_size;
        
        // 应用窗函数并填充数据
        for (size_t i = 0; i < window_size; i++) {
            if (offset + i < sig->length) {
                stft[frame][i].real = sig->data[offset + i] * window[i];
                stft[frame][i].imag = 0.0;
            } else {
                stft[frame][i].real = 0.0;
                stft[frame][i].imag = 0.0;
            }
        }
        
        // 执行FFT
        fft_recursive(stft[frame], window_size, 0);
    }
    
    // 计算时频特征
    
    // 1. 频谱质心随时间的变化率
    double* spectral_centroids = (double*)calloc(num_frames, sizeof(double));
    if (spectral_centroids) {
        for (size_t frame = 0; frame < num_frames; frame++) {
            double weighted_sum = 0.0;
            double total_energy = 0.0;
            
            for (size_t bin = 0; bin < window_size / 2; bin++) {
                double magnitude = complex_abs(stft[frame][bin]);
                double freq = bin * sig->fs / window_size;
                
                weighted_sum += freq * magnitude;
                total_energy += magnitude;
            }
            
            if (total_energy > 0) {
                spectral_centroids[frame] = weighted_sum / total_energy;
            }
        }
        
        // 计算变化率
        double centroid_variation = 0.0;
        if (num_frames > 1) {
            for (size_t frame = 1; frame < num_frames; frame++) {
                centroid_variation += fabs(spectral_centroids[frame] - spectral_centroids[frame - 1]);
            }
            centroid_variation /= (num_frames - 1);
        }
        
        features[0] = centroid_variation;
        free(spectral_centroids);
    }
    
    // 2. 频谱熵随时间的变化
    double avg_entropy = 0.0;
    for (size_t frame = 0; frame < num_frames; frame++) {
        double total_energy = 0.0;
        
        // 首先计算总能量
        for (size_t bin = 0; bin < window_size / 2; bin++) {
            total_energy += complex_abs(stft[frame][bin]);
        }
        
        // 计算熵
        double entropy = 0.0;
        if (total_energy > 0) {
            for (size_t bin = 0; bin < window_size / 2; bin++) {
                double p = complex_abs(stft[frame][bin]) / total_energy;
                if (p > 0) {
                    entropy -= p * log2(p);
                }
            }
        }
        
        avg_entropy += entropy;
    }
    
    if (num_frames > 0) {
        avg_entropy /= num_frames;
    }
    features[1] = avg_entropy;
    
    // 3. 频谱流量（测量频谱随时间的变化）
    double spectral_flux = 0.0;
    if (num_frames > 1) {
        for (size_t frame = 1; frame < num_frames; frame++) {
            double frame_flux = 0.0;
            
            for (size_t bin = 0; bin < window_size / 2; bin++) {
                double diff = complex_abs(stft[frame][bin]) - complex_abs(stft[frame - 1][bin]);
                frame_flux += diff * diff;
            }
            
            spectral_flux += sqrt(frame_flux);
        }
        spectral_flux /= (num_frames - 1);
    }
    features[2] = spectral_flux;
    
    // 4-9: 设置其他时频特征（暂时用0填充）
    for (int i = 3; i < 10; i++) {
        features[i] = 0.0;
    }
    
    // 清理内存
    for (size_t i = 0; i < num_frames; i++) {
        free(stft[i]);
    }
    free(stft);
    free(window);
}

// 提取信号特征
int extract_features(const Signal* sig, SignalFeatures* features) {
    if (!sig || !features || sig->length == 0) {
        return -1;
    }

    // 计算时域特征
    calculate_time_domain_features(sig, features->time_domain_features);

    // 计算频域特征
    calculate_frequency_domain_features(sig, features->freq_domain_features);
    
    // 计算时频特征
    calculate_time_freq_features(sig, features->time_freq_features);

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

// === 目标特性模型实现 ===

/**
 * 创建标准目标模型
 */
TargetModel* create_target_model(const char* target_type, 
                                double length, 
                                double draft) {
    if (!target_type || length <= 0 || draft <= 0) {
        return NULL;
    }
    
    TargetModel* model = (TargetModel*)malloc(sizeof(TargetModel));
    if (!model) {
        return NULL;
    }
    
    // 初始化基本参数
    model->length = length;
    model->width = length * 0.1;  // 典型宽度约为长度的1/10
    model->draft = draft;
    strncpy(model->target_type, target_type, sizeof(model->target_type) - 1);
    model->target_type[sizeof(model->target_type) - 1] = '\0';
    
    // 设置角度数据点（每10度一个点）
    model->angle_count = 37;  // 0-360度，每10度
    model->aspect_angles = (double*)malloc(model->angle_count * sizeof(double));
    model->reflection_coeffs = (double*)malloc(model->angle_count * sizeof(double));
    
    if (!model->aspect_angles || !model->reflection_coeffs) {
        destroy_target_model(model);
        return NULL;
    }
    
    // 根据目标类型设置参考目标强度和角度特性
    if (strcmp(target_type, "SUBMARINE") == 0) {
        // 潜艇模型
        model->target_strength_ref = 10.0 * log10(length * draft);  // 基于几何尺寸
        
        for (size_t i = 0; i < model->angle_count; i++) {
            model->aspect_angles[i] = i * 10.0;  // 0-360度
            double angle_rad = model->aspect_angles[i] * PI / 180.0;
            
            // 潜艇典型的角度特性：首尾方向强，侧面弱
            if (i == 0 || i == 18 || i == 36) {  // 0°, 180°, 360°
                model->reflection_coeffs[i] = 1.0;  // 首尾最强
            } else if (i == 9 || i == 27) {  // 90°, 270°
                model->reflection_coeffs[i] = 0.3;  // 侧面最弱
            } else {
                // 其他角度的平滑过渡
                model->reflection_coeffs[i] = 0.3 + 0.7 * pow(cos(angle_rad), 2);
            }
        }
        
    } else if (strcmp(target_type, "SURFACE_SHIP") == 0) {
        // 水面舰艇模型
        model->target_strength_ref = 15.0 * log10(length * draft);
        
        for (size_t i = 0; i < model->angle_count; i++) {
            model->aspect_angles[i] = i * 10.0;
            double angle_rad = model->aspect_angles[i] * PI / 180.0;
            
            // 水面舰艇：侧面反射强，首尾相对弱
            if (i == 9 || i == 27) {  // 90°, 270°
                model->reflection_coeffs[i] = 1.0;  // 侧面最强
            } else if (i == 0 || i == 18 || i == 36) {  // 0°, 180°, 360°
                model->reflection_coeffs[i] = 0.5;  // 首尾中等
            } else {
                model->reflection_coeffs[i] = 0.5 + 0.5 * pow(sin(angle_rad), 2);
            }
        }
        
    } else if (strcmp(target_type, "UUV") == 0) {
        // 无人潜航器模型
        model->target_strength_ref = 5.0 * log10(length * draft);
        
        for (size_t i = 0; i < model->angle_count; i++) {
            model->aspect_angles[i] = i * 10.0;
            // UUV通常较小，角度特性相对平缓
            model->reflection_coeffs[i] = 0.5 + 0.3 * cos(2.0 * model->aspect_angles[i] * PI / 180.0);
        }
        
    } else if (strcmp(target_type, "TORPEDO") == 0) {
        // 鱼雷模型
        model->target_strength_ref = 0.0 * log10(length * draft);
        
        for (size_t i = 0; i < model->angle_count; i++) {
            model->aspect_angles[i] = i * 10.0;
            double angle_rad = model->aspect_angles[i] * PI / 180.0;
            // 鱼雷：细长形状，首尾方向强
            model->reflection_coeffs[i] = 0.2 + 0.8 * pow(cos(angle_rad), 4);
        }
        
    } else {
        // 默认目标模型
        model->target_strength_ref = 8.0 * log10(length * draft);
        
        for (size_t i = 0; i < model->angle_count; i++) {
            model->aspect_angles[i] = i * 10.0;
            model->reflection_coeffs[i] = 0.7;  // 均匀反射
        }
    }
    
    printf("Created target model: %s, L=%.1fm, D=%.1fm, TS_ref=%.1fdB\n",
           target_type, length, draft, model->target_strength_ref);
    
    return model;
}

/**
 * 销毁目标模型
 */
void destroy_target_model(TargetModel* model) {
    if (model) {
        if (model->aspect_angles) {
            free(model->aspect_angles);
        }
        if (model->reflection_coeffs) {
            free(model->reflection_coeffs);
        }
        free(model);
    }
}

/**
 * 计算目标强度
 */
double calculate_target_strength(const TargetModel* model,
                                double frequency,
                                double aspect_angle) {
    if (!model || frequency <= 0) {
        return -100.0;  // 极小的目标强度
    }
    
    // 将角度规范化到0-360度
    while (aspect_angle < 0) aspect_angle += 360.0;
    while (aspect_angle >= 360.0) aspect_angle -= 360.0;
    
    // 插值计算角度修正因子
    double angle_factor = 1.0;
    
    for (size_t i = 0; i < model->angle_count - 1; i++) {
        if (aspect_angle >= model->aspect_angles[i] && 
            aspect_angle <= model->aspect_angles[i + 1]) {
            
            double ratio = (aspect_angle - model->aspect_angles[i]) / 
                          (model->aspect_angles[i + 1] - model->aspect_angles[i]);
            
            angle_factor = model->reflection_coeffs[i] + 
                          ratio * (model->reflection_coeffs[i + 1] - model->reflection_coeffs[i]);
            break;
        }
    }
    
    // 频率修正（瑞利散射区域的频率依赖性）
    double freq_factor = 1.0;
    double wavelength = 1500.0 / frequency;  // 假设声速1500m/s
    double ka = 2.0 * PI * model->length / wavelength;  // 无量纲频率参数
    
    if (ka < 1.0) {
        // 瑞利散射区域
        freq_factor = ka * ka;
    } else if (ka < 10.0) {
        // 过渡区域
        freq_factor = 1.0 + 0.1 * log10(ka);
    } else {
        // 几何散射区域
        freq_factor = 1.0;
    }
    
    // 计算最终目标强度
    double target_strength = model->target_strength_ref + 
                            20.0 * log10(angle_factor) + 
                            10.0 * log10(freq_factor);
    
    return target_strength;
}

/**
 * 生成对抗信号
 */
Signal* generate_countermeasure_signal(const CountermeasureParams* params,
                                      double sampling_rate) {
    if (!params || sampling_rate <= 0 || params->duration <= 0) {
        return NULL;
    }
    
    size_t signal_length = (size_t)(params->duration * sampling_rate);
    Signal* signal = create_signal(signal_length, sampling_rate);
    if (!signal) {
        return NULL;
    }
    
    // 根据干扰类型生成不同的信号
    if (strcmp(params->jamming_type, "NOISE") == 0) {
        // 噪声干扰
        for (size_t i = 0; i < signal_length; i++) {
            // 生成高斯白噪声
            double noise = 0.0;
            for (int j = 0; j < 12; j++) {
                noise += (double)rand() / RAND_MAX;
            }
            noise = (noise - 6.0) / sqrt(12.0);  // 标准化为N(0,1)
            
            // 应用带宽限制（简单的矩形窗）
            double t = (double)i / sampling_rate;
            double carrier = cos(2.0 * PI * params->center_frequency * t);
            
            signal->data[i] = noise * carrier;
        }
        
    } else if (strcmp(params->jamming_type, "DECEPTION") == 0) {
        // 欺骗干扰（模拟目标回波）
        for (size_t i = 0; i < signal_length; i++) {
            double t = (double)i / sampling_rate;
            
            // 生成调制的正弦波
            double phase = 2.0 * PI * params->center_frequency * t + 
                          params->modulation_index * sin(2.0 * PI * 10.0 * t);
            
            signal->data[i] = cos(phase);
        }
        
    } else if (strcmp(params->jamming_type, "BARRAGE") == 0) {
        // 阻塞干扰（宽带噪声）
        for (size_t i = 0; i < signal_length; i++) {
            double t = (double)i / sampling_rate;
            double signal_sum = 0.0;
            
            // 多频率成分的叠加
            for (int k = 1; k <= 10; k++) {
                double freq = params->center_frequency + (k - 5.5) * params->bandwidth / 10.0;
                signal_sum += cos(2.0 * PI * freq * t + ((double)rand() / RAND_MAX) * 2.0 * PI);
            }
            
            signal->data[i] = signal_sum / 10.0;
        }
        
    } else {
        // 默认：单频干扰
        for (size_t i = 0; i < signal_length; i++) {
            double t = (double)i / sampling_rate;
            signal->data[i] = cos(2.0 * PI * params->center_frequency * t);
        }
    }
    
    // 应用声源级
    double amplitude = pow(10.0, params->source_level / 20.0) / 100.0;  // 归一化
    for (size_t i = 0; i < signal_length; i++) {
        signal->data[i] *= amplitude;
    }
    
    printf("Generated %s countermeasure: SL=%.1fdB, BW=%.0fHz, Duration=%.2fs\n",
           params->jamming_type, params->source_level, params->bandwidth, params->duration);
    
    return signal;
}

/**
 * 评估干扰效果
 */
double evaluate_jamming_effect(const Signal* jamming_signal,
                              const Signal* target_signal,
                              double detection_threshold) {
    if (!jamming_signal || !target_signal) {
        return 0.0;
    }
    
    // 计算目标信号功率
    double target_power = 0.0;
    size_t min_length = (jamming_signal->length < target_signal->length) ? 
                        jamming_signal->length : target_signal->length;
    
    for (size_t i = 0; i < min_length; i++) {
        target_power += target_signal->data[i] * target_signal->data[i];
    }
    target_power /= min_length;
    
    // 计算干扰信号功率
    double jamming_power = 0.0;
    for (size_t i = 0; i < min_length; i++) {
        jamming_power += jamming_signal->data[i] * jamming_signal->data[i];
    }
    jamming_power /= min_length;
    
    // 计算干扰信噪比
    double jsr = 10.0 * log10(jamming_power / (target_power + 1e-12));
    
    // 简化的干扰效果评估
    double effectiveness = 0.0;
    if (jsr > detection_threshold + 10.0) {
        effectiveness = 1.0;  // 完全有效
    } else if (jsr > detection_threshold) {
        effectiveness = (jsr - detection_threshold) / 10.0;  // 部分有效
    } else {
        effectiveness = 0.0;  // 无效
    }
    
    printf("Jamming evaluation: JSR=%.1fdB, Effectiveness=%.2f\n", jsr, effectiveness);
    
    return effectiveness;
}

/**
 * 计算检测概率 (增强版)
 */
double calculate_detection_probability_enhanced(double snr, double false_alarm_rate) {
    if (false_alarm_rate <= 0 || false_alarm_rate >= 1.0) {
        return 0.0;
    }
    
    // 使用Swerling-I目标模型的检测概率公式
    // 这是一个简化的实现，实际应用中需要更复杂的计算
    
    double snr_linear = pow(10.0, snr / 10.0);
    double threshold_factor = -log(false_alarm_rate);
    
    // Marcum Q函数的近似计算
    double pd = 0.0;
    if (snr > -10.0) {
        double x = sqrt(2.0 * snr_linear);
        double y = sqrt(2.0 * threshold_factor);
        
        // 简化的检测概率计算
        if (x > y) {
            pd = 1.0 - exp(-(x - y) * (x - y) / 2.0);
        } else {
            pd = exp(-(y - x) * (y - x) / 2.0) * false_alarm_rate;
        }
        
        // 限制在合理范围内
        if (pd > 1.0) pd = 1.0;
        if (pd < 0.0) pd = 0.0;
    }
    
    return pd;
} 

