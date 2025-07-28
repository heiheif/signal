#include "../include/signal_lib.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/**
 * 创建声速剖面对象
 */
SIGNAL_LIB_API SoundProfile* create_sound_profile(size_t length) {
    if (length == 0) {
        return NULL;
    }
    
    SoundProfile* profile = (SoundProfile*)malloc(sizeof(SoundProfile));
    if (!profile) {
        return NULL;
    }
    
    profile->depth = (double*)malloc(length * sizeof(double));
    profile->speed = (double*)malloc(length * sizeof(double));
    
    if (!profile->depth || !profile->speed) {
        if (profile->depth) free(profile->depth);
        if (profile->speed) free(profile->speed);
        free(profile);
        return NULL;
    }
    
    profile->length = length;
    return profile;
}

/**
 * 销毁声速剖面对象
 */
SIGNAL_LIB_API void destroy_sound_profile(SoundProfile* profile) {
    if (profile) {
        if (profile->depth) free(profile->depth);
        if (profile->speed) free(profile->speed);
        free(profile);
    }
}

/**
 * 使用Mackenzie公式计算声速
 */
SIGNAL_LIB_API double calculate_sound_speed_mackenzie(double temperature, double salinity, double depth) {
    // 参数范围检查和调整（温和处理）
    double T = temperature;
    double S = salinity;
    double D = depth;
    
    // 温度范围检查和调整
    if (T < -10.0) {
        printf("Warning: Temperature %.2f°C is below minimum (-10°C), clamped to -10°C\n", T);
        T = -10.0;
    } else if (T > 40.0) {
        printf("Warning: Temperature %.2f°C is above maximum (40°C), clamped to 40°C\n", T);
        T = 40.0;
    }
    
    // 盐度范围检查和调整
    if (S < 0.0) {
        printf("Warning: Salinity %.2f‰ is below minimum (0‰), clamped to 0‰\n", S);
        S = 0.0;
    } else if (S > 45.0) {
        printf("Warning: Salinity %.2f‰ is above maximum (45‰), clamped to 45‰\n", S);
        S = 45.0;
    }
    
    // 深度范围检查和调整
    if (D < 0.0) {
        printf("Warning: Depth %.2fm is below minimum (0m), clamped to 0m\n", D);
        D = 0.0;
    } else if (D > 12000.0) {
        printf("Warning: Depth %.2fm is above maximum (12000m), clamped to 12000m\n", D);
        D = 12000.0;
    }
    
    // Mackenzie九项公式(1981)完整实现
    // 根据NPL标准的完整Mackenzie公式
    double c = 1448.96 + 4.591*T - 5.304e-2*T*T + 2.374e-4*T*T*T + 
               1.340*(S-35) + 1.630e-2*D + 1.675e-7*D*D - 
               1.025e-2*T*(S-35) - 7.139e-13*T*D*D*D;
    
    return c;
}

/**
 * 使用Chen-Millero公式计算声速
 */
SIGNAL_LIB_API double calculate_sound_speed_chen_millero(double temperature, double salinity, double depth) {
    // 参数范围检查和调整（温和处理）
    double T = temperature;
    double S = salinity;
    double D = depth;
    
    // 温度范围检查和调整
    if (T < -10.0) {
        printf("Warning: Temperature %.2f°C is below minimum (-10°C), clamped to -10°C\n", T);
        T = -10.0;
    } else if (T > 40.0) {
        printf("Warning: Temperature %.2f°C is above maximum (40°C), clamped to 40°C\n", T);
        T = 40.0;
    }
    
    // 盐度范围检查和调整
    if (S < 0.0) {
        printf("Warning: Salinity %.2f‰ is below minimum (0‰), clamped to 0‰\n", S);
        S = 0.0;
    } else if (S > 45.0) {
        printf("Warning: Salinity %.2f‰ is above maximum (45‰), clamped to 45‰\n", S);
        S = 45.0;
    }
    
    // 深度范围检查和调整
    if (D < 0.0) {
        printf("Warning: Depth %.2fm is below minimum (0m), clamped to 0m\n", D);
        D = 0.0;
    } else if (D > 12000.0) {
        printf("Warning: Depth %.2fm is above maximum (12000m), clamped to 12000m\n", D);
        D = 12000.0;
    }
    
    double P = D * 0.1 + 1.01325;  // 正确的压力转换
    
    // Chen-Millero公式(1977)改进实现
    
    // 纯水中的声速项
    double Cw = 1402.388 + 5.03830*T - 5.81090e-2*T*T + 3.3432e-4*T*T*T 
                - 1.47797e-6*T*T*T*T + 3.1419e-9*T*T*T*T*T;
    
    // 压力修正项
    double A = 1.60272e-1 + 1.0268e-5*T + 3.5734e-9*T*T - 3.3603e-12*T*T*T;
    double B = 7.4706e-6 - 1.6618e-8*T + 2.6801e-11*T*T;
    double Cp = A*P + B*P*P;
    
    // 盐度项
    double S_diff = S - 35.0;
    double Cs = 1.39799*S_diff + 1.69202e-3*S_diff*S_diff + (S_diff > 0 ? -1.1244e-7*S_diff*S_diff*S_diff : 0);
    
    // 盐度-温度交叉项
    double Cst = S_diff*(9.4742e-5*T - 1.2580e-5*T*T - 6.4928e-8*T*T*T);
    
    // 盐度-压力交叉项  
    double Csp = S_diff*P*(1.727e-3 - 7.9836e-6*P);
    
    return Cw + Cp + Cs + Cst + Csp;
}

/**
 * 使用简化经验公式计算声速
 */
SIGNAL_LIB_API double calculate_sound_speed_empirical(double temperature, double salinity, double depth) {
    // 参数范围检查和调整（温和处理）
    double T = temperature;
    double S = salinity;
    double D = depth;
    
    // 温度范围检查和调整
    if (T < -10.0) {
        printf("Warning: Temperature %.2f°C is below minimum (-10°C), clamped to -10°C\n", T);
        T = -10.0;
    } else if (T > 40.0) {
        printf("Warning: Temperature %.2f°C is above maximum (40°C), clamped to 40°C\n", T);
        T = 40.0;
    }
    
    // 盐度范围检查和调整
    if (S < 0.0) {
        printf("Warning: Salinity %.2f‰ is below minimum (0‰), clamped to 0‰\n", S);
        S = 0.0;
    } else if (S > 45.0) {
        printf("Warning: Salinity %.2f‰ is above maximum (45‰), clamped to 45‰\n", S);
        S = 45.0;
    }
    
    // 深度范围检查和调整
    if (D < 0.0) {
        printf("Warning: Depth %.2fm is below minimum (0m), clamped to 0m\n", D);
        D = 0.0;
    } else if (D > 12000.0) {
        printf("Warning: Depth %.2fm is above maximum (12000m), clamped to 12000m\n", D);
        D = 12000.0;
    }
    
    // 简化经验公式：c = 1450 + 4.21T - 0.037T² + 1.14(S-35) + 0.175P
    double P = D * 0.1 + 1.01325;  // 深度转换为压力(bar)
    
    double c = 1450.0 + 4.21*T - 0.037*T*T + 1.14*(S-35) + 0.175*P;
    
    return c;
}

/**
 * 根据CTD数据生成声速剖面
 */
SoundProfile* generate_svp_from_ctd(const double* temperatures, const double* salinities, 
                                    const double* depths, size_t length, int method) {
    if (!temperatures || !salinities || !depths || length == 0) {
        return NULL;
    }
    
    SoundProfile* profile = create_sound_profile(length);
    if (!profile) {
        return NULL;
    }
    
    // 根据选择的方法计算声速
    for (size_t i = 0; i < length; i++) {
        profile->depth[i] = depths[i];
        
        if (method == 0) {
            // Mackenzie公式
            profile->speed[i] = calculate_sound_speed_mackenzie(temperatures[i], salinities[i], depths[i]);
        } else if (method == 1) {
            // Chen-Millero公式
            profile->speed[i] = calculate_sound_speed_chen_millero(temperatures[i], salinities[i], depths[i]);
        } else {
            // 简化经验公式
            profile->speed[i] = calculate_sound_speed_empirical(temperatures[i], salinities[i], depths[i]);
        }
        
        // 注意：不再需要检查计算结果，因为所有函数都保证返回有效值
    }
    
    return profile;
}

/**
 * 从声速测量数据创建声速剖面
 */
SIGNAL_LIB_API SoundProfile* create_svp_from_measurements(const double* depths, const double* speeds, size_t length) {
    if (!depths || !speeds || length == 0) {
        return NULL;
    }
    
    SoundProfile* profile = create_sound_profile(length);
    if (!profile) {
        return NULL;
    }
    
    // 复制数据
    memcpy(profile->depth, depths, length * sizeof(double));
    memcpy(profile->speed, speeds, length * sizeof(double));
    
    return profile;
}

/**
 * 融合测量的声速数据和CTD计算的声速数据
 */
SoundProfile* fuse_sound_profiles(const SoundProfile* measured_profile, 
                                 const SoundProfile* calculated_profile,
                                 double weight_measured) {
    if (!measured_profile || !calculated_profile) {
        return NULL;
    }
    
    // 参数检查
    if (weight_measured < 0.0) weight_measured = 0.0;
    if (weight_measured > 1.0) weight_measured = 1.0;
    double weight_calculated = 1.0 - weight_measured;
    
    // 确定融合剖面的深度范围和分辨率
    double min_depth = fmax(measured_profile->depth[0], calculated_profile->depth[0]);
    double max_depth = fmin(measured_profile->depth[measured_profile->length-1], 
                           calculated_profile->depth[calculated_profile->length-1]);
    
    if (min_depth >= max_depth) {
        return NULL;  // 两个剖面没有重叠范围
    }
    
    // 使用计算剖面的深度点在重叠范围内
    size_t valid_count = 0;
    for (size_t i = 0; i < calculated_profile->length; i++) {
        if (calculated_profile->depth[i] >= min_depth && calculated_profile->depth[i] <= max_depth) {
            valid_count++;
        }
    }
    
    if (valid_count == 0) {
        return NULL;
    }
    
    // 创建融合剖面
    SoundProfile* fused_profile = create_sound_profile(valid_count);
    if (!fused_profile) {
        return NULL;
    }
    
    // 填充融合数据
    size_t index = 0;
    for (size_t i = 0; i < calculated_profile->length; i++) {
        double depth = calculated_profile->depth[i];
        if (depth >= min_depth && depth <= max_depth) {
            fused_profile->depth[index] = depth;
            double calc_speed = calculated_profile->speed[i];
            double measured_speed = interpolate_sound_speed(measured_profile, depth);
            
            // 检查插值结果的有效性
            if (measured_speed > 0.0) {
                fused_profile->speed[index] = weight_measured * measured_speed + weight_calculated * calc_speed;
            } else {
                fused_profile->speed[index] = calc_speed;  // 插值失败时使用计算值
            }
            index++;
        }
    }
    
    return fused_profile;
}

/**
 * 在指定深度处插值计算声速值 (改进版：支持三次样条插值)
 */
SIGNAL_LIB_API double interpolate_sound_speed(const SoundProfile* profile, double target_depth) {
    if (!profile || profile->length == 0) {
        return 0.0;
    }
    
    // 处理超出范围的情况
    if (target_depth <= profile->depth[0]) {
        return profile->speed[0];
    }
    
    if (target_depth >= profile->depth[profile->length - 1]) {
        return profile->speed[profile->length - 1];
    }
    
    // 对于小数据集，使用线性插值
    if (profile->length < 4) {
        // 线性插值
        for (size_t i = 0; i < profile->length - 1; i++) {
            if (target_depth >= profile->depth[i] && target_depth <= profile->depth[i + 1]) {
                double ratio = (target_depth - profile->depth[i]) / (profile->depth[i + 1] - profile->depth[i]);
                return profile->speed[i] + ratio * (profile->speed[i + 1] - profile->speed[i]);
            }
        }
    } else {
        // 对于大数据集，使用三次样条插值的简化版本
        for (size_t i = 1; i < profile->length - 2; i++) {
            if (target_depth >= profile->depth[i] && target_depth <= profile->depth[i + 1]) {
                // 计算局部三次插值
                double h0 = profile->depth[i] - profile->depth[i-1];
                double h1 = profile->depth[i+1] - profile->depth[i];
                double h2 = profile->depth[i+2] - profile->depth[i+1];
                
                double c0 = profile->speed[i-1];
                double c1 = profile->speed[i];
                double c2 = profile->speed[i+1];
                double c3 = profile->speed[i+2];
                
                // 计算一阶导数近似
                double d1 = (c2 - c0) / (h0 + h1);
                double d2 = (c3 - c1) / (h1 + h2);
                
                // 归一化参数
                double t = (target_depth - profile->depth[i]) / h1;
                
                // 三次Hermite插值
                double result = c1 * (1 - t) + c2 * t + 
                               t * (1 - t) * ((1 - t) * d1 * h1 - t * d2 * h1);
                
                return result;
            }
        }
        
        // 边界情况回退到线性插值
        for (size_t i = 0; i < profile->length - 1; i++) {
            if (target_depth >= profile->depth[i] && target_depth <= profile->depth[i + 1]) {
                double ratio = (target_depth - profile->depth[i]) / (profile->depth[i + 1] - profile->depth[i]);
                return profile->speed[i] + ratio * (profile->speed[i + 1] - profile->speed[i]);
            }
        }
    }
    
    // 不应该到达这里，但为了安全返回一个值
    return profile->speed[0];
}

/**
 * 检查声速剖面数据的质量
 */
SIGNAL_LIB_API int check_sound_profile_quality(const SoundProfile* profile) {
    if (!profile || profile->length < 2) {
        return -1;
    }
    
    // 检查深度是否单调递增
    for (size_t i = 1; i < profile->length; i++) {
        if (profile->depth[i] <= profile->depth[i-1]) {
            return -2;  // 深度必须单调递增
        }
    }
    
    // 检查声速值是否在合理范围内
    // 海水声速通常在1400-1600 m/s范围，但在极地或深海可能超出此范围
    for (size_t i = 0; i < profile->length; i++) {
        if (profile->speed[i] < 1350.0 || profile->speed[i] > 1650.0) {
            return -3;  // 声速值可能异常
        }
    }
    
    // 检查相邻声速值变化是否合理
    // 调整阈值以适应不同海洋环境
    for (size_t i = 1; i < profile->length; i++) {
        double depth_diff = profile->depth[i] - profile->depth[i-1];
        double speed_diff = fabs(profile->speed[i] - profile->speed[i-1]);
        double change_rate = speed_diff / depth_diff;
        
        // 根据深度调整阈值：浅水区域更严格，深水区域更宽松
        double threshold = (profile->depth[i] < 200.0) ? 2.0 : 5.0;  // m/s per meter
        
        if (change_rate > threshold) {
            return -4;  // 声速变化梯度过大
        }
    }
    
    return 0;  // 数据质量正常
}

/**
 * 使用抛物线模型计算声速剖面(paowu.m的简化模型)
 */
SoundProfile* generate_svp_parabolic_model(const double* depth, size_t length, 
                                          double c0, double eps, double feature_depth) {
    if (!depth || length == 0) {
        return NULL;
    }
    
    // 使用默认参数，如果提供的参数无效
    if (c0 <= 0) c0 = 1500.0;            // 默认基准声速1500m/s
    if (eps <= 0) eps = 0.00737;         // 默认声速变化系数
    if (feature_depth <= 0) feature_depth = 1300.0;  // 默认特征深度1300m
    
    // 创建声速剖面对象
    SoundProfile* profile = create_sound_profile(length);
    if (!profile) {
        return NULL;
    }
    
    // 复制深度数据
    memcpy(profile->depth, depth, length * sizeof(double));
    
    // 计算声速
    for (size_t i = 0; i < length; i++) {
        // 根据 paowu.m 的公式: c = c0 * (1 + eps * (x - 1 + exp(-x)))
        // 其中 x = 2 * (depth - feature_depth) / feature_depth
        double z = depth[i];
        double x = 2.0 * (z - feature_depth) / feature_depth;
        profile->speed[i] = c0 * (1.0 + eps * (x - 1.0 + exp(-x)));
    }
    
    return profile;
}

// === 环境噪声模型实现 ===

/**
 * 创建环境噪声模型
 */
SIGNAL_LIB_API NoiseModel* create_noise_model(const NoiseModelParams* params, 
                              double min_freq, double max_freq, 
                              int model_type) {
    if (!params || min_freq <= 0 || max_freq <= min_freq) {
        return NULL;
    }
    
    NoiseModel* model = (NoiseModel*)malloc(sizeof(NoiseModel));
    if (!model) {
        return NULL;
    }
    
    model->params = *params;
    model->frequency_range[0] = min_freq;
    model->frequency_range[1] = max_freq;
    model->model_type = model_type;
    
    printf("Created noise model: wind=%.1f m/s, shipping=%.2f, type=%d\n",
           params->wind_speed, params->shipping_factor, model_type);
    
    return model;
}

/**
 * 销毁噪声模型
 */
SIGNAL_LIB_API void destroy_noise_model(NoiseModel* model) {
    if (model) {
        free(model);
    }
}

/**
 * 计算Wenz环境噪声谱
 */
SIGNAL_LIB_API double calculate_wenz_noise(double frequency, double wind_speed, double shipping_factor) {
    if (frequency <= 0) {
        return 0.0;
    }
    
    double freq_khz = frequency / 1000.0;  // 转换为kHz
    
    // Wenz海洋噪声模型
    double wind_noise = 0.0;
    double shipping_noise = 0.0;
    double thermal_noise = 0.0;
    
    // 风生噪声 (Knudsen公式修正版)
    if (freq_khz >= 0.1 && freq_khz <= 100.0) {
        wind_noise = 44.0 + 17.0 * log10(wind_speed) - 20.0 * log10(freq_khz);
        if (wind_speed < 1.0) wind_noise = 44.0 - 20.0 * log10(freq_khz);
    }
    
    // 航运噪声
    if (freq_khz >= 0.01 && freq_khz <= 10.0) {
        shipping_noise = 60.0 + 20.0 * shipping_factor - 20.0 * log10(freq_khz);
    }
    
    // 热噪声
    if (freq_khz >= 1.0) {
        thermal_noise = -15.0 + 20.0 * log10(freq_khz);
    }
    
    // 能量叠加
    double total_noise = 10.0 * log10(
        pow(10.0, wind_noise / 10.0) + 
        pow(10.0, shipping_noise / 10.0) + 
        pow(10.0, thermal_noise / 10.0)
    );
    
    return total_noise;
}

/**
 * 计算指定频率的环境噪声谱级
 */
SIGNAL_LIB_API double get_noise_spectrum(const NoiseModel* model, double frequency) {
    if (!model || frequency < model->frequency_range[0] || 
        frequency > model->frequency_range[1]) {
        return 0.0;
    }
    
    if (model->model_type == 0) {
        // Wenz模型
        return calculate_wenz_noise(frequency, model->params.wind_speed, 
                                   model->params.shipping_factor);
    } else {
        // 简化模型
        double freq_khz = frequency / 1000.0;
        double noise_level = model->params.bio_noise_level + 
                           model->params.thermal_noise_ref - 
                           15.0 * log10(freq_khz);
        return noise_level;
    }
}

/**
 * 生成混响信号
 */
SIGNAL_LIB_API Signal* generate_reverberation(ReverbType reverb_type,
                              const Signal* source_signal,
                              double sea_state,
                              double bottom_loss,
                              double range_m) {
    if (!source_signal || sea_state < 0 || sea_state > 9 || range_m <= 0) {
        return NULL;
    }
    
    // 创建混响信号
    Signal* reverb_signal = create_signal(source_signal->length, source_signal->fs);
    if (!reverb_signal) {
        return NULL;
    }
    
    // 计算混响强度
    double reverb_strength = 0.0;
    
    switch (reverb_type) {
        case SURFACE_REVERB:
            // 海面混响强度与海况相关
            reverb_strength = -60.0 + 10.0 * log10(sea_state + 1.0) - 20.0 * log10(range_m / 1000.0);
            break;
            
        case BOTTOM_REVERB:
            // 海底混响强度与底质相关
            reverb_strength = -50.0 - bottom_loss - 30.0 * log10(range_m / 1000.0);
            break;
            
        case VOLUME_REVERB:
            // 体积混响（生物散射等）
            reverb_strength = -80.0 - 40.0 * log10(range_m / 1000.0);
            break;
    }
    
    // 应用混响强度到信号
    double reverb_factor = pow(10.0, reverb_strength / 20.0);
    
    for (size_t i = 0; i < source_signal->length; i++) {
        // 添加随机相位和衰减
        double random_phase = ((double)rand() / RAND_MAX) * 2.0 * PI;
        reverb_signal->data[i] = source_signal->data[i] * reverb_factor * 
                                cos(random_phase + i * 0.01);
    }
    
    printf("Generated %s reverb: strength=%.1f dB, range=%.0f m\n",
           (reverb_type == SURFACE_REVERB) ? "surface" : 
           (reverb_type == BOTTOM_REVERB) ? "bottom" : "volume",
           reverb_strength, range_m);
    
    return reverb_signal;
}

/**
 * 获取标准声速剖面
 */
SIGNAL_LIB_API SoundProfile* get_standard_svp(SoundProfileType profile_type,
                              double max_depth,
                              double latitude) {
    if (max_depth <= 0) {
        return NULL;
    }
    
    size_t point_count = (size_t)(max_depth / 10.0) + 1;  // 每10m一个点
    if (point_count < 10) point_count = 10;
    
    SoundProfile* profile = create_sound_profile(point_count);
    if (!profile) {
        return NULL;
    }
    
    // 生成深度数组
    for (size_t i = 0; i < point_count; i++) {
        profile->depth[i] = i * max_depth / (point_count - 1);
    }
    
    switch (profile_type) {
        case SURFACE_DUCT:
            // 表面波导剖面
            for (size_t i = 0; i < point_count; i++) {
                double z = profile->depth[i];
                if (z <= 50.0) {
                    profile->speed[i] = 1520.0 - 2.0 * z / 50.0;  // 表面混合层
                } else if (z <= 200.0) {
                    profile->speed[i] = 1518.0 - 15.0 * (z - 50.0) / 150.0;  // 跃层
                } else {
                    profile->speed[i] = 1503.0 + 0.02 * (z - 200.0);  // 深层递增
                }
            }
            break;
            
        case DEEP_SOUND_CHANNEL:
            // 深海声道剖面 (Munk模型简化版)
            for (size_t i = 0; i < point_count; i++) {
                double z = profile->depth[i];
                double z_norm = 2.0 * (z - 1300.0) / 1000.0;  // 归一化深度
                double eps = 0.00737;
                profile->speed[i] = 1500.0 * (1.0 + eps * (z_norm - 1.0 + exp(-z_norm)));
            }
            break;
            
        case ARCTIC_PROFILE:
            // 北极剖面（低温高盐）
            for (size_t i = 0; i < point_count; i++) {
                double z = profile->depth[i];
                if (z <= 100.0) {
                    profile->speed[i] = 1430.0 + 0.5 * z / 100.0;  // 冷水层
                } else {
                    profile->speed[i] = 1430.5 + 0.02 * (z - 100.0);  // 深层递增
                }
            }
            break;
            
        case TROPICAL_PROFILE:
            // 热带剖面（高温）
            for (size_t i = 0; i < point_count; i++) {
                double z = profile->depth[i];
                if (z <= 100.0) {
                    profile->speed[i] = 1540.0 - 3.0 * z / 100.0;  // 温跃层
                } else {
                    profile->speed[i] = 1537.0 + 0.015 * (z - 100.0);  // 深层递增
                }
            }
            break;
            
        default:
            // 默认线性剖面
            for (size_t i = 0; i < point_count; i++) {
                profile->speed[i] = 1500.0 + 0.017 * profile->depth[i];
            }
            break;
    }
    
    printf("Generated standard SVP: type=%d, points=%zu, max_depth=%.0f m\n",
           profile_type, point_count, max_depth);
    
    return profile;
} 