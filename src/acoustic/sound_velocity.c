#include "../include/signal_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * 创建声速剖面对象
 */
SoundProfile* create_sound_profile(size_t length) {
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
void destroy_sound_profile(SoundProfile* profile) {
    if (profile) {
        if (profile->depth) free(profile->depth);
        if (profile->speed) free(profile->speed);
        free(profile);
    }
}

/**
 * 使用Mackenzie公式计算声速
 */
double calculate_sound_speed_mackenzie(double temperature, double salinity, double depth) {
    // Mackenzie公式(1981): c = 1448.96 + 4.591T - 0.05304T² + 0.0002374T³ + 1.340(S-35) + 0.0163D
    double T = temperature;
    double S = salinity;
    double D = depth;
    
    return 1448.96 + 4.591*T - 0.05304*T*T + 0.0002374*T*T*T + 1.340*(S-35) + 0.0163*D;
}

/**
 * 使用Chen-Millero公式计算声速
 */
double calculate_sound_speed_chen_millero(double temperature, double salinity, double depth) {
    // Chen-Millero公式(1977)
    // 这是一个简化的实现，完整版本较为复杂，涉及多个压力和温度修正项
    
    double T = temperature;
    double S = salinity;
    double P = depth / 10.0;  // 转换为压力（粗略近似，深度单位为米，压力单位为bar）
    
    // 基础声速项
    double Cw = 1402.388 + 5.03830*T - 5.81090e-2*T*T + 3.3432e-4*T*T*T - 1.47797e-6*T*T*T*T + 3.1419e-9*T*T*T*T*T;
    
    // 压力项
    double A = 1.603e-2*T + 2.5e-7*T*T - 1.322e-8*T*T*T;
    double B = 7.139e-13*T*T*T;
    double Cp = A*P + B*P*P;
    
    // 盐度项
    double Cs = (1.39799)*(S-35) + 1.69202e-3*(S-35)*(S-35);
    
    // 组合项
    double result = Cw + Cp + Cs;
    
    return result;
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
        } else {
            // Chen-Millero公式
            profile->speed[i] = calculate_sound_speed_chen_millero(temperatures[i], salinities[i], depths[i]);
        }
    }
    
    return profile;
}

/**
 * 从声速测量数据创建声速剖面
 */
SoundProfile* create_svp_from_measurements(const double* depths, const double* speeds, size_t length) {
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
    
    // 创建新的融合剖面
    // 这里使用计算剖面的深度点，简化处理
    SoundProfile* fused_profile = create_sound_profile(calculated_profile->length);
    if (!fused_profile) {
        return NULL;
    }
    
    // 复制深度数据
    memcpy(fused_profile->depth, calculated_profile->depth, calculated_profile->length * sizeof(double));
    
    // 对每个深度点计算融合的声速值
    for (size_t i = 0; i < calculated_profile->length; i++) {
        double depth = calculated_profile->depth[i];
        double calc_speed = calculated_profile->speed[i];
        
        // 在测量剖面中找到最接近的深度点
        double measured_speed = interpolate_sound_speed(measured_profile, depth);
        
        // 融合计算
        fused_profile->speed[i] = weight_measured * measured_speed + weight_calculated * calc_speed;
    }
    
    return fused_profile;
}

/**
 * 在指定深度处插值计算声速值
 */
double interpolate_sound_speed(const SoundProfile* profile, double target_depth) {
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
    
    // 线性插值
    for (size_t i = 0; i < profile->length - 1; i++) {
        if (target_depth >= profile->depth[i] && target_depth <= profile->depth[i + 1]) {
            double ratio = (target_depth - profile->depth[i]) / (profile->depth[i + 1] - profile->depth[i]);
            return profile->speed[i] + ratio * (profile->speed[i + 1] - profile->speed[i]);
        }
    }
    
    // 不应该到达这里，但为了安全返回一个值
    return profile->speed[0];
}

/**
 * 检查声速剖面数据的质量
 */
int check_sound_profile_quality(const SoundProfile* profile) {
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
    // 海水声速通常在1400-1600 m/s范围
    for (size_t i = 0; i < profile->length; i++) {
        if (profile->speed[i] < 1400.0 || profile->speed[i] > 1600.0) {
            return -3;  // 声速值可能异常
        }
    }
    
    // 检查相邻声速值变化是否合理
    // 突变可能表示测量错误
    for (size_t i = 1; i < profile->length; i++) {
        double change_rate = fabs(profile->speed[i] - profile->speed[i-1]) / 
                            (profile->depth[i] - profile->depth[i-1]);
        if (change_rate > 1.0) {  // 声速梯度阈值（每米变化超过1m/s可能不合理）
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