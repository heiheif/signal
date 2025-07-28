 #include "../include/signal_lib.hpp"

// 被动声纳方程实现
SIGNAL_LIB_API double calculate_passive_sonar(double sl, double tl, double nl, double di, double dt) {
    // SL - 声源级
    // TL - 传播损失
    // NL - 噪声级
    // DI - 指向性指数
    // DT - 检测阈值
    
    // 计算声纳方程
    double se = sl - 2 * tl - (nl - di) - dt;
    return se;
}

// 计算传播损失
SIGNAL_LIB_API double calculate_transmission_loss(double distance, double alpha, double spreading_factor) {
    // distance - 传播距离(m)
    // alpha - 吸收系数(dB/km)
    // spreading_factor - 扩展因子(通常为10-20)
    
    // 计算几何扩展损失
    double geometric_loss = spreading_factor * log10(distance);
    
    // 计算吸收损失(将距离转换为km)
    double absorption_loss = alpha * (distance / 1000.0);
    
    // 总传播损失
    return geometric_loss + absorption_loss;
}

// 计算海洋环境噪声级
SIGNAL_LIB_API double calculate_ambient_noise(double wind_speed, double shipping_density, double freq) {
    // wind_speed - 风速(m/s)
    // shipping_density - 船舶密度(0-1)
    // freq - 频率(Hz)
    
    // 风生噪声(Knudsen公式)
    double wind_noise = 44 + 17 * log10(wind_speed) - 20 * log10(freq/1000.0);
    
    // 船舶噪声
    double shipping_noise = 60 + 20 * shipping_density - 20 * log10(freq/1000.0);
    
    // 热噪声
    double thermal_noise = -15 + 20 * log10(freq/1000.0);
    
    // 总噪声级(能量叠加)
    double total_noise = 10 * log10(pow(10, wind_noise/10) + 
                                  pow(10, shipping_noise/10) + 
                                  pow(10, thermal_noise/10));
    
    return total_noise;
}

// 计算阵增益
SIGNAL_LIB_API double calculate_array_gain(int array_elements, double element_spacing, 
                          double signal_direction, double freq) {
    // array_elements - 阵元数量
    // element_spacing - 阵元间距(m)
    // signal_direction - 信号到达方向(度)
    // freq - 频率(Hz)
    
    double wavelength = 1500.0 / freq;  // 假设声速为1500m/s
    double k = 2 * PI / wavelength;     // 波数
    
    // 计算阵列方向性函数
    double sum_real = 0.0;
    double sum_imag = 0.0;
    for (int i = 0; i < array_elements; i++) {
        double phase = k * i * element_spacing * sin(signal_direction * PI / 180.0);
        sum_real += cos(phase);
        sum_imag += sin(phase);
    }
    
    // 计算阵增益
    double array_pattern = sqrt(sum_real * sum_real + sum_imag * sum_imag);
    double array_gain = 20 * log10(array_pattern / sqrt(array_elements));
    
    return array_gain;
}

// 计算检测概率
SIGNAL_LIB_API double calculate_detection_probability(double snr, double threshold, double time_bandwidth) {
    // snr - 信噪比(dB)
    // threshold - 检测阈值(dB)
    // time_bandwidth - 时间带宽积
    
    // 将dB转换为线性比
    double snr_linear = pow(10, snr/10);
    double threshold_linear = pow(10, threshold/10);
    
    // 使用改进的检测概率计算公式
    double d = sqrt(2 * time_bandwidth * snr_linear);
    double x = threshold_linear / sqrt(2 * time_bandwidth);
    
    // Q函数近似
    double q = 0.5 * erfc(x - d/2);
    
    return q;
}

// 计算最大检测距离
SIGNAL_LIB_API double calculate_max_detection_range(double sl, double nl, double di, double dt,
                                   double alpha, double spreading_factor) {
    // 使用二分法求解最大检测距离
    double min_range = 1.0;      // 最小距离1m
    double max_range = 100000.0; // 最大距离100km
    double tolerance = 0.1;      // 精度0.1m
    
    while (max_range - min_range > tolerance) {
        double mid_range = (min_range + max_range) / 2;
        double tl = calculate_transmission_loss(mid_range, alpha, spreading_factor);
        double se = calculate_passive_sonar(sl, tl, nl, di, dt);
        
        if (se > 0) {
            min_range = mid_range;
        } else {
            max_range = mid_range;
        }
    }
    
    return (min_range + max_range) / 2;
}