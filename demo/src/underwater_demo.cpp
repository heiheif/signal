#include <iostream>
#include <signal_lib.h>
#include <vector>

void run_underwater_demo() {
    std::cout << "\n=== 水声传播与声纳系统演示 ===\n\n";

    // 1. 声速剖面设置
    const size_t profile_points = 10;
    SoundProfile profile;
    profile.length = profile_points;
    profile.depth = new double[profile_points];
    profile.speed = new double[profile_points];

    // 模拟典型的声速剖面
    for (size_t i = 0; i < profile_points; i++) {
        profile.depth[i] = i * 100.0;  // 0-900m，间隔100m
        // 简化的声速剖面模型
        profile.speed[i] = 1500.0 - 0.05 * profile.depth[i] + 
                          0.000075 * profile.depth[i] * profile.depth[i];
    }

    // 2. 声线追踪演示
    std::cout << "执行声线追踪计算...\n";
    const size_t angle_count = 5;
    double angles[angle_count] = {-30.0, -15.0, 0.0, 15.0, 30.0};  // 发射角度（度）
    double** ray_paths = new double*[angle_count];
    size_t* path_lengths = new size_t[angle_count];

    // 转换角度为弧度
    for (size_t i = 0; i < angle_count; i++) {
        angles[i] = angles[i] * PI / 180.0;
    }

    int result = ray_direction_calc(&profile, 50.0, angles, angle_count, 
                                  ray_paths, path_lengths);
    
    if (result == 0) {
        std::cout << "声线追踪计算完成！\n";
        // 这里可以添加声线路径的可视化代码
    } else {
        std::cout << "声线追踪计算失败！\n";
    }

    // 3. 被动声纳性能分析
    std::cout << "\n执行被动声纳性能分析...\n";
    
    // 设置声纳参数
    double sl = 160.0;  // 声源级 (dB)
    double nl = 60.0;   // 噪声级 (dB)
    double di = 15.0;   // 指向性指数 (dB)
    double dt = 10.0;   // 检测阈值 (dB)
    double alpha = 0.036;  // 吸收系数 (dB/km)
    double spreading = 15.0;  // 扩展因子

    // 计算最大检测距离
    double max_range = calculate_max_detection_range(sl, nl, di, dt, alpha, spreading);
    std::cout << "最大检测距离: " << max_range << " 米\n";

    // 计算1000米处的性能
    double test_range = 1000.0;
    double tl = calculate_transmission_loss(test_range, alpha, spreading);
    double se = calculate_passive_sonar(sl, tl, nl, di, dt);
    std::cout << "在" << test_range << "米处:" << std::endl;
    std::cout << "传播损失: " << tl << " dB" << std::endl;
    std::cout << "声纳方程余量: " << se << " dB" << std::endl;

    // 4. 环境噪声分析
    std::cout << "\n环境噪声分析...\n";
    double wind_speed = 5.0;  // 风速 (m/s)
    double shipping = 0.5;    // 船舶密度
    double freq = 1000.0;     // 频率 (Hz)
    
    double ambient_noise = calculate_ambient_noise(wind_speed, shipping, freq);
    std::cout << "环境噪声级: " << ambient_noise << " dB\n";

    // 5. 阵列性能分析
    std::cout << "\n阵列性能分析...\n";
    int array_elements = 16;           // 阵元数量
    double element_spacing = 0.75;     // 阵元间距 (m)
    double signal_direction = 45.0;    // 信号到达方向 (度)
    
    double array_gain = calculate_array_gain(array_elements, element_spacing, 
                                           signal_direction, freq);
    std::cout << "阵列增益: " << array_gain << " dB\n";

    // 清理内存
    delete[] profile.depth;
    delete[] profile.speed;
    delete[] path_lengths;
    for (size_t i = 0; i < angle_count; i++) {
        delete[] ray_paths[i];
    }
    delete[] ray_paths;

    std::cout << "\n水声传播与声纳系统演示完成\n";
}

/**
 * 水声对抗仿真综合演示
 * 展示新增功能的使用方法
 */

void demonstrate_enhanced_features() {
    printf("\n=== 水声对抗仿真系统功能演示 ===\n");
    
    // === 1. 环境噪声模型演示 ===
    printf("\n--- 1. 环境噪声模型演示 ---\n");
    
    NoiseModelParams noise_params = {
        .wind_speed = 5.0,          // 风速 5 m/s
        .shipping_factor = 0.3,     // 中等航运活动
        .bio_noise_level = 50.0,    // 生物噪声级
        .thermal_noise_ref = -15.0  // 热噪声参考级
    };
    
    NoiseModel* noise_model = create_noise_model(&noise_params, 100.0, 10000.0, 0);
    if (noise_model) {
        printf("环境噪声谱级 (Wenz模型):\n");
        double test_freqs[] = {500, 1000, 2000, 5000, 8000};
        for (int i = 0; i < 5; i++) {
            double noise_level = get_noise_spectrum(noise_model, test_freqs[i]);
            printf("  %.0f Hz: %.1f dB re 1μPa²/Hz\n", test_freqs[i], noise_level);
        }
        destroy_noise_model(noise_model);
    }
    
    // === 2. 标准声速剖面演示 ===
    printf("\n--- 2. 标准声速剖面演示 ---\n");
    
    SoundProfile* surface_duct = get_standard_svp(SURFACE_DUCT, 500.0, 30.0);
    if (surface_duct) {
        printf("表面波导声速剖面 (前5个点):\n");
        for (size_t i = 0; i < 5 && i < surface_duct->length; i++) {
            printf("  深度 %.0fm: 声速 %.1f m/s\n", 
                   surface_duct->depth[i], surface_duct->speed[i]);
        }
        destroy_sound_profile(surface_duct);
    }
    
    SoundProfile* deep_channel = get_standard_svp(DEEP_SOUND_CHANNEL, 2000.0, 0.0);
    if (deep_channel) {
        printf("深海声道剖面 (声轴附近):\n");
        for (size_t i = 130; i < 135 && i < deep_channel->length; i++) {
            printf("  深度 %.0fm: 声速 %.1f m/s\n", 
                   deep_channel->depth[i], deep_channel->speed[i]);
        }
        destroy_sound_profile(deep_channel);
    }
    
    // === 3. 目标特性模型演示 ===
    printf("\n--- 3. 目标特性模型演示 ---\n");
    
    TargetModel* submarine = create_target_model("SUBMARINE", 100.0, 10.0);
    if (submarine) {
        printf("潜艇目标强度 (长度100m, 吃水10m):\n");
        double test_angles[] = {0, 45, 90, 135, 180};
        for (int i = 0; i < 5; i++) {
            double ts = calculate_target_strength(submarine, 1000.0, test_angles[i]);
            printf("  方位角 %.0f°: TS = %.1f dB\n", test_angles[i], ts);
        }
        destroy_target_model(submarine);
    }
    
    TargetModel* uuv = create_target_model("UUV", 6.0, 0.8);
    if (uuv) {
        printf("UUV目标强度 (长度6m, 吃水0.8m):\n");
        double ts_0 = calculate_target_strength(uuv, 1000.0, 0.0);
        double ts_90 = calculate_target_strength(uuv, 1000.0, 90.0);
        printf("  正首方向: TS = %.1f dB\n", ts_0);
        printf("  正横方向: TS = %.1f dB\n", ts_90);
        destroy_target_model(uuv);
    }
    
    // === 4. 对抗信号生成演示 ===
    printf("\n--- 4. 对抗信号生成演示 ---\n");
    
    CountermeasureParams jamming_params = {
        .source_level = 180.0,      // 声源级 180 dB
        .bandwidth = 2000.0,        // 带宽 2 kHz
        .duration = 1.0,            // 持续时间 1 秒
        .center_frequency = 1000.0, // 中心频率 1 kHz
        .modulation_index = 0.5     // 调制指数
    };
    
    strcpy(jamming_params.jamming_type, "NOISE");
    Signal* noise_jamming = generate_countermeasure_signal(&jamming_params, 8000.0);
    
    strcpy(jamming_params.jamming_type, "DECEPTION");
    Signal* deception_jamming = generate_countermeasure_signal(&jamming_params, 8000.0);
    
    if (noise_jamming && deception_jamming) {
        // 创建目标信号用于干扰效果评估
        Signal* target_signal = generate_sine_wave(1000.0, 1.0, 8000.0);
        if (target_signal) {
            double noise_effect = evaluate_jamming_effect(noise_jamming, target_signal, 10.0);
            double deception_effect = evaluate_jamming_effect(deception_jamming, target_signal, 10.0);
            
            printf("干扰效果评估:\n");
            printf("  噪声干扰效果: %.2f\n", noise_effect);
            printf("  欺骗干扰效果: %.2f\n", deception_effect);
            
            destroy_signal(target_signal);
        }
    }
    
    // === 5. 信号处理增强演示 ===
    printf("\n--- 5. 信号处理增强演示 ---\n");
    
    // 生成LFM信号
    Signal* lfm_signal = generate_lfm_signal(0.1, 800.0, 1200.0, 8000.0);
    if (lfm_signal) {
        printf("LFM信号生成成功: 长度 %zu 样点\n", lfm_signal->length);
        
        // 生成相位编码信号
        int barker_code[] = {1, 1, 1, -1, -1, 1, -1};
        Signal* coded_signal = generate_phase_coded_signal(barker_code, 7, 0.001, 1000.0, 8000.0);
        
        if (coded_signal) {
            printf("Barker码信号生成成功: 长度 %zu 样点\n", coded_signal->length);
            
            // 脉冲压缩演示
            Signal* compressed = pulse_compression(lfm_signal, lfm_signal);
            if (compressed) {
                printf("脉冲压缩完成: 输出长度 %zu 样点\n", compressed->length);
                destroy_signal(compressed);
            }
            
            destroy_signal(coded_signal);
        }
        
        destroy_signal(lfm_signal);
    }
    
    // === 6. 检测概率计算演示 ===
    printf("\n--- 6. 检测概率计算演示 ---\n");
    
    printf("检测概率计算 (虚警率 1e-6):\n");
    double snr_values[] = {5, 10, 15, 20};
    for (int i = 0; i < 4; i++) {
                 double pd = calculate_detection_probability_enhanced(snr_values[i], 1e-6);
        printf("  SNR = %.0f dB: Pd = %.3f\n", snr_values[i], pd);
    }
    
    // === 7. 混响信号生成演示 ===
    printf("\n--- 7. 混响信号生成演示 ---\n");
    
    Signal* source_pulse = generate_sine_wave(1000.0, 0.1, 8000.0);
    if (source_pulse) {
        Signal* surface_reverb = generate_reverberation(SURFACE_REVERB, source_pulse, 3.0, 0.0, 5000.0);
        Signal* bottom_reverb = generate_reverberation(BOTTOM_REVERB, source_pulse, 0.0, 10.0, 5000.0);
        
        if (surface_reverb) {
            printf("海面混响信号生成成功\n");
            destroy_signal(surface_reverb);
        }
        
        if (bottom_reverb) {
            printf("海底混响信号生成成功\n");
            destroy_signal(bottom_reverb);
        }
        
        destroy_signal(source_pulse);
    }
    
    // 清理资源
    if (noise_jamming) destroy_signal(noise_jamming);
    if (deception_jamming) destroy_signal(deception_jamming);
    
    printf("\n=== 功能演示完成 ===\n");
}

/**
 * 红蓝对抗场景演示
 */
void demonstrate_red_blue_scenario() {
    printf("\n=== 红蓝对抗场景演示 ===\n");
    
    // 场景设置
    printf("场景设置: 红方UUV vs 蓝方护卫舰\n");
    
    // 1. 环境设置
    SoundProfile* env_profile = get_standard_svp(SURFACE_DUCT, 200.0, 35.0);
    NoiseModelParams env_noise = {3.0, 0.2, 45.0, -15.0};
    NoiseModel* ocean_noise = create_noise_model(&env_noise, 500.0, 5000.0, 0);
    
    // 2. 目标建模
    TargetModel* red_uuv = create_target_model("UUV", 8.0, 1.0);          // 红方UUV
    TargetModel* blue_ship = create_target_model("SURFACE_SHIP", 120.0, 8.0); // 蓝方护卫舰
    
    if (env_profile && ocean_noise && red_uuv && blue_ship) {
        // 3. 声场计算
        printf("\n计算声场传播特性...\n");
        
        SoundFieldResult* field_result = compute_sound_field_pe(
            1000.0,    // 频率 1kHz
            50.0,      // 声源深度 50m
            env_profile,
            200,       // 深度点数
            100,       // 距离点数
            10000.0    // 最大距离 10km
        );
        
        if (field_result) {
            // 4. 目标检测能力评估
            double red_ts = calculate_target_strength(red_uuv, 1000.0, 90.0);  // 侧面探测
            double blue_ts = calculate_target_strength(blue_ship, 1000.0, 90.0);
            double noise_level = get_noise_spectrum(ocean_noise, 1000.0);
            
            double red_detection_range = calculate_detection_range(field_result, red_ts, noise_level, 15.0);
            double blue_detection_range = calculate_detection_range(field_result, blue_ts, noise_level, 15.0);
            
            printf("\n目标检测能力分析:\n");
            printf("  红方UUV目标强度: %.1f dB\n", red_ts);
            printf("  蓝方护卫舰目标强度: %.1f dB\n", blue_ts);
            printf("  环境噪声级: %.1f dB\n", noise_level);
            printf("  红方UUV检测距离: %.0f m\n", red_detection_range);
            printf("  蓝方护卫舰检测距离: %.0f m\n", blue_detection_range);
            
            // 5. 对抗措施效果评估
            printf("\n对抗措施效果评估:\n");
            
            CountermeasureParams red_decoy = {
                .source_level = 160.0,
                .bandwidth = 1000.0,
                .duration = 2.0,
                .center_frequency = 1000.0,
                .modulation_index = 0.3
            };
            strcpy(red_decoy.jamming_type, "DECEPTION");
            
            Signal* decoy_signal = generate_countermeasure_signal(&red_decoy, 8000.0);
            Signal* uuv_echo = generate_sine_wave(1000.0, 0.1, 8000.0);
            
            if (decoy_signal && uuv_echo) {
                double decoy_effectiveness = evaluate_jamming_effect(decoy_signal, uuv_echo, 15.0);
                printf("  红方声诱饵效果: %.2f (%.0f%% 有效)\n", 
                       decoy_effectiveness, decoy_effectiveness * 100.0);
                
                destroy_signal(decoy_signal);
                destroy_signal(uuv_echo);
            }
            
            // 6. 战术建议
            printf("\n战术建议:\n");
            if (red_detection_range < blue_detection_range) {
                printf("  红方优势: UUV隐蔽性好，建议采用隐蔽接近战术\n");
            } else {
                printf("  蓝方优势: 探测距离远，建议红方使用声诱饵掩护\n");
            }
            
            if (red_detection_range > 0 && red_detection_range < 3000.0) {
                printf("  建议红方在%.0fm外实施攻击，避免被发现\n", red_detection_range + 500.0);
            }
            
            destroy_sound_field_result(field_result);
        }
    }
    
    // 清理资源
    if (env_profile) destroy_sound_profile(env_profile);
    if (ocean_noise) destroy_noise_model(ocean_noise);
    if (red_uuv) destroy_target_model(red_uuv);
    if (blue_ship) destroy_target_model(blue_ship);
    
    printf("\n=== 红蓝对抗场景演示完成 ===\n");
} 