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