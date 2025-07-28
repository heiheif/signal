#include <iostream>
#include <signal_lib.h>
#include <vector>

void run_signal_analysis_demo() {
    std::cout << "\n=== 信号分析与特征提取演示 ===\n\n";

    // 1. 生成测试信号
    std::cout << "生成测试信号...\n";
    double fs = 48000.0;  // 采样率 (Hz)
    double duration = 2.0; // 信号持续时间 (s)
    
    // 生成复合信号：包含两个频率分量
    Signal* test_signal = generate_basic_signal(1000.0, fs, duration);  // 1kHz基频
    if (!test_signal) {
        std::cout << "信号生成失败！\n";
        return;
    }

    // 添加2kHz分量
    Signal* high_freq = generate_basic_signal(2000.0, fs, duration);
    if (high_freq) {
        for (size_t i = 0; i < test_signal->length; i++) {
            test_signal->data[i] += 0.5 * high_freq->data[i];  // 添加半幅值的高频分量
        }
        destroy_signal(high_freq);
    }

    // 2. 特征提取
    std::cout << "\n执行特征提取...\n";
    
    // 使用新的SignalFeatures结构体
    SignalFeatures features;
    
    int feature_result = extract_features(test_signal, &features);
    if (feature_result == 0) {
        // 显示时域特征
        std::cout << "时域特征：\n";
        std::cout << "均值: " << features.time_domain_features[0] << "\n";
        std::cout << "方差: " << features.time_domain_features[1] << "\n";
        std::cout << "标准差: " << features.time_domain_features[2] << "\n";
        std::cout << "偏度: " << features.time_domain_features[3] << "\n";
        std::cout << "峰度: " << features.time_domain_features[4] << "\n";

        // 显示频域特征
        std::cout << "\n频域特征：\n";
        std::cout << "中心频率: " << features.freq_domain_features[0] << " Hz\n";
        std::cout << "主频: " << features.freq_domain_features[1] << " Hz\n";
        std::cout << "频带宽度: " << features.freq_domain_features[2] << " Hz\n";
        
        // 显示时频特征
        std::cout << "\n时频特征：\n";
        std::cout << "频谱质心变化率: " << features.time_freq_features[0] << "\n";
        std::cout << "频谱熵: " << features.time_freq_features[1] << "\n";
        std::cout << "频谱流量: " << features.time_freq_features[2] << "\n";
    }

    // 3. 线谱分析
    std::cout << "\n执行线谱分析...\n";
    LineSpectrum spectrum_result;
    spectrum_result.frequencies = new double[10];  // 假设最多10个峰值
    spectrum_result.amplitudes = new double[10];
    spectrum_result.peak_count = 0;

    int spectrum_status = analyze_line_spectrum(test_signal, 0.1, 100.0, &spectrum_result);
    if (spectrum_status == 0) {
        std::cout << "检测到 " << spectrum_result.peak_count << " 个频率峰值：\n";
        for (size_t i = 0; i < spectrum_result.peak_count; i++) {
            std::cout << "频率: " << spectrum_result.frequencies[i] 
                     << " Hz, 幅值: " << spectrum_result.amplitudes[i] << "\n";
        }
    }

    // 4. 信号滤波演示
    std::cout << "\n执行匹配滤波处理...\n";
    // 使用1kHz的正弦信号作为参考
    Signal* reference = generate_basic_signal(1000.0, fs, 0.1);  // 0.1s参考信号
    if (reference) {
        Signal* filtered = matched_filter(test_signal, reference);
        if (filtered) {
            std::cout << "匹配滤波完成\n";
            // 这里可以添加滤波结果的分析代码
            destroy_signal(filtered);
        }
        destroy_signal(reference);
    }

    // 5. 频谱分析
    std::cout << "\n执行频谱分析...\n";
    size_t spec_length;
    
    // 计算所需的FFT长度（2的幂）
    size_t fft_length = 1;
    while (fft_length < test_signal->length) {
        fft_length <<= 1;
    }
    
    // 分配足够的内存
    Complex* spectrum = new Complex[fft_length];
    int spec_result = spectrum_analysis(test_signal, spectrum, &spec_length);
    if (spec_result == 0) {
        std::cout << "频谱分析完成\n";
        // 显示一些频谱特征
        double max_amplitude = 0.0;
        size_t max_freq_index = 0;
        for (size_t i = 0; i < spec_length/2; i++) {  // 只看正频率部分
            double amplitude = complex_abs(spectrum[i]);
            if (amplitude > max_amplitude) {
                max_amplitude = amplitude;
                max_freq_index = i;
            }
        }
        double freq_resolution = fs / spec_length;  // 使用实际的FFT长度计算频率分辨率
        std::cout << "主要频率分量: " << (max_freq_index * freq_resolution) << " Hz\n";
        std::cout << "最大幅值: " << max_amplitude << "\n";
    } else {
        std::cout << "频谱分析失败\n";
    }

    // 清理内存
    delete[] spectrum_result.frequencies;
    delete[] spectrum_result.amplitudes;
    delete[] spectrum;
    destroy_signal(test_signal);

    std::cout << "\n信号分析与特征提取演示完成\n";
} 