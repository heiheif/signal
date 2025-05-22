#include <iostream>
#include <signal_lib.h>
#include <vector>
#include <cmath>

void run_communication_demo() {
    std::cout << "\n=== 通信信号生成与处理演示 ===\n\n";

    // 1. 基本参数设置
    double fs = 48000.0;  // 采样率 (Hz)
    double duration = 1.0; // 信号持续时间 (s)

    // 2. 生成连续波信号
    std::cout << "生成连续波(CW)信号...\n";
    double cw_freq = 1000.0;  // 1kHz
    Signal* cw_signal = generate_cw(cw_freq, fs, duration, 1.0, 0.0);
    if (cw_signal) {
        std::cout << "CW信号生成成功，长度: " << cw_signal->length << " 采样点\n";
    }

    // 3. 生成线性调频信号
    std::cout << "\n生成线性调频(LFM)信号...\n";
    double f_start = 1000.0;  // 起始频率 1kHz
    double f_end = 5000.0;    // 终止频率 5kHz
    Signal* lfm_signal = generate_lfm(f_start, f_end, fs, duration);
    if (lfm_signal) {
        std::cout << "LFM信号生成成功，带宽: " << (f_end - f_start) << " Hz\n";
    }

    // 4. 生成双曲调频信号
    std::cout << "\n生成双曲调频(HFM)信号...\n";
    Signal* hfm_signal = generate_hfm(f_start, f_end, fs, duration);
    if (hfm_signal) {
        std::cout << "HFM信号生成成功\n";
    }

    // 5. 生成QPSK信号
    std::cout << "\n生成QPSK信号...\n";
    int symbol_count = 100;  // 符号数量
    int samples_per_symbol = 48;  // 每个符号的采样点数
    std::vector<int> symbols(symbol_count);
    // 生成随机符号序列
    for (int i = 0; i < symbol_count; i++) {
        symbols[i] = rand() % 4;  // 0-3，对应QPSK的四个相位
    }
    Signal* psk_signal = generate_psk(2000.0, fs, symbol_count, 
                                    samples_per_symbol, symbols.data());
    if (psk_signal) {
        std::cout << "QPSK信号生成成功，符号率: " << (fs/samples_per_symbol) << " Baud\n";
    }

    // 6. 生成4FSK信号
    std::cout << "\n生成4FSK信号...\n";
    double freqs[4] = {1000.0, 2000.0, 3000.0, 4000.0};  // 四个频率
    Signal* fsk_signal = generate_fsk(freqs, 4, fs, symbol_count, 
                                    samples_per_symbol, symbols.data());
    if (fsk_signal) {
        std::cout << "4FSK信号生成成功\n";
    }

    // 7. 信号处理演示
    if (lfm_signal && hfm_signal) {
        std::cout << "\n执行匹配滤波处理...\n";
        // 使用LFM信号作为参考信号对HFM信号进行匹配滤波
        Signal* matched_output = matched_filter(hfm_signal, lfm_signal);
        if (matched_output) {
            std::cout << "匹配滤波处理完成\n";
            // 这里可以添加输出信号的分析代码
            destroy_signal(matched_output);
        }
    }

    // 8. 频谱分析
    if (psk_signal) {
        std::cout << "\n执行频谱分析...\n";
        size_t spec_length;
        
        // 计算所需的FFT长度（2的幂）
        size_t fft_length = 1;
        while (fft_length < psk_signal->length) {
            fft_length <<= 1;
        }
        
        // 分配足够的内存
        Complex* spectrum = nullptr;
        try {
            spectrum = new Complex[fft_length];
            std::cout << "分配内存成功，FFT长度: " << fft_length << "\n";
            
            int spec_result = spectrum_analysis(psk_signal, spectrum, &spec_length);
            if (spec_result == 0) {
                std::cout << "频谱分析完成\n";
                
                // 分析频谱特征
                double max_amplitude = 0.0;
                size_t max_freq_index = 0;
                double total_power = 0.0;
                
                // 只分析正频率部分（0到fs/2）
                for (size_t i = 0; i < spec_length/2; i++) {
                    double amplitude = sqrt(spectrum[i].real * spectrum[i].real + 
                                         spectrum[i].imag * spectrum[i].imag);
                    total_power += amplitude * amplitude;
                    
                    if (amplitude > max_amplitude) {
                        max_amplitude = amplitude;
                        max_freq_index = i;
                    }
                }
                
                // 计算并显示结果
                double freq_resolution = fs / spec_length;
                std::cout << "频谱分析结果：\n";
                std::cout << "- 主要频率分量: " << (max_freq_index * freq_resolution) << " Hz\n";
                std::cout << "- 最大幅值: " << max_amplitude << "\n";
                std::cout << "- 总功率: " << total_power << "\n";
                std::cout << "- 频率分辨率: " << freq_resolution << " Hz\n";
            } else {
                std::cout << "频谱分析失败，错误代码: " << spec_result << "\n";
            }
        } catch (const std::bad_alloc& e) {
            std::cout << "内存分配失败！需要的内存大小: " << 
                     (fft_length * sizeof(Complex)) << " bytes\n";
        }
        
        // 清理内存
        delete[] spectrum;
    }

    // 清理内存
    if (cw_signal) destroy_signal(cw_signal);
    if (lfm_signal) destroy_signal(lfm_signal);
    if (hfm_signal) destroy_signal(hfm_signal);
    if (psk_signal) destroy_signal(psk_signal);
    if (fsk_signal) destroy_signal(fsk_signal);

    std::cout << "\n通信信号生成与处理演示完成\n";
} 