#ifndef SIGNAL_LIB_H
#define SIGNAL_LIB_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// === 基本定义和常量 ===
#define PI 3.14159265358979323846
#define MAX_SIGNAL_LENGTH 1000000

// === 基础数据结构定义 ===

/**
 * @brief 信号基本数据结构
 * @details 用于存储一维时域信号数据，包含采样率信息
 */
typedef struct {
    double* data;      // 信号数据数组
    size_t length;     // 信号长度
    double fs;         // 采样率(Hz)
} Signal;

/**
 * @brief 复数数据结构
 * @details 用于FFT等复数运算
 */
typedef struct {
    double real;       // 实部
    double imag;       // 虚部
} Complex;

/**
 * @brief 线谱分析结果结构
 * @details 存储频谱分析得到的线谱信息
 */
typedef struct {
    double* frequencies;  // 频率数组
    double* amplitudes;   // 幅值数组
    size_t peak_count;    // 峰值数量
} LineSpectrum;

/**
 * @brief 声速剖面数据结构
 * @details 存储海洋声速随深度的变化数据
 */
typedef struct {
    double* depth;     // 深度数组(m)
    double* speed;     // 声速数组(m/s)
    size_t length;     // 数据点数
} SoundProfile;

/**
 * @brief 信号特征结构
 */
typedef struct {
    double time_domain_features[5];   // 时域特征
    double freq_domain_features[3];   // 频域特征
    double time_freq_features[10];    // 时频特征
} SignalFeatures;

// === 基础函数声明 ===

/**
 * @brief 创建信号对象
 * @param length 信号长度
 * @param fs 采样率(Hz)
 * @return 成功返回Signal指针，失败返回NULL
 */
Signal* create_signal(size_t length, double fs);

/**
 * @brief 销毁信号对象，释放内存
 * @param sig 要销毁的信号对象指针
 */
void destroy_signal(Signal* sig);

/**
 * @brief 复数加法运算
 * @param a 第一个复数
 * @param b 第二个复数
 * @return 两个复数的和
 */
Complex complex_add(Complex a, Complex b);

/**
 * @brief 复数减法运算
 * @param a 被减数
 * @param b 减数
 * @return 两个复数的差
 */
Complex complex_subtract(Complex a, Complex b);

/**
 * @brief 复数乘法运算
 * @param a 第一个复数
 * @param b 第二个复数
 * @return 两个复数的积
 */
Complex complex_multiply(Complex a, Complex b);

/**
 * @brief 复数除法运算
 * @param a 被除数
 * @param b 除数
 * @return 两个复数的商
 */
Complex complex_divide(Complex a, Complex b);

/**
 * @brief 计算复数的模
 * @param z 输入复数
 * @return 复数的模
 */
double complex_abs(Complex z);

/**
 * @brief 计算复数的相位
 * @param z 输入复数
 * @return 复数的相位(弧度)
 */
double complex_phase(Complex z);

/**
 * @brief 对数据应用汉宁窗
 * @param data 输入数据数组
 * @param length 数据长度
 */
void apply_hanning_window(double* data, size_t length);

/**
 * @brief 对数据应用汉明窗
 * @param data 输入数据数组
 * @param length 数据长度
 */
void apply_hamming_window(double* data, size_t length);

/**
 * @brief 快速傅里叶变换
 * @param data 输入/输出复数数组
 * @param n 数据长度(必须是2的幂)
 */
void fft(Complex* data, size_t n);

/**
 * @brief 逆快速傅里叶变换
 * @param data 输入/输出复数数组
 * @param n 数据长度(必须是2的幂)
 */
void ifft(Complex* data, size_t n);

/**
 * @brief 递归实现的FFT/IFFT
 * @param data 输入/输出复数数组
 * @param n 数据长度(必须是2的幂)
 * @param inverse 0表示FFT，1表示IFFT
 */
void fft_recursive(Complex* data, size_t n, int inverse);

// === 信号处理函数 ===

/**
 * @brief 生成基本正弦信号
 * @param freq 信号频率(Hz)
 * @param fs 采样率(Hz)
 * @param duration 信号持续时间(s)
 * @return 成功返回Signal指针，失败返回NULL
 */
Signal* generate_basic_signal(double freq, double fs, double duration);

/**
 * @brief 对信号进行频谱分析
 * @param sig 输入信号
 * @param spectrum 输出频谱(复数形式)
 * @param spec_length 输出频谱长度
 * @return 成功返回0，失败返回-1
 */
int spectrum_analysis(const Signal* sig, Complex* spectrum, size_t* spec_length);

/**
 * @brief 生成连续波(CW)信号
 * @param freq 载波频率(Hz)
 * @param fs 采样率(Hz)
 * @param duration 信号持续时间(s)
 * @param amplitude 信号幅度
 * @param phase 初始相位(弧度)
 * @return 成功返回Signal指针，失败返回NULL
 */
Signal* generate_cw(double freq, double fs, double duration, double amplitude, double phase);

/**
 * @brief 生成线性调频(LFM)信号
 * @param f_start 起始频率(Hz)
 * @param f_end 终止频率(Hz)
 * @param fs 采样率(Hz)
 * @param duration 信号持续时间(s)
 * @return 成功返回Signal指针，失败返回NULL
 */
Signal* generate_lfm(double f_start, double f_end, double fs, double duration);

/**
 * @brief 生成双曲调频(HFM)信号
 * @param f_start 起始频率(Hz)
 * @param f_end 终止频率(Hz)
 * @param fs 采样率(Hz)
 * @param duration 信号持续时间(s)
 * @return 成功返回Signal指针，失败返回NULL
 */
Signal* generate_hfm(double f_start, double f_end, double fs, double duration);

/**
 * @brief 生成相位键控(PSK)信号
 * @param carrier_freq 载波频率(Hz)
 * @param fs 采样率(Hz)
 * @param symbol_count 符号数量
 * @param samples_per_symbol 每个符号的采样点数
 * @param symbols 符号序列数组
 * @return 成功返回Signal指针，失败返回NULL
 */
Signal* generate_psk(double carrier_freq, double fs, int symbol_count, 
                    int samples_per_symbol, const int* symbols);

/**
 * @brief 生成频率键控(FSK)信号
 * @param freqs 频率数组
 * @param freq_count 频率数量
 * @param fs 采样率(Hz)
 * @param symbol_count 符号数量
 * @param samples_per_symbol 每个符号的采样点数
 * @param symbols 符号序列数组
 * @return 成功返回Signal指针，失败返回NULL
 */
Signal* generate_fsk(double* freqs, int freq_count, double fs, 
                    int symbol_count, int samples_per_symbol, const int* symbols);

// === 信号检测函数 ===

/**
 * @brief 匹配滤波器
 * @param input 输入信号
 * @param reference 参考信号
 * @return 成功返回滤波后的Signal指针，失败返回NULL
 */
Signal* matched_filter(const Signal* input, const Signal* reference);

/**
 * @brief 线谱分析
 * @param sig 输入信号
 * @param min_peak_height 最小峰值高度
 * @param min_peak_distance 最小峰值间距
 * @param result 分析结果
 * @return 成功返回0，失败返回-1
 */
int analyze_line_spectrum(const Signal* sig, double min_peak_height,
                         double min_peak_distance, LineSpectrum* result);

/**
 * @brief 提取信号特征
 * @param sig 输入信号
 * @param time_features 时域特征输出数组
 * @param freq_features 频域特征输出数组
 * @return 成功返回0，失败返回-1
 */
int extract_features(const Signal* sig, double* time_features, double* freq_features);

// === 水声信号处理函数 ===

/**
 * @brief 声线追踪计算
 * @param profile 声速剖面
 * @param source_depth 声源深度(m)
 * @param angles 发射角数组(弧度)
 * @param angle_count 发射角数量
 * @param ray_paths 声线路径数组
 * @param path_lengths 各声线路径长度
 * @return 成功返回0，失败返回-1
 */
int ray_direction_calc(const SoundProfile* profile, double source_depth,
                      double* angles, size_t angle_count,
                      double** ray_paths, size_t* path_lengths);

/**
 * @brief 抛物方程声场计算
 * @param profile 声速剖面
 * @param freq 声波频率(Hz)
 * @param source_depth 声源深度(m)
 * @param field 输出声场(复数形式)
 * @param range_points 距离采样点数
 * @param depth_points 深度采样点数
 * @return 成功返回0，失败返回-1
 */
int parabolic_equation_field(const SoundProfile* profile, double freq,
                           double source_depth, Complex** field,
                           size_t range_points, size_t depth_points);

/**
 * @brief 计算被动声纳方程
 * @param sl 声源级(dB)
 * @param tl 传播损失(dB)
 * @param nl 噪声级(dB)
 * @param di 指向性指数(dB)
 * @param dt 检测阈值(dB)
 * @return 声纳方程结果(dB)
 */
double calculate_passive_sonar(double sl, double tl, double nl, double di, double dt);

/**
 * @brief 计算传播损失
 * @param distance 传播距离(m)
 * @param alpha 吸收系数(dB/km)
 * @param spreading_factor 扩展因子(通常为10-20)
 * @return 传播损失(dB)
 */
double calculate_transmission_loss(double distance, double alpha, double spreading_factor);

/**
 * @brief 计算环境噪声级
 * @param wind_speed 风速(m/s)
 * @param shipping_density 船舶密度(0-1)
 * @param freq 频率(Hz)
 * @return 噪声级(dB)
 */
double calculate_ambient_noise(double wind_speed, double shipping_density, double freq);

/**
 * @brief 计算阵列增益
 * @param array_elements 阵元数量
 * @param element_spacing 阵元间距(m)
 * @param signal_direction 信号到达方向(度)
 * @param freq 频率(Hz)
 * @return 阵增益(dB)
 */
double calculate_array_gain(int array_elements, double element_spacing, 
                          double signal_direction, double freq);

/**
 * @brief 计算检测概率
 * @param snr 信噪比(dB)
 * @param threshold 检测阈值(dB)
 * @param time_bandwidth 时间带宽积
 * @return 检测概率(0-1)
 */
double calculate_detection_probability(double snr, double threshold, double time_bandwidth);

/**
 * @brief 计算最大检测距离
 * @param sl 声源级(dB)
 * @param nl 噪声级(dB)
 * @param di 指向性指数(dB)
 * @param dt 检测阈值(dB)
 * @param alpha 吸收系数(dB/km)
 * @param spreading_factor 扩展因子
 * @return 最大检测距离(m)
 */
double calculate_max_detection_range(double sl, double nl, double di, double dt,
                                   double alpha, double spreading_factor);

#ifdef __cplusplus
}
#endif

#endif // SIGNAL_LIB_H 

