#ifndef SIGNAL_LIB_HPP
#define SIGNAL_LIB_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "signal_struct.hpp"

// 动态库导出宏定义
#ifdef _WIN32
    #ifdef SIGNAL_LIB_EXPORTS
        #define SIGNAL_LIB_API __declspec(dllexport)
    #else
        #define SIGNAL_LIB_API __declspec(dllimport)
    #endif
#else
    #define SIGNAL_LIB_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma region 基础数学运算函数
// === 基础数学运算函数 ===

/**
 * @brief 复数加法运算
 * @param a 第一个复数
 * @param b 第二个复数
 * @return 两个复数的和
 */
SIGNAL_LIB_API Complex complex_add(Complex a, Complex b);

/**
 * @brief 复数减法运算
 * @param a 被减数
 * @param b 减数
 * @return 两个复数的差
 */
SIGNAL_LIB_API Complex complex_subtract(Complex a, Complex b);

/**
 * @brief 复数乘法运算
 * @param a 第一个复数
 * @param b 第二个复数
 * @return 两个复数的积
 */
SIGNAL_LIB_API Complex complex_multiply(Complex a, Complex b);

/**
 * @brief 复数除法运算
 * @param a 被除数
 * @param b 除数
 * @return 两个复数的商
 */
SIGNAL_LIB_API Complex complex_divide(Complex a, Complex b);

/**
 * @brief 计算复数的模
 * @param z 输入复数
 * @return 复数的模
 */
SIGNAL_LIB_API double complex_abs(Complex z);

/**
 * @brief 计算复数的相位
 * @param z 输入复数
 * @return 复数的相位(弧度)
 */
SIGNAL_LIB_API double complex_phase(Complex z);

/**
 * @brief 对数据应用汉宁窗
 * @param data 输入数据数组
 * @param length 数据长度
 */
SIGNAL_LIB_API void apply_hanning_window(double* data, size_t length);

/**
 * @brief 对数据应用汉明窗
 * @param data 输入数据数组
 * @param length 数据长度
 */
SIGNAL_LIB_API void apply_hamming_window(double* data, size_t length);

/**
 * @brief 快速傅里叶变换
 * @param data 输入/输出复数数组
 * @param n 数据长度(必须是2的幂)
 */
SIGNAL_LIB_API void fft(Complex* data, size_t n);

/**
 * @brief 逆快速傅里叶变换
 * @param data 输入/输出复数数组
 * @param n 数据长度(必须是2的幂)
 */
SIGNAL_LIB_API void ifft(Complex* data, size_t n);

/**
 * @brief 递归实现的FFT/IFFT
 * @param data 输入/输出复数数组
 * @param n 数据长度(必须是2的幂)
 * @param inverse 0表示FFT，1表示IFFT
 */
SIGNAL_LIB_API void fft_recursive(Complex* data, size_t n, int inverse);
#pragma endregion

// =================================== core interface ===================================

/**
 * 判断被动声呐是否检测到目标
 * 
 * @param SL 目标的声源级 (dB)
 * @param TL 传输损失 (dB)
 * @param NL 噪声级 (dB)
 * @param DI 指向性指数 (dB)
 * @param threshold 设备的检测阈值 (dB)
 * @return 如果目标被检测到，返回 true，否则返回 false
 */
SIGNAL_LIB_API bool passiveSonarDetection(double SL, double TL, double NL, double DI, double threshold) ;
/**
 * 判断主动声呐（单基地模式）是否检测到目标
 * 
 * @param SL 声呐的声源级 (dB)
 * @param TL 单向传输损失 (dB)
 * @param TS 目标强度 (dB)
 * @param NL 噪声级 (dB)
 * @param DI 指向性指数 (dB)
 * @param threshold 设备的检测阈值 (dB)
 * @return 如果目标被检测到，返回 true，否则返回 false
 */
SIGNAL_LIB_API bool activeSonarDetectionMonostatic(double SL, double TL, double TS, double NL, double DI, double threshold) ;

/**
 * 判断主动声呐（双基地模式）是否检测到目标
 * 
 * @param SL 声呐的声源级 (dB)
 * @param TL1 从声源到目标的传输损失 (dB)
 * @param TL2 从目标到接收器的传输损失 (dB)
 * @param TS 目标强度 (dB)
 * @param NL 噪声级 (dB)
 * @param DI 指向性指数 (dB)
 * @param threshold 设备的检测阈值 (dB)
 * @return 如果目标被检测到，返回 true，否则返回 false
 */
SIGNAL_LIB_API bool activeSonarDetectionBistatic(double SL, double TL1, double TL2, double TS, double NL, double DI, double threshold) ;


/**
 * TL为传播衰减，分三种算法，采用三选一
 * 1. 球面扩展衰减: TL = 20*lg(R)
 * 2. 柱面扩展衰减: TL = 10*lg(R)  
 * @param cal_type 计算类型：0=球面扩展，1=柱面扩展，2=声场传播模型
 * @param R 距离，单位米
 * @return 传输损失(dB)
 */
SIGNAL_LIB_API double calculateTL(TLCalType cal_type, double R);


/**
 * TL 为传播衰减，如上方法，直接传递损失系数和空气吸收系数，计算TL
 * 
 * @param TL_coefficient 损失系数
 * @param air_absorption_coefficient 空气吸收系数(dB/km)
 * @param R 距离，单位米
 * @return 传输损失(dB)
 */
SIGNAL_LIB_API double calculateTL(double TL_coefficient, double air_absorption_coefficient, double R);


/**
 * 计算指向性指数
 * 假设阵列由 ( n ) 个阵元组成，阵元间距为 ( d )，工作频率为 ( f )，声速为 ( c )，则 DI 近似为：DI≈10lg⁡(n)
 * 更精确的计算考虑阵元间距和工作频率的影响
 * @param params 指向性指数计算参数
 * @return 指向性指数(dB)
 */
SIGNAL_LIB_API double calculateDI(DIParams* params);

// =================================== core interface ===================================


#pragma region 高级声场计算接口
// === 高级声场计算接口 ===
/**
 * @brief 创建声场计算参数结构体
 * @param sound_profile 声速剖面
 * @param frequency_hz 声源频率(Hz)
 * @param source_depth_m 声源深度(m)
 * @param max_range_m 最大计算距离(m)
 * @param max_depth_m 最大计算深度(m)
 * @return 成功返回SoundFieldParams指针，失败返回NULL
 */
SIGNAL_LIB_API SoundFieldParams* create_sound_field_params(
    const SoundProfile* sound_profile,
    double frequency_hz,
    double source_depth_m,
    double max_range_m,
    double max_depth_m
);

/**
 * @brief 设置声场计算的网格参数
 * @param params 声场参数结构体
 * @param range_points 距离方向网格点数
 * @param depth_points 深度方向网格点数
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int set_sound_field_grid(SoundFieldParams* params, size_t range_points, size_t depth_points);

/**
 * @brief 设置声场计算的高级参数
 * @param params 声场参数结构体
 * @param gaussian_beam_width_factor 高斯源束宽因子
 * @param boundary_condition_type 边界条件类型
 * @param bottom_attenuation_db_lambda 海底衰减(dB/波长)
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int set_sound_field_advanced_params(
    SoundFieldParams* params,
    double gaussian_beam_width_factor,
    int boundary_condition_type,
    double bottom_attenuation_db_lambda
);

/**
 * @brief 基于抛物方程方法计算声场传输损失
 * @param params 声场计算参数
 * @param result 输出结果结构体
 * @return 成功返回0，失败返回负值错误代码
 * @details 
 * 使用Crank-Nicolson隐式格式求解抛物方程：
 * ?ψ/?r = (ik0)^(-1) * [??ψ/?z? + k0?(n?(z)-1)ψ]
 * 
 * 算法特点：
 * - 支持任意声速剖面
 * - 自动应用Hankel函数修正
 * - 数值稳定的隐式格式
 * - 高斯源函数初始化
 * - 支持多种边界条件
 * 
 * 错误代码：
 * -1: 参数无效
 * -2: 内存分配失败
 * -3: 矩阵奇异，数值不稳定
 * -4: 计算过程中出现非有限值
 */
SIGNAL_LIB_API int compute_sound_field_pe(const SoundFieldParams* params, SoundFieldResult* result);

/**
 * @brief 创建声场结果结构体
 * @param range_points 距离方向点数
 * @param depth_points 深度方向点数
 * @return 成功返回SoundFieldResult指针，失败返回NULL
 */
SIGNAL_LIB_API SoundFieldResult* create_sound_field_result(size_t range_points, size_t depth_points);

/**
 * @brief 销毁声场参数结构体，释放内存
 * @param params 要销毁的参数结构体指针
 */
SIGNAL_LIB_API void destroy_sound_field_params(SoundFieldParams* params);

/**
 * @brief 销毁声场结果结构体，释放内存
 * @param result 要销毁的结果结构体指针
 */
SIGNAL_LIB_API void destroy_sound_field_result(SoundFieldResult* result);

/**
 * @brief 从声场结果中提取指定距离处的传输损失剖面
 * @param result 声场计算结果
 * @param range_m 目标距离(m)
 * @param depth_profile 输出深度数组(m)
 * @param tl_profile 输出传输损失数组(dB)
 * @param profile_length 输出数组长度
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int extract_range_profile(
    const SoundFieldResult* result,
    double range_m,
    double** depth_profile,
    double** tl_profile,
    size_t* profile_length
);

/**
 * @brief 从声场结果中提取指定深度处的传输损失剖面
 * @param result 声场计算结果
 * @param depth_m 目标深度(m)
 * @param range_profile 输出距离数组(m)
 * @param tl_profile 输出传输损失数组(dB)
 * @param profile_length 输出数组长度
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int extract_depth_profile(
    const SoundFieldResult* result,
    double depth_m,
    double** range_profile,
    double** tl_profile,
    size_t* profile_length
);

/**
 * @brief 计算声场结果的统计信息
 * @param result 声场计算结果
 * @param min_tl_db 最小传输损失(dB)
 * @param max_tl_db 最大传输损失(dB)
 * @param mean_tl_db 平均传输损失(dB)
 * @param convergence_range_m 声场收敛距离(m)
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int analyze_sound_field_statistics(
    const SoundFieldResult* result,
    double* min_tl_db,
    double* max_tl_db,
    double* mean_tl_db,
    double* convergence_range_m
);
#pragma endregion

#pragma region 环境噪声模型
/**
 * @brief 创建环境噪声模型
 * @param params 噪声模型参数
 * @param min_freq 最小频率(Hz)
 * @param max_freq 最大频率(Hz)
 * @param model_type 模型类型：0=Wenz模型, 1=简化模型
 * @return 成功返回NoiseModel指针，失败返回NULL
 */
SIGNAL_LIB_API NoiseModel* create_noise_model(const NoiseModelParams* params, 
                                            double min_freq, double max_freq, 
                                            int model_type);

/**
 * @brief 销毁噪声模型
 * @param model 噪声模型指针
 */
SIGNAL_LIB_API void destroy_noise_model(NoiseModel* model);

/**
 * @brief 计算指定频率的环境噪声谱级
 * @param model 噪声模型
 * @param frequency 频率(Hz)
 * @return 噪声谱级(dB re 1μPa²/Hz)
 */
SIGNAL_LIB_API double get_noise_spectrum(const NoiseModel* model, double frequency);
#pragma endregion

#pragma region 混响信号和标准声速剖面
/**
 * @brief 生成混响信号
 * @param reverb_type 混响类型
 * @param source_signal 源信号
 * @param sea_state 海况等级[0-9]
 * @param bottom_loss 海底损失(dB)
 * @param range_m 距离(m)
 * @return 成功返回混响信号，失败返回NULL
 */
SIGNAL_LIB_API Signal* generate_reverberation(ReverbType reverb_type,
                                            const Signal* source_signal,
                                            double sea_state,
                                            double bottom_loss,
                                            double range_m);

/**
 * @brief 获取标准声速剖面
 * @param profile_type 声速剖面类型
 * @param max_depth 最大深度(m)
 * @param latitude 纬度(度)，可选参数，默认为0
 * @return 成功返回SoundProfile指针，失败返回NULL
 */
SIGNAL_LIB_API SoundProfile* get_standard_svp(SoundProfileType profile_type,
                                             double max_depth,
                                             double latitude);

/**
 * @brief 计算Wenz环境噪声谱
 * @param frequency 频率(Hz)
 * @param wind_speed 风速(m/s)
 * @param shipping_factor 航运因子[0-1]
 * @return 噪声谱级(dB re 1μPa²/Hz)
 */
SIGNAL_LIB_API double calculate_wenz_noise(double frequency, double wind_speed, double shipping_factor);
#pragma endregion

#pragma region 目标声学特性模型
/**
 * @brief 创建标准目标模型
 * @param target_type 目标类型："SUBMARINE", "SURFACE_SHIP", "UUV", "TORPEDO"
 * @param length 目标长度(m)
 * @param draft 吃水深度(m)
 * @return 成功返回TargetModel指针，失败返回NULL
 */
SIGNAL_LIB_API TargetModel* create_target_model(const char* target_type, 
                                               double length, 
                                               double draft);

/**
 * @brief 销毁目标模型
 * @param model 目标模型指针
 */
SIGNAL_LIB_API void destroy_target_model(TargetModel* model);

/**
 * @brief 计算目标强度
 * @param model 目标模型
 * @param frequency 频率(Hz)
 * @param aspect_angle 目标方位角(度)，0度为正首方向
 * @return 目标强度(dB)
 */
SIGNAL_LIB_API double calculate_target_strength(const TargetModel* model,
                                               double frequency,
                                               double aspect_angle);
#pragma endregion

#pragma region 对抗装备和干扰效果
/**
 * @brief 生成对抗信号
 * @param params 对抗装备参数
 * @param sampling_rate 采样率(Hz)
 * @return 成功返回对抗信号，失败返回NULL
 */
SIGNAL_LIB_API Signal* generate_countermeasure_signal(const CountermeasureParams* params,
                                                    double sampling_rate);

/**
 * @brief 评估干扰效果
 * @param jamming_signal 干扰信号
 * @param target_signal 目标信号
 * @param detection_threshold 检测阈值(dB)
 * @return 干扰效果评估值，>1表示干扰有效
 */
SIGNAL_LIB_API double evaluate_jamming_effect(const Signal* jamming_signal,
                                             const Signal* target_signal,
                                             double detection_threshold);

/**
 * @brief 计算检测概率 (增强版)
 * @param snr 信噪比(dB)
 * @param false_alarm_rate 虚警率
 * @return 检测概率[0-1]
 */
SIGNAL_LIB_API double calculate_detection_probability_enhanced(double snr, double false_alarm_rate);

/**
 * @brief 脉冲压缩处理
 * @param received_signal 接收信号
 * @param transmitted_signal 发射信号（参考信号）
 * @return 成功返回压缩后的信号，失败返回NULL
 */
SIGNAL_LIB_API Signal* pulse_compression(const Signal* received_signal, 
                                       const Signal* transmitted_signal);

/**
 * @brief 多普勒处理
 * @param received_signal 接收信号
 * @param pulse_width 脉冲宽度(s)
 * @param doppler_bins 输出多普勒频率bins数组长度
 * @return 成功返回多普勒频谱数组，失败返回NULL
 */
SIGNAL_LIB_API double* doppler_processing(const Signal* received_signal,
                                        double pulse_width,
                                        size_t* doppler_bins);

/**
 * @brief 计算检测距离
 * @param sound_field 声场计算结果
 * @param target_strength 目标强度(dB)
 * @param noise_level 噪声级(dB)
 * @param detection_threshold 检测阈值(dB)
 * @return 检测距离(m)，-1表示无法检测
 */
SIGNAL_LIB_API double calculate_detection_range(const SoundFieldResult* sound_field,
                                               double target_strength,
                                               double noise_level,
                                               double detection_threshold);

/**
 * @brief 生成线性调频(LFM)信号
 * @param duration 信号时长(s)
 * @param start_freq 起始频率(Hz)
 * @param end_freq 结束频率(Hz)
 * @param sampling_rate 采样率(Hz)
 * @return 成功返回LFM信号，失败返回NULL
 */
SIGNAL_LIB_API Signal* generate_lfm_signal(double duration,
                                          double start_freq,
                                          double end_freq,
                                          double sampling_rate);

/**
 * @brief 生成相位编码信号
 * @param code_sequence 编码序列数组
 * @param code_length 编码长度
 * @param chip_duration 码元持续时间(s)
 * @param carrier_freq 载波频率(Hz)
 * @param sampling_rate 采样率(Hz)
 * @return 成功返回相位编码信号，失败返回NULL
 */
SIGNAL_LIB_API Signal* generate_phase_coded_signal(const int* code_sequence,
                                                  size_t code_length,
                                                  double chip_duration,
                                                  double carrier_freq,
                                                  double sampling_rate);
#pragma endregion

#pragma region 信号对象管理
// === 信号对象管理 ===

/**
 * @brief 创建信号对象
 * @param length 信号长度
 * @param fs 采样率(Hz)
 * @return 成功返回Signal指针，失败返回NULL
 */
SIGNAL_LIB_API Signal* create_signal(size_t length, double fs);

/**
 * @brief 销毁信号对象，释放内存
 * @param sig 要销毁的信号对象指针
 */
SIGNAL_LIB_API void destroy_signal(Signal* sig);
#pragma endregion

#pragma region 基础信号生成
// === 基础信号生成 ===

/**
 * @brief 生成基本正弦信号
 * @param freq 信号频率(Hz)
 * @param fs 采样率(Hz)
 * @param duration 信号持续时间(s)
 * @return 成功返回Signal指针，失败返回NULL
 */
SIGNAL_LIB_API Signal* generate_basic_signal(double freq, double fs, double duration);

/**
 * @brief 生成连续波(CW)信号
 * @param freq 载波频率(Hz)
 * @param fs 采样率(Hz)
 * @param duration 信号持续时间(s)
 * @param amplitude 信号幅度
 * @param phase 初始相位(弧度)
 * @return 成功返回Signal指针，失败返回NULL
 */
SIGNAL_LIB_API Signal* generate_cw(double freq, double fs, double duration, double amplitude, double phase);

/**
 * @brief 生成线性调频(LFM)信号
 * @param f_start 起始频率(Hz)
 * @param f_end 终止频率(Hz)
 * @param fs 采样率(Hz)
 * @param duration 信号持续时间(s)
 * @return 成功返回Signal指针，失败返回NULL
 */
SIGNAL_LIB_API Signal* generate_lfm(double f_start, double f_end, double fs, double duration);

/**
 * @brief 生成双曲调频(HFM)信号
 * @param f_start 起始频率(Hz)
 * @param f_end 终止频率(Hz)
 * @param fs 采样率(Hz)
 * @param duration 信号持续时间(s)
 * @return 成功返回Signal指针，失败返回NULL
 */
SIGNAL_LIB_API Signal* generate_hfm(double f_start, double f_end, double fs, double duration);

/**
 * @brief 生成相位键控(PSK)信号
 * @param carrier_freq 载波频率(Hz)
 * @param fs 采样率(Hz)
 * @param symbol_count 符号数量
 * @param samples_per_symbol 每个符号的采样点数
 * @param symbols 符号序列数组
 * @return 成功返回相位编码信号，失败返回NULL
 */
SIGNAL_LIB_API Signal* generate_psk(double carrier_freq, double fs, int symbol_count, 
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
SIGNAL_LIB_API Signal* generate_fsk(double* freqs, int freq_count, double fs, 
                    int symbol_count, int samples_per_symbol, const int* symbols);

/**
 * @brief 生成船舶辐射噪声信号
 * @param motion_state 船舶运动状态参数
 * @param fs 采样率(Hz)
 * @param freq_min 最小频率(Hz)
 * @param freq_max 最大频率(Hz)
 * @param source_level_db 声源级(dB re 1μPa@1m)
 * @return 成功返回船舶噪声信号，失败返回NULL
 * @details 基于船舶类型、尺寸、速度等参数仿真辐射噪声谱特性
 */
SIGNAL_LIB_API Signal* generate_ship_noise(const ShipMotionState* motion_state,
                                          double fs,
                                          double freq_min,
                                          double freq_max,
                                          double source_level_db);

/**
 * @brief 生成OFDM(正交频分复用)信号
 * @param params OFDM信号参数
 * @param fs 采样率(Hz)
 * @param duration 信号持续时间(s)
 * @return 成功返回OFDM信号，失败返回NULL
 * @details 实现基本的OFDM调制，包括子载波映射、IFFT变换和循环前缀添加
 */
SIGNAL_LIB_API Signal* generate_ofdm(const OFDMParams* params, double fs, double duration);

/**
 * @brief 生成DSSS(直接序列扩频)信号
 * @param params DSSS信号参数
 * @param fs 采样率(Hz)
 * @param duration 信号持续时间(s)
 * @return 成功返回DSSS信号，失败返回NULL
 * @details 实现基本的直接序列扩频调制，包括数据调制和扩频码调制
 */
SIGNAL_LIB_API Signal* generate_dsss(const DSSSParams* params, double fs, double duration);

/**
 * @brief 生成组合信号(对应MATLAB的gen_signal01函数)
 * @param signal_type 信号类型(对应MATLAB中的signal_kind)
 * @param fs 采样率(Hz)
 * @param duration 信号持续时间(s)
 * @param freq_min 最小频率(Hz)
 * @param freq_max 最大频率(Hz)
 * @return 成功返回组合信号，失败返回NULL
 * @details 根据signal_type生成对应的组合信号，包括基本信号+空白+LFM等组合模式
 */
SIGNAL_LIB_API Signal* generate_composite_signal(CompositeSignalType signal_type,
                                                double fs,
                                                double duration,
                                                double freq_min,
                                                double freq_max);

/**
 * @brief 创建OFDM参数结构体
 * @param carrier_freq 载波频率(Hz)
 * @param bandwidth 信号带宽(Hz)
 * @param num_subcarriers 子载波数量
 * @param cp_ratio 循环前缀比例[0-1]
 * @return 成功返回OFDMParams指针，失败返回NULL
 */
SIGNAL_LIB_API OFDMParams* create_ofdm_params(double carrier_freq, double bandwidth, 
                                              int num_subcarriers, double cp_ratio);

/**
 * @brief 销毁OFDM参数结构体
 * @param params 要销毁的OFDM参数指针
 */
SIGNAL_LIB_API void destroy_ofdm_params(OFDMParams* params);

/**
 * @brief 创建DSSS参数结构体
 * @param carrier_freq 载波频率(Hz)
 * @param bit_rate 比特率(bps)
 * @param chip_rate 码片速率(cps)
 * @param code_length 扩频码长度
 * @return 成功返回DSSSParams指针，失败返回NULL
 */
SIGNAL_LIB_API DSSSParams* create_dsss_params(double carrier_freq, double bit_rate, 
                                              double chip_rate, size_t code_length);

/**
 * @brief 销毁DSSS参数结构体  
 * @param params 要销毁的DSSS参数指针
 */
SIGNAL_LIB_API void destroy_dsss_params(DSSSParams* params);

/**
 * @brief 生成伪随机序列(用于扩频码)
 * @param sequence 输出序列数组
 * @param length 序列长度
 * @param seed 随机种子
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int generate_pn_sequence(int* sequence, size_t length, unsigned int seed);

/**
 * @brief 在信号中添加空白间隔
 * @param signal1 第一个信号
 * @param signal2 第二个信号  
 * @param gap_duration 间隔时间(s)
 * @return 成功返回拼接后的信号，失败返回NULL
 */
SIGNAL_LIB_API Signal* concatenate_signals_with_gap(const Signal* signal1, 
                                                   const Signal* signal2,
                                                   double gap_duration);

/**
 * @brief 对信号进行频谱分析
 * @param sig 输入信号
 * @param spectrum 输出频谱(复数形式)
 * @param spec_length 输出频谱长度
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int spectrum_analysis(const Signal* sig, Complex* spectrum, size_t* spec_length);

/**
 * @brief 匹配滤波器
 * @param input 输入信号
 * @param reference 参考信号
 * @return 成功返回滤波后的Signal指针，失败返回NULL
 */
SIGNAL_LIB_API Signal* matched_filter(const Signal* input, const Signal* reference);

/**
 * @brief 线谱分析
 * @param sig 输入信号
 * @param min_peak_height 最小峰值高度
 * @param min_peak_distance 最小峰值间距
 * @param result 分析结果
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int analyze_line_spectrum(const Signal* sig, double min_peak_height,
                         double min_peak_distance, LineSpectrum* result);

/**
 * @brief 提取信号特征
 * @param sig 输入信号
 * @param features 输出的信号特征结构体
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int extract_features(const Signal* sig, SignalFeatures* features);
#pragma endregion

#pragma region 声速剖面计算与管理
// === 声速剖面计算与管理 ===

/**
 * @brief 创建声速剖面对象
 * @param length 数据点数
 * @return 成功返回SoundProfile指针，失败返回NULL
 */
SIGNAL_LIB_API SoundProfile* create_sound_profile(size_t length);

/**
 * @brief 销毁声速剖面对象，释放内存
 * @param profile 要销毁的声速剖面对象指针
 */
SIGNAL_LIB_API void destroy_sound_profile(SoundProfile* profile);

/**
 * @brief 使用Mackenzie公式(1981)根据温度、盐度和深度计算声速
 * @param temperature 温度(°C)，建议范围: -10.0 ~ 40.0，超范围时自动调整并警告
 * @param salinity 盐度(‰)，建议范围: 0.0 ~ 45.0，超范围时自动调整并警告
 * @param depth 深度(m)，建议范围: 0.0 ~ 12000.0，超范围时自动调整并警告
 * @return 声速(m/s)，始终返回有效值
 * @details 改进的Mackenzie公式，包含完整的压力修正项。采用温和错误处理，超范围参数会被限制到有效范围内。
 */
SIGNAL_LIB_API double calculate_sound_speed_mackenzie(double temperature, double salinity, double depth);

/**
 * @brief 使用Chen-Millero公式(1977)根据温度、盐度和深度计算声速
 * @param temperature 温度(°C)，建议范围: -10.0 ~ 40.0，超范围时自动调整并警告
 * @param salinity 盐度(‰)，建议范围: 0.0 ~ 45.0，超范围时自动调整并警告
 * @param depth 深度(m)，建议范围: 0.0 ~ 12000.0，超范围时自动调整并警告
 * @return 声速(m/s)，始终返回有效值
 * @details 改进的Chen-Millero公式，包含交叉修正项。采用温和错误处理，超范围参数会被限制到有效范围内。
 */
SIGNAL_LIB_API double calculate_sound_speed_chen_millero(double temperature, double salinity, double depth);

/**
 * @brief 使用简化经验公式根据温度、盐度和深度计算声速
 * @param temperature 温度(°C)，建议范围: -10.0 ~ 40.0，超范围时自动调整并警告
 * @param salinity 盐度(‰)，建议范围: 0.0 ~ 45.0，超范围时自动调整并警告
 * @param depth 深度(m)，建议范围: 0.0 ~ 12000.0，超范围时自动调整并警告
 * @return 声速(m/s)，始终返回有效值
 * @details 简化经验公式: c = 1450 + 4.21T - 0.037T2 + 1.14(S-35) + 0.175P。采用温和错误处理，超范围参数会被限制到有效范围内。
 */
SIGNAL_LIB_API double calculate_sound_speed_empirical(double temperature, double salinity, double depth);

/**
 * @brief 使用抛物线模型计算声速剖面(paowu.m的简化模型)
 * @param depth 深度数组(m)
 * @param length 数据点数
 * @param c0 基准声速(m/s)，默认1500m/s
 * @param eps 声速变化系数，默认0.00737
 * @param feature_depth 特征深度(m)，默认1300m
 * @return 成功返回SoundProfile指针，失败返回NULL
 * @details 公式: c = c0 * (1 + eps * (x - 1 + exp(-x))), 其中 x = 2 * (depth - feature_depth) / feature_depth
 */
SIGNAL_LIB_API SoundProfile* generate_svp_parabolic_model(const double* depth, size_t length, 
                                           double c0, double eps, double feature_depth);

/**
 * @brief 根据CTD数据生成声速剖面
 * @param temperatures 温度数组(°C)
 * @param salinities 盐度数组(‰)
 * @param depths 深度数组(m)
 * @param length 数据点数
 * @param method 声速计算方法(0: Mackenzie, 1: Chen-Millero, 2: 简化经验公式)
 * @return 成功返回SoundProfile指针，失败返回NULL
 */
SIGNAL_LIB_API SoundProfile* generate_svp_from_ctd(const double* temperatures, const double* salinities, 
                                    const double* depths, size_t length, int method);

/**
 * @brief 从声速测量数据创建声速剖面
 * @param depths 深度数组(m)
 * @param speeds 声速数组(m/s)
 * @param length 数据点数
 * @return 成功返回SoundProfile指针，失败返回NULL
 */
SIGNAL_LIB_API SoundProfile* create_svp_from_measurements(const double* depths, const double* speeds, size_t length);

/**
 * @brief 融合测量的声速数据和CTD计算的声速数据
 * @param measured_profile 测量的声速剖面
 * @param calculated_profile 从CTD数据计算的声速剖面
 * @param weight_measured 测量数据的权重(0.0-1.0)
 * @return 成功返回融合后的SoundProfile指针，失败返回NULL
 */
SIGNAL_LIB_API SoundProfile* fuse_sound_profiles(const SoundProfile* measured_profile, 
                                 const SoundProfile* calculated_profile,
                                 double weight_measured);

/**
 * @brief 在指定深度处插值计算声速值
 * @param profile 声速剖面
 * @param target_depth 目标深度(m)
 * @return 插值计算的声速值(m/s)
 */
SIGNAL_LIB_API double interpolate_sound_speed(const SoundProfile* profile, double target_depth);

/**
 * @brief 检查声速剖面数据的质量
 * @param profile 声速剖面
 * @return 成功返回0，检测到异常返回负值
 */
SIGNAL_LIB_API int check_sound_profile_quality(const SoundProfile* profile);
#pragma endregion

#pragma region 声线追踪与声场计算
// === 声线追踪与声场计算 ===
// 注意：声线追踪与声场计算函数已移至后面API接口部分统一声明
#pragma endregion

#pragma region 增强数学运算接口
// === 增强数学运算接口（使用Eigen等第三方库） ===

/**
 * @brief 使用Eigen库进行高精度复数运算
 * @details 替代原生C++复数运算，提供更好的数值稳定性
 */

/**
 * @brief 创建Eigen兼容的复数
 * @param real 实部
 * @param imag 虚部
 * @return Complex结构体
 */
SIGNAL_LIB_API Complex create_eigen_complex(double real, double imag);

/**
 * @brief 使用Eigen进行高精度复数乘法
 * @param a 第一个复数
 * @param b 第二个复数
 * @return 乘法结果
 */
SIGNAL_LIB_API Complex eigen_complex_multiply(Complex a, Complex b);

/**
 * @brief 使用Eigen进行高精度复数除法
 * @param a 被除数
 * @param b 除数
 * @return 除法结果，自动处理数值稳定性
 */
SIGNAL_LIB_API Complex eigen_complex_divide(Complex a, Complex b);

/**
 * @brief 使用Eigen计算复数的模（避免溢出/下溢）
 * @param z 输入复数
 * @return 复数的模，使用稳定算法
 */
SIGNAL_LIB_API double eigen_complex_abs(Complex z);

/**
 * @brief 使用FFTW进行高效FFT变换（如果可用）
 * @param data 输入/输出复数数组
 * @param n 数据长度
 * @param forward 1表示正向FFT，0表示反向FFT
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int enhanced_fft(Complex* data, size_t n, int forward);

/**
 * @brief 使用Eigen进行矩阵LU分解
 * @param matrix 输入矩阵（行优先存储）
 * @param rows 矩阵行数
 * @param cols 矩阵列数
 * @param L 输出下三角矩阵
 * @param U 输出上三角矩阵
 * @param P 输出置换矩阵
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int eigen_lu_decomposition(const double* matrix, size_t rows, size_t cols,
                                         double** L, double** U, int** P);

/**
 * @brief 使用Eigen求解线性方程组
 * @param A 系数矩阵
 * @param b 右端向量
 * @param x 解向量
 * @param n 矩阵大小
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int eigen_solve_linear_system(const double* A, const double* b, double* x, size_t n);

/**
 * @brief 使用Eigen进行复数矩阵求解（用于抛物方程）
 * @param A 复数系数矩阵
 * @param b 复数右端向量
 * @param x 复数解向量
 * @param n 矩阵大小
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int eigen_solve_complex_system(const Complex* A, const Complex* b, Complex* x, size_t n);

/**
 * @brief 使用Eigen进行数值积分（自适应Simpson法则）
 * @param func 被积函数指针
 * @param a 积分下限
 * @param b 积分上限
 * @param tolerance 精度要求
 * @param result 积分结果
 * @return 成功返回0，失败返回-1
 */
typedef double (*IntegrandFunction)(double x, void* params);
SIGNAL_LIB_API int adaptive_integration(IntegrandFunction func, double a, double b, 
                                       double tolerance, void* params, double* result);

/**
 * @brief 使用Eigen进行特征值分解
 * @param matrix 输入对称矩阵
 * @param n 矩阵大小
 * @param eigenvalues 输出特征值
 * @param eigenvectors 输出特征向量
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int eigen_eigenvalue_decomposition(const double* matrix, size_t n,
                                                 double* eigenvalues, double* eigenvectors);

/**
 * @brief 使用高精度库进行Bessel函数计算
 * @param order 阶数
 * @param x 自变量
 * @return Bessel函数值
 */
SIGNAL_LIB_API double enhanced_bessel_j(int order, double x);

/**
 * @brief 使用高精度库进行Hankel函数计算
 * @param order 阶数
 * @param x 自变量
 * @return Hankel函数值（复数）
 */
SIGNAL_LIB_API Complex enhanced_hankel_h1(int order, double x);

/**
 * @brief 使用稳定算法进行多项式求根
 * @param coefficients 多项式系数（从高次到低次）
 * @param degree 多项式次数
 * @param roots 输出根（复数数组）
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int stable_polynomial_roots(const double* coefficients, int degree, Complex* roots);

/**
 * @brief 使用高精度算法进行样条插值
 * @param x 已知点x坐标
 * @param y 已知点y坐标
 * @param n 已知点数量
 * @param xi 插值点x坐标
 * @param yi 输出插值结果
 * @param ni 插值点数量
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int cubic_spline_interpolation(const double* x, const double* y, size_t n,
                                             const double* xi, double* yi, size_t ni);

/**
 * @brief 检查并设置数学库优化选项
 * @param use_eigen 是否使用Eigen优化
 * @param use_fftw 是否使用FFTW（如果可用）
 * @param use_lapack 是否使用LAPACK（如果可用）
 * @param num_threads 并行线程数
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int configure_math_libraries(int use_eigen, int use_fftw, int use_lapack, int num_threads);

/**
 * @brief 获取当前数学库配置信息
 * @param info 输出配置信息字符串
 * @param max_length 字符串最大长度
 * @return 成功返回0，失败返回-1
 */
SIGNAL_LIB_API int get_math_library_info(char* info, size_t max_length);
#pragma endregion

// === 动态库导出缺失函数声明 ===
SIGNAL_LIB_API void generate_chirp_signal(double* signal, int length, double fs, double f0, double f1, double duration);
SIGNAL_LIB_API int parabolic_equation_solver(double* field, int nx, int nz, double dx, double dz, double freq, double* sound_speed, void* source);
SIGNAL_LIB_API int parabolic_equation_solver_detailed(double* field, int nx, int nz, double dx, double dz, double freq, double* sound_speed, void* source, int option);

#ifdef __cplusplus
}
#endif

#endif // SIGNAL_LIB_HPP 

