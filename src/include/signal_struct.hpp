#pragma once

#pragma region 基本定义和常量
// === 基本定义和常量 ===
#define PI 3.14159265358979323846
#define MAX_SIGNAL_LENGTH 1000000
#pragma endregion

// 主动声纳输入参数结构体
typedef struct {
    double SL; // 声呐的声源级 (dB)
    double TL; // 单向传输损失 (dB)
    double TS; // 目标强度 (dB)
    double NL; // 噪声级 (dB)
    double DI; // 指向性指数 (dB)
    double threshold; // 设备的检测阈值 (dB)
} ActiveSonarInputParams;

// 被动声纳输入参数结构体
typedef struct {
    double SL; // 声呐的声源级 (dB)
    double TL; // 单向传输损失 (dB)
    double NL; // 噪声级 (dB)
    double DI; // 指向性指数 (dB)
} PassiveSonarInputParams;

// 损失计算的参数类型 
enum TLCalType : int
{
    //球面扩展衰减
    TL_CAL_TYPE_SPHERE = 0,
    //柱面扩展衰减
    TL_CAL_TYPE_CYLINDER = 1,
    //声场传播模型
    TL_CAL_TYPE_MODEL = 2
};

// 指向性指数参数
struct DIParams
{
    int array_num; // 阵元个数
    double array_spacing; // 阵元间距
};

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
 * @brief 信号特征结构
 * @details 存储信号的各种域特征，用于信号识别和分类
 */
typedef struct {
    double time_domain_features[5];   // 时域特征：均值、方差、偏度、峰度、过零率
    double freq_domain_features[3];   // 频域特征：中心频率、带宽、频谱熵
    double time_freq_features[10];    // 时频特征：能量分布、调制特征等
} SignalFeatures;
#pragma endregion

#pragma region 水声环境数据结构
// === 水声环境数据结构 ===

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
 * @brief 抛物方程计算参数结构
 * @details 包含抛物方程声场计算所需的所有参数
 */
typedef struct {
    const SoundProfile* profile;          // 声速剖面
    double frequency;                      // 频率(Hz)
    double source_depth_m;                 // 声源深度(m)
    size_t range_points;                   // 距离采样点数
    size_t depth_points;                   // 深度采样点数
    double max_range_m;                    // 最大距离(m)
    double total_depth_m;                  // 总深度(m)
    double range_step_m;                   // 距离步长(m)，0表示自动计算
    double gaussian_source_width_factor;   // 高斯源宽度因子，0表示使用默认值
    double bottom_attenuation_dB_lambda;   // 海底衰减(dB/波长)
} parabolic_equation_params_t;

/**
 * @brief 声场计算参数结构体
 * @details 包含PE声场计算所需的全部参数，支持灵活配置
 */
typedef struct {
    const SoundProfile* sound_profile;    // 声速剖面数据
    double frequency_hz;                  // 声源频率(Hz)
    double source_depth_m;                // 声源深度(m)
    double max_range_m;                   // 最大计算距离(m)
    double max_depth_m;                   // 最大计算深度(m)
    size_t range_points;                  // 距离方向网格点数
    size_t depth_points;                  // 深度方向网格点数
    double range_step_m;                  // 距离步长(m)，0表示自动计算
    double depth_step_m;                  // 深度步长(m)，0表示自动计算
    double gaussian_beam_width_factor;    // 高斯源束宽因子，默认10.0
    double reference_sound_speed_ms;      // 参考声速(m/s)，默认1500.0
    int boundary_condition_type;          // 边界条件类型：0=自由表面+刚性底面，1=自由表面+吸收底面
    double bottom_attenuation_db_lambda;  // 海底衰减(dB/波长)，默认0.5
} SoundFieldParams;

/**
 * @brief 声场计算结果结构体
 * @details 存储计算得到的传输损失矩阵和相关信息
 */
typedef struct {
    double** transmission_loss_db;        // 传输损失矩阵(dB)，[depth_index][range_index]
    Complex** complex_field;              // 复数声场矩阵，[depth_index][range_index]
    size_t range_points;                  // 距离方向点数
    size_t depth_points;                  // 深度方向点数
    double range_step_m;                  // 实际距离步长(m)
    double depth_step_m;                  // 实际深度步长(m)
    double max_range_m;                   // 最大距离(m)
    double max_depth_m;                   // 最大深度(m)
    double computation_time_seconds;      // 计算耗时(秒)
    int computation_status;               // 计算状态：0=成功，负值=错误代码
} SoundFieldResult;
#pragma endregion

#pragma region 环境噪声与混响模型
// === 环境噪声与混响模型 ===

/**
 * @brief 环境噪声模型参数结构体
 */
typedef struct {
    double wind_speed;        // 风速(m/s)
    double shipping_factor;   // 航运活动因子[0-1]
    double bio_noise_level;   // 生物噪声级(dB)
    double thermal_noise_ref; // 热噪声参考级(dB)
} NoiseModelParams;

/**
 * @brief 环境噪声模型结构体
 */
typedef struct {
    NoiseModelParams params;
    double frequency_range[2]; // 频率范围[min_freq, max_freq] Hz
    int model_type;           // 噪声模型类型：0=Wenz模型, 1=简化模型
} NoiseModel;

/**
 * @brief 混响类型枚举
 */
typedef enum {
    SURFACE_REVERB = 0,  // 海面混响
    BOTTOM_REVERB = 1,   // 海底混响
    VOLUME_REVERB = 2    // 体积混响
} ReverbType;

/**
 * @brief 标准声速剖面类型枚举
 */
typedef enum {
    SURFACE_DUCT = 0,        // 表面波导
    DEEP_SOUND_CHANNEL = 1,  // 深海声道
    ARCTIC_PROFILE = 2,      // 北极剖面
    TROPICAL_PROFILE = 3,    // 热带剖面
    USER_DEFINED = 99        // 用户自定义
} SoundProfileType;
#pragma endregion

#pragma region 目标特性与对抗装备模型
// === 目标特性与对抗装备模型 ===

/**
 * @brief 目标声学特性模型结构体
 */
typedef struct {
    double length;              // 目标长度(m)
    double width;               // 目标宽度(m)
    double draft;               // 吃水深度(m)
    double target_strength_ref; // 参考目标强度(dB)，通常在正横方向
    char target_type[32];       // 目标类型："SUBMARINE", "SURFACE_SHIP", "UUV", "TORPEDO"
    double* reflection_coeffs;  // 不同角度的反射系数
    double* aspect_angles;      // 对应的方位角(度)
    size_t angle_count;         // 角度数据点数
} TargetModel;

/**
 * @brief 对抗装备参数结构体
 */
typedef struct {
    double source_level;        // 声源级(dB re 1μPa@1m)
    double bandwidth;           // 带宽(Hz)
    double duration;            // 持续时间(s)
    char jamming_type[20];      // 干扰类型："NOISE", "DECEPTION", "BARRAGE"
    double center_frequency;    // 中心频率(Hz)
    double modulation_index;    // 调制指数
} CountermeasureParams;

/**
 * @brief 船舶运动状态参数结构体
 * @details 用于船舶噪声仿真的运动状态描述
 */
typedef struct {
    int type;                   // 目标类型：0=水面舰船, 1=潜艇
    double duration;            // 持续时间(s)
    double velocity;            // 速度(m/s)
    double depth;               // 深度(m)
    double length;              // 舰船长度(m)
    double displacement;        // 排水量(吨)
    double engine_power;        // 发动机功率(kW)
} ShipMotionState;

/**
 * @brief OFDM信号参数结构体
 * @details 正交频分复用信号生成参数
 */
typedef struct {
    double carrier_freq;        // 载波频率(Hz)
    double bandwidth;           // 信号带宽(Hz)
    int num_subcarriers;        // 子载波数量
    double symbol_duration;     // 符号持续时间(s)
    double cp_ratio;            // 循环前缀比例[0-1]
    int* data_bits;            // 数据比特序列
    size_t data_length;         // 数据长度
} OFDMParams;

/**
 * @brief DSSS信号参数结构体  
 * @details 直接序列扩频信号生成参数
 */
typedef struct {
    double carrier_freq;        // 载波频率(Hz)
    double bit_rate;           // 比特率(bps)
    double chip_rate;          // 码片速率(cps)
    int* spreading_code;       // 扩频码序列
    size_t code_length;        // 扩频码长度
    int* data_bits;           // 数据比特序列
    size_t data_length;        // 数据长度
} DSSSParams;

/**
 * @brief 组合信号类型枚举
 * @details 定义复合信号的类型，对应MATLAB中的signal_kind
 */
typedef enum {
    SIGNAL_SHIP_NOISE = 0,      // 船舶辐射噪声
    SIGNAL_CW = 1,              // 连续波
    SIGNAL_LFM = 2,             // 线性调频
    SIGNAL_HFM = 3,             // 双曲调频
    SIGNAL_4FSK_COMPOSITE = 4,   // 4FSK组合信号(包含CW+LFM+FSK+LFM)
    SIGNAL_4PSK_COMPOSITE = 5,   // 4PSK组合信号(包含CW+LFM+PSK+LFM)
    SIGNAL_OFDM_COMPOSITE = 6,   // OFDM组合信号(包含CW+LFM+OFDM+LFM)
    SIGNAL_DSSS_COMPOSITE = 7    // DSSS组合信号(包含CW+LFM+DSSS+LFM)
} CompositeSignalType;