#include "signal_lib.hpp"


bool passiveSonarDetection(double SL, double TL, double NL, double DI, double threshold) {
    double DT = SL - TL - NL + DI;  // 计算检测阈值（接收信噪比）
    return DT >= threshold;         // 与设备阈值比较
}

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
bool activeSonarDetectionMonostatic(double SL, double TL, double TS, double NL, double DI, double threshold) {
    double DT = SL - 2 * TL + TS - NL + DI;  // 计算检测阈值（接收信噪比）
    return DT >= threshold;                   // 与设备阈值比较
}

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
bool activeSonarDetectionBistatic(double SL, double TL1, double TL2, double TS, double NL, double DI, double threshold) {
    double DT = SL - TL1 - TL2 + TS - NL + DI;  // 计算检测阈值（接收信噪比）
    return DT >= threshold;                      // 与设备阈值比较
}



SIGNAL_LIB_API double calculateTL(TLCalType cal_type, double R) {
    if (R <= 0) {
        return 0.0;  // 距离无效时返回0
    }
    
    double TL = 0.0;
    
    switch (cal_type) {
        case TL_CAL_TYPE_SPHERE:
            // 球面扩展衰减: TL = 20*lg(R)
            TL = 20.0 * log10(R);
            break;
            
        case TL_CAL_TYPE_CYLINDER:
            // 柱面扩展衰减: TL = 10*lg(R)
            TL = 10.0 * log10(R);
            break;
            
        case TL_CAL_TYPE_MODEL:
            TL = -1; // 不支持声场模型在此处计算
            break;
            
        default:
            // 默认使用球面扩展
            TL = 20.0 * log10(R);
            break;
    }
    
    return TL;
}

SIGNAL_LIB_API double calculateTL(double TL_coefficient, double air_absorption_coefficient, double R) {
    return TL_coefficient * log10(R) + air_absorption_coefficient * (R / 1000);
}


SIGNAL_LIB_API double calculateDI(DIParams* params) {
    if (!params || params->array_num <= 0) {
        return 0.0;  // 参数无效时返回0
    }
    
    double DI = 0.0;
    int n = params->array_num;
    
    // 基础指向性指数: DI ≈ 10*lg(n)
    DI = 10.0 * log10((double)n);
    
    // 如果提供了阵元间距，可以进行更精确的计算
    if (params->array_spacing > 0) {
        // 考虑阵元间距的影响
        // 对于均匀线阵，更精确的DI计算为:
        // DI = 10*lg(n) + 10*lg(2*d/lambda)
        // 其中lambda为波长，d为阵元间距
        
        // 假设工作频率为典型值10kHz，声速1500m/s
        double freq = 10000.0;  // Hz
        double c = 1500.0;      // m/s
        double lambda = c / freq;  // 波长
        
        if (lambda > 0) {
            double spacing_factor = 2.0 * params->array_spacing / lambda;
            if (spacing_factor > 1.0) {
                DI += 10.0 * log10(spacing_factor);
            }
        }
    }
    
    return DI;
}