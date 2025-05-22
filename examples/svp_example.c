#include "../src/include/signal_lib.h"
#include <stdio.h>

/**
 * 声速剖面计算示例程序
 * 展示如何使用声速计算、声速剖面生成和管理相关函数
 */
int main() {
    printf("============ 声速剖面(SVP)计算示例 ============\n\n");
    
    // 1. 使用Mackenzie公式计算单点声速
    printf("1. 使用Mackenzie公式计算单点声速:\n");
    double temperature = 15.0;    // 温度15°C
    double salinity = 35.0;       // 盐度35‰
    double depth = 100.0;         // 深度100米
    double speed = calculate_sound_speed_mackenzie(temperature, salinity, depth);
    printf("   温度: %.1f°C, 盐度: %.1f‰, 深度: %.1f m\n", temperature, salinity, depth);
    printf("   计算得到声速: %.2f m/s\n\n", speed);
    
    // 2. 使用Chen-Millero公式计算单点声速
    printf("2. 使用Chen-Millero公式计算单点声速:\n");
    speed = calculate_sound_speed_chen_millero(temperature, salinity, depth);
    printf("   温度: %.1f°C, 盐度: %.1f‰, 深度: %.1f m\n", temperature, salinity, depth);
    printf("   计算得到声速: %.2f m/s\n\n", speed);
    
    // 3. 从CTD数据生成声速剖面
    printf("3. 从CTD数据生成声速剖面:\n");
    
    // 定义CTD数据
    #define DATA_POINTS 5
    double temperatures[DATA_POINTS] = {20.0, 18.0, 15.0, 10.0, 5.0};
    double salinities[DATA_POINTS] = {34.5, 34.8, 35.0, 35.2, 35.3};
    double depths[DATA_POINTS] = {0.0, 50.0, 100.0, 200.0, 500.0};
    
    // 使用Mackenzie公式生成剖面
    SoundProfile* profile_mackenzie = generate_svp_from_ctd(
        temperatures, salinities, depths, DATA_POINTS, 0);
    
    // 使用Chen-Millero公式生成剖面
    SoundProfile* profile_chen = generate_svp_from_ctd(
        temperatures, salinities, depths, DATA_POINTS, 1);
    
    if (profile_mackenzie && profile_chen) {
        printf("   深度(m)  |  Mackenzie声速(m/s)  |  Chen-Millero声速(m/s)\n");
        printf("   --------------------------------------------------------\n");
        for (int i = 0; i < DATA_POINTS; i++) {
            printf("   %7.1f  |       %7.2f       |       %7.2f\n", 
                profile_mackenzie->depth[i],
                profile_mackenzie->speed[i],
                profile_chen->speed[i]);
        }
        printf("\n");
    } else {
        printf("   生成声速剖面失败!\n\n");
    }
    
    // 4. 从直接测量数据创建声速剖面
    printf("4. 模拟从声速仪直接测量数据创建声速剖面:\n");
    double measured_depths[DATA_POINTS] = {0.0, 50.0, 100.0, 200.0, 500.0};
    double measured_speeds[DATA_POINTS] = {1538.0, 1533.5, 1521.2, 1498.6, 1487.3};
    
    SoundProfile* measured_profile = create_svp_from_measurements(
        measured_depths, measured_speeds, DATA_POINTS);
    
    if (measured_profile) {
        printf("   深度(m)  |  测量声速(m/s)\n");
        printf("   ------------------------\n");
        for (int i = 0; i < DATA_POINTS; i++) {
            printf("   %7.1f  |    %7.2f\n", 
                measured_profile->depth[i],
                measured_profile->speed[i]);
        }
        printf("\n");
    } else {
        printf("   创建测量声速剖面失败!\n\n");
    }
    
    // 5. 融合测量数据和计算数据
    printf("5. 融合测量数据和计算数据(权重比例0.7:0.3):\n");
    SoundProfile* fused_profile = fuse_sound_profiles(
        measured_profile, profile_mackenzie, 0.7);
    
    if (fused_profile) {
        printf("   深度(m)  |  测量声速(m/s)  |  计算声速(m/s)  |  融合声速(m/s)\n");
        printf("   ----------------------------------------------------------------\n");
        for (int i = 0; i < DATA_POINTS; i++) {
            printf("   %7.1f  |    %7.2f     |    %7.2f     |    %7.2f\n", 
                fused_profile->depth[i],
                measured_profile->speed[i],
                profile_mackenzie->speed[i],
                fused_profile->speed[i]);
        }
        printf("\n");
    } else {
        printf("   融合声速剖面失败!\n\n");
    }
    
    // 6. 插值计算中间深度的声速
    printf("6. 插值计算中间深度的声速:\n");
    double target_depth = 75.0;  // 75米深度
    double interp_speed = interpolate_sound_speed(measured_profile, target_depth);
    printf("   在深度%.1f米处，插值计算的声速为: %.2f m/s\n\n", target_depth, interp_speed);
    
    // 7. 质量检查
    printf("7. 声速剖面数据质量检查:\n");
    int quality = check_sound_profile_quality(measured_profile);
    printf("   测量声速剖面质量检查结果: %d (%s)\n", 
        quality, quality == 0 ? "通过" : "存在问题");
    
    quality = check_sound_profile_quality(profile_mackenzie);
    printf("   Mackenzie计算声速剖面质量检查结果: %d (%s)\n\n", 
        quality, quality == 0 ? "通过" : "存在问题");
    
    // 释放资源
    destroy_sound_profile(profile_mackenzie);
    destroy_sound_profile(profile_chen);
    destroy_sound_profile(measured_profile);
    destroy_sound_profile(fused_profile);
    
    printf("============ 示例程序结束 ============\n");
    
    return 0;
} 