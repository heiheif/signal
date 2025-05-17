#include "../include/signal_lib.h"

// 声线追踪计算函数
int ray_direction_calc(const SoundProfile* profile, double source_depth,
                      double* angles, size_t angle_count,
                      double** ray_paths, size_t* path_lengths) {
    if (!profile || !angles || !ray_paths || !path_lengths ||
        angle_count == 0 || source_depth < 0) {
        return -1;
    }

    // 设置计算参数
    double dr = 100.0;  // 水平距离步长(m)
    double max_range = 10000.0;  // 最大传播距离(m)
    size_t max_points = (size_t)(max_range / dr) + 1;

    // 为每条声线分配内存
    for (size_t i = 0; i < angle_count; i++) {
        ray_paths[i] = (double*)malloc(2 * max_points * sizeof(double));  // [x,z]坐标对
        if (!ray_paths[i]) {
            // 内存分配失败，释放已分配的内存
            for (size_t j = 0; j < i; j++) {
                free(ray_paths[j]);
            }
            return -1;
        }
        path_lengths[i] = 0;
    }

    // 对每个发射角度进行声线追踪
    for (size_t i = 0; i < angle_count; i++) {
        double theta = angles[i] * PI / 180.0;  // 角度转弧度
        double sin_theta = sin(theta);
        
        // 初始化声线位置
        double x = 0.0;
        double z = source_depth;
        size_t point_count = 0;

        // 记录初始点
        ray_paths[i][2*point_count] = x;
        ray_paths[i][2*point_count + 1] = z;
        point_count++;

        // 声线追踪主循环
        while (x < max_range && z >= 0.0 && z <= profile->depth[profile->length-1] && 
               point_count < max_points) {
            // 在声速剖面中查找当前深度对应的声速
            size_t idx = 0;
            while (idx < profile->length - 1 && z > profile->depth[idx + 1]) {
                idx++;
            }
            double c = profile->speed[idx];
            
            // 计算下一个点的位置
            x += dr;
            // 计算声速梯度并更新声线角度
            double dc_dz = 0.0;
            if (idx < profile->length - 1) {
                dc_dz = (profile->speed[idx+1] - c) / (profile->depth[idx+1] - profile->depth[idx]);
            }
            sin_theta -= (dr * dc_dz) / c;  // 声线方程的数值解
            z += dr * sin_theta;
            
            // 记录当前点
            ray_paths[i][2*point_count] = x;
            ray_paths[i][2*point_count + 1] = z;
            point_count++;
        }

        path_lengths[i] = point_count;
    }

    return 0;
} 

