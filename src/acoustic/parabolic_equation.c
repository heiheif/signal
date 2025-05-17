#include "../include/signal_lib.h"

// 抛物方程声场计算函数
int parabolic_equation_field(const SoundProfile* profile, double freq,
                           double source_depth, Complex** field,
                           size_t range_points, size_t depth_points) {
    if (!profile || !field || range_points == 0 || depth_points == 0 ||
        freq <= 0 || source_depth < 0) {
        return -1;
    }

    // 初始化计算参数
    double k0 = 2.0 * PI * freq / profile->speed[0];  // 参考波数
    double dr = 100.0;  // 水平距离步长(m)
    double dz = profile->depth[profile->length-1] / (depth_points - 1);  // 深度步长(m)

    // 分配声场数组内存
    Complex* current_field = (Complex*)malloc(depth_points * sizeof(Complex));
    Complex* next_field = (Complex*)malloc(depth_points * sizeof(Complex));
    if (!current_field || !next_field) {
        free(current_field);
        free(next_field);
        return -1;
    }

    // 初始化声源场（高斯分布）
    for (size_t i = 0; i < depth_points; i++) {
        double z = i * dz;
        double gaussian = exp(-pow(z - source_depth, 2) / (2.0 * pow(10.0, 2)));
        current_field[i].real = gaussian;
        current_field[i].imag = 0.0;
    }

    // 保存初始场
    memcpy(field[0], current_field, depth_points * sizeof(Complex));

    // Crank-Nicolson差分格式求解
    for (size_t r = 1; r < range_points; r++) {
        // 分配三对角矩阵系数内存
        Complex* a = (Complex*)malloc(depth_points * sizeof(Complex));
        Complex* b = (Complex*)malloc(depth_points * sizeof(Complex));
        Complex* c = (Complex*)malloc(depth_points * sizeof(Complex));
        Complex* d = (Complex*)malloc(depth_points * sizeof(Complex));
        
        if (!a || !b || !c || !d) {
            free(current_field);
            free(next_field);
            free(a);
            free(b);
            free(c);
            free(d);
            return -1;
        }

        // 计算差分方程系数
        double alpha = dr / (2.0 * k0 * dz * dz);
        for (size_t i = 0; i < depth_points; i++) {
            // 在声速剖面中查找当前深度对应的声速
            double z = i * dz;
            size_t idx = 0;
            while (idx < profile->length - 1 && z > profile->depth[idx + 1]) {
                idx++;
            }
            double sound_speed = profile->speed[idx];  // 当前深度的声速
            double n = profile->speed[0] / sound_speed;

            // 构建三对角矩阵系数
            a[i].real = -alpha;
            a[i].imag = 0.0;
            b[i].real = 1.0 + 2.0 * alpha - k0 * k0 * dr * (n * n - 1.0) / 4.0;
            b[i].imag = 0.0;
            c[i].real = -alpha;
            c[i].imag = 0.0;
            d[i].real = current_field[i].real;  // 右侧项
            d[i].imag = current_field[i].imag;
        }

        // 求解三对角矩阵方程
        // Thomas算法前向消元
        for (size_t i = 1; i < depth_points; i++) {
            Complex temp1 = complex_divide(a[i], b[i-1]);
            Complex temp2 = complex_multiply(temp1, c[i-1]);
            b[i] = complex_subtract(b[i], temp2);
            
            temp1 = complex_divide(a[i], b[i-1]);
            temp2 = complex_multiply(temp1, d[i-1]);
            d[i] = complex_subtract(d[i], temp2);
        }

        // 回代求解
        next_field[depth_points-1] = complex_divide(d[depth_points-1], b[depth_points-1]);
        for (size_t i = depth_points - 1; i > 0; i--) {
            Complex temp = complex_multiply(c[i-1], next_field[i]);
            temp = complex_subtract(d[i-1], temp);
            next_field[i-1] = complex_divide(temp, b[i-1]);
        }

        // 保存当前步结果
        memcpy(field[r], next_field, depth_points * sizeof(Complex));
        memcpy(current_field, next_field, depth_points * sizeof(Complex));

        // 释放临时内存
        free(a);
        free(b);
        free(c);
        free(d);
    }

    // 释放内存
    free(current_field);
    free(next_field);
    return 0;
} 

