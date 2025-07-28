#include "../include/signal_lib.hpp"

// 辅助函数：构造复数
static Complex make_complex(double real, double imag) {
    Complex c;
    c.real = real;
    c.imag = imag;
    return c;
}

// 抛物方程声场计算函数（标准PE算法核心）
SIGNAL_LIB_API int parabolic_equation_field(const SoundProfile* profile, double freq,
                            double source_depth, Complex** field,
                            size_t range_points, size_t depth_points) {
    // 参数校验
    if (!profile || !field || range_points == 0 || depth_points < 2 || freq <= 0) {
        printf("ERROR: Invalid parameters for parabolic equation\n");
        return -1;
    }
    
    printf("=== PARABOLIC EQUATION COMPUTATION ===\n");
    printf("Range points: %zu, Depth points: %zu\n", range_points, depth_points);
    printf("Frequency: %.1f Hz, Source depth: %.1f m\n", freq, source_depth);
    
    // 计算常量
    double c0 = profile->speed[0];  // 参考声速
    double omega = 2.0 * PI * freq;
    double k0 = omega / c0;
    double total_depth = profile->depth[profile->length - 1];
    double h = total_depth / (depth_points - 1);  // 深度步长
    double h2 = h * h;
    
    // 设置距离步长 (改进版：基于波长自适应选择)
    double lambda = 2.0 * PI / k0;  // 波长
    double deltar_optimal = lambda / 15.0;  // 步长应小于波长的1/15，确保数值稳定性
    double rmax = (range_points - 1) * deltar_optimal;  // 根据最优步长计算总距离
    double deltar = rmax / (range_points - 1);
    
    printf("Computational grid: h=%.2f m, deltar=%.2f m (λ/%.1f)\n", h, deltar, lambda/deltar);
    printf("k0=%.6f, total_depth=%.1f m, wavelength=%.2f m\n", k0, total_depth, lambda);
    
    // 分配网格数组
    double* z = (double*)malloc(depth_points * sizeof(double));
    double* c = (double*)malloc(depth_points * sizeof(double));
    double* n = (double*)malloc(depth_points * sizeof(double));
    if (!z || !c || !n) {
        printf("ERROR: Memory allocation failed for grids\n");
        free(z); free(c); free(n);
        return -1;
    }
    
    // 初始化深度、声速、折射率
    for (size_t i = 0; i < depth_points; i++) {
        z[i] = i * h;
        
        // 分段线性插值声速
        c[i] = c0;  // 默认值
        for (size_t j = 0; j < profile->length - 1; j++) {
            if (z[i] >= profile->depth[j] && z[i] <= profile->depth[j+1]) {
                double ratio = (z[i] - profile->depth[j]) / 
                              (profile->depth[j+1] - profile->depth[j]);
                c[i] = profile->speed[j] + ratio * (profile->speed[j+1] - profile->speed[j]);
                break;
            }
        }
        n[i] = c0 / c[i];  // 折射率
    }
    
    printf("Sound speed range: [%.1f, %.1f] m/s\n", c[0], c[depth_points-1]);
    
    // 分配工作数组
    Complex* psi_current = (Complex*)calloc(depth_points, sizeof(Complex));
    Complex* psi_next = (Complex*)calloc(depth_points, sizeof(Complex));
    Complex* rhs = (Complex*)calloc(depth_points, sizeof(Complex));
    
    // 三对角矩阵系数（简化LU分解版本）
    Complex* a = (Complex*)malloc(depth_points * sizeof(Complex));  // 下对角
    Complex* b = (Complex*)malloc(depth_points * sizeof(Complex));  // 对角
    Complex* c_coeff = (Complex*)malloc(depth_points * sizeof(Complex));  // 上对角
    Complex* d = (Complex*)malloc(depth_points * sizeof(Complex));  // 右端项
    
    if (!psi_current || !psi_next || !rhs || !a || !b || !c_coeff || !d) {
        printf("ERROR: Memory allocation failed for computation arrays\n");
        goto cleanup;
    }
    
    // 高斯源初始化 (统一公式版本)
    printf("Initializing Gaussian source...\n");
    double fac = 10.0;  // 束宽因子，使源更集中 (与MATLAB版本一致)
    double gaussian_width_factor = sqrt(0.5) * fac / k0;  // 统一的高斯宽度因子
    
    for (size_t i = 0; i < depth_points; i++) {
        double depth_diff = z[i] - source_depth;
        // 使用统一的高斯公式，确保一致性
        double gaussian = exp(-0.5 * (depth_diff / gaussian_width_factor) * (depth_diff / gaussian_width_factor));
        psi_current[i].real = gaussian;
        psi_current[i].imag = 0.0;
    }
    
    // 拷贝初始声场到输出
    memcpy(field[0], psi_current, depth_points * sizeof(Complex));
    
    printf("Starting range marching...\n");
    
    // 距离步进主循环
    for (size_t ir = 1; ir < range_points; ir++) {
        if (ir % 100 == 0) {
            printf("Range step: %zu/%zu\n", ir, range_points-1);
        }
        
        // 构造Crank-Nicolson格式系数矩阵
        // 方程 SPE: 2ik_0 ∂ψ/∂r + ∂²ψ/∂z² + k0²(n²-1)ψ = 0
        
        Complex alpha = make_complex(0.0, 2.0 * k0 / deltar);  // 2ik0/Δr
        
        for (size_t i = 0; i < depth_points; i++) {
            double n2_minus_1 = n[i] * n[i] - 1.0;
            // 系数矩阵 C（三对角形式）
            if (i == 0) {
                // 表面边界（Neumann边界：dψ/dz = 0，等价于[1]=[0]）
                a[i] = make_complex(0.0, 0.0);
                b[i] = make_complex(2.0/h2 + k0*k0*n2_minus_1/2.0, alpha.imag);
                c_coeff[i] = make_complex(-2.0/h2, 0.0); // 表面边界2倍系数
            } else if (i == depth_points - 1) {
                // 底面边界（刚性）
                a[i] = make_complex(-1.0/(2.0*h2), 0.0);
                b[i] = make_complex(2.0/h2 + k0*k0*n2_minus_1/2.0, alpha.imag);
                c_coeff[i] = make_complex(0.0, 0.0);
            } else {
                // 内部点
                a[i] = make_complex(-1.0/(2.0*h2), 0.0);
                b[i] = make_complex(2.0/h2 + k0*k0*n2_minus_1/2.0, alpha.imag);
                c_coeff[i] = make_complex(-1.0/(2.0*h2), 0.0);
            }
        }
        
        // 构造右端项（显式部分）
        for (size_t i = 0; i < depth_points; i++) {
            if (i == 0) {
                // 计算右端项 (显式部分)
                double n2_minus_1 = n[i] * n[i] - 1.0;
                Complex term1 = complex_multiply(make_complex(-2.0/h2 - k0*k0*n2_minus_1/2.0, -alpha.imag), psi_current[i]);
                Complex term2 = complex_multiply(make_complex(2.0/h2, 0.0), psi_current[i+1]);
                rhs[i] = complex_add(term1, term2);
            } else if (i == depth_points - 1) {
                // 刚性底面边界
                Complex term1 = complex_multiply(make_complex(1.0/(2.0*h2), 0.0), psi_current[i-1]);
                Complex term2 = complex_multiply(make_complex(-2.0/h2 - k0*k0*(n[i]*n[i]-1.0)/2.0, -alpha.imag), psi_current[i]);
                rhs[i] = complex_add(term1, term2);
            } else {
                // 内部点 - 显式部分
                double n2_minus_1 = n[i] * n[i] - 1.0;
                Complex term1 = complex_multiply(make_complex(1.0/(2.0*h2), 0.0), psi_current[i-1]);
                Complex term2 = complex_multiply(make_complex(-2.0/h2 - k0*k0*n2_minus_1/2.0, -alpha.imag), psi_current[i]);
                Complex term3 = complex_multiply(make_complex(1.0/(2.0*h2), 0.0), psi_current[i+1]);
                
                rhs[i] = complex_add(complex_add(term1, term2), term3);
            }
        }
        
        // 解三对角方程 Cx = rhs（简化LU分解/Thomas算法）
        // 优先尝试增强复数线性方程组求解器
        Complex* A_matrix = (Complex*)calloc(depth_points * depth_points, sizeof(Complex));
        int use_eigen_solver = (A_matrix != NULL);
        
        if (use_eigen_solver) {
            // 构造三对角矩阵
            for (size_t i = 0; i < depth_points; i++) {
                // 对角元素
                A_matrix[i * depth_points + i] = b[i];
                // 上对角元素
                if (i < depth_points - 1) {
                    A_matrix[i * depth_points + (i + 1)] = c_coeff[i];
                }
                // 下对角元素
                if (i > 0) {
                    A_matrix[i * depth_points + (i - 1)] = a[i];
                }
            }
            // 使用Eigen求解复数线性方程组
            int solve_result = eigen_solve_complex_system(A_matrix, rhs, psi_next, depth_points);
            free(A_matrix);
            if (solve_result != 0) {
                use_eigen_solver = 0;  // 回退到原始算法
            }
        }
        // 如果Eigen求解失败或不可用，使用原始Thomas算法
        if (!use_eigen_solver) {
            // 前向消元
            for (size_t i = 1; i < depth_points; i++) {
                if (complex_abs(b[i-1]) < 1e-12) {
                    printf("ERROR: Matrix is singular at step %zu, index %zu\n", ir, i-1);
                    goto cleanup;
                }
                Complex factor = complex_divide(a[i], b[i-1]);
                b[i] = complex_subtract(b[i], complex_multiply(factor, c_coeff[i-1]));
                rhs[i] = complex_subtract(rhs[i], complex_multiply(factor, rhs[i-1]));
            }
            // 回代
            if (complex_abs(b[depth_points-1]) < 1e-12) {
                printf("ERROR: Matrix is singular at final step %zu\n", ir);
                goto cleanup;
            }
            psi_next[depth_points-1] = complex_divide(rhs[depth_points-1], b[depth_points-1]);
            for (int i = (int)depth_points - 2; i >= 0; i--) {
                if (complex_abs(b[i]) < 1e-12) {
                    printf("ERROR: Matrix is singular at step %zu, back substitution index %d\n", ir, i);
                    goto cleanup;
                }
                Complex temp = complex_multiply(c_coeff[i], psi_next[i+1]);
                psi_next[i] = complex_divide(complex_subtract(rhs[i], temp), b[i]);
            }
        }
        // 数值稳定性检查
        for (size_t i = 0; i < depth_points; i++) {
            if (!isfinite(psi_next[i].real) || !isfinite(psi_next[i].imag)) {
                printf("ERROR: Non-finite values detected at range step %zu\n", ir);
                goto cleanup;
            }
        }
        // 拷贝结果
        memcpy(field[ir], psi_next, depth_points * sizeof(Complex));
        // 交换指针，准备下一个步进
        Complex* temp = psi_current;
        psi_current = psi_next;
        psi_next = temp;
    }
    printf("Parabolic equation computation completed successfully\n");
cleanup:
    free(z); free(c); free(n);
    free(psi_current); free(psi_next); free(rhs);
    free(a); free(b); free(c_coeff); free(d);
    return 0;
}

// 抛物方程声场计算函数（详细参数版）
SIGNAL_LIB_API int parabolic_equation_field_detailed(const parabolic_equation_params_t* params, Complex** field) {
    // 参数校验
    if (!params || !params->profile || !field || 
        params->range_points == 0 || params->depth_points < 2 || params->frequency <= 0) {
        printf("ERROR: Invalid parameters for detailed parabolic equation\n");
        return -1;
    }
    const SoundProfile* profile = params->profile;
    double freq = params->frequency;
    double source_depth = params->source_depth_m;
    size_t range_points = params->range_points;
    size_t depth_points = params->depth_points;
    printf("=== DETAILED PARABOLIC EQUATION COMPUTATION ===\n");
    printf("Range points: %zu, Depth points: %zu\n", range_points, depth_points);
    printf("Frequency: %.1f Hz, Source depth: %.1f m\n", freq, source_depth);
    printf("Max range: %.1f m, Total depth: %.1f m\n", params->max_range_m, params->total_depth_m);
    // 计算常量
    double c0 = profile->speed[0];  // 参考声速
    double omega = 2.0 * PI * freq;
    double k0 = omega / c0;
    double total_depth = params->total_depth_m;
    double h = total_depth / (depth_points - 1);  // 深度步长
    double h2 = h * h;
    // 设置距离步长 (使用详细参数)
    double lambda = 2.0 * PI / k0;  // 波长
    double rmax = params->max_range_m;
    double deltar = (params->range_step_m > 0) ? params->range_step_m : 
                    (rmax / (range_points - 1));
    // 检查步长是否满足稳定性条件
    double deltar_max = lambda / 15.0;  // 最大允许步长
    if (deltar > deltar_max) {
        printf("Warning: Range step %.2f m exceeds stability limit %.2f m (λ/15)\n", deltar, deltar_max);
    }
    // 使用用户指定的高斯源宽度因子
    double gaussian_width_factor = (params->gaussian_source_width_factor > 0) ? 
                                   params->gaussian_source_width_factor : 
                                   (sqrt(0.5) * 10.0 / k0);  // 默认值
    printf("Computational grid: h=%.2f m, deltar=%.2f m (λ/%.1f)\n", h, deltar, lambda/deltar);
    printf("k0=%.6f, gaussian_width=%.2f m\n", k0, gaussian_width_factor);
    printf("Bottom attenuation: %.3f dB/wavelength\n", params->bottom_attenuation_dB_lambda);
    // 分配网格数组
    double* z = (double*)malloc(depth_points * sizeof(double));
    double* c = (double*)malloc(depth_points * sizeof(double));
    double* n = (double*)malloc(depth_points * sizeof(double));
    if (!z || !c || !n) {
        printf("ERROR: Memory allocation failed for grids\n");
        free(z); free(c); free(n);
        return -1;
    }
    // 初始化深度、声速、折射率
    for (size_t i = 0; i < depth_points; i++) {
        z[i] = i * h;
        // 分段线性插值声速
        c[i] = c0;  // 默认值
        for (size_t j = 0; j < profile->length - 1; j++) {
            if (z[i] >= profile->depth[j] && z[i] <= profile->depth[j+1]) {
                double ratio = (z[i] - profile->depth[j]) / 
                              (profile->depth[j+1] - profile->depth[j]);
                c[i] = profile->speed[j] + ratio * (profile->speed[j+1] - profile->speed[j]);
                break;
            }
        }
        n[i] = c0 / c[i];  // 折射率
    }
    printf("Sound speed range: [%.1f, %.1f] m/s\n", c[0], c[depth_points-1]);
    // 分配工作数组
    Complex* psi_current = (Complex*)calloc(depth_points, sizeof(Complex));
    Complex* psi_next = (Complex*)calloc(depth_points, sizeof(Complex));
    Complex* rhs = (Complex*)calloc(depth_points, sizeof(Complex));
    // 三对角矩阵系数
    Complex* a = (Complex*)malloc(depth_points * sizeof(Complex));  // 下对角
    Complex* b = (Complex*)malloc(depth_points * sizeof(Complex));  // 对角
    Complex* c_coeff = (Complex*)malloc(depth_points * sizeof(Complex));  // 上对角
    Complex* d = (Complex*)malloc(depth_points * sizeof(Complex));  // 右端项
    if (!psi_current || !psi_next || !rhs || !a || !b || !c_coeff || !d) {
        printf("ERROR: Memory allocation failed for computation arrays\n");
        goto cleanup_detailed;
    }
    // 高斯源初始化（使用详细参数）
    printf("Initializing Gaussian source with width factor %.2f...\n", gaussian_width_factor);
    for (size_t i = 0; i < depth_points; i++) {
        double depth_diff = z[i] - source_depth;
        // 使用给定的高斯宽度，初始化源
        double gaussian = exp(-0.5 * (depth_diff / gaussian_width_factor) * (depth_diff / gaussian_width_factor));
        // 可加初始相位，这里设为0
        double phase = 0.0;
        psi_current[i] = make_complex(gaussian * cos(phase), gaussian * sin(phase));
    }
    // 拷贝初始声场到输出
    memcpy(field[0], psi_current, depth_points * sizeof(Complex));
    printf("Starting range marching...\n");
    // 距离步进主循环
    for (size_t ir = 1; ir < range_points; ir++) {
        if (ir % 100 == 0) {
            printf("Range step: %zu/%zu\n", ir, range_points-1);
        }
        // 构造Crank-Nicolson格式系数矩阵
        Complex alpha = make_complex(0.0, 2.0 * k0 / deltar);  // 2ik0/Δr
        for (size_t i = 0; i < depth_points; i++) {
            double n2_minus_1 = n[i] * n[i] - 1.0;
            // 系数矩阵 C（三对角形式）
            if (i == 0) {
                // 表面边界（Neumann边界）
                a[i] = make_complex(0.0, 0.0);
                b[i] = make_complex(2.0/h2 + k0*k0*n2_minus_1/2.0, alpha.imag);
                c_coeff[i] = make_complex(-2.0/h2, 0.0);
            } else if (i == depth_points - 1) {
                // 底面边界（吸收）
                double lambda = 2.0 * PI / k0;  // 波长
                double attenuation_per_step = params->bottom_attenuation_dB_lambda * deltar / lambda;
                double attenuation_factor = exp(-attenuation_per_step * log(10.0) / 20.0);  // dB转幅值
                a[i] = make_complex(-1.0/(2.0*h2), 0.0);
                b[i] = make_complex(2.0/h2 + k0*k0*n2_minus_1/2.0, alpha.imag + (1.0 - attenuation_factor));
                c_coeff[i] = make_complex(0.0, 0.0);
            } else {
                // 内部点
                a[i] = make_complex(-1.0/(2.0*h2), 0.0);
                b[i] = make_complex(2.0/h2 + k0*k0*n2_minus_1/2.0, alpha.imag);
                c_coeff[i] = make_complex(-1.0/(2.0*h2), 0.0);
            }
        }
        // 构造右端项（显式部分）
        for (size_t i = 0; i < depth_points; i++) {
            if (i == 0) {
                // 计算右端项 (显式部分)
                double n2_minus_1 = n[i] * n[i] - 1.0;
                Complex term1 = complex_multiply(make_complex(-2.0/h2 - k0*k0*n2_minus_1/2.0, -alpha.imag), psi_current[i]);
                Complex term2 = complex_multiply(make_complex(2.0/h2, 0.0), psi_current[i+1]);
                rhs[i] = complex_add(term1, term2);
            } else if (i == depth_points - 1) {
                // 刚性底面边界
                Complex term1 = complex_multiply(make_complex(1.0/(2.0*h2), 0.0), psi_current[i-1]);
                Complex term2 = complex_multiply(make_complex(-2.0/h2 - k0*k0*(n[i]*n[i]-1.0)/2.0, -alpha.imag), psi_current[i]);
                rhs[i] = complex_add(term1, term2);
            } else {
                // 内部点 - 显式部分
                double n2_minus_1 = n[i] * n[i] - 1.0;
                Complex term1 = complex_multiply(make_complex(1.0/(2.0*h2), 0.0), psi_current[i-1]);
                Complex term2 = complex_multiply(make_complex(-2.0/h2 - k0*k0*n2_minus_1/2.0, -alpha.imag), psi_current[i]);
                Complex term3 = complex_multiply(make_complex(1.0/(2.0*h2), 0.0), psi_current[i+1]);
                rhs[i] = complex_add(complex_add(term1, term2), term3);
            }
        }
        // 优先尝试增强复数线性方程组求解器
        Complex* A_matrix_detailed = (Complex*)calloc(depth_points * depth_points, sizeof(Complex));
        int use_eigen_solver_detailed = (A_matrix_detailed != NULL);
        if (use_eigen_solver_detailed) {
            // 构造三对角矩阵
            for (size_t i = 0; i < depth_points; i++) {
                // 对角元素
                A_matrix_detailed[i * depth_points + i] = b[i];
                // 上对角元素
                if (i < depth_points - 1) {
                    A_matrix_detailed[i * depth_points + (i + 1)] = c_coeff[i];
                }
                // 下对角元素
                if (i > 0) {
                    A_matrix_detailed[i * depth_points + (i - 1)] = a[i];
                }
            }
            // 使用Eigen求解复数线性方程组
            int solve_result_detailed = eigen_solve_complex_system(A_matrix_detailed, rhs, psi_next, depth_points);
            free(A_matrix_detailed);
            if (solve_result_detailed != 0) {
                use_eigen_solver_detailed = 0;  // 回退到原始算法
            }
        }
        // 如果Eigen求解失败或不可用，使用原始Thomas算法
        if (!use_eigen_solver_detailed) {
            // 解三对角方程（Thomas算法 - 使用增强复数运算）
            // 前向消元
            for (size_t i = 1; i < depth_points; i++) {
                if (complex_abs(b[i-1]) < 1e-12) {
                    printf("ERROR: Matrix is singular at step %zu, index %zu\n", ir, i-1);
                    goto cleanup_detailed;
                }
                Complex factor = complex_divide(a[i], b[i-1]);
                b[i] = complex_subtract(b[i], complex_multiply(factor, c_coeff[i-1]));
                rhs[i] = complex_subtract(rhs[i], complex_multiply(factor, rhs[i-1]));
            }
            // 回代
            if (complex_abs(b[depth_points-1]) < 1e-12) {
                printf("ERROR: Matrix is singular at final step %zu\n", ir);
                goto cleanup_detailed;
            }
            psi_next[depth_points-1] = complex_divide(rhs[depth_points-1], b[depth_points-1]);
            for (int i = (int)depth_points - 2; i >= 0; i--) {
                if (complex_abs(b[i]) < 1e-12) {
                    printf("ERROR: Matrix is singular at step %zu, back substitution index %d\n", ir, i);
                    goto cleanup_detailed;
                }
                Complex temp = complex_multiply(c_coeff[i], psi_next[i+1]);
                psi_next[i] = complex_divide(complex_subtract(rhs[i], temp), b[i]);
            }
        }
        // 数值稳定性检查
        for (size_t i = 0; i < depth_points; i++) {
            if (!isfinite(psi_next[i].real) || !isfinite(psi_next[i].imag)) {
                printf("ERROR: Non-finite values detected at range step %zu\n", ir);
                goto cleanup_detailed;
            }
        }
        // 拷贝结果
        memcpy(field[ir], psi_next, depth_points * sizeof(Complex));
        // 交换指针，准备下一个步进
        Complex* temp = psi_current;
        psi_current = psi_next;
        psi_next = temp;
    }
    printf("Detailed parabolic equation computation completed successfully\n");
cleanup_detailed:
    free(z); free(c); free(n);
    free(psi_current); free(psi_next); free(rhs);
    free(a); free(b); free(c_coeff); free(d);
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
SIGNAL_LIB_API int parabolic_equation_solver(double* field, int nx, int nz, double dx, double dz, double freq, double* sound_speed, void* source) {
    // TODO: 实现抛物方程声场计算，这里仅为占位
    if (field && nx > 0 && nz > 0) {
        for (int i = 0; i < nx * nz; ++i) field[i] = 0.0;
    }
    return 0;
}
SIGNAL_LIB_API int parabolic_equation_solver_detailed(double* field, int nx, int nz, double dx, double dz, double freq, double* sound_speed, void* source, int option) {
    // TODO: 实现详细版抛物方程声场计算，这里仅为占位
    if (field && nx > 0 && nz > 0) {
        for (int i = 0; i < nx * nz; ++i) field[i] = 0.0;
    }
    return 0;
}
#ifdef __cplusplus
}
#endif
