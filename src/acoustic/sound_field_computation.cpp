#include "../include/signal_lib.hpp"
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// 包含Eigen库头文件用于矩阵运算
#include "../../3party/Eigen/Eigen"
#include "../../3party/Eigen/Sparse"
#include "../../3party/Eigen/SparseLU"

using namespace Eigen;
using namespace std;

// 复数类型定义，与Eigen兼容
typedef std::complex<double> cdbl;

// 辅助函数：创建复数
static Complex make_complex(double real, double imag) {
    Complex c;
    c.real = real;
    c.imag = imag;
    return c;
}

// 辅助函数：Complex转换为cdbl
static cdbl complex_to_cdbl(Complex c) {
    return cdbl(c.real, c.imag);
}

// 辅助函数：cdbl转换为Complex
static Complex cdbl_to_complex(cdbl c) {
    Complex result;
    result.real = c.real();
    result.imag = c.imag();
    return result;
}

// 辅助函数：分配二维数组
static double** allocate_2d_array(size_t rows, size_t cols) {
    double** array = (double**)malloc(rows * sizeof(double*));
    if (!array) return NULL;
    
    for (size_t i = 0; i < rows; i++) {
        array[i] = (double*)calloc(cols, sizeof(double));
        if (!array[i]) {
            // 释放已分配的内存
            for (size_t j = 0; j < i; j++) {
                free(array[j]);
            }
            free(array);
            return NULL;
        }
    }
    return array;
}

// 辅助函数：分配二维复数数组
static Complex** allocate_2d_complex_array(size_t rows, size_t cols) {
    Complex** array = (Complex**)malloc(rows * sizeof(Complex*));
    if (!array) return NULL;
    
    for (size_t i = 0; i < rows; i++) {
        array[i] = (Complex*)calloc(cols, sizeof(Complex));
        if (!array[i]) {
            // 释放已分配的内存
            for (size_t j = 0; j < i; j++) {
                free(array[j]);
            }
            free(array);
            return NULL;
        }
    }
    return array;
}

// 辅助函数：释放二维数组
static void free_2d_array(double** array, size_t rows) {
    if (!array) return;
    for (size_t i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

// 辅助函数：释放二维复数数组
static void free_2d_complex_array(Complex** array, size_t rows) {
    if (!array) return;
    for (size_t i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

// 创建声场计算参数结构体
SIGNAL_LIB_API SoundFieldParams* create_sound_field_params(
    const SoundProfile* sound_profile,
    double frequency_hz,
    double source_depth_m,
    double max_range_m,
    double max_depth_m
) {
    // 参数验证
    if (!sound_profile || frequency_hz <= 0 || source_depth_m < 0 || 
        max_range_m <= 0 || max_depth_m <= 0 || source_depth_m > max_depth_m) {
        printf("ERROR: Invalid parameters for sound field computation\n");
        return NULL;
    }
    
    SoundFieldParams* params = (SoundFieldParams*)malloc(sizeof(SoundFieldParams));
    if (!params) {
        printf("ERROR: Memory allocation failed for SoundFieldParams\n");
        return NULL;
    }
    
    // 设置基本参数
    params->sound_profile = sound_profile;
    params->frequency_hz = frequency_hz;
    params->source_depth_m = source_depth_m;
    params->max_range_m = max_range_m;
    params->max_depth_m = max_depth_m;
    
    // 设置默认网格参数
    params->range_points = 1000;
    params->depth_points = 250;
    params->range_step_m = 0.0;  // 自动计算
    params->depth_step_m = 0.0;  // 自动计算
    
    // 设置默认高级参数
    params->gaussian_beam_width_factor = 10.0;
    params->reference_sound_speed_ms = 1500.0;
    params->boundary_condition_type = 0;  // 自由表面+刚性底面
    params->bottom_attenuation_db_lambda = 0.5;
    
    printf("Created sound field parameters: f=%.1f Hz, source_depth=%.1f m, max_range=%.1f m\n",
           frequency_hz, source_depth_m, max_range_m);
    
    return params;
}

// 设置声场计算的网格参数
SIGNAL_LIB_API int set_sound_field_grid(SoundFieldParams* params, size_t range_points, size_t depth_points) {
    if (!params || range_points < 10 || depth_points < 10) {
        printf("ERROR: Invalid grid parameters\n");
        return -1;
    }
    
    params->range_points = range_points;
    params->depth_points = depth_points;
    
    // 重新计算步长
    params->range_step_m = params->max_range_m / (range_points - 1);
    params->depth_step_m = params->max_depth_m / (depth_points - 1);
    
    printf("Set grid: %zu x %zu points, dr=%.2f m, dz=%.2f m\n",
           range_points, depth_points, params->range_step_m, params->depth_step_m);
    
    return 0;
}

// 设置声场计算的高级参数
SIGNAL_LIB_API int set_sound_field_advanced_params(
    SoundFieldParams* params,
    double gaussian_beam_width_factor,
    int boundary_condition_type,
    double bottom_attenuation_db_lambda
) {
    if (!params || gaussian_beam_width_factor <= 0 || 
        boundary_condition_type < 0 || boundary_condition_type > 1 ||
        bottom_attenuation_db_lambda < 0) {
        printf("ERROR: Invalid advanced parameters\n");
        return -1;
    }
    
    params->gaussian_beam_width_factor = gaussian_beam_width_factor;
    params->boundary_condition_type = boundary_condition_type;
    params->bottom_attenuation_db_lambda = bottom_attenuation_db_lambda;
    
    printf("Set advanced params: beam_width_factor=%.1f, boundary_type=%d, bottom_atten=%.2f dB/λ\n",
           gaussian_beam_width_factor, boundary_condition_type, bottom_attenuation_db_lambda);
    
    return 0;
}

// 创建声场结果结构体
SIGNAL_LIB_API SoundFieldResult* create_sound_field_result(size_t range_points, size_t depth_points) {
    if (range_points == 0 || depth_points == 0) {
        printf("ERROR: Invalid dimensions for sound field result\n");
        return NULL;
    }
    
    SoundFieldResult* result = (SoundFieldResult*)malloc(sizeof(SoundFieldResult));
    if (!result) {
        printf("ERROR: Memory allocation failed for SoundFieldResult\n");
        return NULL;
    }
    
    // 分配传输损失矩阵
    result->transmission_loss_db = allocate_2d_array(depth_points, range_points);
    if (!result->transmission_loss_db) {
        free(result);
        return NULL;
    }
    
    // 分配复数声场矩阵
    result->complex_field = allocate_2d_complex_array(depth_points, range_points);
    if (!result->complex_field) {
        free_2d_array(result->transmission_loss_db, depth_points);
        free(result);
        return NULL;
    }
    
    result->range_points = range_points;
    result->depth_points = depth_points;
    result->range_step_m = 0.0;
    result->depth_step_m = 0.0;
    result->max_range_m = 0.0;
    result->max_depth_m = 0.0;
    result->computation_time_seconds = 0.0;
    result->computation_status = 0;
    
    return result;
}

// 基于抛物方程方法计算声场传输损失 (完全按照标准代码实现)
SIGNAL_LIB_API int compute_sound_field_pe(const SoundFieldParams* params, SoundFieldResult* result) {
    if (!params || !result || !params->sound_profile) {
        printf("ERROR: Invalid parameters for PE computation\n");
        return -1;
    }
    
    clock_t start_time = clock();
    printf("=== 高级PE声场计算开始 ===\n");
    printf("频率: %.1f Hz, 声源深度: %.1f m\n", params->frequency_hz, params->source_depth_m);
    
    // 完全按照标准代码的参数设置
    double zs = params->source_depth_m;
    double f = params->frequency_hz;
    double omega = 2 * PI * f;
    double c0 = 1500.0;  // 参考声速
    double d = params->max_depth_m;   // 最大深度
    int nz = (int)params->depth_points;        // 深度网格点数
    double h = d / nz;   // 深度步长
    double h2 = h * h;
    
    int nr = (int)params->range_points;               // 距离网格点数
    double rmax = params->max_range_m;      // 最大距离
    double deltar = rmax / nr;   // 距离步长
    
    printf("计算域: %.1f x %.1f m, 网格: %d x %d\n", rmax, d, nr, nz);
    printf("计算参数: h=%.2f m, Δr=%.2f m\n", h, deltar);
    
    // 更新结果结构体的参数
    result->range_step_m = deltar;
    result->depth_step_m = h;
    result->max_range_m = rmax;
    result->max_depth_m = d;
    
    try {
        vector<double> z(nz);
        for (int i = 0; i < nz; ++i)
            z[i] = i * h;
        
        // 计算声速剖面 - 从数据库加载真实声速剖面数据
        vector<double> c(nz);
        for (int i = 0; i < nz; ++i) {
            c[i] = interpolate_sound_speed(params->sound_profile, z[i]);
            if (c[i] <= 0) {
                printf("ERROR: Invalid sound speed at depth %.1f m\n", z[i]);
                return -2;
            }
        }
        
        printf("声速剖面已映射到网格: 表面声速=%.1f m/s, 底部声速=%.1f m/s\n", c[0], c[nz-1]);
        
        // 高斯初始条件
        double k0 = omega / c0;
        vector<cdbl> psi(nz, 0.0);
        double fac = params->gaussian_beam_width_factor;
        if (fac <= 0 || !std::isfinite(fac)) {
            fac = 10.0;  // 默认束宽因子
        }
        
        for (int i = 0; i < nz; ++i) {
            double arg = pow((k0 / fac), 2) * pow(z[i] - zs, 2);
            psi[i] = sqrt(k0 / fac) * exp(-arg);
        }
        
        printf("高斯源初始化完成，束宽因子=%.1f\n", fac);
        
        // 构建系数矩阵
        vector<double> n(nz);
        for (int i = 0; i < nz; ++i)
            n[i] = c0 / c[i];
        
        // 填充矩阵A
        SparseMatrix<cdbl> A(nz, nz), B(nz, nz), C(nz, nz);
        vector<Triplet<cdbl>> triplets;
        
        for (int i = 1; i < nz - 1; ++i) {
            triplets.emplace_back(i, i - 1, 1.0 / h2);
            triplets.emplace_back(i, i + 1, 1.0 / h2);
            double val = -2.0 / h2 + pow(k0, 2)*(pow(n[i], 2) - 1);
            triplets.emplace_back(i, i, val);
        }
        triplets.emplace_back(0, 0, 1.0 / h2);
        triplets.emplace_back(nz - 1, nz - 1, 1.0 / h2);
        A.setFromTriplets(triplets.begin(), triplets.end());
        
        // 构建B和C矩阵
        SparseMatrix<cdbl> I(nz, nz);
        I.setIdentity();
        B = (2.0 * cdbl(0, 1) * k0 / deltar)* I - 0.5*A;
        C = (2.0 * cdbl(0, 1) * k0 / deltar)* I + 0.5*A;
        
        printf("矩阵构建完成，开始LU分解...\n");
        
        // LU分解
        SparseLU<SparseMatrix<cdbl>> solver;
        solver.compute(C);
        if (solver.info() != Success) {
            printf("ERROR: LU decomposition failed\n");
            return -3;
        }
        
        printf("开始距离步进计算...\n");
        
        // 步进计算
        vector<vector<cdbl>> psi_history(nr, vector<cdbl>(nz, 0.0));
        psi_history[0] = psi;
        for (int ir = 1; ir < nr; ++ir) {
            if (ir % 100 == 0) {
                printf("距离步进: %d/%d (%.1f%%)\n", ir, nr-1, 100.0 * ir / (nr-1));
            }
            
            VectorXcd rhs = B * VectorXcd::Map(&psi_history[ir - 1][0], nz);
            VectorXcd sol = solver.solve(rhs);
            
            if (solver.info() != Success) {
                printf("ERROR: Linear system solution failed at range step %d\n", ir);
                return -3;
            }
            
            for (int i = 0; i < nz; ++i) {
                if (!isfinite(sol[i].real()) || !isfinite(sol[i].imag())) {
                    printf("ERROR: Non-finite values at range step %d\n", ir);
                    return -4;
                }
                psi_history[ir][i] = sol[i];
            }
        }
        
        printf("计算传输损失和Hankel函数修正...\n");
        
        // 计算结果场
        for (int j = 0; j < nr; ++j) {
            double r = max(j*deltar, 1e-10);
            cdbl hank = sqrt(2 / (PI*k0)) * exp(cdbl(0, 1)*(k0*r - PI / 4)) / sqrt(r);
            for (int i = 0; i < nz; ++i) {
                cdbl field_value = psi_history[j][i] * hank;
                double tl = 20 * log10(max(abs(field_value), 1e-10));
                
                result->transmission_loss_db[i][j] = tl;
                result->complex_field[i][j] = cdbl_to_complex(field_value);
            }
        }
        
        clock_t end_time = clock();
        result->computation_time_seconds = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        result->computation_status = 0;
        
        printf("=== 高级PE声场计算完成 ===\n");
        printf("计算耗时: %.2f 秒\n", result->computation_time_seconds);
        printf("传输损失范围: [%.1f, %.1f] dB\n", 
               result->transmission_loss_db[0][0], 
               result->transmission_loss_db[nz-1][nr-1]);
        printf("计算使用了来自数据库的真实声速剖面数据\n");
        
        return 0;
        
    } catch (const std::exception& e) {
        printf("ERROR: Exception in PE computation: %s\n", e.what());
        result->computation_status = -2;
        return -2;
    }
}


// 销毁声场参数结构体，释放内存
SIGNAL_LIB_API void destroy_sound_field_params(SoundFieldParams* params) {
    if (params) {
        free(params);
    }
}

// 销毁声场结果结构体，释放内存
SIGNAL_LIB_API void destroy_sound_field_result(SoundFieldResult* result) {
    if (!result) return;
    
    if (result->transmission_loss_db) {
        free_2d_array(result->transmission_loss_db, result->depth_points);
    }
    
    if (result->complex_field) {
        free_2d_complex_array(result->complex_field, result->depth_points);
    }
    
    free(result);
}

// 从声场结果中提取指定距离处的传输损失剖面
SIGNAL_LIB_API int extract_range_profile(
    const SoundFieldResult* result,
    double range_m,
    double** depth_profile,
    double** tl_profile,
    size_t* profile_length
) {
    if (!result || !depth_profile || !tl_profile || !profile_length || range_m < 0) {
        return -1;
    }
    
    // 找到最接近的距离索引
    size_t range_index = (size_t)(range_m / result->range_step_m + 0.5);
    if (range_index >= result->range_points) {
        range_index = result->range_points - 1;
    }
    
    // 分配输出数组
    *depth_profile = (double*)malloc(result->depth_points * sizeof(double));
    *tl_profile = (double*)malloc(result->depth_points * sizeof(double));
    
    if (!*depth_profile || !*tl_profile) {
        free(*depth_profile);
        free(*tl_profile);
        return -1;
    }
    
    // 填充数据
    for (size_t i = 0; i < result->depth_points; i++) {
        (*depth_profile)[i] = i * result->depth_step_m;
        (*tl_profile)[i] = result->transmission_loss_db[i][range_index];
    }
    
    *profile_length = result->depth_points;
    return 0;
}

// 从声场结果中提取指定深度处的传输损失剖面
SIGNAL_LIB_API int extract_depth_profile(
    const SoundFieldResult* result,
    double depth_m,
    double** range_profile,
    double** tl_profile,
    size_t* profile_length
) {
    if (!result || !range_profile || !tl_profile || !profile_length || depth_m < 0) {
        return -1;
    }
    
    // 找到最接近的深度索引
    size_t depth_index = (size_t)(depth_m / result->depth_step_m + 0.5);
    if (depth_index >= result->depth_points) {
        depth_index = result->depth_points - 1;
    }
    
    // 分配输出数组
    *range_profile = (double*)malloc(result->range_points * sizeof(double));
    *tl_profile = (double*)malloc(result->range_points * sizeof(double));
    
    if (!*range_profile || !*tl_profile) {
        free(*range_profile);
        free(*tl_profile);
        return -1;
    }
    
    // 填充数据
    for (size_t j = 0; j < result->range_points; j++) {
        (*range_profile)[j] = j * result->range_step_m;
        (*tl_profile)[j] = result->transmission_loss_db[depth_index][j];
    }
    
    *profile_length = result->range_points;
    return 0;
}

// 计算声场结果的统计信息
SIGNAL_LIB_API int analyze_sound_field_statistics(
    const SoundFieldResult* result,
    double* min_tl_db,
    double* max_tl_db,
    double* mean_tl_db,
    double* convergence_range_m
) {
    if (!result || !min_tl_db || !max_tl_db || !mean_tl_db || !convergence_range_m) {
        return -1;
    }
    
    double min_val = result->transmission_loss_db[0][0];
    double max_val = result->transmission_loss_db[0][0];
    double sum = 0.0;
    size_t count = 0;
    
    // 计算统计值
    for (size_t i = 0; i < result->depth_points; i++) {
        for (size_t j = 0; j < result->range_points; j++) {
            double val = result->transmission_loss_db[i][j];
            if (isfinite(val)) {
                min_val = fmin(min_val, val);
                max_val = fmax(max_val, val);
                sum += val;
                count++;
            }
        }
    }
    
    *min_tl_db = min_val;
    *max_tl_db = max_val;
    *mean_tl_db = (count > 0) ? sum / count : 0.0;
    
    // 估算收敛距离（简化方法：找到传输损失变化小于1dB的距离）
    *convergence_range_m = result->max_range_m;  // 默认值
    
    if (result->range_points > 10) {
        size_t mid_depth = result->depth_points / 2;
        for (size_t j = 10; j < result->range_points - 1; j++) {
            double current_tl = result->transmission_loss_db[mid_depth][j];
            double next_tl = result->transmission_loss_db[mid_depth][j + 1];
            if (fabs(next_tl - current_tl) < 1.0) {
                *convergence_range_m = j * result->range_step_m;
                break;
            }
        }
    }
    
    return 0;
} 