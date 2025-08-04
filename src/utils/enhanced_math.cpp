#include "../include/signal_lib.hpp"
#include "../../3party/Eigen/Eigen"
#include "../../3party/Eigen/Dense"
#include "../../3party/Eigen/Sparse"
#include <complex>
#include <cmath>
#include <algorithm>

using namespace Eigen;

// 全局配置变量
static struct {
    bool use_eigen;
    bool use_fftw;
    bool use_lapack;
    int num_threads;
    bool initialized;
} math_config = {true, false, false, 1, false};

// 初始化数学库配置
static void initialize_math_config() {
    if (!math_config.initialized) {
        math_config.use_eigen = true;
        math_config.use_fftw = false;
        math_config.use_lapack = false;
        math_config.num_threads = 1;
        
        // 设置Eigen线程数
        Eigen::setNbThreads(math_config.num_threads);
        
        math_config.initialized = true;
    }
}

// 创建Eigen兼容的复数
SIGNAL_LIB_API Complex create_eigen_complex(double real, double imag) {
    Complex result;
    result.real = real;
    result.imag = imag;
    return result;
}

// 使用Eigen进行高精度复数乘法
SIGNAL_LIB_API Complex eigen_complex_multiply(Complex a, Complex b) {
    initialize_math_config();
    
    if (math_config.use_eigen) {
        std::complex<double> ca(a.real, a.imag);
        std::complex<double> cb(b.real, b.imag);
        std::complex<double> result = ca * cb;
        
        Complex output;
        output.real = result.real();
        output.imag = result.imag();
        return output;
    } else {
        // 回退到原始实现
        return complex_multiply(a, b);
    }
}

// 使用Eigen进行高精度复数除法
SIGNAL_LIB_API Complex eigen_complex_divide(Complex a, Complex b) {
    initialize_math_config();
    
    if (math_config.use_eigen) {
        std::complex<double> ca(a.real, a.imag);
        std::complex<double> cb(b.real, b.imag);
        
        // 检查除零
        if (std::abs(cb) < std::numeric_limits<double>::epsilon()) {
            Complex result = {0.0, 0.0};
            return result;
        }
        
        std::complex<double> result = ca / cb;
        
        Complex output;
        output.real = result.real();
        output.imag = result.imag();
        return output;
    } else {
        // 回退到原始实现
        return complex_divide(a, b);
    }
}

// 使用Eigen计算复数的模（避免溢出/下溢）
SIGNAL_LIB_API double eigen_complex_abs(Complex z) {
    initialize_math_config();
    
    if (math_config.use_eigen) {
        std::complex<double> cz(z.real, z.imag);
        return std::abs(cz);  // std::abs使用了稳定的算法
    } else {
        // 回退到原始实现
        return complex_abs(z);
    }
}

// 使用Eigen进行矩阵LU分解
SIGNAL_LIB_API int eigen_lu_decomposition(const double* matrix, size_t rows, size_t cols,
                                         double** L, double** U, int** P) {
    if (!matrix || !L || !U || !P || rows != cols) {
        return -1;
    }
    
    initialize_math_config();
    
    try {
        // 创建Eigen矩阵
        MatrixXd A = Map<const MatrixXd>(matrix, rows, cols);
        
        // 执行LU分解
        PartialPivLU<MatrixXd> lu(A);
        
        // 获取L和U矩阵
        MatrixXd L_matrix = lu.matrixLU().triangularView<UnitLower>();
        MatrixXd U_matrix = lu.matrixLU().triangularView<Upper>();
        
        // 分配输出内存
        *L = (double*)malloc(rows * cols * sizeof(double));
        *U = (double*)malloc(rows * cols * sizeof(double));
        *P = (int*)malloc(rows * sizeof(int));
        
        if (!*L || !*U || !*P) {
            free(*L);
            free(*U);
            free(*P);
            return -1;
        }
        
        // 复制结果
        Map<MatrixXd>(*L, rows, cols) = L_matrix;
        Map<MatrixXd>(*U, rows, cols) = U_matrix;
        
        // 获取置换信息
        VectorXi perm = lu.permutationP().indices();
        for (size_t i = 0; i < rows; i++) {
            (*P)[i] = perm[i];
        }
        
        return 0;
    } catch (const std::exception& e) {
        printf("Eigen LU decomposition error: %s\n", e.what());
        return -1;
    }
}

// 使用Eigen求解线性方程组
SIGNAL_LIB_API int eigen_solve_linear_system(const double* A, const double* b, double* x, size_t n) {
    if (!A || !b || !x || n == 0) {
        return -1;
    }
    
    initialize_math_config();
    
    try {
        // 创建Eigen矩阵和向量
        MatrixXd mat_A = Map<const MatrixXd>(A, n, n);
        VectorXd vec_b = Map<const VectorXd>(b, n);
        
        // 选择合适的求解器
        VectorXd vec_x;
        if (n < 100) {
            // 小矩阵使用直接法
            vec_x = mat_A.lu().solve(vec_b);
        } else {
            // 大矩阵使用迭代法
            ConjugateGradient<MatrixXd> solver;
            solver.compute(mat_A);
            vec_x = solver.solve(vec_b);
            
            if (solver.info() != Success) {
                // 如果迭代法失败，回退到直接法
                vec_x = mat_A.lu().solve(vec_b);
            }
        }
        
        // 复制结果
        Map<VectorXd>(x, n) = vec_x;
        
        return 0;
    } catch (const std::exception& e) {
        printf("Eigen solve linear system error: %s\n", e.what());
        return -1;
    }
}

// 使用Eigen进行复数矩阵求解（用于抛物方程）
SIGNAL_LIB_API int eigen_solve_complex_system(const Complex* A, const Complex* b, Complex* x, size_t n) {
    if (!A || !b || !x || n == 0) {
        return -1;
    }
    
    initialize_math_config();
    
    try {
        // 创建复数矩阵和向量
        MatrixXcd mat_A(n, n);
        VectorXcd vec_b(n);
        
        // 转换数据格式
        for (size_t i = 0; i < n; i++) {
            vec_b[i] = std::complex<double>(b[i].real, b[i].imag);
            for (size_t j = 0; j < n; j++) {
                mat_A(i, j) = std::complex<double>(A[i * n + j].real, A[i * n + j].imag);
            }
        }
        
        // 求解复数线性方程组
        VectorXcd vec_x;
        if (n < 100) {
            // 小矩阵使用LU分解
            vec_x = mat_A.lu().solve(vec_b);
        } else {
            // 大矩阵使用稀疏求解器
            SparseMatrix<std::complex<double>> sparse_A = mat_A.sparseView();
            SparseLU<SparseMatrix<std::complex<double>>> solver;
            solver.compute(sparse_A);
            if (solver.info() == Success) {
                vec_x = solver.solve(vec_b);
            } else {
                // 回退到密集矩阵求解
                vec_x = mat_A.lu().solve(vec_b);
            }
        }
        
        // 转换结果
        for (size_t i = 0; i < n; i++) {
            x[i].real = vec_x[i].real();
            x[i].imag = vec_x[i].imag();
        }
        
        return 0;
    } catch (const std::exception& e) {
        printf("Eigen solve complex system error: %s\n", e.what());
        return -1;
    }
}

// 自适应Simpson积分
static double simpson_adaptive(IntegrandFunction func, double a, double b, 
                              double tolerance, void* params, int max_depth) {
    if (max_depth <= 0) {
        // 达到最大深度，使用简单Simpson公式
        double h = (b - a) / 2.0;
        double fa = func(a, params);
        double fm = func(a + h, params);
        double fb = func(b, params);
        return h / 3.0 * (fa + 4.0 * fm + fb);
    }
    
    double h = (b - a) / 2.0;
    double c = a + h;
    
    // 计算整个区间的Simpson积分
    double fa = func(a, params);
    double fm = func(c, params);
    double fb = func(b, params);
    double S = h / 3.0 * (fa + 4.0 * fm + fb);
    
    // 计算两个子区间的Simpson积分
    double h2 = h / 2.0;
    double f_ac = func(a + h2, params);
    double f_cb = func(c + h2, params);
    
    double S1 = h2 / 3.0 * (fa + 4.0 * f_ac + fm);
    double S2 = h2 / 3.0 * (fm + 4.0 * f_cb + fb);
    double S12 = S1 + S2;
    
    // 检查精度
    if (std::abs(S12 - S) < 15.0 * tolerance) {
        return S12 + (S12 - S) / 15.0;  // Richardson外推
    } else {
        // 递归细分
        return simpson_adaptive(func, a, c, tolerance / 2.0, params, max_depth - 1) +
               simpson_adaptive(func, c, b, tolerance / 2.0, params, max_depth - 1);
    }
}

// 使用自适应Simpson法则进行数值积分
SIGNAL_LIB_API int adaptive_integration(IntegrandFunction func, double a, double b, 
                                       double tolerance, void* params, double* result) {
    if (!func || !result || tolerance <= 0.0) {
        return -1;
    }
    
    try {
        *result = simpson_adaptive(func, a, b, tolerance, params, 20);  // 最大递归深度20
        return 0;
    } catch (const std::exception& e) {
        printf("Adaptive integration error: %s\n", e.what());
        return -1;
    }
}

// 使用Eigen进行特征值分解
SIGNAL_LIB_API int eigen_eigenvalue_decomposition(const double* matrix, size_t n,
                                                 double* eigenvalues, double* eigenvectors) {
    if (!matrix || !eigenvalues || !eigenvectors || n == 0) {
        return -1;
    }
    
    initialize_math_config();
    
    try {
        // 创建对称矩阵
        MatrixXd A = Map<const MatrixXd>(matrix, n, n);
        
        // 确保矩阵是对称的
        A = (A + A.transpose()) / 2.0;
        
        // 执行特征值分解
        SelfAdjointEigenSolver<MatrixXd> solver(A);
        
        if (solver.info() != Success) {
            return -1;
        }
        
        // 复制特征值
        VectorXd vals = solver.eigenvalues();
        Map<VectorXd>(eigenvalues, n) = vals;
        
        // 复制特征向量
        MatrixXd vecs = solver.eigenvectors();
        Map<MatrixXd>(eigenvectors, n, n) = vecs;
        
        return 0;
    } catch (const std::exception& e) {
        printf("Eigen eigenvalue decomposition error: %s\n", e.what());
        return -1;
    }
}

// 增强的Bessel函数计算（使用渐近展开和级数展开）
SIGNAL_LIB_API double enhanced_bessel_j(int order, double x) {
    if (x < 0.0) {
        // 处理负参数
        if (order % 2 == 0) {
            return enhanced_bessel_j(order, -x);
        } else {
            return -enhanced_bessel_j(order, -x);
        }
    }
    
    if (x == 0.0) {
        return (order == 0) ? 1.0 : 0.0;
    }
    
    // 对于大的x值，使用渐近展开
    if (x > 20.0) {
        double phase = x - (order + 0.5) * PI / 2.0;
        return sqrt(2.0 / (PI * x)) * cos(phase);
    }
    
    // 对于小的x值，使用级数展开
    double result = 0.0;
    double term = pow(x / 2.0, order);
    
    // 计算阶乘
    for (int k = 1; k <= order; k++) {
        term /= k;
    }
    
    // 级数求和
    for (int k = 0; k < 50; k++) {  // 最多50项
        double factorial_k = 1.0;
        for (int i = 1; i <= k; i++) {
            factorial_k *= i;
        }
        
        double factorial_k_order = 1.0;
        for (int i = 1; i <= k + order; i++) {
            factorial_k_order *= i;
        }
        
        double sign = (k % 2 == 0) ? 1.0 : -1.0;
        double term_k = sign * pow(x * x / 4.0, k) / (factorial_k * factorial_k_order);
        
        result += term * term_k;
        
        // 检查收敛
        if (std::abs(term_k) < 1e-15) {
            break;
        }
    }
    
    return result;
}

// 增强的Hankel函数计算
SIGNAL_LIB_API Complex enhanced_hankel_h1(int order, double x) {
    if (x <= 0.0) {
        Complex result = {0.0, 0.0};
        return result;
    }
    
    // 对于大的x值，使用渐近展开
    if (x > 20.0) {
        double phase = x - (order + 0.5) * PI / 2.0;
        double amplitude = sqrt(2.0 / (PI * x));
        
        Complex result;
        result.real = amplitude * cos(phase);
        result.imag = amplitude * sin(phase);
        return result;
    }
    
    // 对于小的x值，使用级数展开计算Bessel函数，然后构造Hankel函数
    double j_n = enhanced_bessel_j(order, x);
    
    // 简化的Neumann函数计算
    double y_n;
    if (order == 0) {
        y_n = (2.0 / PI) * (log(x / 2.0) + 0.5772156649);  // 欧拉常数
    } else {
        // 对于高阶，使用递推关系或近似
        y_n = -enhanced_bessel_j(-order, x);
    }
    
    Complex result;
    result.real = j_n;
    result.imag = y_n;
    return result;
}

// 使用三次样条插值
SIGNAL_LIB_API int cubic_spline_interpolation(const double* x, const double* y, size_t n,
                                             const double* xi, double* yi, size_t ni) {
    if (!x || !y || !xi || !yi || n < 2 || ni == 0) {
        return -1;
    }
    
    initialize_math_config();
    
    try {
        // 使用Eigen进行三次样条插值
        VectorXd x_vec = Map<const VectorXd>(x, n);
        VectorXd y_vec = Map<const VectorXd>(y, n);
        
        // 计算二阶导数
        MatrixXd A = MatrixXd::Zero(n, n);
        VectorXd b = VectorXd::Zero(n);
        
        // 自然边界条件
        A(0, 0) = 1.0;
        A(n-1, n-1) = 1.0;
        
        // 内部点
        for (size_t i = 1; i < n-1; i++) {
            double h1 = x_vec[i] - x_vec[i-1];
            double h2 = x_vec[i+1] - x_vec[i];
            
            A(i, i-1) = h1;
            A(i, i) = 2.0 * (h1 + h2);
            A(i, i+1) = h2;
            
            b[i] = 6.0 * ((y_vec[i+1] - y_vec[i]) / h2 - (y_vec[i] - y_vec[i-1]) / h1);
        }
        
        // 求解二阶导数
        VectorXd d2y = A.lu().solve(b);
        
        // 进行插值
        for (size_t j = 0; j < ni; j++) {
            double xi_val = xi[j];
            
            // 找到区间
            size_t i = 0;
            for (i = 0; i < n-1; i++) {
                if (xi_val <= x_vec[i+1]) {
                    break;
                }
            }
            
            if (i >= n-1) i = n-2;
            
            double h = x_vec[i+1] - x_vec[i];
            double t = (xi_val - x_vec[i]) / h;
            
            // 三次样条公式
            yi[j] = y_vec[i] * (1.0 - t) + y_vec[i+1] * t +
                   h * h / 6.0 * ((1.0 - t) * (1.0 - t) * (1.0 - t) - (1.0 - t)) * d2y[i] +
                   h * h / 6.0 * (t * t * t - t) * d2y[i+1];
        }
        
        return 0;
    } catch (const std::exception& e) {
        printf("Cubic spline interpolation error: %s\n", e.what());
        return -1;
    }
}

// 配置数学库
SIGNAL_LIB_API int configure_math_libraries(int use_eigen, int use_fftw, int use_lapack, int num_threads) {
    math_config.use_eigen = (use_eigen != 0);
    math_config.use_fftw = (use_fftw != 0);
    math_config.use_lapack = (use_lapack != 0);
    math_config.num_threads = (num_threads > 0) ? num_threads : 1;
    
    // 设置Eigen线程数
    if (math_config.use_eigen) {
        Eigen::setNbThreads(math_config.num_threads);
    }
    
    math_config.initialized = true;
    return 0;
}

// 获取数学库信息
SIGNAL_LIB_API int get_math_library_info(char* info, size_t max_length) {
    if (!info || max_length == 0) {
        return -1;
    }
    
    initialize_math_config();
    
    snprintf(info, max_length,
        "Math Library Configuration:\n"
        "- Eigen: %s (Version: %d.%d.%d)\n"
        "- FFTW: %s\n"
        "- LAPACK: %s\n"
        "- Threads: %d\n",
        math_config.use_eigen ? "Enabled" : "Disabled",
        EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION,
        math_config.use_fftw ? "Enabled" : "Disabled",
        math_config.use_lapack ? "Enabled" : "Disabled",
        math_config.num_threads);
    
    return 0;
}

// 增强的FFT实现（回退到原始实现，但添加了错误检查）
SIGNAL_LIB_API int enhanced_fft(Complex* data, size_t n, int forward) {
    if (!data || n == 0) {
        return -1;
    }
    
    // 检查n是否为2的幂
    if ((n & (n - 1)) != 0) {
        return -1;  // n不是2的幂
    }
    
    try {
        if (forward) {
            fft(data, n);
        } else {
            ifft(data, n);
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

// 稳定的多项式求根（使用特征值方法）
SIGNAL_LIB_API int stable_polynomial_roots(const double* coefficients, int degree, Complex* roots) {
    if (!coefficients || !roots || degree <= 0) {
        return -1;
    }
    
    initialize_math_config();
    
    try {
        // 构造伴随矩阵
        MatrixXd companion = MatrixXd::Zero(degree, degree);
        
        // 设置伴随矩阵
        for (int i = 0; i < degree - 1; i++) {
            companion(i + 1, i) = 1.0;
        }
        
        // 最后一列是系数
        for (int i = 0; i < degree; i++) {
            companion(i, degree - 1) = -coefficients[degree - i] / coefficients[0];
        }
        
        // 计算特征值（即多项式的根）
        EigenSolver<MatrixXd> solver(companion);
        
        if (solver.info() != Success) {
            return -1;
        }
        
        // 提取根
        VectorXcd eigenvals = solver.eigenvalues();
        for (int i = 0; i < degree; i++) {
            roots[i].real = eigenvals[i].real();
            roots[i].imag = eigenvals[i].imag();
        }
        
        return 0;
    } catch (const std::exception& e) {
        printf("Stable polynomial roots error: %s\n", e.what());
        return -1;
    }
} 