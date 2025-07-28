#include <iostream>
#include <signal_lib.h>

// 声明演示函数
void run_underwater_demo();
void run_communication_demo();
void run_signal_analysis_demo();

// 演示函数声明
void demonstrate_signal_generation();
void demonstrate_spectrum_analysis();
void demonstrate_matched_filter();
void demonstrate_sound_propagation();

// 新增功能演示函数声明
void demonstrate_enhanced_features();
void demonstrate_red_blue_scenario();

int main() {
    std::cout << "水声信号处理库演示程序\n";
    std::cout << "========================\n\n";

    while (true) {
        std::cout << "\n请选择演示模块：\n";
        std::cout << "1. 水声传播与声纳系统演示\n";
        std::cout << "2. 通信信号生成与处理演示\n";
        std::cout << "3. 信号分析与特征提取演示\n";
        std::cout << "0. 退出程序\n";
        std::cout << "请输入选项（0-3）：";

        int choice;
        std::cin >> choice;

        switch (choice) {
            case 0:
                std::cout << "程序结束\n";
                return 0;
            case 1:
                run_underwater_demo();
                break;
            case 2:
                run_communication_demo();
                break;
            case 3:
                run_signal_analysis_demo();
                break;
            default:
                std::cout << "无效选项，请重新选择\n";
        }
    }

    // 运行各种演示
    demonstrate_signal_generation();
    demonstrate_spectrum_analysis();
    demonstrate_matched_filter();
    demonstrate_sound_propagation();
    
    // === 新增功能演示 ===
    printf("\n" + std::string(50, '=') + "\n");
    printf("开始新增功能演示...\n");
    printf(std::string(50, '=') + "\n");
    
    // 调用新增功能演示
    demonstrate_enhanced_features();
    demonstrate_red_blue_scenario();
    
    printf("\n" + std::string(50, '=') + "\n");
    printf("所有演示完成!\n");
    printf(std::string(50, '=') + "\n");

    return 0;
} 