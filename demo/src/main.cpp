#include <iostream>
#include <signal_lib.h>

// 声明演示函数
void run_underwater_demo();
void run_communication_demo();
void run_signal_analysis_demo();

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

    return 0;
} 