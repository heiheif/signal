cmake_minimum_required(VERSION 3.10)
project(signal_processing_demo)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加头文件路径
include_directories(${CMAKE_SOURCE_DIR}/include)

# 添加源文件
add_executable(signal_demo 
    src/main.cpp
    src/underwater_demo.cpp
    src/communication_demo.cpp
    src/signal_analysis_demo.cpp
)

# 链接信号处理库
target_link_libraries(signal_demo 
    ${CMAKE_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE}/signal_lib.lib
) 