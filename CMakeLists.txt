cmake_minimum_required(VERSION 3.10)
project(SignalLibTest)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加编译选项
if(MSVC)
    add_compile_options(/utf-8 /wd4819)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    # 启用多线程编译
    add_compile_options(/MP)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 添加头文件目录
include_directories(../src/include)
include_directories(../3party)
include_directories(3party/gtest-1.8.0/googletest/include)
include_directories(3party/gtest-1.8.0/googlemock/include)

# 查找gtest库
set(GTEST_LIBRARY ${CMAKE_SOURCE_DIR}/3party/gtest-1.8.0/build64/googlemock/gtest/Release/gtest.lib)
set(GTEST_MAIN_LIBRARY ${CMAKE_SOURCE_DIR}/3party/gtest-1.8.0/build64/googlemock/gtest/Release/gtest_main.lib)
set(GMOCK_LIBRARY ${CMAKE_SOURCE_DIR}/3party/gtest-1.8.0/build64/googlemock/Release/gmock.lib)
set(GMOCK_MAIN_LIBRARY ${CMAKE_SOURCE_DIR}/3party/gtest-1.8.0/build64/googlemock/Release/gmock_main.lib)

# 检查是否找到所有库
foreach(lib ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} ${GMOCK_LIBRARY} ${GMOCK_MAIN_LIBRARY})
    if(NOT EXISTS ${lib})
        message(FATAL_ERROR "Not found: ${lib}")
    endif()
endforeach()

message(STATUS "Found gtest: ${GTEST_LIBRARY}")
message(STATUS "Found gtest_main: ${GTEST_MAIN_LIBRARY}")
message(STATUS "Found gmock: ${GMOCK_LIBRARY}")
message(STATUS "Found gmock_main: ${GMOCK_MAIN_LIBRARY}")

# 添加信号处理库（动态库）
file(GLOB_RECURSE SIGNAL_LIB_SOURCES "src/*.cpp")
message(STATUS "Found source files: ${SIGNAL_LIB_SOURCES}")
add_library(signal_lib SHARED ${SIGNAL_LIB_SOURCES})
target_include_directories(signal_lib PUBLIC src/include)
target_compile_definitions(signal_lib PRIVATE SIGNAL_LIB_EXPORTS)
set_target_properties(signal_lib PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# 创建测试可执行文件
#add_executable(test_all_interfaces test_all_interfaces.cpp)

# 链接库
# 先链接主库，再链接gtest/gmock
# gtest/gmock是静态库，signal_lib是动态库
# 这样test程序运行时依赖signal_lib.dll
#
# target_link_libraries(test_all_interfaces 
#     signal_lib
#     ${GTEST_LIBRARY}
#     ${GTEST_MAIN_LIBRARY}
#     ${GMOCK_LIBRARY}
#     ${GMOCK_MAIN_LIBRARY}
# )

# # 启用测试
# enable_testing()

# 添加测试
#add_test(NAME AllInterfacesTest COMMAND test_all_interfaces)

# 设置测试环境变量
#set_tests_properties(AllInterfacesTest PROPERTIES
#    ENVIRONMENT "PATH=${CMAKE_BINARY_DIR}/bin;$ENV{PATH}"
#) 