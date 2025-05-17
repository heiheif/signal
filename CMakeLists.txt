cmake_minimum_required(VERSION 3.10)
project(signal_processing VERSION 1.0.0 LANGUAGES C)

# 设置C标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 添加编译选项
if(MSVC)
    add_compile_options(/utf-8 /wd4819)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    # 添加标准库搜索路径
    include_directories("$ENV{VSINSTALLDIR}/VC/Tools/MSVC/$ENV{VCToolsVersion}/include")
    include_directories("$ENV{WindowsSdkDir}/Include/$ENV{WindowsSDKVersion}/ucrt")
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 添加头文件目录
include_directories(${CMAKE_SOURCE_DIR}/src/include)

# 收集源文件
file(GLOB_RECURSE SOURCES 
    "src/utils/*.c"
    "src/signal/*.c"
    "src/acoustic/*.c"
    "src/processing/*.c"
)

# 创建静态库
add_library(signal_lib STATIC ${SOURCES})

# 设置输出目录
set_target_properties(signal_lib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 链接数学库
if(NOT MSVC)
    target_link_libraries(signal_lib m)
endif()

# 设置头文件包含目录
target_include_directories(signal_lib
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 安装配置
include(GNUInstallDirs)
install(TARGETS signal_lib
    EXPORT signal_lib-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES src/include/signal_lib.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 导出配置
install(EXPORT signal_lib-targets
    FILE signal_lib-targets.cmake
    NAMESPACE signal_lib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/signal_lib
)

# 创建配置文件
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "signal_lib-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/signal_lib-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/signal_lib-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/signal_lib
)

# 安装配置文件
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/signal_lib-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/signal_lib-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/signal_lib
) 