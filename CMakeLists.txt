cmake_minimum_required(VERSION 3.10)
project(spmvBenchmark)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

# # 添加子目录
# add_subdirectory(src)      # 源文件目录
# add_subdirectory(include)  # 头文件目录（如果有）

# # 手动列出源文件（推荐）
# set(SOURCES
#     main.cpp
#     utils.cpp
#     # 添加更多cpp文件...
# )

# 或者只包含当前目录的cpp文件（不递归子目录）
file(GLOB SOURCES "*.cpp")

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME}  PUBLIC OpenMP::OpenMP_CXX)
endif()

# 可选：添加编译选项
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)

