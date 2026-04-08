# 定义项目名称
project(image_fusion_orin)

# GryFlux 要求至少使用 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 1. 查找依赖包 (OpenCV / CUDA / TensorRT)
# ==========================================
# 查找 OpenCV
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/include/aarch64-linux-gnu
        /usr/local/include
)
find_library(TENSORRT_LIB nvinfer
    PATHS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /usr/local/lib
        /usr/local/TensorRT/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
    message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_INCLUDE_DIR and TENSORRT_LIB.")
endif()

# ==========================================
# 2. 收集源文件并生成可执行文件
# ==========================================
# 收集当前应用目录下的所有 .cpp 文件 (包含 source, nodes, context 等)
file(GLOB_RECURSE APP_SOURCES
    "*.cpp" 
)

# 这里的 ${SRC_DIR} 是在根目录 CMakeLists.txt 中定义的 GryFlux 框架源码路径
# 将框架源码和应用源码一起编译成可执行程序
add_executable(${PROJECT_NAME}
    ${APP_SOURCES}
    ${SRC_DIR}
)

# ==========================================
# 3. 包含头文件目录
# ==========================================
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}         # 包含当前目录以支持相对路径引用
    ${OpenCV_INCLUDE_DIRS}              # OpenCV 头文件
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${INC_DIR}                          # GryFlux 框架头文件目录 (假设根目录已定义 INC_DIR)
)

# ==========================================
# 4. 链接库
# ==========================================
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${dynamic_libs}                     # 基础动态库
    ${OpenCV_LIBS}                      # OpenCV 库
    CUDA::cudart
    ${TENSORRT_LIB}
    stdc++fs                            # 用于 std::filesystem 支持
)

# 如果启用了性能分析宏，则添加定义
if(GRYFLUX_BUILD_PROFILING)
    target_compile_definitions(${PROJECT_NAME} PRIVATE GRYFLUX_BUILD_PROFILING=1)
endif()

# ==========================================
# 5. 安装配置 (执行 make install 时拷贝到 bin 目录)
# ==========================================
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
