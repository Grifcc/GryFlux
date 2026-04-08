# 1. 定义项目名称
project(deepsort_track_orin)

# GryFlux 要求至少使用 C++17 [cite: 1]
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 2. 查找依赖包
# ==========================================
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 【关键新增】DeepSORT 的数学核心依赖 Eigen3
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Dense
        PATHS
            /usr/include/eigen3
            /usr/local/include/eigen3
    )
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR "Eigen3 not found. Please install eigen3-dev or set EIGEN3_INCLUDE_DIR.")
    endif()
    set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()

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
# 3. 收集源文件
# ==========================================
# 收集当前工程目录下所有的 .cpp (包含 nodes, utils, context 等子目录)
file(GLOB_RECURSE APP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" 
)

# 生成可执行文件
# 注意：${SRC_DIR} 通常是在根目录 CMake 中定义的框架源码列表
add_executable(${PROJECT_NAME}
    ${APP_SOURCES}
    ${SRC_DIR}
)

# ==========================================
# 4. 包含头文件目录
# ==========================================
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}         # 包含当前目录 (用于找到 packet/)
    ${CMAKE_CURRENT_SOURCE_DIR}/utils   # 【优化】直接包含 utils，代码里可以直接 #include "datatype.h"
    ${OpenCV_INCLUDE_DIRS}              # OpenCV 头文件 [cite: 1]
    ${EIGEN3_INCLUDE_DIRS}              # 【新增】Eigen 头文件
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# ==========================================
# 5. 链接库
# ==========================================
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${dynamic_libs}    # 框架定义的通用库
    ${OpenCV_LIBS}     # OpenCV 库
    CUDA::cudart
    ${TENSORRT_LIB}
)

if(TARGET Eigen3::Eigen)
    target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
endif()

# ==========================================
# 6. 安装规则
# ==========================================
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin)
