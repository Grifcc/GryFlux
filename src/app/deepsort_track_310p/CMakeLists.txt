# 1. 定义项目名称 (更新为 track)
project(deepsort_track_310p)

# GryFlux 要求至少使用 C++17 [cite: 1]
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 2. 查找依赖包
# ==========================================
find_package(OpenCV REQUIRED)

# 【关键新增】DeepSORT 的数学核心依赖 Eigen3
find_package(Eigen3 REQUIRED)

# 配置 Ascend ACL 路径 [cite: 1]
set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Ascend Toolkit Path")

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
    ${ASCEND_PATH}/include              # ACL 头文件 [cite: 1, 2]
)

# ==========================================
# 5. 链接库
# ==========================================
# 添加 ACL 动态库的搜索路径 [cite: 1]
target_link_directories(${PROJECT_NAME} PRIVATE ${ASCEND_PATH}/lib64)

# 链接库文件 [cite: 1]
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${dynamic_libs}    # 框架定义的通用库
    ${OpenCV_LIBS}     # OpenCV 库
    ascendcl           # 昇腾 ACL 核心库
)

# ==========================================
# 6. 安装规则
# ==========================================
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin)