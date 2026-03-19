# 定义项目名称
project(image_fusion_310p)

# GryFlux 要求至少使用 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 1. 查找依赖包 (OpenCV & Ascend ACL)
# ==========================================
# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 配置 Ascend ACL 路径
set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Ascend Toolkit Path")

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
    ${ASCEND_PATH}/include              # ACL 头文件
    ${INC_DIR}                          # GryFlux 框架头文件目录 (假设根目录已定义 INC_DIR)
)

# ==========================================
# 4. 链接库
# ==========================================
# 添加 ACL 动态库的搜索路径
target_link_directories(${PROJECT_NAME} PRIVATE ${ASCEND_PATH}/lib64)

# 使用根目录中定义的 dynamic_libs (包含 pthread, dl 等基础库)
# 并链接业务所需的 OpenCV 和 ACL 库
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${dynamic_libs}                     # 基础动态库
    ${OpenCV_LIBS}                      # OpenCV 库
    ascendcl                            # 昇腾 ACL 核心库
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