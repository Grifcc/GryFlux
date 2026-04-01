set(CUR_APP_NAME classification_app_gryflux)

# 1. 收集你写的 ResNet 源文件
file(GLOB_RECURSE APP_SOURCES "*.cpp")

# 2. 收集 GryFlux 框架底层真正干活的源码
file(GLOB_RECURSE GRYFLUX_CORE_SRC 
    "${PROJECT_SOURCE_DIR}/src/framework/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/utils/*.cpp"
)

# 3. 将你的代码和框架代码揉在一起生成可执行文件
add_executable(${CUR_APP_NAME} ${APP_SOURCES} ${GRYFLUX_CORE_SRC})

# 4. 强制 C++17 标准
target_compile_features(${CUR_APP_NAME} PRIVATE cxx_std_17)

# 5. 查找 OpenCV
find_package(OpenCV REQUIRED)

# 6. 定义华为工具链路径
set(ASCEND_TOOLKIT_HOME "/usr/local/Ascend/ascend-toolkit/latest")

# 7. 指定头文件搜索路径
target_include_directories(${CUR_APP_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
    ${ASCEND_TOOLKIT_HOME}/runtime/include
    ${OpenCV_INCLUDE_DIRS}
)

# 8. 指定库文件搜索路径 
target_link_directories(${CUR_APP_NAME} PRIVATE 
    ${ASCEND_TOOLKIT_HOME}/acllib/lib64
)

# 9. 链接系统库、华为 ACL 和 OpenCV
target_link_libraries(${CUR_APP_NAME}
    PRIVATE
    ascendcl         
    ${OpenCV_LIBS}   
    pthread
    dl
)