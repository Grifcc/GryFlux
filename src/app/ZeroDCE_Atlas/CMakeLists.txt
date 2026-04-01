# 1. 收集 ZeroDCE 模块下所有的源码文件
set(ZERODCE_SRC
    ZeroDCE_Atlas.cpp
    source/ZeroDceDataSource.cpp
    context/AtlasContext.cpp                         
    consumer/ResultConsumer/ZeroDceResultConsumer.cpp 
    consumer/DiskWriter/AsyncDiskWriter.cpp         
    nodes/Preprocess/PreprocessNode.cpp
    nodes/Infer/InferNode.cpp
    nodes/Postprocess/PostprocessNode.cpp
)

# 2. 自动搜索并加入 GryFlux 框架的核心源码和工具源码
file(GLOB FRAMEWORK_SRC 
    "${CMAKE_SOURCE_DIR}/src/framework/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/*.cpp"
)

# 3. 将你的代码和框架代码打包在一起编译
add_executable(zero_dce_app ${ZERODCE_SRC} ${FRAMEWORK_SRC})

find_package(OpenCV REQUIRED)
# 4. 指定头文件搜索路径
target_include_directories(zero_dce_app PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}           
    ${CMAKE_SOURCE_DIR}/src       
    ${CMAKE_SOURCE_DIR}/include   
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/Ascend/ascend-toolkit/latest/include 
    /usr/local/Ascend/ascend-toolkit/latest/acllib/include
)

target_link_directories(zero_dce_app PRIVATE 
    /usr/local/Ascend/ascend-toolkit/latest/lib64
    /usr/local/Ascend/ascend-toolkit/latest/acllib/lib64
    /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/acllib/lib64
)

# 5. 链接动态库
target_link_libraries(zero_dce_app PRIVATE 
    ascendcl            
    opencv_core 
    opencv_imgcodecs 
    opencv_imgproc
    opencv_dnn
    pthread
)