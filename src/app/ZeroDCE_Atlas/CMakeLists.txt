# 1. 收集 ZeroDCE 模块下所有的源码文件
set(ZERODCE_SRC
    ZeroDCE_Atlas.cpp
    source/ZeroDceDataSource.cpp
    context/AtlasContext.cpp
    context/acl_environment.cpp
    consumer/ResultConsumer/ZeroDceResultConsumer.cpp 
    consumer/DiskWriter/AsyncDiskWriter.cpp         
    nodes/Input/InputNode.cpp
    nodes/Preprocess/PreprocessNode.cpp
    nodes/Infer/InferNode.cpp
    nodes/Postprocess/PostprocessNode.cpp
    nodes/Output/OutputNode.cpp
)

# 2. 将你的代码和框架代码打包在一起编译
set(CUR_APP_NAME zero_dce_app)
add_executable(${CUR_APP_NAME} ${ZERODCE_SRC} ${SRC_DIR})

find_package(OpenCV REQUIRED)
# 3. 指定头文件搜索路径
set(ASCEND_TOOLKIT_HOME "$ENV{ASCEND_TOOLKIT_HOME}" CACHE PATH "Ascend toolkit root")
if(NOT ASCEND_TOOLKIT_HOME)
    set(ASCEND_TOOLKIT_HOME "/usr/local/Ascend/ascend-toolkit/latest")
endif()

target_include_directories(${CUR_APP_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}           
    ${CMAKE_SOURCE_DIR}/src       
    ${CMAKE_SOURCE_DIR}/include   
    ${OpenCV_INCLUDE_DIRS}
    ${ASCEND_TOOLKIT_HOME}/include
    ${ASCEND_TOOLKIT_HOME}/acllib/include
)

target_link_directories(${CUR_APP_NAME} PRIVATE 
    ${ASCEND_TOOLKIT_HOME}/lib64
    ${ASCEND_TOOLKIT_HOME}/acllib/lib64
    ${ASCEND_TOOLKIT_HOME}/x86_64-linux/acllib/lib64
)

# 4. 链接动态库
target_link_libraries(${CUR_APP_NAME} PRIVATE 
    ascendcl            
    opencv_core 
    opencv_imgcodecs 
    opencv_imgproc
    opencv_dnn
    pthread
)

install(TARGETS ${CUR_APP_NAME} DESTINATION bin)
