set(APP_NAME yolox_detection_orin)
set(APP_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(APP_3RDPARTY_DIR ${APP_ROOT_DIR}/3rdparty)

set(YOLOX_ORIN_OPENCV_ROOT
    "${APP_3RDPARTY_DIR}/opencv"
    CACHE PATH "Root directory or config directory for app-local OpenCV")
set(YOLOX_ORIN_CUDA_ROOT
    "${APP_3RDPARTY_DIR}/cuda"
    CACHE PATH "CUDA toolkit root for yolox_detection_orin")
set(YOLOX_ORIN_TENSORRT_ROOT
    "${APP_3RDPARTY_DIR}/TensorRT"
    CACHE PATH "TensorRT root for yolox_detection_orin")

find_package(OpenCV CONFIG REQUIRED
    PATHS
        ${YOLOX_ORIN_OPENCV_ROOT}
        ${YOLOX_ORIN_OPENCV_ROOT}/lib/cmake/opencv4
        ${YOLOX_ORIN_OPENCV_ROOT}/lib64/cmake/opencv4
    NO_DEFAULT_PATH
)

set(CUDAToolkit_ROOT ${YOLOX_ORIN_CUDA_ROOT})
find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS
        ${YOLOX_ORIN_TENSORRT_ROOT}/include
        ${YOLOX_ORIN_TENSORRT_ROOT}/include/aarch64-linux-gnu
        ${YOLOX_ORIN_TENSORRT_ROOT}/targets/aarch64-linux-gnu/include
    NO_DEFAULT_PATH
)

find_library(TENSORRT_LIB nvinfer
    HINTS
        ${YOLOX_ORIN_TENSORRT_ROOT}/lib
        ${YOLOX_ORIN_TENSORRT_ROOT}/lib64
        ${YOLOX_ORIN_TENSORRT_ROOT}/lib/aarch64-linux-gnu
        ${YOLOX_ORIN_TENSORRT_ROOT}/targets/aarch64-linux-gnu/lib
        ${YOLOX_ORIN_TENSORRT_ROOT}/targets/aarch64-linux/lib
    NO_DEFAULT_PATH
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
    message(FATAL_ERROR
        "TensorRT was not found for yolox_detection_orin. "
        "Populate src/app/yolox_detection_orin/3rdparty/TensorRT or pass "
        "-DYOLOX_ORIN_TENSORRT_ROOT=<TensorRT root>.")
endif()

set(APP_SOURCES
    consumer/result_consumer.cpp
    context/infercontext.cpp
    source/image_data_source.cpp
    yolox_detection_orin.cpp
    nodes/Input/InputNode.cpp
    nodes/Preprocess/PreprocessNode.cpp
    nodes/Inference/InferenceNode.cpp
    nodes/Postprocess/PostprocessNode.cpp
    nodes/Output/OutputNode.cpp
)

add_executable(${APP_NAME}
    ${APP_SOURCES}
    ${SRC_DIR}
)

target_include_directories(${APP_NAME} PRIVATE
    ${APP_ROOT_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(${APP_NAME} PRIVATE
    ${dynamic_libs}
    ${OpenCV_LIBS}
    CUDA::cudart
    ${TENSORRT_LIB}
)

install(TARGETS ${APP_NAME} DESTINATION bin)
