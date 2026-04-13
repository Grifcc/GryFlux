set(APP_NAME image_fusion_orin)
set(APP_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(APP_3RDPARTY_DIR ${APP_ROOT_DIR}/3rdparty)

set(IMAGE_FUSION_ORIN_OPENCV_ROOT
    "${APP_3RDPARTY_DIR}/opencv"
    CACHE PATH "Root directory or config directory for app-local OpenCV")
set(IMAGE_FUSION_ORIN_CUDA_ROOT
    "${APP_3RDPARTY_DIR}/cuda"
    CACHE PATH "CUDA toolkit root for image_fusion_orin")
set(IMAGE_FUSION_ORIN_TENSORRT_ROOT
    "${APP_3RDPARTY_DIR}/TensorRT"
    CACHE PATH "TensorRT root for image_fusion_orin")

find_package(OpenCV CONFIG REQUIRED
    PATHS
        ${IMAGE_FUSION_ORIN_OPENCV_ROOT}
        ${IMAGE_FUSION_ORIN_OPENCV_ROOT}/lib/cmake/opencv4
        ${IMAGE_FUSION_ORIN_OPENCV_ROOT}/lib64/cmake/opencv4
        ${IMAGE_FUSION_ORIN_OPENCV_ROOT}/lib/x86_64-linux-gnu/cmake/opencv4
        ${IMAGE_FUSION_ORIN_OPENCV_ROOT}/lib/aarch64-linux-gnu/cmake/opencv4
    NO_DEFAULT_PATH
)

set(CUDAToolkit_ROOT ${IMAGE_FUSION_ORIN_CUDA_ROOT})
find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS
        ${IMAGE_FUSION_ORIN_TENSORRT_ROOT}/include
        ${IMAGE_FUSION_ORIN_TENSORRT_ROOT}/include/x86_64-linux-gnu
        ${IMAGE_FUSION_ORIN_TENSORRT_ROOT}/include/aarch64-linux-gnu
        ${IMAGE_FUSION_ORIN_TENSORRT_ROOT}/targets/x86_64-linux-gnu/include
        ${IMAGE_FUSION_ORIN_TENSORRT_ROOT}/targets/aarch64-linux-gnu/include
    NO_DEFAULT_PATH
)

find_library(TENSORRT_LIB nvinfer
    HINTS
        ${IMAGE_FUSION_ORIN_TENSORRT_ROOT}/lib
        ${IMAGE_FUSION_ORIN_TENSORRT_ROOT}/lib64
        ${IMAGE_FUSION_ORIN_TENSORRT_ROOT}/lib/x86_64-linux-gnu
        ${IMAGE_FUSION_ORIN_TENSORRT_ROOT}/lib/aarch64-linux-gnu
        ${IMAGE_FUSION_ORIN_TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib
        ${IMAGE_FUSION_ORIN_TENSORRT_ROOT}/targets/aarch64-linux-gnu/lib
    NO_DEFAULT_PATH
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
    message(FATAL_ERROR
        "TensorRT was not found for image_fusion_orin. Populate "
        "src/app/image_fusion_orin/3rdparty/TensorRT or pass "
        "-DIMAGE_FUSION_ORIN_TENSORRT_ROOT=<TensorRT root>.")
endif()

set(APP_SOURCES
    image_fusion_orin.cpp
    consumer/result_consumer.cpp
    context/infercontext.cpp
    source/fusion_data_source.cpp
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
