set(APP_NAME deepsort_track_orin)
set(APP_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(APP_3RDPARTY_DIR ${APP_ROOT_DIR}/3rdparty)

set(DEEPSORT_ORIN_OPENCV_ROOT
    "${APP_3RDPARTY_DIR}/opencv"
    CACHE PATH "Root directory or config directory for app-local OpenCV")
set(DEEPSORT_ORIN_CUDA_ROOT
    "${APP_3RDPARTY_DIR}/cuda"
    CACHE PATH "CUDA toolkit root for deepsort_track_orin")
set(DEEPSORT_ORIN_TENSORRT_ROOT
    "${APP_3RDPARTY_DIR}/TensorRT"
    CACHE PATH "TensorRT root for deepsort_track_orin")
set(DEEPSORT_ORIN_EIGEN_ROOT
    "${APP_3RDPARTY_DIR}/Eigen"
    CACHE PATH "Eigen root for deepsort_track_orin")

find_package(OpenCV CONFIG REQUIRED
    PATHS
        ${DEEPSORT_ORIN_OPENCV_ROOT}
        ${DEEPSORT_ORIN_OPENCV_ROOT}/lib/cmake/opencv4
        ${DEEPSORT_ORIN_OPENCV_ROOT}/lib64/cmake/opencv4
    NO_DEFAULT_PATH
)

set(CUDAToolkit_ROOT ${DEEPSORT_ORIN_CUDA_ROOT})
find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS
        ${DEEPSORT_ORIN_TENSORRT_ROOT}/include
        ${DEEPSORT_ORIN_TENSORRT_ROOT}/include/aarch64-linux-gnu
        ${DEEPSORT_ORIN_TENSORRT_ROOT}/targets/aarch64-linux-gnu/include
    NO_DEFAULT_PATH
)

find_library(TENSORRT_LIB nvinfer
    HINTS
        ${DEEPSORT_ORIN_TENSORRT_ROOT}/lib
        ${DEEPSORT_ORIN_TENSORRT_ROOT}/lib64
        ${DEEPSORT_ORIN_TENSORRT_ROOT}/lib/aarch64-linux-gnu
        ${DEEPSORT_ORIN_TENSORRT_ROOT}/targets/aarch64-linux-gnu/lib
        ${DEEPSORT_ORIN_TENSORRT_ROOT}/targets/aarch64-linux/lib
    NO_DEFAULT_PATH
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
    message(FATAL_ERROR
        "TensorRT was not found for deepsort_track_orin. "
        "Populate src/app/deepsort_track_orin/3rdparty/TensorRT or pass "
        "-DDEEPSORT_ORIN_TENSORRT_ROOT=<TensorRT root>.")
endif()

find_package(Eigen3 QUIET CONFIG
    PATHS
        ${DEEPSORT_ORIN_EIGEN_ROOT}
        ${DEEPSORT_ORIN_EIGEN_ROOT}/share/eigen3/cmake
        ${DEEPSORT_ORIN_EIGEN_ROOT}/lib/cmake/eigen3
        ${DEEPSORT_ORIN_EIGEN_ROOT}/lib64/cmake/eigen3
    NO_DEFAULT_PATH
)

if(NOT Eigen3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Dense
        HINTS
            ${DEEPSORT_ORIN_EIGEN_ROOT}/include/eigen3
            ${DEEPSORT_ORIN_EIGEN_ROOT}/eigen3
            ${DEEPSORT_ORIN_EIGEN_ROOT}/include
        NO_DEFAULT_PATH
    )
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR
            "Eigen was not found for deepsort_track_orin. Populate "
            "src/app/deepsort_track_orin/3rdparty/Eigen or pass "
            "-DDEEPSORT_ORIN_EIGEN_ROOT=<Eigen root>.")
    endif()
    set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()

set(APP_SOURCES
    consumer/result_consumer.cpp
    context/infercontext.cpp
    context/reid_context.cpp
    source/image_data_source.cpp
    deepsort_track_orin.cpp
    utils/deepsort_tracker.cpp
    utils/kalman_filter.cpp
    utils/linear_assignment.cpp
    utils/nn_matching.cpp
    utils/track.cpp
    nodes/Input/InputNode.cpp
    nodes/Preprocess/PreprocessNode.cpp
    nodes/DetectionInference/DetectionInferenceNode.cpp
    nodes/Postprocess/PostprocessNode.cpp
    nodes/ReidPreprocess/ReidPreprocessNode.cpp
    nodes/ReidInference/ReidInferenceNode.cpp
    nodes/Output/OutputNode.cpp
)

add_executable(${APP_NAME}
    ${APP_SOURCES}
    ${SRC_DIR}
)

target_include_directories(${APP_NAME} PRIVATE
    ${APP_ROOT_DIR}
    ${APP_ROOT_DIR}/utils
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(${APP_NAME} PRIVATE
    ${dynamic_libs}
    ${OpenCV_LIBS}
    CUDA::cudart
    ${TENSORRT_LIB}
)

if(TARGET Eigen3::Eigen)
    target_link_libraries(${APP_NAME} PRIVATE Eigen3::Eigen)
endif()

install(TARGETS ${APP_NAME} DESTINATION bin)
