set(RESNET_ORIN_TARGET_NAME resnet_Orin)
set(RESNET_ORIN_COMPAT_TARGET classification_app_gryflux)
set(RESNET_ORIN_INSTALL_COMPONENT resnet_Orin)

set(RESNET_ORIN_3RDPARTY_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty"
    CACHE PATH "App-local 3rdparty root for resnet_Orin")
set(RESNET_ORIN_USE_LOCAL_3RDPARTY OFF
    CACHE BOOL "Resolve OpenCV/CUDA/TensorRT from resnet_Orin/3rdparty")

set(RESNET_ORIN_OPENCV_ROOT
    "${RESNET_ORIN_3RDPARTY_ROOT}/opencv"
    CACHE PATH "OpenCV root under resnet_Orin 3rdparty")
set(RESNET_ORIN_CUDA_ROOT
    "${RESNET_ORIN_3RDPARTY_ROOT}/cuda"
    CACHE PATH "CUDA root under resnet_Orin 3rdparty")
set(RESNET_ORIN_TENSORRT_ROOT
    "${RESNET_ORIN_3RDPARTY_ROOT}/tensorrt"
    CACHE PATH "TensorRT root under resnet_Orin 3rdparty")
set(RESNET_ORIN_MODELS_ROOT
    "${RESNET_ORIN_3RDPARTY_ROOT}/models"
    CACHE PATH "Model root under resnet_Orin 3rdparty")

if(NOT EXISTS "${RESNET_ORIN_3RDPARTY_ROOT}")
    message(FATAL_ERROR
        "resnet_Orin expects app-local 3rdparty root at: ${RESNET_ORIN_3RDPARTY_ROOT}")
endif()

function(resnet_orin_require_dir dir_path label)
    if(NOT EXISTS "${dir_path}")
        message(FATAL_ERROR "resnet_Orin missing ${label}: ${dir_path}")
    endif()
endfunction()

function(resnet_orin_find_header out_var header_name)
    cmake_parse_arguments(ARG "" "" "HINTS" ${ARGN})
    find_path(${out_var}
        NAMES ${header_name}
        HINTS ${ARG_HINTS}
        NO_DEFAULT_PATH
    )
    if(NOT ${out_var})
        message(FATAL_ERROR "resnet_Orin failed to find header ${header_name} from app-local 3rdparty")
    endif()
endfunction()

function(resnet_orin_find_library out_var lib_name)
    cmake_parse_arguments(ARG "" "" "HINTS" ${ARGN})
    find_library(${out_var}
        NAMES ${lib_name}
        HINTS ${ARG_HINTS}
        NO_DEFAULT_PATH
    )
    if(NOT ${out_var})
        message(FATAL_ERROR "resnet_Orin failed to find library ${lib_name} from app-local 3rdparty")
    endif()
endfunction()

file(GLOB_RECURSE RESNET_ORIN_NODE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*/*.cpp"
)

file(GLOB_RECURSE RESNET_ORIN_LOCAL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/context/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/consumer/*.cpp"
)

add_executable(${RESNET_ORIN_TARGET_NAME}
    resnet_Orin.cpp
    ${RESNET_ORIN_NODE_SOURCES}
    ${RESNET_ORIN_LOCAL_SOURCES}
    ${SRC_DIR}
)

set(RESNET_ORIN_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set(RESNET_ORIN_LINK_LIBS
    ${dynamic_libs}
    dl
)
set(RESNET_ORIN_RUNTIME_LIBS "")

if(RESNET_ORIN_USE_LOCAL_3RDPARTY)
    resnet_orin_require_dir("${RESNET_ORIN_OPENCV_ROOT}" "OpenCV root")
    resnet_orin_require_dir("${RESNET_ORIN_CUDA_ROOT}" "CUDA root")
    resnet_orin_require_dir("${RESNET_ORIN_TENSORRT_ROOT}" "TensorRT root")

    set(RESNET_ORIN_OPENCV_INCLUDE_HINTS
        "${RESNET_ORIN_OPENCV_ROOT}/include"
        "${RESNET_ORIN_OPENCV_ROOT}/include/opencv4"
    )
    set(RESNET_ORIN_OPENCV_LIB_HINTS
        "${RESNET_ORIN_OPENCV_ROOT}/lib"
        "${RESNET_ORIN_OPENCV_ROOT}/lib64"
    )
    set(RESNET_ORIN_CUDA_INCLUDE_HINTS
        "${RESNET_ORIN_CUDA_ROOT}/include"
        "${RESNET_ORIN_CUDA_ROOT}/targets/aarch64-linux/include"
    )
    set(RESNET_ORIN_CUDA_LIB_HINTS
        "${RESNET_ORIN_CUDA_ROOT}/lib64"
        "${RESNET_ORIN_CUDA_ROOT}/targets/aarch64-linux/lib"
    )
    set(RESNET_ORIN_TENSORRT_INCLUDE_HINTS
        "${RESNET_ORIN_TENSORRT_ROOT}/include"
        "${RESNET_ORIN_TENSORRT_ROOT}/include/aarch64-linux-gnu"
    )
    set(RESNET_ORIN_TENSORRT_LIB_HINTS
        "${RESNET_ORIN_TENSORRT_ROOT}/lib"
        "${RESNET_ORIN_TENSORRT_ROOT}/lib64"
    )

    resnet_orin_find_header(RESNET_ORIN_OPENCV_INCLUDE_DIR opencv2/opencv.hpp
        HINTS ${RESNET_ORIN_OPENCV_INCLUDE_HINTS})
    resnet_orin_find_header(RESNET_ORIN_CUDA_INCLUDE_DIR cuda_runtime.h
        HINTS ${RESNET_ORIN_CUDA_INCLUDE_HINTS})
    resnet_orin_find_header(RESNET_ORIN_TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${RESNET_ORIN_TENSORRT_INCLUDE_HINTS})

    resnet_orin_find_library(RESNET_ORIN_OPENCV_CORE_LIB opencv_core
        HINTS ${RESNET_ORIN_OPENCV_LIB_HINTS})
    resnet_orin_find_library(RESNET_ORIN_OPENCV_IMGPROC_LIB opencv_imgproc
        HINTS ${RESNET_ORIN_OPENCV_LIB_HINTS})
    resnet_orin_find_library(RESNET_ORIN_OPENCV_IMGCODECS_LIB opencv_imgcodecs
        HINTS ${RESNET_ORIN_OPENCV_LIB_HINTS})
    resnet_orin_find_library(RESNET_ORIN_CUDART_LIB cudart
        HINTS ${RESNET_ORIN_CUDA_LIB_HINTS})
    resnet_orin_find_library(RESNET_ORIN_NVINFER_LIB nvinfer
        HINTS ${RESNET_ORIN_TENSORRT_LIB_HINTS})
    resnet_orin_find_library(RESNET_ORIN_NVINFER_PLUGIN_LIB nvinfer_plugin
        HINTS ${RESNET_ORIN_TENSORRT_LIB_HINTS})

    list(APPEND RESNET_ORIN_INCLUDE_DIRS
        ${RESNET_ORIN_OPENCV_INCLUDE_DIR}
        ${RESNET_ORIN_CUDA_INCLUDE_DIR}
        ${RESNET_ORIN_TENSORRT_INCLUDE_DIR}
    )
    list(APPEND RESNET_ORIN_LINK_LIBS
        ${RESNET_ORIN_OPENCV_CORE_LIB}
        ${RESNET_ORIN_OPENCV_IMGPROC_LIB}
        ${RESNET_ORIN_OPENCV_IMGCODECS_LIB}
        ${RESNET_ORIN_CUDART_LIB}
        ${RESNET_ORIN_NVINFER_LIB}
        ${RESNET_ORIN_NVINFER_PLUGIN_LIB}
    )
    list(APPEND RESNET_ORIN_RUNTIME_LIBS
        ${RESNET_ORIN_OPENCV_CORE_LIB}
        ${RESNET_ORIN_OPENCV_IMGPROC_LIB}
        ${RESNET_ORIN_OPENCV_IMGCODECS_LIB}
        ${RESNET_ORIN_CUDART_LIB}
        ${RESNET_ORIN_NVINFER_LIB}
        ${RESNET_ORIN_NVINFER_PLUGIN_LIB}
    )
else()
    find_package(OpenCV REQUIRED)
    find_package(CUDA REQUIRED)
    find_library(RESNET_ORIN_NVINFER_LIB
        NAMES nvinfer
        HINTS /usr/lib/aarch64-linux-gnu
    )
    find_library(RESNET_ORIN_NVINFER_PLUGIN_LIB
        NAMES nvinfer_plugin
        HINTS /usr/lib/aarch64-linux-gnu
    )
    if(NOT RESNET_ORIN_NVINFER_LIB OR NOT RESNET_ORIN_NVINFER_PLUGIN_LIB)
        message(FATAL_ERROR "resnet_Orin failed to find TensorRT system libraries")
    endif()

    list(APPEND RESNET_ORIN_INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        /usr/include/aarch64-linux-gnu
    )
    list(APPEND RESNET_ORIN_LINK_LIBS
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${RESNET_ORIN_NVINFER_LIB}
        ${RESNET_ORIN_NVINFER_PLUGIN_LIB}
    )
endif()

target_include_directories(${RESNET_ORIN_TARGET_NAME} PRIVATE
    ${RESNET_ORIN_INCLUDE_DIRS}
)

target_link_libraries(${RESNET_ORIN_TARGET_NAME} PRIVATE
    ${RESNET_ORIN_LINK_LIBS}
)

install(TARGETS ${RESNET_ORIN_TARGET_NAME}
    DESTINATION bin
    COMPONENT ${RESNET_ORIN_INSTALL_COMPONENT}
)

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
    DESTINATION "share/${RESNET_ORIN_TARGET_NAME}"
    COMPONENT ${RESNET_ORIN_INSTALL_COMPONENT}
)

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/chart.svg"
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/timeline_resnet.json"
    DESTINATION "share/${RESNET_ORIN_TARGET_NAME}/assets"
    COMPONENT ${RESNET_ORIN_INSTALL_COMPONENT}
)

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/README.md"
    DESTINATION "share/${RESNET_ORIN_TARGET_NAME}/3rdparty"
    COMPONENT ${RESNET_ORIN_INSTALL_COMPONENT}
)

if(RESNET_ORIN_USE_LOCAL_3RDPARTY AND RESNET_ORIN_RUNTIME_LIBS)
    list(REMOVE_DUPLICATES RESNET_ORIN_RUNTIME_LIBS)
    install(FILES ${RESNET_ORIN_RUNTIME_LIBS}
        DESTINATION lib
        COMPONENT ${RESNET_ORIN_INSTALL_COMPONENT}
    )
endif()

add_custom_target(${RESNET_ORIN_COMPAT_TARGET}
    DEPENDS ${RESNET_ORIN_TARGET_NAME}
)
