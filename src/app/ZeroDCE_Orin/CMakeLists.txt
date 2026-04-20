set(ZERODCE_ORIN_TARGET_NAME zero_dce_orin)
set(ZERODCE_ORIN_INSTALL_COMPONENT ZeroDCE_Orin)

file(GLOB_RECURSE ZERODCE_ORIN_NODE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*/*.cpp"
)

file(GLOB_RECURSE ZERODCE_ORIN_LOCAL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/context/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/consumer/*.cpp"
)

add_executable(${ZERODCE_ORIN_TARGET_NAME}
    ZeroDCE_Orin.cpp
    ${ZERODCE_ORIN_NODE_SOURCES}
    ${ZERODCE_ORIN_LOCAL_SOURCES}
    ${SRC_DIR}
)

set_target_properties(${ZERODCE_ORIN_TARGET_NAME} PROPERTIES
    OUTPUT_NAME zero_dce_app
)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

find_library(ZERODCE_ORIN_NVINFER_LIB
    NAMES nvinfer
    HINTS /usr/lib/aarch64-linux-gnu
)
find_library(ZERODCE_ORIN_NVINFER_PLUGIN_LIB
    NAMES nvinfer_plugin
    HINTS /usr/lib/aarch64-linux-gnu
)

if(NOT ZERODCE_ORIN_NVINFER_LIB OR NOT ZERODCE_ORIN_NVINFER_PLUGIN_LIB)
    message(FATAL_ERROR "ZeroDCE_Orin failed to find TensorRT system libraries")
endif()

target_include_directories(${ZERODCE_ORIN_TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    /usr/include/aarch64-linux-gnu
)

target_link_libraries(${ZERODCE_ORIN_TARGET_NAME} PRIVATE
    ${dynamic_libs}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${ZERODCE_ORIN_NVINFER_LIB}
    ${ZERODCE_ORIN_NVINFER_PLUGIN_LIB}
    dl
)

install(TARGETS ${ZERODCE_ORIN_TARGET_NAME}
    DESTINATION bin
    COMPONENT ${ZERODCE_ORIN_INSTALL_COMPONENT}
)
