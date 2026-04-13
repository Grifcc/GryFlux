cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(yolox_detection_310p LANGUAGES CXX C)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    get_filename_component(GRYFLUX_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

    add_library(project_includes INTERFACE)
    target_include_directories(project_includes INTERFACE
        ${GRYFLUX_ROOT_DIR}/include
    )

    set(dynamic_libs
        pthread
        project_includes
    )

    aux_source_directory(${GRYFLUX_ROOT_DIR}/src/framework SRC_DIR)
    aux_source_directory(${GRYFLUX_ROOT_DIR}/src/utils SRC_DIR)
endif()

set(APP_NAME yolox_detection_310p)
set(APP_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(APP_3RDPARTY_DIR ${APP_ROOT_DIR}/3rdparty)

set(YOLOX_310P_OPENCV_ROOT
    "${APP_3RDPARTY_DIR}/opencv"
    CACHE PATH "Root directory or config directory for app-local OpenCV")
set(YOLOX_310P_ASCEND_ROOT
    "${APP_3RDPARTY_DIR}/ascend-toolkit/latest"
    CACHE PATH "Ascend Toolkit root for yolox_detection_310p")

if(EXISTS "${YOLOX_310P_OPENCV_ROOT}")
    find_package(OpenCV CONFIG REQUIRED
        PATHS
            ${YOLOX_310P_OPENCV_ROOT}
            ${YOLOX_310P_OPENCV_ROOT}/lib/cmake/opencv4
            ${YOLOX_310P_OPENCV_ROOT}/lib64/cmake/opencv4
        NO_DEFAULT_PATH
    )
else()
    find_package(OpenCV REQUIRED)
endif()

find_path(ASCEND_ACL_INCLUDE_DIR acl/acl.h
    HINTS
        ${YOLOX_310P_ASCEND_ROOT}/include
        /usr/local/Ascend/ascend-toolkit/latest/include
)

find_library(ASCEND_ACL_LIB ascendcl
    HINTS
        ${YOLOX_310P_ASCEND_ROOT}/lib64
        /usr/local/Ascend/ascend-toolkit/latest/lib64
)

set(ASCEND_RUNTIME_DIRS
    ${YOLOX_310P_ASCEND_ROOT}/lib64
    ${YOLOX_310P_ASCEND_ROOT}/x86_64-linux/lib64
    ${YOLOX_310P_ASCEND_ROOT}/aarch64-linux/lib64
    /usr/local/Ascend/ascend-toolkit/latest/lib64
    /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/lib64
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64
)
list(REMOVE_DUPLICATES ASCEND_RUNTIME_DIRS)

if(NOT ASCEND_ACL_INCLUDE_DIR OR NOT ASCEND_ACL_LIB)
    message(FATAL_ERROR
        "Ascend ACL was not found for yolox_detection_310p. "
        "Populate src/app/yolox_detection_310p/3rdparty/ascend-toolkit/latest or pass "
        "-DYOLOX_310P_ASCEND_ROOT=<Ascend Toolkit root>.")
endif()

set(APP_SOURCES
    consumer/result_consumer.cpp
    context/infercontext.cpp
    source/image_data_source.cpp
    yolox_detection_310p.cpp
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
    ${ASCEND_ACL_INCLUDE_DIR}
)

target_link_libraries(${APP_NAME} PRIVATE
    ${dynamic_libs}
    ${OpenCV_LIBS}
    ${ASCEND_ACL_LIB}
)

target_link_options(${APP_NAME} PRIVATE
    -Wl,--disable-new-dtags
)

set_target_properties(${APP_NAME} PROPERTIES
    BUILD_RPATH "${ASCEND_RUNTIME_DIRS}"
    INSTALL_RPATH "${ASCEND_RUNTIME_DIRS}"
)

install(TARGETS ${APP_NAME} DESTINATION bin)
