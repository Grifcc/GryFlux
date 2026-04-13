cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(image_fusion_310p LANGUAGES CXX C)
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

set(APP_NAME image_fusion_310p)
set(APP_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(APP_3RDPARTY_DIR ${APP_ROOT_DIR}/3rdparty)

set(IMAGE_FUSION_310P_OPENCV_ROOT
    "${APP_3RDPARTY_DIR}/opencv"
    CACHE PATH "Root directory or config directory for app-local OpenCV")
set(IMAGE_FUSION_310P_ASCEND_ROOT
    "${APP_3RDPARTY_DIR}/ascend-toolkit/latest"
    CACHE PATH "Ascend Toolkit root for image_fusion_310p")

if(EXISTS "${IMAGE_FUSION_310P_OPENCV_ROOT}")
    find_package(OpenCV CONFIG REQUIRED
        PATHS
            ${IMAGE_FUSION_310P_OPENCV_ROOT}
            ${IMAGE_FUSION_310P_OPENCV_ROOT}/lib/cmake/opencv4
            ${IMAGE_FUSION_310P_OPENCV_ROOT}/lib64/cmake/opencv4
        NO_DEFAULT_PATH
    )
else()
    find_package(OpenCV REQUIRED)
endif()

find_path(ASCEND_ACL_INCLUDE_DIR acl/acl.h
    HINTS
        ${IMAGE_FUSION_310P_ASCEND_ROOT}/include
        /usr/local/Ascend/ascend-toolkit/latest/include
)

find_library(ASCEND_ACL_LIB ascendcl
    HINTS
        ${IMAGE_FUSION_310P_ASCEND_ROOT}/lib64
        /usr/local/Ascend/ascend-toolkit/latest/lib64
)

if(NOT ASCEND_ACL_INCLUDE_DIR OR NOT ASCEND_ACL_LIB)
    message(FATAL_ERROR
        "Ascend ACL was not found for image_fusion_310p. "
        "Populate src/app/image_fusion_310p/3rdparty/ascend-toolkit/latest or pass "
        "-DIMAGE_FUSION_310P_ASCEND_ROOT=<Ascend Toolkit root>.")
endif()

set(ASCEND_RUNTIME_DIRS
    ${IMAGE_FUSION_310P_ASCEND_ROOT}/lib64
    ${IMAGE_FUSION_310P_ASCEND_ROOT}/x86_64-linux/lib64
    ${IMAGE_FUSION_310P_ASCEND_ROOT}/aarch64-linux/lib64
    /usr/local/Ascend/ascend-toolkit/latest/lib64
    /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/lib64
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64
)
list(REMOVE_DUPLICATES ASCEND_RUNTIME_DIRS)

set(APP_SOURCES
    consumer/result_consumer.cpp
    context/infercontext.cpp
    source/fusion_data_source.cpp
    image_fusion_310p.cpp
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
