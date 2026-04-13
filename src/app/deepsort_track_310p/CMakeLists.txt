cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(deepsort_track_310p LANGUAGES CXX C)
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

set(APP_NAME deepsort_track_310p)
set(APP_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(APP_3RDPARTY_DIR ${APP_ROOT_DIR}/3rdparty)

set(DEEPSORT_310P_OPENCV_ROOT
    "${APP_3RDPARTY_DIR}/opencv"
    CACHE PATH "Root directory or config directory for app-local OpenCV")
set(DEEPSORT_310P_ASCEND_ROOT
    "${APP_3RDPARTY_DIR}/ascend-toolkit/latest"
    CACHE PATH "Ascend Toolkit root for deepsort_track_310p")
set(DEEPSORT_310P_EIGEN_ROOT
    "${APP_3RDPARTY_DIR}/Eigen"
    CACHE PATH "Eigen root for deepsort_track_310p")

if(EXISTS "${DEEPSORT_310P_OPENCV_ROOT}")
    find_package(OpenCV CONFIG REQUIRED
        PATHS
            ${DEEPSORT_310P_OPENCV_ROOT}
            ${DEEPSORT_310P_OPENCV_ROOT}/lib/cmake/opencv4
            ${DEEPSORT_310P_OPENCV_ROOT}/lib64/cmake/opencv4
        NO_DEFAULT_PATH
    )
else()
    find_package(OpenCV REQUIRED)
endif()

find_package(Eigen3 QUIET CONFIG
    PATHS
        ${DEEPSORT_310P_EIGEN_ROOT}
        ${DEEPSORT_310P_EIGEN_ROOT}/share/eigen3/cmake
        ${DEEPSORT_310P_EIGEN_ROOT}/lib/cmake/eigen3
        ${DEEPSORT_310P_EIGEN_ROOT}/lib64/cmake/eigen3
    NO_DEFAULT_PATH
)

if(NOT Eigen3_FOUND)
    find_package(Eigen3 REQUIRED)
endif()

find_path(ASCEND_ACL_INCLUDE_DIR acl/acl.h
    HINTS
        ${DEEPSORT_310P_ASCEND_ROOT}/include
        /usr/local/Ascend/ascend-toolkit/latest/include
)

find_library(ASCEND_ACL_LIB ascendcl
    HINTS
        ${DEEPSORT_310P_ASCEND_ROOT}/lib64
        /usr/local/Ascend/ascend-toolkit/latest/lib64
)

if(NOT ASCEND_ACL_INCLUDE_DIR OR NOT ASCEND_ACL_LIB)
    message(FATAL_ERROR
        "Ascend ACL was not found for deepsort_track_310p. "
        "Populate src/app/deepsort_track_310p/3rdparty/ascend-toolkit/latest or pass "
        "-DDEEPSORT_310P_ASCEND_ROOT=<Ascend Toolkit root>.")
endif()

set(ASCEND_RUNTIME_DIRS
    ${DEEPSORT_310P_ASCEND_ROOT}/lib64
    ${DEEPSORT_310P_ASCEND_ROOT}/x86_64-linux/lib64
    ${DEEPSORT_310P_ASCEND_ROOT}/aarch64-linux/lib64
    /usr/local/Ascend/ascend-toolkit/latest/lib64
    /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/lib64
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64
)
list(REMOVE_DUPLICATES ASCEND_RUNTIME_DIRS)

set(APP_SOURCES
    consumer/result_consumer.cpp
    context/infercontext.cpp
    context/reid_context.cpp
    source/image_data_source.cpp
    deepsort_track_310p.cpp
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
    ${ASCEND_ACL_INCLUDE_DIR}
)

target_link_libraries(${APP_NAME} PRIVATE
    ${dynamic_libs}
    ${OpenCV_LIBS}
    ${ASCEND_ACL_LIB}
)

if(TARGET Eigen3::Eigen)
    target_link_libraries(${APP_NAME} PRIVATE Eigen3::Eigen)
else()
    target_include_directories(${APP_NAME} PRIVATE ${EIGEN3_INCLUDE_DIRS})
endif()

target_link_options(${APP_NAME} PRIVATE
    -Wl,--disable-new-dtags
)

set_target_properties(${APP_NAME} PROPERTIES
    BUILD_RPATH "${ASCEND_RUNTIME_DIRS}"
    INSTALL_RPATH "${ASCEND_RUNTIME_DIRS}"
)

install(TARGETS ${APP_NAME} DESTINATION bin)
