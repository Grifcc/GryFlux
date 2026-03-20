# realesrgan app
# Default mode aligns with tracker: EasyStream 3rdparty + aarch64 cross-build.
option(REALESRGAN_USE_EASYSTREAM_3RDPARTY "Use EasyStream bundled OpenCV/RKNN/Eigen" ON)

file(GLOB_RECURSE REALESRGAN_NODE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*/*.cpp"
)

file(GLOB_RECURSE REALESRGAN_APP_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/context/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/consumer/*.cpp"
)

add_executable(realesrgan
    realesrgan.cpp
    ${REALESRGAN_NODE_SOURCES}
    ${REALESRGAN_APP_SOURCES}
    ${SRC_DIR}
)

target_include_directories(realesrgan PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(realesrgan
    ${dynamic_libs}
    dl
)

if(REALESRGAN_USE_EASYSTREAM_3RDPARTY)
    set(EASYSTREAM_ROOT "${CMAKE_SOURCE_DIR}/../EasyStream" CACHE PATH "EasyStream repo root (for 3rdparty deps)")

    set(OPENCV_INCLUDE_DIR "${EASYSTREAM_ROOT}/3rdparty/opencv/include")
    set(OPENCV_INCLUDE_DIR_OPENCV4 "${OPENCV_INCLUDE_DIR}/opencv4")
    set(OPENCV_LIB_DIR "${EASYSTREAM_ROOT}/3rdparty/opencv/lib")

    set(RKNN_INCLUDE_DIR "${EASYSTREAM_ROOT}/3rdparty/librknn_api/include")
    set(RKNN_LIB_DIR "${EASYSTREAM_ROOT}/3rdparty/librknn_api/aarch64")

    set(EIGEN_INCLUDE_DIR "${EASYSTREAM_ROOT}/3rdparty/Eigen")

    if(NOT EXISTS "${OPENCV_INCLUDE_DIR_OPENCV4}")
        message(FATAL_ERROR "OpenCV include dir not found: ${OPENCV_INCLUDE_DIR_OPENCV4}")
    endif()
    if(NOT EXISTS "${OPENCV_LIB_DIR}")
        message(FATAL_ERROR "OpenCV lib dir not found: ${OPENCV_LIB_DIR}")
    endif()
    if(NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
        message(FATAL_ERROR "RKNN header not found: ${RKNN_INCLUDE_DIR}/rknn_api.h")
    endif()
    if(NOT EXISTS "${RKNN_LIB_DIR}")
        message(FATAL_ERROR "RKNN lib dir not found: ${RKNN_LIB_DIR}")
    endif()
    if(NOT EXISTS "${EIGEN_INCLUDE_DIR}")
        message(FATAL_ERROR "Eigen include dir not found: ${EIGEN_INCLUDE_DIR}")
    endif()

    target_include_directories(realesrgan PRIVATE
        ${OPENCV_INCLUDE_DIR_OPENCV4}
        ${RKNN_INCLUDE_DIR}
        ${EIGEN_INCLUDE_DIR}
    )

    target_link_directories(realesrgan PRIVATE
        ${OPENCV_LIB_DIR}
        ${RKNN_LIB_DIR}
    )

    target_link_libraries(realesrgan
        rknnrt
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_videoio
    )
else()
    if(NOT RKNN_INCLUDE_DIR)
        message(FATAL_ERROR "RKNN_INCLUDE_DIR is required in system mode. Example: -DRKNN_INCLUDE_DIR=/path/to/librknn_api/include")
    endif()

    find_package(OpenCV REQUIRED)
    set(RKNN_LIBRARY "/usr/lib/librknnrt.so" CACHE FILEPATH "Path to librknnrt library")

    target_include_directories(realesrgan PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${RKNN_INCLUDE_DIR}
    )

    target_link_libraries(realesrgan
        ${OpenCV_LIBS}
        ${RKNN_LIBRARY}
    )
endif()

install(TARGETS realesrgan DESTINATION bin)
