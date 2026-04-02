# fusionnetv2 app
# Use app-local third-party dependencies only.

set(_FUSIONNETV2_3RDPARTY_CANDIDATES
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty"

)

set(_FUSIONNETV2_3RDPARTY_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")
foreach(_candidate IN LISTS _FUSIONNETV2_3RDPARTY_CANDIDATES)
    if(EXISTS "${_candidate}")
        set(_FUSIONNETV2_3RDPARTY_DEFAULT "${_candidate}")
        break()
    endif()
endforeach()

set(FUSIONNETV2_3RDPARTY_ROOT "${_FUSIONNETV2_3RDPARTY_DEFAULT}" CACHE PATH "FusionNetV2 app-local 3rdparty root")
message(STATUS "FusionNetV2 3rdparty root: ${FUSIONNETV2_3RDPARTY_ROOT}")

file(GLOB_RECURSE FUSIONNETV2_NODE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*/*.cpp"
)

file(GLOB_RECURSE FUSIONNETV2_APP_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/context/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/consumer/*.cpp"
)

add_executable(fusionnetv2
    fusionnetv2.cpp
    ${FUSIONNETV2_NODE_SOURCES}
    ${FUSIONNETV2_APP_SOURCES}
    ${SRC_DIR}
)

target_include_directories(fusionnetv2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(fusionnetv2
    ${dynamic_libs}
    dl
)

set(OPENCV_INCLUDE_DIR "${FUSIONNETV2_3RDPARTY_ROOT}/opencv/include")
set(OPENCV_INCLUDE_DIR_OPENCV4 "${OPENCV_INCLUDE_DIR}/opencv4")
set(OPENCV_LIB_DIR "${FUSIONNETV2_3RDPARTY_ROOT}/opencv/lib")

set(RKNN_INCLUDE_DIR "${FUSIONNETV2_3RDPARTY_ROOT}/librknn_api/include")
set(RKNN_LIB_DIR "${FUSIONNETV2_3RDPARTY_ROOT}/librknn_api/aarch64")
set(RKNN_RUNTIME_LIB "${RKNN_LIB_DIR}/librknnrt.so")

set(OPENCV_CORE_LIB "${OPENCV_LIB_DIR}/libopencv_core.so")
set(OPENCV_IMGPROC_LIB "${OPENCV_LIB_DIR}/libopencv_imgproc.so")
set(OPENCV_IMGCODECS_LIB "${OPENCV_LIB_DIR}/libopencv_imgcodecs.so")

if(NOT EXISTS "${OPENCV_INCLUDE_DIR_OPENCV4}")
    message(FATAL_ERROR "OpenCV include dir not found: ${OPENCV_INCLUDE_DIR_OPENCV4}")
endif()
if(NOT EXISTS "${OPENCV_LIB_DIR}")
    message(FATAL_ERROR "OpenCV lib dir not found: ${OPENCV_LIB_DIR}")
endif()
if(NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
    message(FATAL_ERROR "RKNN header not found: ${RKNN_INCLUDE_DIR}/rknn_api.h")
endif()
if(NOT EXISTS "${RKNN_RUNTIME_LIB}")
    message(FATAL_ERROR "RKNN runtime not found: ${RKNN_RUNTIME_LIB}")
endif()
if(NOT EXISTS "${OPENCV_CORE_LIB}")
    message(FATAL_ERROR "OpenCV runtime lib not found: ${OPENCV_CORE_LIB}")
endif()
if(NOT EXISTS "${OPENCV_IMGPROC_LIB}")
    message(FATAL_ERROR "OpenCV runtime lib not found: ${OPENCV_IMGPROC_LIB}")
endif()
if(NOT EXISTS "${OPENCV_IMGCODECS_LIB}")
    message(FATAL_ERROR "OpenCV runtime lib not found: ${OPENCV_IMGCODECS_LIB}")
endif()

target_include_directories(fusionnetv2 PRIVATE
    ${OPENCV_INCLUDE_DIR_OPENCV4}
    ${RKNN_INCLUDE_DIR}
)

target_link_directories(fusionnetv2 PRIVATE
    ${OPENCV_LIB_DIR}
    ${RKNN_LIB_DIR}
)

set_target_properties(fusionnetv2 PROPERTIES
    BUILD_RPATH "${OPENCV_LIB_DIR};${RKNN_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN/../lib"
)

target_link_libraries(fusionnetv2
    "${RKNN_RUNTIME_LIB}"
    "${OPENCV_CORE_LIB}"
    "${OPENCV_IMGPROC_LIB}"
    "${OPENCV_IMGCODECS_LIB}"
)

set(OPENCV_RUNTIME_LIBS)
foreach(OPENCV_SO_NAME IN ITEMS
    libopencv_core.so
    libopencv_imgproc.so
    libopencv_imgcodecs.so
)
    if(EXISTS "${OPENCV_LIB_DIR}/${OPENCV_SO_NAME}")
        list(APPEND OPENCV_RUNTIME_LIBS "${OPENCV_LIB_DIR}/${OPENCV_SO_NAME}")
    endif()
    file(GLOB OPENCV_VERSIONED_MATCHES "${OPENCV_LIB_DIR}/${OPENCV_SO_NAME}.*")
    if(OPENCV_VERSIONED_MATCHES)
        list(APPEND OPENCV_RUNTIME_LIBS ${OPENCV_VERSIONED_MATCHES})
    endif()
endforeach()
list(REMOVE_DUPLICATES OPENCV_RUNTIME_LIBS)

install(FILES
    "${RKNN_RUNTIME_LIB}"
    ${OPENCV_RUNTIME_LIBS}
    DESTINATION lib
)

install(TARGETS fusionnetv2 DESTINATION bin)
