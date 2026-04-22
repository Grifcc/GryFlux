file(GLOB_RECURSE REALESRGAN_NODE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*/*.cpp"
)

file(GLOB_RECURSE REALESRGAN_CONSUMER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/consumer/*.cpp"
)

file(GLOB_RECURSE REALESRGAN_CONTEXT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/context/*.cpp"
)

set(REALESRGAN_MAIN_FILE "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    set(REALESRGAN_MAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/realesrgan.cpp")
    set(REALESRGAN_MAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/realesrgan.cpp")
endif()

if(REALESRGAN_MAIN_FILE STREQUAL "")
    message(STATUS "main.cpp/realesrgan.cpp not found under src/app/super-resolution_realesrgan_Ascend, skip building target 'realesrgan'")
    return()
endif()

find_package(OpenCV REQUIRED)

set(ASCEND_ROOT "/usr/local/Ascend/ascend-toolkit/latest")
set(ASCEND_ARCH_DIR "")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(ASCEND_ARCH_DIR "aarch64-linux")
else()
    set(ASCEND_ARCH_DIR "x86_64-linux")
endif()

find_path(ASCEND_ACL_INCLUDE_DIR
    NAMES acl/acl.h
    PATHS
        ${ASCEND_ROOT}/${ASCEND_ARCH_DIR}/include
        ${ASCEND_ROOT}/runtime/include
        /usr/local/Ascend/nnrt/latest/${ASCEND_ARCH_DIR}/include
        /usr/local/Ascend/nnrt/latest/runtime/include
)

find_library(ASCENDCL_LIBRARY
    NAMES ascendcl
    PATHS
        ${ASCEND_ROOT}/${ASCEND_ARCH_DIR}/lib64
        ${ASCEND_ROOT}/runtime/lib64
        /usr/local/Ascend/nnrt/latest/${ASCEND_ARCH_DIR}/lib64
        /usr/local/Ascend/nnrt/latest/runtime/lib64
)

if(NOT ASCEND_ACL_INCLUDE_DIR OR NOT ASCENDCL_LIBRARY)
    message(WARNING
        "Ascend CANN ACL not found, skip building target 'realesrgan'. "
        "Need acl/acl.h and libascendcl.so under ${ASCEND_ROOT} or /usr/local/Ascend/nnrt/latest"
    )
    return()
endif()

get_filename_component(ASCENDCL_LIBRARY_DIR ${ASCENDCL_LIBRARY} DIRECTORY)

add_executable(realesrgan
    ${REALESRGAN_MAIN_FILE}
    ${REALESRGAN_NODE_SOURCES}
    ${REALESRGAN_CONSUMER_SOURCES}
    ${REALESRGAN_CONTEXT_SOURCES}
    ${SRC_DIR}
)

target_include_directories(realesrgan PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${ASCEND_ACL_INCLUDE_DIR}
)

target_link_directories(realesrgan PRIVATE
    ${ASCENDCL_LIBRARY_DIR}
)

target_link_libraries(realesrgan PRIVATE
    ${dynamic_libs}
    ${OpenCV_LIBS}
    ${ASCENDCL_LIBRARY}
)

install(TARGETS realesrgan DESTINATION bin)
