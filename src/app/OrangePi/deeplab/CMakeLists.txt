 file(GLOB_RECURSE DEEPLAB_NODE_SOURCES CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*/*.cpp"
  )

  file(GLOB_RECURSE DEEPLAB_CONSUMER_SOURCES CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/consumer/*.cpp"
  )

  set(DEEPLAB_MAIN_FILE "")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/go.cpp")
      set(DEEPLAB_MAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/go.cpp")
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deeplab.cpp")
      set(DEEPLAB_MAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/deeplab.cpp")
  endif()

  if(DEEPLAB_MAIN_FILE STREQUAL "")
      message(STATUS "go.cpp/deeplab.cpp not found under src/app/OrangePi/deeplab, skip building target 'deeplab'")
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
          "Ascend CANN ACL not found, skip building target 'deeplab'. "
          "Need acl/acl.h and libascendcl.so under ${ASCEND_ROOT} or /usr/local/Ascend/nnrt/latest"
      )
      return()
  endif()

  get_filename_component(ASCENDCL_LIBRARY_DIR ${ASCENDCL_LIBRARY} DIRECTORY)

  add_executable(deeplab
      ${DEEPLAB_MAIN_FILE}
      ${DEEPLAB_NODE_SOURCES}
      ${DEEPLAB_CONSUMER_SOURCES}
      ${SRC_DIR}
  )

  target_include_directories(deeplab PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${OpenCV_INCLUDE_DIRS}
      ${ASCEND_ACL_INCLUDE_DIR}
  )

  target_link_directories(deeplab PRIVATE
      ${ASCENDCL_LIBRARY_DIR}
  )

  target_link_libraries(deeplab PRIVATE
      ${dynamic_libs}
      ${OpenCV_LIBS}
      ${ASCENDCL_LIBRARY}
  )

  install(TARGETS deeplab DESTINATION bin)
