cmake_minimum_required(VERSION 3.16.0)
project(GryFlux VERSION 1.0.0 LANGUAGES CXX C)

message(STATUS "Graph profiling build switch: compile with -DGRYFLUX_BUILD_PROFILING=1 (or 0) and rebuild")

# Set common compile warning suppressions and linking options
function(set_common_compile_flags)
    set(common_flags "-Wl,--allow-shlib-undefined -Wno-class-memaccess -Wno-deprecated-declarations -Wno-sign-compare")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${common_flags}" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${common_flags}" PARENT_SCOPE)
endfunction()

# Set optimization level and debug flags based on build type
function(set_build_type_flags)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-Wall -O0 -g)
        # add_definitions(-DDEBUG)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-Wall -O3)
        add_definitions(-DNDEBUG)
    else()
        add_compile_options(-Wall -O0 -g)
        # add_definitions(-DDEBUG)
    endif()
    
    set(CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE} "${CMAKE_C_FLAGS} ${compile_flags}" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE} "${CMAKE_CXX_FLAGS} ${compile_flags}" PARENT_SCOPE)
endfunction()

# C++ Standard Settings
set(CXX_STD "17" CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD ${CXX_STD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Unified output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

# Build options
option(BUILD_EXAMPLE "Build the example application" ON)

# Apply common compile flags
set_common_compile_flags()

# Set build type flags
set_build_type_flags()

# Runtime Path Configuration
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_BUILD_RPATH "$ORIGIN/../lib")

# Include directories setup
set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR}/include)

add_library(project_includes INTERFACE)
target_include_directories(project_includes INTERFACE
    ${COMMON_HEADER_DIRS}
)

# Dynamic libraries setup
set(dynamic_libs 
  pthread 
  project_includes
)

# Link filesystem library for GCC < 9
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    list(APPEND dynamic_libs stdc++fs)
    message(STATUS "Adding stdc++fs library for GCC ${CMAKE_CXX_COMPILER_VERSION}")
endif()

# Source files collection
aux_source_directory(${PROJECT_SOURCE_DIR}/src/common SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/framework SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/utils SRC_DIR)

# Add subdirectories
add_subdirectory(src/app)

if(BUILD_TEST)
    add_subdirectory(src/tests)
endif()

# Installation Configuration
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/install)
endif()

message(STATUS "Build configuration complete")
