# example
# Collect all node implementations under nodes/ automatically.
# NOTE: GLOB is convenient for samples; CONFIGURE_DEPENDS makes CMake re-scan on file add/remove.
file(GLOB_RECURSE EXAMPLE_NODE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*/*.cpp"
)

# Exclude legacy monolithic implementation (kept for reference only).
list(FILTER EXAMPLE_NODE_SOURCES EXCLUDE REGEX "/nodes/dag_nodes\\.cpp$")

add_executable(example
    example.cpp
    ${EXAMPLE_NODE_SOURCES}
    # Framework sources
    ${SRC_DIR}
)

# Add example directory as include path (allows #include "packet/xxx.h" style)
target_include_directories(example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(example
    ${dynamic_libs}
)

install(TARGETS example DESTINATION bin)
