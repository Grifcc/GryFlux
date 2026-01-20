# example
add_executable(example
    example.cpp
    nodes/dag_nodes.cpp
    # Framework sources
    ${SRC_DIR}
)

# Add example directory as include path (allows #include "packet/xxx.h" style)
target_include_directories(example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(example
    ${dynamic_libs}
)

install(TARGETS example DESTINATION bin)
