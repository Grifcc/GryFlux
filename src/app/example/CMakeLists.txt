# Parallel Calc example
add_executable(example
    example.cpp
    context/adder_context.cpp
    context/multiplier_context.cpp
    packet/calc_packet.cpp
    nodes/input/input_node.cpp
    nodes/add/add_nodes.cpp
    nodes/mul/mul_nodes.cpp
    nodes/output/output_node.cpp
    ${SRC_DIR}
)

target_include_directories(example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(example
    ${dynamic_libs}
)

install(TARGETS example DESTINATION bin)
