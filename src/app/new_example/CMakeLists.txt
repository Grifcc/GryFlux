# new_example
add_executable(new_example
    new_example.cpp
    # Context resources
    context/simulated_npu_context.cpp
    context/simulated_tracker_context.cpp
    # Data packets
    packet/simple_data_packet.cpp
    # Task nodes
    nodes/dag/dag_nodes.cpp
    nodes/input/input_node.cpp
    nodes/preprocess/preprocess_node.cpp
    nodes/inference/inference_node.cpp
    nodes/postprocess/postprocess_node.cpp
    nodes/output/output_node.cpp
    # Framework sources
    ${SRC_DIR}
)

# Add example directory as include path (allows #include "packet/xxx.h" style)
target_include_directories(new_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(new_example
    ${dynamic_libs}
)

install(TARGETS new_example DESTINATION bin)
