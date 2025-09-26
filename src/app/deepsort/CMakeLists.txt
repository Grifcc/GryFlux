cmake_minimum_required(VERSION 3.16)

add_library(deepsort_includes INTERFACE)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(RKNN_RUNTIME_LIB /usr/lib/librknnrt.so CACHE FILEPATH "Path to rknn runtime shared library")

set(APP_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/package
    ${CMAKE_CURRENT_SOURCE_DIR}/tasks
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/deepsort/include
)

set(YOLOX_SHARED_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../yolox
    ${CMAKE_CURRENT_SOURCE_DIR}/../yolox/source
)

target_include_directories(deepsort_includes INTERFACE
    ${APP_INCLUDE_DIRS}
    ${YOLOX_SHARED_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../yolox/runtime
)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../yolox/source/producer PRODUCER_SOURCES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../yolox/tasks/image_preprocess PREPROCESS_SOURCES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../yolox/tasks/object_detector DETECTOR_SOURCES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../yolox/tasks/rk_runner RK_RUNNER_SOURCES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/deepsort_tracker TRACKER_TASK_SOURCES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/track_result_sender TRACK_RESULT_SOURCES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/sink/write_consumer CONSUMER_SOURCES)
set(RUNTIME_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/rknn_fp.cpp
)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/modules/deepsort/src DEEPSORT_SOURCES)

add_executable(deepsort_stream
    deepsort_stream.cpp
    ${SRC_DIR}
    ${PRODUCER_SOURCES}
    ${PREPROCESS_SOURCES}
    ${DETECTOR_SOURCES}
    ${RK_RUNNER_SOURCES}
    ${TRACKER_TASK_SOURCES}
    ${TRACK_RESULT_SOURCES}
    ${CONSUMER_SOURCES}
    ${RUNTIME_SOURCES}
    ${DEEPSORT_SOURCES}
)

target_include_directories(deepsort_stream PRIVATE
    ${APP_INCLUDE_DIRS}
    ${YOLOX_SHARED_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../yolox/runtime
)

target_link_libraries(deepsort_stream
    project_includes
    deepsort_includes
    ${dynamic_libs}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${RKNN_RUNTIME_LIB}
)

target_compile_definitions(deepsort_stream PRIVATE
    EIGEN_MPL2_ONLY
)

install(TARGETS deepsort_stream RUNTIME DESTINATION ./)
