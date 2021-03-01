cmake_minimum_required(VERSION 3.17)
project(cpp_2020_random_forest)
set(CMAKE_CXX_STANDARD 17)

set(THREADS_PREFER_PTHREAD_FLAG ON)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost REQUIRED)
find_package(Threads REQUIRED)

add_executable(
        cpp_2020_random_forest
        main.cpp

        src/decision_tree.cpp
        src/decision_tree.h

        src/data.cpp
        src/data.h

        src/metrics.cpp
        src/metrics.h

        src/time_measurement.h
        src/random_forest.cpp src/random_forest.h)

target_link_libraries(cpp_2020_random_forest ${Boost_LIBRARIES})
target_link_libraries(cpp_2020_random_forest Threads::Threads)
include_directories( ${Boost_INCLUDE_DIRS} )

add_executable(sandbox sandbox.cpp)

