cmake_minimum_required(VERSION 3.17)
project(cpp_2020_random_forest)
set(CMAKE_CXX_STANDARD 17)

set(THREADS_PREFER_PTHREAD_FLAG ON)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
#find_package(Boost REQUIRED COMPONENTS algorithm)
find_package(Threads REQUIRED)

add_executable(cpp_2020_random_forest main.cpp)

target_link_libraries(cpp_2020_random_forest ${Boost_LIBRARIES})
target_link_libraries(cpp_2020_random_forest Threads::Threads)
include_directories( ${Boost_INCLUDE_DIRS} )