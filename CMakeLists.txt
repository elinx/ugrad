cmake_minimum_required(VERSION 3.13)

project(ugrad)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(ugrad include/engine.hpp src/engine.cpp)
target_include_directories(ugrad PUBLIC include)
add_subdirectory(examples)

