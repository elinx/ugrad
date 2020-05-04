cmake_minimum_required(VERSION 3.13)

set(CMAKE_USER_MAKE_RULES_OVERRIDE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/c_flag_overrides.cmake)
set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cxx_flag_overrides.cmake)

project(ugrad)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(dependency)

add_library(ugrad include/engine.hpp)
target_include_directories(ugrad PUBLIC include)
set_target_properties(ugrad PROPERTIES LINKER_LANGUAGE CXX)

add_subdirectory(examples)
add_subdirectory(tests)
