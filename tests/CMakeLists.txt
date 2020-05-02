add_executable(value_test value_test.cpp)
target_link_libraries(value_test gtest_main)
add_test(NAME value_test COMMAND value_test)