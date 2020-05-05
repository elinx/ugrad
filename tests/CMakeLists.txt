add_executable(value_test value_test.cpp)
target_link_libraries(value_test ugrad gtest_main)
add_test(NAME value_test COMMAND value_test)

add_executable(nn_test nn_test.cpp)
target_link_libraries(nn_test ugrad gtest_main)
add_test(NAME nn_test COMMAND nn_test)
