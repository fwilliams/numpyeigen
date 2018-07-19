cmake_minimum_required(VERSION 2.8.12)
project(cmake_example)

make_module(test_module ${TEST_DIRECTORY}/test_module.cpp
BINDING_SOURCES ${TEST_DIRECTORY}/mutate_matrix.cpp ${TEST_DIRECTORY}/matrix_add.cpp)
target_link_libraries(test_module PRIVATE igl::core)
#target_include_directories(test_module PRIVATE ${SRC_DIRECTORY})

add_test(NAME test_nocopy
         COMMAND python ${TEST_DIRECTORY}/test_nocopy.py)
