cmake_minimum_required(VERSION 2.8.12)
project(cmake_example)

pybind11_add_module(test_module SHARED test_module.cpp)
target_link_libraries(test_module PRIVATE igl::core)
add_custom_target(run_binding_generator
  DEPENDS ${TEST_DIRECTORY}/mutate_matrix.cpp
  COMMAND python ${SRC_DIRECTORY}/binding_dsl.py ${TEST_DIRECTORY}/mutate_matrix.cpp -o ${TEST_DIRECTORY}/mutate_matrix.out.cpp)
add_dependencies(test_module run_binding_generator)

target_include_directories(test_module PRIVATE ${SRC_DIRECTORY})

add_test(NAME test_nocopy
         COMMAND python ${TEST_DIRECTORY}/test_nocopy.py)
