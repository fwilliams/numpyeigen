cmake_minimum_required(VERSION 2.8.12)
project(cmake_example)

# A module for writing bindings with our framework
npe_add_module(numpyeigen_test
  BINDING_SOURCES mutate_matrix.cpp matrix_add.cpp binding_example.cpp)
target_link_libraries(numpyeigen_test PRIVATE igl::core)

# A vanilla pybind11 module used to write functions to compare against pyeigen
pybind11_add_module(numpyeigen_helpers SHARED pybind11_helper_module.cpp)
target_link_libraries(numpyeigen_helpers PRIVATE igl::core)

add_test(NAME test_nocopy
         COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_nocopy.py)
