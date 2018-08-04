cmake_minimum_required(VERSION 2.8.12)
project(cmake_example)

SET(CMAKE_CXX_FLAGS  "-Wno-deprecated-declarations" )

# A module for writing bindings with our framework
npe_add_module(numpyeigen_test
  BINDING_SOURCES
    mutate_matrix.cpp
    matrix_add.cpp
    binding_example.cpp
    sparse_matrix_add.cpp
    sparse_matrix_passthru.cpp
    mutate_sparse_matrix.cpp
    default_arg.cpp)
target_link_libraries(numpyeigen_test PRIVATE igl::core)

# A vanilla pybind11 module used to write functions to compare against pyeigen
execute_process(COMMAND python -c "import numpy as np;import sys;sys.stdout.write(np.get_include())"
  OUTPUT_VARIABLE NP_INCLUDE_DIR)
execute_process(
  COMMAND
    python -c "from sysconfig import get_paths;import sys;sys.stdout.write(get_paths()['include'])"
  OUTPUT_VARIABLE PYTHON_INCLUDE_DIR)
pybind11_add_module(numpyeigen_helpers SHARED pybind11_helper_module.cpp)
target_link_libraries(numpyeigen_helpers PRIVATE igl::core)
target_include_directories(numpyeigen_helpers PRIVATE ${NPE_SRC_DIR} ${PYTHON_INCLUDE_DIR} ${NP_INCLUDE_DIR})


add_test(NAME test_nocopy
         COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_dense_binding.py)
add_test(NAME sparse_test
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_sparse_binding.py)
add_test(NAME default_arg_test
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_default_arguments.py)
