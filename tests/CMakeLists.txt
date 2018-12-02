cmake_minimum_required(VERSION 3.2)
project(NumpyEigenTests)

set(NPE_WITH_EIGEN ON)
set(NPE_PYTHON_EXECUTABLE "python")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
include(numpyeigen)

# A module for writing bindings with our framework
npe_add_module(numpyeigen_test
  BINDING_SOURCES
    mutate_matrix.cpp
    matrix_add.cpp
    binding_example.cpp
    sparse_matrix_add.cpp
    sparse_matrix_passthru.cpp
    mutate_sparse_matrix.cpp
    default_arg.cpp
    docstring.cpp
    string_expr_docstring.cpp
    no_numpy.cpp
    dtype.cpp
    default_matches.cpp)
target_link_libraries(numpyeigen_test PRIVATE Eigen3::Eigen)

# A vanilla pybind11 module used to write functions to compare against pyeigen
execute_process(COMMAND python -c "import numpy as np;import sys;sys.stdout.write(np.get_include())"
  OUTPUT_VARIABLE NP_INCLUDE_DIR)
execute_process(
  COMMAND
    python -c "from sysconfig import get_paths;import sys;sys.stdout.write(get_paths()['include'])"
  OUTPUT_VARIABLE PYTHON_INCLUDE_DIR)
pybind11_add_module(numpyeigen_helpers SHARED pybind11_helper_module.cpp)
target_link_libraries(numpyeigen_helpers PRIVATE Eigen3::Eigen)
target_include_directories(numpyeigen_helpers PRIVATE ${NPE_ROOT_DIR}/src ${PYTHON_INCLUDE_DIR} ${NP_INCLUDE_DIR})


add_test(NAME test_dense_bindings
         COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_dense_binding.py)
add_test(NAME test_sparse_bindings
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_sparse_binding.py)
add_test(NAME test_npe_call_interface
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_npe_call_interface.py)
add_test(NAME test_1d_arrays
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_1d_arrays.py)
add_test(NAME test_docstring
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_docstring.py)
add_test(NAME test_default_matches
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_default_matches.py)
