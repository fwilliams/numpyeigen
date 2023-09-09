cmake_minimum_required(VERSION 3.2)
project(NumpyEigenTests)

# A module for writing bindings with our framework
npe_add_module(numpyeigen_test
  BINDING_SOURCES
  matrix_add.cpp
  mutate_matrix.cpp
  sparse_matrix_add.cpp
  binding_example.cpp
  mutate_sparse_matrix.cpp
  sparse_matrix_passthru.cpp
  docstring.cpp
  default_arg.cpp
  no_numpy.cpp
  string_expr_docstring.cpp
  redirect_io.cpp
  dtype.cpp
  default_matches.cpp
  sparse_like.cpp
  bool_arg.cpp
  function_1d.cpp
  long_and_int.cpp
  dense_like.cpp)

# A vanilla pybind11 module used to write functions to compare against pyeigen
execute_process(COMMAND python -c "import numpy as np;import sys;sys.stdout.write(np.get_include())"
  OUTPUT_VARIABLE NP_INCLUDE_DIR)
execute_process(
  COMMAND
    python -c "from sysconfig import get_paths;import sys;sys.stdout.write(get_paths()['include'])"
  OUTPUT_VARIABLE PYTHON_INCLUDE_DIR)
pybind11_add_module(numpyeigen_helpers SHARED pybind11_helper_module.cpp)
if(TARGET Eigen3::Eigen)
  # If an imported target already exists, use it
  target_link_libraries(numpyeigen_helpers PUBLIC Eigen3::Eigen)
else()
  target_include_directories(numpyeigen_helpers PUBLIC ${NUMPYEIGEN_EXTERNAL}/eigen)
endif()
target_include_directories(numpyeigen_helpers PRIVATE ${NPE_ROOT_DIR}/src ${PYTHON_INCLUDE_DIR} ${NP_INCLUDE_DIR})

function (add_test_helper name)
  add_test(NAME ${name}
	COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py)
  set_tests_properties(${name} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/Release\;${CMAKE_CURRENT_BINARY_DIR}")
endfunction()

add_test_helper(test_dense_binding)
add_test_helper(test_sparse_binding)
add_test_helper(test_npe_call_interface)
add_test_helper(test_1d_arrays)
add_test_helper(test_docstring)
add_test_helper(test_default_matches)
add_test_helper(test_slices)

