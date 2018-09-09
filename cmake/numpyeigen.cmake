# Include this file to set up NumpyEigen with your CMake project
#
#
# Setting the following variables before including this file will affect behavior:
#
# NPE_PYTHON_EXECUTABLE
#     Set the Python interpreter which is in turn used to find the correct
#     library and header files to link against.
# NPE_PYTHON_VERSION
#     Request a specific version of Python. NOTE: If NPE_PYTHON_EXECUTABLE is set
#     then we ignore this variable and issue a warning.
# NPE_WITH_EIGEN
#     Build NumpyEigen modules with the bundled version of Eigen. We include an
#     up-to-date version of Eigen for convenience, set this to use it.
# NPE_ROOT_DIR
#     The root directory of the NumpyEigen repository. If you're including the
#     original version of this file in the NumpyEigen repository, this will
#     be set automatically.
#


cmake_minimum_required(VERSION 3.2)

# Utility options to enable various vectorization optimizations in Eigen
set(NPE_EXTRA_CXX_FLAGS "")
if(NOT MSVC)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-fopenmp" COMPILER_SUPPORT_OPENMP)
  if(COMPILER_SUPPORT_OPENMP)
    option(NPE_OPENMP  "Enable/Disable OpenMP" OFF)
    if(NPE_OPENMP)
      set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-fopenmp")
      message(STATUS "Enabling OpenMP")
    endif()
  endif()

  option(NPE_SSE2 "Enable/Disable SSE2" OFF)
  if(NPE_SSE2)
    set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-msse2")
    message(STATUS "Enabling SSE2 for NumpyEigen modules")
  endif()

  option(NPE_SSE3 "Enable/Disable SSE3" OFF)
  if(NPE_SSE3)
    set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-msse3")
    message(STATUS "Enabling SSE3 for NumpyEigen modules")
  endif()

  option(NPE_SSSE3 "Enable/Disable SSSE2" OFF)
  if(NPE_SSSE3)
    set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-mssse3")
    message(STATUS "Enabling SSSE3 for NumpyEigen modules")
  endif()

  option(NPE_SSE4_1 "Enable/Disable SSE4.1" OFF)
  if(NPE_SSE4_1)
    set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-msse4.1")
    message(STATUS "Enabling SSE4.1 for NumpyEigen modules")
  endif()

  option(NPE_SSE4_2 "Enable/Disable SSE4.2" OFF)
  if (NPE_SSE4_2)
    set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-msse4.2")
    message(STATUS "Enabling SSE4.2 for NumpyEigen modules")
  endif()

  option(NPE_AVX "Enable/Disable AVX" OFF)
  if(NPE_AVX)
    set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-mavx")
    message(STATUS "Enabling AVX for NumpyEigen modules")
  endif()

  option(NPE_AVX2 "Enable/Disable AVX" OFF)
  if(NPE_AVX2)
    set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-mavx2")
    message(STATUS "Enabling AVX2 for NumpyEigen modules")
  endif()

  option(NPE_AVX512 "Enable/Disable AVX" OFF)
  if(NPE_AVX512)
    set(NPE_EXTRA_CXX_FLAGS ${NPE_EXTRA_CXX_FLAGS} "-mavx512f" "-DEIGEN_ENABLE_AVX512")
    message(STATUS "Enabling AVX512 for NumpyEigen modules")
  endif()
else()
  warning("Windows build of NumpyEigen is alpha and not considered stable")
endif()


# If you haven't set NPE_ROOT_DIR, then we're going to try and set it
if(NOT NPE_ROOT_DIR)
  set(NPE_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
  if (NOT EXISTS ${NPE_ROOT_DIR}/src/npe.h)
    message(FATAL_ERROR "Could not find NumpyEigen. Set NPE_ROOT_DIR to point to the root directory of NumpyEigen.")
  endif()
  set(NPE_ROOT_DIR ${NPE_ROOT_DIR} PARENT_SCOPE)
# If you have set NPE_ROOT_DIR, then we'lll check that its valid
elseif(NOT EXISTS ${NPE_ROOT_DIR}/src/npe.h)
  message(FATAL_ERROR "NPE_ROOT_DIR (${NPE_ROOT_DIR}) does not point to a valid root directory for NumpyEigen.")
endif()

# If you specified NPE_PYTHON_EXECUTABLE, then we use this as the python interpreter
# which is in turn used to find the correct headers and link libraries.
# If NPE_PYTHON_EXECUTABLE is unset, then we just use whatever findPythonInterp turns up
#
# If you specified NPE_PYTHON_VERSION, then we will try and find a Python interpreter with
# that version. IF NPE_PYTHON_EXECUTABLE is set, then the version string is ignored
# and we use the interpreter specified by NPE_PYTHON_EXECUTABLE
if (NPE_PYTHON_EXECUTABLE)
  if (NPE_PYTHON_VERSION)
    message(WARNING "You set both NPE_PYTHON_EXECUTABLE and NPE_PYTHON_VERSION. NPE_PYTHON_EXECUTABLE will take precedence and the versions may not match.")
  endif()
  set(PYTHON_EXECUTABLE ${NPE_PYTHON_EXECUTABLE})
elseif(NPE_PYTHON_VERSION)
  set(PYBIND11_PYTHON_VERSION "${NPE_PYTHON_VERSION}")
endif()
# Include the bundled pybind11
add_subdirectory(${NPE_ROOT_DIR}/external/pybind11 ${CMAKE_BINARY_DIR}/pybind11)

# If you want to build with the included Eigen, then we'll find the Eigen package and attempt
# to use it.
if(${NPE_WITH_EIGEN})
  if (TARGET Eigen::Eigen3)
    message(WARNING "You enabled NPE_WITH_EIGEN but a target named Eigen3::Eigen already exists. NumpyEigen will use the existing Eigen target instead.")
  else()
    set(ENV{EIGEN3_ROOT} ${NPE_ROOT_DIR}/external/eigen)
    find_package(Eigen3 3.2 REQUIRED)
  endif()
endif()



# Create a Python module from the NumpyEigen src1.cpp, ..., srcN.cpp
#
# npe_add_module(
#     BINDING_SOURCES src1.cpp src2.cpp ... srcN.cpp
#     [MODULE | SHARED]
#     [EXClUDE_FROM_ALL]
#     [THIN_LTO])
#
# MODULE or SHARED specifies the type of library to build. The default is MODULE. These are
# the same as for add_library (https://cmake.org/cmake/help/v3.0/command/add_library.html)
#
# EXCLUDE_FROM_ALL is the same as for add_library
# (https://cmake.org/cmake/help/v3.0/command/add_library.html)
#
# THIN_LTO enables link time optimizations for some compilers (see for example
# https://clang.llvm.org/docs/ThinLTO.html)
#
# NOTE:This function assumes the variable NPE_ROOT_DIR is set to the root directory of NumpyEigen.
#
function(npe_add_module target_name)
  set(options MODULE SHARED EXCLUDE_FROM_ALL NO_EXTRAS THIN_LTO)
  set(multiValueArgs BINDING_SOURCES)
  cmake_parse_arguments(npe_add_module "${options}" "" "${multiValueArgs}" ${ARGN})

  # If you included this file, then PYTHON_EXECUTABLE should always be set
  if (NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "No Python interpreter is defined. We expect the variable PYTHON_EXECUTABLE to be set!")
  endif()

  # You can globally set extra C++ flags to pass when compiling the module
  if(NOT NPE_EXTRA_CXX_FLAGS)
    set(NPE_EXTRA_CXX_FLAGS "")
  endif()

  # Directory containing NumpyEigen source code
  set(NPE_SRC_DIR "${NPE_ROOT_DIR}/src")

  # NumpyEigen uses the C preprocessor for parsing. Here we find a valid command to invoke the C preprocessor
  if (MSVC)
    set(C_PREPROCESSOR_CMD "${CMAKE_CXX_COMPILER} /E")
  else()
    set(C_PREPROCESSOR_CMD "${CMAKE_CXX_COMPILER} -w -E")
  endif()

  # Get the path to the Python header files
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "from sysconfig import get_paths;import sys;sys.stdout.write(get_paths()['include'])"
    OUTPUT_VARIABLE NPE_PYTHON_INCLUDE_DIR)

  # Get the path to the Numpy header files
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import numpy as np;import sys;sys.stdout.write(np.get_include())"
    OUTPUT_VARIABLE NPE_NUMPY_INCLUDE_DIR)


  # For each binding source file add a "target" which runs the NumpyEigen compiler when the binding code changes
  foreach(binding_source ${npe_add_module_BINDING_SOURCES})
    get_filename_component(name ${binding_source} NAME_WE)

    set(bound_function_source_filename "${CMAKE_CURRENT_BINARY_DIR}/${name}.out.cpp")
    add_custom_command(OUTPUT ${bound_function_source_filename}
      DEPENDS ${binding_source} ${NPE_SRC_DIR}/codegen_function.py ${NPE_SRC_DIR}/codegen_module.py
      COMMAND ${PYTHON_EXECUTABLE} ${NPE_SRC_DIR}/codegen_function.py ${binding_source} ${C_PREPROCESSOR_CMD} -o ${bound_function_source_filename}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    list(APPEND function_sources "${bound_function_source_filename}")
  endforeach()

  # Add a target which generates the C++ code for the whole output module
  set(module_source_filename ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.module.cpp)
  add_custom_command(OUTPUT ${module_source_filename}
    DEPENDS ${npe_add_module_BINDING_SOURCES} ${NPE_SRC_DIR}/codegen_function.py ${NPE_SRC_DIR}/codegen_module.py
    COMMAND ${PYTHON_EXECUTABLE} ${NPE_SRC_DIR}/codegen_module.py -o ${module_source_filename} -m ${target_name} -f ${npe_add_module_BINDING_SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  # Register the generated C++ code for the module as a pybind11 module so it gets compiled
  pybind11_add_module(${target_name} SHARED ${module_source_filename} ${NPE_SRC_DIR}/npe_typedefs.cpp ${function_sources})
  target_include_directories(${target_name} PUBLIC ${NPE_SRC_DIR} ${NPE_PYTHON_INCLUDE_DIR} ${NPE_NUMPY_INCLUDE_DIR})
  target_compile_options(${target_name} PUBLIC ${NPE_EXTRA_CXX_FLAGS})

  # If you set NPE_WITH_EIGEN, then we compile against the bundled version of Eigen
  if(${NPE_WITH_EIGEN})
    target_link_libraries(${target_name} PUBLIC Eigen3::Eigen)
  endif()

  # We require C++ 14 for auto return types
  set_target_properties(${target_name} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)
endfunction()

