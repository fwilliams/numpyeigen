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
#  set(NPE_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/.. PARENT_SCOPE)
  set(NPE_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

  if (NOT EXISTS ${NPE_ROOT_DIR}/src/npe.h)
    error("Could not find NumpyEigen. Set NPE_ROOT_DIR to point to the root directory of NumpyEigen.")
  endif()
# If you have set NPE_ROOT_DIR, then we'lll check that its valid
elseif(NOT EXISTS ${NPE_ROOT_DIR}/src/npe.h)
  error("NPE_ROOT_DIR (${NPE_ROOT_DIR}) does not point to a valid root directory for NumpyEigen.")
endif()

# Include the bundled pybind11 version
add_subdirectory(${NPE_ROOT_DIR}/external/pybind11 ${CMAKE_BINARY_DIR}/pybind11)

# If you want to build with the included Eigen, then we'll find the Eigen package and attempt
# to use it.
if(${NPE_WITH_EIGEN})
  if (TARGET Eigen::Eigen3)
    warning("You enabled NPE_WITH_EIGEN but a target named Eigen3::Eigen already exists. NumpyEigen will use the existing Eigen target instead.")
  else()
    set(ENV{EIGEN3_ROOT} ${NPE_ROOT_DIR}/external/eigen)
    find_package(Eigen3 3.2 REQUIRED)
  endif()
endif()


# This function assumes the variable NPE_ROOT_DIR is set to the root directory of NumpyEigen.
function(npe_add_module target_name)
  set(NPE_SRC_DIR "${NPE_ROOT_DIR}/src")

  find_program(CPP_PROGRAM, cpp)

  if (MSVC)
    set(C_PREPROCESSOR_CMD "${CMAKE_CXX_COMPILER} /E")
  else()
    find_program(cpp, FIND_CPP)
    if (NOT ${FIND_CPP})
      error("Failed to find C preprocessor. This is needed to generate bindings!")
    endif()
    set(C_PREPROCESSOR_CMD "cpp")
  endif()

  execute_process(COMMAND python -c "import numpy as np;import sys;sys.stdout.write(np.get_include())"
    OUTPUT_VARIABLE NP_INCLUDE_DIR)
  execute_process(
    COMMAND
      python -c "from sysconfig import get_paths;import sys;sys.stdout.write(get_paths()['include'])"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR)

  message("Current source dir ${CMAKE_CURRENT_SOURCE_DIR}")
  set(options MODULE SHARED EXCLUDE_FROM_ALL NO_EXTRAS THIN_LTO)
  set(multiValueArgs TARGET_SOURCES BINDING_SOURCES)
  cmake_parse_arguments(make_module "${options}" "" "${multiValueArgs}" ${ARGN})

  foreach(binding_source ${make_module_BINDING_SOURCES})
    get_filename_component(name ${binding_source} NAME_WE)

    set(bound_function_source_filename "${CMAKE_CURRENT_BINARY_DIR}/${name}.out.cpp")
    add_custom_command(OUTPUT ${bound_function_source_filename}
      DEPENDS ${binding_source} ${NPE_SRC_DIR}/codegen_function.py ${NPE_SRC_DIR}/codegen_module.py
      COMMAND python ${NPE_SRC_DIR}/codegen_function.py ${binding_source} ${C_PREPROCESSOR_CMD} -o ${bound_function_source_filename}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    list(APPEND function_sources "${bound_function_source_filename}")
  endforeach()

  set(module_source_filename ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.module.cpp)
  add_custom_command(OUTPUT ${module_source_filename}
    DEPENDS ${make_module_BINDING_SOURCES} ${NPE_SRC_DIR}/codegen_function.py ${NPE_SRC_DIR}/codegen_module.py
    COMMAND python ${NPE_SRC_DIR}/codegen_module.py -o ${module_source_filename} -m ${target_name} -f ${make_module_BINDING_SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  pybind11_add_module(${target_name} SHARED ${module_source_filename} ${NPE_SRC_DIR}/npe_typedefs.cpp ${function_sources})
  target_include_directories(${target_name} PUBLIC ${NPE_SRC_DIR} ${PYTHON_INCLUDE_DIR} ${NP_INCLUDE_DIR})

  if(NOT NPE_EXTRA_CXX_FLAGS)
    set(NPE_EXTRA_CXX_FLAGS "")
  endif()

  target_compile_options(${target_name} PUBLIC ${NPE_EXTRA_CXX_FLAGS})

  if(${NPE_WITH_EIGEN})
    target_link_libraries(${target_name} PUBLIC Eigen3::Eigen)
  endif()

  # python ${make_module_BINDING_SOURCE}
endfunction()

