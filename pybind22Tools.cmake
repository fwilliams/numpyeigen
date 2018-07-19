add_subdirectory(pybind11)

set(PYBIND22_SRC_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Empty list of binding file names
function(bind_function module_name filename)
  get_filename_component(name ${filename} NAME_WE)
  get_filename_component(dir ${filename} DIRECTORY)
  set(generator_target run_function_binding_generator_${module_name}_${name})
  add_custom_target(${generator_target}
    DEPENDS ${filename}
    COMMAND python ${SRC_DIRECTORY}/codegen_function.py ${filename} -o ${dir}/${name}.out.cpp)
  add_dependencies(${module_name} ${generator_target})
endfunction()

function(make_module target_name)
  message("Current source dir ${CMAKE_CURRENT_SOURCE_DIR}")
  set(options MODULE SHARED EXCLUDE_FROM_ALL NO_EXTRAS THIN_LTO)
  set(multiValueArgs TARGET_SOURCES BINDING_SOURCES)
  cmake_parse_arguments(make_module "${options}" "" "${multiValueArgs}" ${ARGN})

  foreach(binding_source ${make_module_BINDING_SOURCES})
    get_filename_component(name ${binding_source} NAME_WE)

    set(bound_function_source_filename "${CMAKE_CURRENT_BINARY_DIR}/${name}.out.cpp")

    set(function_generator_target run_function_binding_generator_${target_name}_${name})
    add_custom_target(${function_generator_target}
      DEPENDS ${binding_source}
      COMMAND python ${SRC_DIRECTORY}/codegen_function.py ${binding_source} -o ${bound_function_source_filename}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    list(APPEND function_sources "${bound_function_source_filename}")
    file(GENERATE OUTPUT ${bound_function_source_filename} CONTENT "")
  endforeach()

  set(module_source_filename ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.module.cpp)

  set(module_generator_target run_module_binding_generator_${target_name})
  add_custom_target(${module_generator_target}
    DEPENDS ${module_source_filename}
    COMMAND python ${PYBIND22_SRC_DIRECTORY}/codegen_module.py -o ${module_source_filename} -m ${target_name} -f ${make_module_BINDING_SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  file(GENERATE OUTPUT ${module_source_filename} CONTENT "")
  pybind11_add_module(${target_name} SHARED ${module_source_filename} ${PYBIND22_SRC_DIRECTORY}/binding_typedefs.cpp ${function_sources})
  target_include_directories(${target_name} PRIVATE ${PYBIND22_SRC_DIRECTORY})
  add_dependencies(${target_name} ${module_generator_target})

  foreach(binding_source ${make_module_BINDING_SOURCES})
    get_filename_component(name ${binding_source} NAME_WE)
    set(function_generator_target run_function_binding_generator_${target_name}_${name})
    add_dependencies(${target_name} ${function_generator_target})
  endforeach()
  # python ${make_module_BINDING_SOURCE}
endfunction()

