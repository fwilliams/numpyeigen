set(NUMPYEIGEN_SRC_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Empty list of binding file names
function(bind_function module_name filename)
  get_filename_component(name ${filename} NAME_WE)
  get_filename_component(dir ${filename} DIRECTORY)
  set(generator_target run_function_binding_generator_${module_name}_${name})
  add_custom_target(${generator_target}
    DEPENDS ${filename}
    COMMAND python ${NUMPYEIGEN_SRC_DIRECTORY}/codegen_function.py ${filename} -o ${dir}/${name}.out.cpp)
  add_dependencies(${module_name} ${generator_target})
endfunction()

function(npe_add_module target_name)
  message("Current source dir ${CMAKE_CURRENT_SOURCE_DIR}")
  set(options MODULE SHARED EXCLUDE_FROM_ALL NO_EXTRAS THIN_LTO)
  set(multiValueArgs TARGET_SOURCES BINDING_SOURCES)
  cmake_parse_arguments(make_module "${options}" "" "${multiValueArgs}" ${ARGN})

  foreach(binding_source ${make_module_BINDING_SOURCES})
    get_filename_component(name ${binding_source} NAME_WE)

    set(bound_function_source_filename "${CMAKE_CURRENT_BINARY_DIR}/${name}.out.cpp")
    add_custom_command(OUTPUT ${bound_function_source_filename}
      DEPENDS ${make_module_BINDING_SOURCES} ${NUMPYEIGEN_SRC_DIRECTORY}/codegen_function.py ${NUMPYEIGEN_SRC_DIRECTORY}/codegen_module.py
      COMMAND python ${NUMPYEIGEN_SRC_DIRECTORY}/codegen_function.py ${binding_source} -o ${bound_function_source_filename}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    list(APPEND function_sources "${bound_function_source_filename}")
  endforeach()

  set(module_source_filename ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.module.cpp)
  add_custom_command(OUTPUT ${module_source_filename}
    DEPENDS ${make_module_BINDING_SOURCES} ${NUMPYEIGEN_SRC_DIRECTORY}/codegen_function.py ${NUMPYEIGEN_SRC_DIRECTORY}/codegen_module.py
    COMMAND python ${NUMPYEIGEN_SRC_DIRECTORY}/codegen_module.py -o ${module_source_filename} -m ${target_name} -f ${make_module_BINDING_SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  pybind11_add_module(${target_name} SHARED ${module_source_filename} ${NUMPYEIGEN_SRC_DIRECTORY}/numpyeigen_typedefs.cpp ${function_sources})
  target_include_directories(${target_name} PRIVATE ${NUMPYEIGEN_SRC_DIRECTORY})

  # python ${make_module_BINDING_SOURCE}
endfunction()

