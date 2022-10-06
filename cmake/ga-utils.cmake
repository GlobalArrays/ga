#
# module: GlobalArrays.cmake
# author: Bruce Palmer
# description: Define utility functions.
# 
# DISCLAIMER
#
# This material was prepared as an account of work sponsored by an
# agency of the United States Government.  Neither the United States
# Government nor the United States Department of Energy, nor Battelle,
# nor any of their employees, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
# ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY,
# COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT,
# SOFTWARE, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
# INFRINGE PRIVATELY OWNED RIGHTS.
#
#
# ACKNOWLEDGMENT
#
# This software and its documentation were produced with United States
# Government support under Contract Number DE-AC06-76RLO-1830 awarded by
# the United States Department of Energy.  The United States Government
# retains a paid-up non-exclusive, irrevocable worldwide license to
# reproduce, prepare derivative works, perform publicly and display
# publicly by or for the US Government, including the right to
# distribute to other US Government contractors.
#

# This is used to specify a time out for global array unit tests. It's 60
# seconds by default, but may need to be longer on some platforms.
if (NOT GLOBALARRAYS_TEST_TIMEOUT) 
  set (GLOBALARRAYS_TEST_TIMEOUT 240 
    CACHE STRING "Time out for global array unit tests.")
endif ()

# -------------------------------------------------------------
# ga_add_parallel_test
# -------------------------------------------------------------
function(ga_add_parallel_test test_name test_srcs)
  set(GA_TEST_NPROCS 4)
  if(MPI_PR)
    set(GA_TEST_NPROCS 5)
  endif()
  if(DEFINED ARGV2)
    set(GA_TEST_NPROCS ${ARGV2})
  endif()

  set(__ga_mpiexec ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${GA_TEST_NPROCS})
  if(MPI_LAUNCH_CMD)
    set(__ga_mpiexec ${MPI_LAUNCH_CMD} ${MPIEXEC_NUMPROC_FLAG} ${GA_TEST_NPROCS})
  endif()

  separate_arguments(test_srcs)
  set(__ga_test_exe "${test_name}.x")
  add_executable (${__ga_test_exe} ${test_srcs})
  target_link_libraries(${__ga_test_exe} ga)

  add_test(NAME ${test_name} COMMAND ${__ga_mpiexec} ${CMAKE_CURRENT_BINARY_DIR}/${__ga_test_exe})
  set_tests_properties(${test_name}
    PROPERTIES 
    TIMEOUT ${GLOBALARRAYS_TEST_TIMEOUT}
  )
endfunction(ga_add_parallel_test)


function(ga_is_valid __variable __out)
  set(${__out} FALSE PARENT_SCOPE)
  if(DEFINED ${__variable} AND (NOT "${${__variable}}" STREQUAL ""))
      set(${__out} TRUE PARENT_SCOPE)
  endif()
endfunction()

#
# Sets an option's value if the user doesn't supply one.
#
# Syntax: ga_option <name> <value>
#   - name: The name of the variable to store the option's value under,
#           e.g. CMAKE_BUILD_TYPE for the option containing the build's type
#   - value: The default value to set the variable to, e.g. to default to a
#            Debug build for the build type set value to Debug
#
function(ga_option name value)
    ga_is_valid(${name} was_set)
    if(was_set)
        message(STATUS "Value of ${name} was set by user to : ${${name}}")
    else()
        set(${name} ${value} PARENT_SCOPE)
        message(STATUS "Setting value of ${name} to default : ${value}")
    endif()
endfunction()

function(ga_path_exists __variable __out)
    ga_is_valid(${__variable} was_set)
    set(${__out} FALSE PARENT_SCOPE)
    if(NOT was_set)
        return()
    endif()

    get_filename_component(_fullpath "${${__variable}}" REALPATH)
    if(EXISTS ${_fullpath})
      set(${__out} TRUE PARENT_SCOPE)
    endif()
endfunction()
