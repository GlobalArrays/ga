#
# module: GlobalArrays.cmake
# author: Bruce Palmer
# description: Define some functions used for testing
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

# This is used to specify a time out for global array unit tests. It's 5
# seconds by default, but may need to be longer on some platforms.
if (NOT GLOBALARRAYS_TEST_TIMEOUT) 
  set (GLOBALARRAYS_TEST_TIMEOUT 5 
    CACHE STRING "Time out for global array unit tests.")
endif ()

# -------------------------------------------------------------
# ga_add_parallel_test
# -------------------------------------------------------------
function(ga_add_parallel_test test_name test_program)
  set(the_test_name "${test_name}_parallel")
  add_test("${the_test_name}"
    ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ${test_program} ${MPIEXEC_POSTFLAGS})
  set_tests_properties("${the_test_name}"
    PROPERTIES 
    PASS_REGULAR_EXPRESSION "No errors detected"
    FAIL_REGULAR_EXPRESSION "failure detected"
    TIMEOUT ${GLOBALARRAYS_TEST_TIMEOUT}
  )
endfunction(ga_add_parallel_test)

# -------------------------------------------------------------
# ga_add_parallel_run_test
#
# This provides a way to consistly add a test that just runs a program
# on multiple processors using ${MPI_EXEC}. Success or failure is
# based on the exit code.
# -------------------------------------------------------------
function(ga_add_parallel_run_test test_name test_program test_input)
  set(the_test_name "${test_name}_parallel")
  add_test("${the_test_name}"
    ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ${test_program} ${MPIEXEC_POSTFLAGS} ${test_input})
  set_tests_properties("${the_test_name}"
    PROPERTIES 
    TIMEOUT ${GLOBALARRAYS_TEST_TIMEOUT}
  )
endfunction(ga_add_parallel_run_test)
