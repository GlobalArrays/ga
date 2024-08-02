#
# module: CMakeLists.txt
# author: Bruce Palmer
# description: CMake build for GA. Only MPI-based runtimes are supported.
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
# -*- mode: cmake -*-
# -------------------------------------------------------------
# file: CMakeLists.txt
# -------------------------------------------------------------

include_directories(BEFORE 
  ${PROJECT_SOURCE_DIR}/global/src
  ${PROJECT_BINARY_DIR}/global/src
  ${PROJECT_SOURCE_DIR}/ma 
  ${PROJECT_BINARY_DIR}/ma
  ${PROJECT_BINARY_DIR}/gaf2c
  ${PROJECT_SOURCE_DIR}/comex/src-armci
  # ${PROJECT_SOURCE_DIR}/tcgmsg
  ${PROJECT_SOURCE_DIR}/LinAlg/lapack+blas
  ${PROJECT_SOURCE_DIR}/global/testing
  ${PROJECT_BINARY_DIR})


set(TEST_NPROCS_1 1)
set(TEST_NPROCS_2 2)
if(MPI_PR)
  set(TEST_NPROCS_1 2)
  set(TEST_NPROCS_2 3)
endif()

# -------------------------------------------------------------
# Build test executables
# -------------------------------------------------------------

# This test uses random() and srandom() which are not available on Windows
if(NOT MSVC)
  ga_add_parallel_test (mulmatpatchc "mulmatpatchc.c util.c" ${TEST_NPROCS_1})
endif()

ga_add_parallel_test (elempatch     "elempatch.c util.c" )
ga_add_parallel_test (getmem        "getmem.c util.c"    )
ga_add_parallel_test (mtest         "mtest.c util.c"     )
ga_add_parallel_test (normc         "normc.c util.c"     )
ga_add_parallel_test (matrixc       "matrixc.c"          )
ga_add_parallel_test (ntestc        "ntestc.c"           )
ga_add_parallel_test (nbtestc       "nbtestc.c"          )
ga_add_parallel_test (ntestfc       "ntestfc.c"          )
ga_add_parallel_test (packc         "packc.c util.c"     )
ga_add_parallel_test (print         "print.c util.c"     )
ga_add_parallel_test (scan_addc     "scan_addc.c util.c" )
ga_add_parallel_test (scan_copyc    "scan_copyc.c util.c")
ga_add_parallel_test (testc         "testc.c util.c"     )
ga_add_parallel_test (testmult      "testmult.c util.c"  )
ga_add_parallel_test (testmatmultc  "testmatmultc.c"     )
ga_add_parallel_test (testmultrect  "testmultrect.c"     )
ga_add_parallel_test (gemmtest      "gemmtest.c"         )
ga_add_parallel_test (read_only     "read_only.c"        )
ga_add_parallel_test (cache_test    "cache_test.c"       )
ga_add_parallel_test (unpackc       "unpackc.c util.c"   )
ga_add_parallel_test (perf2         "perf2.c"            )
ga_add_parallel_test (ga-mpi        "ga-mpi.c util.c"    )
ga_add_parallel_test (lock          "lock.c util.c"      )
ga_add_parallel_test (comm_init     "comm_init.c util.c" )

ga_add_parallel_test (patch_enumc         "patch_enumc.c util.c"        )
ga_add_parallel_test (simple_groups_commc "simple_groups_commc.c util.c")
ga_add_parallel_test (thread_perf_contig  "thread_perf_contig.c util.c" )
ga_add_parallel_test (thread_perf_strided "thread_perf_strided.c util.c")
ga_add_parallel_test (threadsafec         "threadsafec.c util.c"        )

if(LAPACK_FOUND)
  ga_add_parallel_test (ga_lu "ga_lu.c util.c")
endif()

if (ENABLE_FORTRAN)
  ga_add_parallel_test (bin            "bin.F ffflush.F"                    Fortran)
  ga_add_parallel_test (blktest        "blktest.F ffflush.F"                Fortran)
  ga_add_parallel_test (g2test         "g2test.F ffflush.F util.c"          Fortran)
  ga_add_parallel_test (g3test         "g3test.F ffflush.F util.c"          Fortran)
  ga_add_parallel_test (ga_shift       "ga_shift.F ffflush.F"               Fortran)
  ga_add_parallel_test (ghosts         "ghosts.F ffflush.F util.c"          Fortran)
  ga_add_parallel_test (jacobi         "jacobi.F ffflush.F"                 Fortran)
  ga_add_parallel_test (mir_perf2      "mir_perf2.F ffflush.F util.c"       Fortran)
  ga_add_parallel_test (mmatrix        "mmatrix.F ffflush.F util.c"         Fortran)
  ga_add_parallel_test (mulmatpatch    "mulmatpatch.F ffflush.F testblas.F" Fortran)
  ga_add_parallel_test (nbtest         "nbtest.F ffflush.F util.c"          Fortran)
  ga_add_parallel_test (nb2test        "nb2test.F ffflush.F util.c"         Fortran)
  ga_add_parallel_test (ndim           "ndim.F ffflush.F util.c"            Fortran)
  ga_add_parallel_test (patch          "patch.F ffflush.F testblas.F"       Fortran)
  ga_add_parallel_test (patch2         "patch2.F ffflush.F"                 Fortran)
  ga_add_parallel_test (patch_enumf    "patch_enumf.F ffflush.F"            Fortran)
  ga_add_parallel_test (perfmod        "perfmod.F ffflush.F util.c"         Fortran)
  ga_add_parallel_test (perform        "perform.F ffflush.F util.c"         Fortran)
  ga_add_parallel_test (perf           "perf.F ffflush.F util.c"            Fortran)
  ga_add_parallel_test (pg2test        "pg2test.F ffflush.F util.c"         Fortran)
  ga_add_parallel_test (pgtest         "pgtest.F ffflush.F util.c"          Fortran)
  ga_add_parallel_test (scan           "scan.F ffflush.F"                   Fortran)

  ga_add_parallel_test (sparse         "sparse.F ffflush.F"             Fortran)
  ga_add_parallel_test (sprsmatmult    "sprsmatmult.F ffflush.F"        Fortran)
  ga_add_parallel_test (stride         "stride.F ffflush.F"             Fortran)
  ga_add_parallel_test (test           "test.F ffflush.F util.c"        Fortran)
  ga_add_parallel_test (testmatmult    "testmatmult.F ffflush.F util.c" Fortran)
  ga_add_parallel_test (testsolve      "testsolve.F ffflush.F"          Fortran)
  ga_add_parallel_test (overlay        "overlay.F ffflush.F util.c"     Fortran)
  ga_add_parallel_test (testeig        "testeig.F ffflush.F"            Fortran)

  ga_add_parallel_test (simple_groups      "simple_groups.F ffflush.F"      Fortran)
  ga_add_parallel_test (simple_groups_comm "simple_groups_comm.F ffflush.F" Fortran)

  # Generated Tests
  function(generate_ngatests testname)
    add_custom_command(
      OUTPUT ${CMAKE_BINARY_DIR}/${testname}.F
      COMMAND m4 ${PROJECT_SOURCE_DIR}/global/testing/${testname}.m4 > ${CMAKE_BINARY_DIR}/${testname}.F
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/global/testing
      COMMENT "Generating ${testname} test"
    )
  endfunction()

  generate_ngatests(nga-onesided)
  generate_ngatests(nga-patch)
  generate_ngatests(nga-periodic)
  generate_ngatests(nga-scatter)
  generate_ngatests(ngatest)
  generate_ngatests(nga-util)

  set (NGATEST_GEN_SRCS
      ${CMAKE_BINARY_DIR}/nga-onesided.F
      ${CMAKE_BINARY_DIR}/nga-patch.F
      ${CMAKE_BINARY_DIR}/nga-periodic.F
      ${CMAKE_BINARY_DIR}/nga-scatter.F
      ${CMAKE_BINARY_DIR}/ngatest.F
      ${CMAKE_BINARY_DIR}/nga-util.F
  )
  add_custom_target(
    GenerateNGATests ALL
    DEPENDS ${NGATEST_GEN_SRCS}
  )
  set_source_files_properties(
    ${NGATEST_GEN_SRCS}
    PROPERTIES GENERATED TRUE
  )

  ga_add_parallel_test (nga-onesided   "${CMAKE_BINARY_DIR}/nga-onesided.F ffflush.F util.c" ${TEST_NPROCS_2} Fortran)
  ga_add_parallel_test (nga-patch      "${CMAKE_BINARY_DIR}/nga-patch.F    ffflush.F util.c" ${TEST_NPROCS_1} Fortran)
  ga_add_parallel_test (nga-periodic   "${CMAKE_BINARY_DIR}/nga-periodic.F ffflush.F util.c"  Fortran                )
  ga_add_parallel_test (nga-scatter    "${CMAKE_BINARY_DIR}/nga-scatter.F  ffflush.F util.c"  Fortran                )
  ga_add_parallel_test (ngatest        "${CMAKE_BINARY_DIR}/ngatest.F      ffflush.F util.c" ${TEST_NPROCS_1} Fortran)
  ga_add_parallel_test (nga-util       "${CMAKE_BINARY_DIR}/nga-util.F     ffflush.F util.c"  Fortran                )

  add_dependencies(nga-onesided.x GenerateNGATests)
  add_dependencies(nga-patch.x    GenerateNGATests)
  add_dependencies(nga-periodic.x GenerateNGATests)
  add_dependencies(nga-scatter.x  GenerateNGATests)
  add_dependencies(ngatest.x      GenerateNGATests)
  add_dependencies(nga-util.x     GenerateNGATests)
  

endif()

