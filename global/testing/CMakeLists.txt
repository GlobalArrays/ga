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

# This test uses random() and srandom() which are not available on
# Windows
#if (NOT MSVC)
#   add_executable (mulmatpatchc.x mulmatpatchc.c util.c)
#   ga_add_parallel_test(mulmatpatchc mulmatpatchc.x ${TEST_NPROCS_1})
#endif (NOT MSVC)
if (ENABLE_CUDA OR ENABLE_HIP)
  add_executable (c_test.x c_test.c)
  add_executable (c_test_pg.x c_test_pg.c)
  add_executable (copy_dev_test.x copy_dev_test.c)
  ga_add_parallel_test (c_test c_test.x ${TEST_NPROCS_2})
  ga_add_parallel_test (c_test_pg c_test_pg.x ${TEST_NPROCS_2})
  ga_add_parallel_test (copy_dev_test copy_dev_test.x ${TEST_NPROCS_2})
endif()
add_executable (normc.x normc.c util.c)
#FIXME:Works upto 10 ranks
ga_add_parallel_test(normc normc.x ${TEST_NPROCS_4}) 
add_executable (ntestc.x ntestc.c)
ga_add_parallel_test(ntestc ntestc.x)
add_executable (ntestfc.x ntestfc.c)
ga_add_parallel_test(ntestfc ntestfc.x)
add_executable (packc.x packc.c util.c)
ga_add_parallel_test(packc packc.x)
add_executable (patch_enumc.x patch_enumc.c util.c)
ga_add_parallel_test(patch_enumc patch_enumc.x)
add_executable (perf2.x perf2.c util.c)
ga_add_parallel_test(perf2 perf2.x)
add_executable (print.x print.c util.c)
ga_add_parallel_test(print print.x ${TEST_NPROCS_4})
add_executable (scan_addc.x scan_addc.c util.c)
ga_add_parallel_test(scan_addc scan_addc.x)
add_executable (scan_copyc.x scan_copyc.c util.c)
ga_add_parallel_test(scan_copyc scan_copyc.x)
add_executable (simple_groups_commc.x simple_groups_commc.c util.c)
ga_add_parallel_test(simple_groups_commc simple_groups_commc.x)
#add_executable (sprsmatvec.x sprsmatvec.c util.c)
add_executable (testc.x testc.c util.c)
ga_add_parallel_test(testc testc.x)
add_executable (testmatmultc.x testmatmultc.c util.c)
ga_add_parallel_test(testmatmultc testmatmultc.x)
add_executable (testmult.x testmult.c util.c)
ga_add_parallel_test(testmult testmult.x)
add_executable (testmultrect.x testmultrect.c util.c)
ga_add_parallel_test(testmultrect testmultrect.x)
add_executable (unpackc.x unpackc.c util.c)
ga_add_parallel_test(unpack unpackc.x)
if (ENABLE_FORTRAN)
  add_executable (bin.x bin.F ffflush.F)
  ga_add_parallel_test(bin bin.x ${TEST_NPROCS_4})
#  add_executable (blktest.x blktest.F ffflush.F)
  add_executable (d2test.x d2test.F ffflush.F util.c)
  ga_add_parallel_test(d2test d2test.x)
  add_executable (g2test.x g2test.F ffflush.F util.c)
  ga_add_parallel_test(g2test g2test.x)
  add_executable (g3test.x g3test.F ffflush.F util.c)
  ga_add_parallel_test(g3test g3test.x)
  add_executable (ga_shift.x ga_shift.F ffflush.F)
  ga_add_parallel_test(ga_shift ga_shift.x)
  add_executable (ghosts.x ghosts.F ffflush.F util.c)
  ga_add_parallel_test(ghosts ghosts.x)
  add_executable (jacobi.x jacobi.F ffflush.F)
  ga_add_parallel_test(jacobi jacobi.x)
#  add_executable (merge.x merge.F ffflush.F)
  add_executable (mir_perf1.x mir_perf1.F ffflush.F util.c)
  ga_add_parallel_test(mir_perf1 mir_perf1.x ${TEST_NPROCS_4})
  add_executable (mir_perf2.x mir_perf2.F ffflush.F util.c)
  ga_add_parallel_test(mir_perf2 mir_perf2.x ${TEST_NPROCS_4})
  add_executable (mmatrix.x mmatrix.F ffflush.F util.c)
  ga_add_parallel_test(mmatrix mmatrix.x)
#  add_executable (mulmatpatch.x mulmatpatch.F ffflush.F testblas.F)
#  add_executable (nb2test.x nb2test.F ffflush.F util.c)
#  ga_add_parallel_test(nb2test nb2test.x)
#  add_executable (nbtest.x nbtest.F ffflush.F util.c)
#  ga_add_parallel_test(nbtest nbtest.x)
  add_executable (ndim.x ndim.F ffflush.F util.c)
  ga_add_parallel_test(ndim ndim.x)
#  add_executable (nga-onesided.x nga-onesided.F ffflush.F util.c)
#  add_executable (nga-patch.x nga-patch.F ffflush.F util.c)
#  add_executable (nga-periodic.x nga-periodic.F ffflush.F util.c)
#  add_executable (nga-scatter.x nga-scatter.F ffflush.F)
#  add_executable (nga_test.x nga_test.F ffflush.F)
#  add_executable (ngatest.x ngatest.F ffflush.F util.c)
#  add_executable (nga-util.x nga-util.F ffflush.F)
  add_executable (patch2.x patch2.F ffflush.F)
  ga_add_parallel_test(patch2 patch2.x)
  add_executable (patch_enumf.x patch_enumf.F ffflush.F)
  ga_add_parallel_test(patch_enumf patch_enumf.x)
  add_executable (patch.x patch.F ffflush.F testblas.F)
  ga_add_parallel_test(patch patch.x)
  add_executable (perf.x perf.F ffflush.F util.c)
  ga_add_parallel_test(perf perf.x ${TEST_NPROCS_4})
  add_executable (perfmod.x perfmod.F ffflush.F util.c)
  ga_add_parallel_test(perfmod perfmod.x ${TEST_NPROCS_4})
  add_executable (perform.x perform.F ffflush.F util.c)
  ga_add_parallel_test(perform perform.x ${TEST_NPROCS_4})
  add_executable (pg2test.x pg2test.F ffflush.F util.c)
  ga_add_parallel_test(pg2test pg2test.x)
#  add_executable (pg2testmatmult.x pg2testmatmult.F ffflush.F util.c)
  add_executable (pgtest.x pgtest.F ffflush.F util.c)
  ga_add_parallel_test(pgtest pgtest.x)
#  add_executable (pgtestmatmult.x pgtestmatmult.F ffflush.F util.c)
  add_executable (random.x random.F ffflush.F util.c)
  ga_add_parallel_test(random random.x)
  add_executable (scan.x scan.F ffflush.F)
  ga_add_parallel_test(scan scan.x)
  add_executable (simple_groups_comm.x simple_groups_comm.F ffflush.F)
  ga_add_parallel_test(simple_groups_comm simple_groups_comm.x)
  add_executable (simple_groups.x simple_groups.F ffflush.F)
  ga_add_parallel_test(simple_groups simple_groups.x)
  add_executable (sparse.x sparse.F ffflush.F)
  ga_add_parallel_test(sparse sparse.x)
  add_executable (sprsmatmult.x sprsmatmult.F ffflush.F)
  ga_add_parallel_test(sprsmatmult sprsmatmult.x)
  add_executable (stride.x stride.F ffflush.F)
  ga_add_parallel_test(stride stride.x)
#  add_executable (testblas.x testblas.F ffflush.F)
  add_executable (test.x test.F ffflush.F util.c)
  ga_add_parallel_test(main_test test.x)
#  add_executable (testmatmult.x testmatmult.F ffflush.F util.c)
  add_executable (testsolve.x testsolve.F ffflush.F)
  ga_add_parallel_test(testsolve testsolve.x)
if (ENABLE_SCALAPACK)
  add_executable (testspd.x testspd.F ffflush.F)
  ga_add_parallel_test(testspd testspd.x)
  add_executable (testeig.x testeig.F ffflush.F)
  ga_add_parallel_test(testeig testeig.x)
endif()
  add_executable (types-test.x types-test.F ffflush.F util.c)
  ga_add_parallel_test(types-test types-test.x)
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
target_link_libraries(ga-mpi.x ga)
target_link_libraries(gatscat.x ga)
target_link_libraries(getmem.x ga)
#target_link_libraries(ipc.clean.x ga)
target_link_libraries(lock.x ga)
target_link_libraries(mtest.x ga)
#if (NOT MSVC)
#   target_link_libraries(mulmatpatchc.x ga)
#endif (NOT MSVC)
if (ENABLE_CUDA OR ENABLE_HIP)
target_link_libraries(c_test.x ga)
target_link_libraries(c_test_pg.x ga)
target_link_libraries(copy_dev_test.x ga)
endif()
target_link_libraries(normc.x ga)
target_link_libraries(ntestc.x ga)
target_link_libraries(ntestfc.x ga)
target_link_libraries(packc.x ga)
target_link_libraries(patch_enumc.x ga)
target_link_libraries(perf2.x ga)
target_link_libraries(print.x ga)
target_link_libraries(scan_addc.x ga)
target_link_libraries(scan_copyc.x ga)
target_link_libraries(simple_groups_commc.x ga)
#target_link_libraries(sprsmatvec.x ga)
target_link_libraries(testc.x ga)
target_link_libraries(testmatmultc.x ga)
target_link_libraries(testmult.x ga)
target_link_libraries(testmultrect.x ga)
target_link_libraries(testmult.x ga)
target_link_libraries(unpackc.x ga)

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
      COMMAND ${ga_m4_cmd} ${PROJECT_SOURCE_DIR}/global/testing/${testname}.m4 > ${CMAKE_BINARY_DIR}/${testname}.F
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

