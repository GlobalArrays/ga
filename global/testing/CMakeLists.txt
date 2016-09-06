#
# module: CMakeLists.txt
# author: Bruce Palmer
# description: implements a primative CMake build that can be used to build
#              GA on Windows-based systems. Only MPI-based runtimes are
#              supported.
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

include_directories(BEFORE ${PROJECT_SOURCE_DIR}/global/src
  ${PROJECT_SOURCE_DIR}/ma ${PROJECT_SOURCE_DIR}/gaf2c
  ${PROJECT_SOURCE_DIR}/comex/src-armci
  ${PROJECT_SOURCE_DIR}/tcgmsg
  ${PROJECT_SOURCE_DIR}/LinAlg/lapack+blas
  ${PROJECT_SOURCE_DIR}/global/testing)

if (NOT MSVC)
set(ctargetlibs ${MPI_C_LIBRARIES} m)
else()
set(ctargetlibs ${MPI_C_LIBRARIES})
endif()

# -------------------------------------------------------------
# Build test executables
# -------------------------------------------------------------
add_executable (big.x big.c util.c)
add_executable (elempatch.x elempatch.c util.c)
if (LAPACK_FOUND)
add_executable (ga_lu.x ga_lu.c util.c)
endif()
add_executable (ga-mpi.x ga-mpi.c util.c)
add_executable (gatscat.x gatscat.c util.c)
add_executable (getmem.x getmem.c util.c)
#add_executable (ipc.clean.x ipc.clean.c util.c)
add_executable (lock.x lock.c util.c)
add_executable (mtest.x mtest.c util.c)

# This test uses random() and srandom() which are not available on
# Windoze
if (NOT MSVC)
   add_executable (mulmatpatchc.x mulmatpatchc.c util.c)
endif (NOT MSVC)
add_executable (normc.x normc.c util.c)
add_executable (ntestc.x ntestc.c util.c)
add_executable (ntestfc.x ntestfc.c util.c)
add_executable (packc.x packc.c util.c)
add_executable (patch_enumc.x patch_enumc.c util.c)
add_executable (perf2.x perf2.c util.c)
add_executable (print.x print.c util.c)
add_executable (scan_addc.x scan_addc.c util.c)
add_executable (scan_copyc.x scan_copyc.c util.c)
add_executable (simple_groups_commc.x simple_groups_commc.c util.c)
#add_executable (sprsmatvec.x sprsmatvec.c util.c)
add_executable (testc.x testc.c util.c)
add_executable (testmatmultc.x testmatmultc.c util.c)
add_executable (testmult.x testmult.c util.c)
add_executable (testmultrect.x testmultrect.c util.c)
add_executable (unpackc.x unpackc.c util.c)
if (ENABLE_FORTRAN)
  add_executable (bin.x bin.F ffflush.F)
#  add_executable (blktest.x blktest.F ffflush.F)
  add_executable (d2test.x d2test.F ffflush.F util.c)
  add_executable (g2test.x g2test.F ffflush.F util.c)
  add_executable (g3test.x g3test.F ffflush.F util.c)
  add_executable (ga_shift.x ga_shift.F ffflush.F)
  add_executable (ghosts.x ghosts.F ffflush.F util.c)
  add_executable (jacobi.x jacobi.F ffflush.F)
#  add_executable (merge.x merge.F ffflush.F)
  add_executable (mir_perf1.x mir_perf1.F ffflush.F util.c)
  add_executable (mir_perf2.x mir_perf2.F ffflush.F util.c)
  add_executable (mmatrix.x mmatrix.F ffflush.F util.c)
#  add_executable (mulmatpatch.x mulmatpatch.F ffflush.F testblas.F)
  add_executable (nb2test.x nb2test.F ffflush.F util.c)
  add_executable (ndim.x ndim.F ffflush.F util.c)
#  add_executable (nga-onesided.x nga-onesided.F ffflush.F util.c)
#  add_executable (nga-patch.x nga-patch.F ffflush.F util.c)
#  add_executable (nga-periodic.x nga-periodic.F ffflush.F util.c)
#  add_executable (nga-scatter.x nga-scatter.F ffflush.F)
#  add_executable (nga_test.x nga_test.F ffflush.F)
#  add_executable (ngatest.x ngatest.F ffflush.F util.c)
#  add_executable (nga-util.x nga-util.F ffflush.F)
  add_executable (patch2.x patch2.F ffflush.F)
  add_executable (patch_enumf.x patch_enumf.F ffflush.F)
  add_executable (patch.x patch.F ffflush.F)
  add_executable (perf.x perf.F ffflush.F util.c)
  add_executable (perfmod.x perfmod.F ffflush.F util.c)
  add_executable (perform.x perform.F ffflush.F util.c)
  add_executable (pg2test.x pg2test.F ffflush.F)
#  add_executable (pg2testmatmult.x pg2testmatmult.F ffflush.F util.c)
  add_executable (pgtest.x pgtest.F ffflush.F)
#  add_executable (pgtestmatmult.x pgtestmatmult.F ffflush.F util.c)
  add_executable (random.x random.F ffflush.F util.c)
  add_executable (scan.x scan.F ffflush.F)
  add_executable (simple_groups_comm.x simple_groups_comm.F ffflush.F)
  add_executable (simple_groups.x simple_groups.F ffflush.F)
  add_executable (sparse.x sparse.F ffflush.F)
  add_executable (sprsmatmult.x sprsmatmult.F ffflush.F)
  add_executable (stride.x stride.F ffflush.F)
#  add_executable (testblas.x testblas.F ffflush.F)
  add_executable (testeig.x testeig.F ffflush.F)
  add_executable (test.x test.F ffflush.F)
#  add_executable (testmatmult.x testmatmult.F ffflush.F util.c)
  add_executable (testsolve.x testsolve.F ffflush.F)
#  add_executable (testspd.x testspd.F ffflush.F)
  add_executable (types-test.x types-test.F ffflush.F)
endif()
target_link_libraries(big.x ga ${ctargetlibs})
target_link_libraries(elempatch.x ga ${ctargetlibs})
if (LAPACK_FOUND)
target_link_libraries(ga_lu.x ga ${ctargetlibs})
endif()
target_link_libraries(ga-mpi.x ga ${ctargetlibs})
target_link_libraries(gatscat.x ga ${ctargetlibs})
target_link_libraries(getmem.x ga ${ctargetlibs})
#target_link_libraries(ipc.clean.x ga ${ctargetlibs})
target_link_libraries(lock.x ga ${ctargetlibs})
target_link_libraries(mtest.x ga ${ctargetlibs})
if (NOT MSVC)
   target_link_libraries(mulmatpatchc.x ga ${ctargetlibs})
endif (NOT MSVC)
target_link_libraries(normc.x ga ${ctargetlibs})
target_link_libraries(ntestc.x ga ${ctargetlibs})
target_link_libraries(ntestfc.x ga ${ctargetlibs})
target_link_libraries(packc.x ga ${ctargetlibs})
target_link_libraries(patch_enumc.x ga ${ctargetlibs})
target_link_libraries(perf2.x ga ${ctargetlibs})
target_link_libraries(print.x ga ${ctargetlibs})
target_link_libraries(scan_addc.x ga ${ctargetlibs})
target_link_libraries(scan_copyc.x ga ${ctargetlibs})
target_link_libraries(simple_groups_commc.x ga ${ctargetlibs})
#target_link_libraries(sprsmatvec.x ga ${ctargetlibs})
target_link_libraries(testc.x ga ${ctargetlibs})
target_link_libraries(testmatmultc.x ga ${ctargetlibs})
target_link_libraries(testmult.x ga ${ctargetlibs})
target_link_libraries(testmultrect.x ga ${ctargetlibs})
target_link_libraries(testmult.x ga ${ctargetlibs})
target_link_libraries(unpackc.x ga ${ctargetlibs})
if (ENABLE_FORTRAN)
  target_link_libraries(bin.x ga ${MPI_Fortran_LIBRARIES})
#  target_link_libraries(blktest.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(d2test.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(g2test.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(g3test.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(ga_shift.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(ghosts.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(jacobi.x ga ${MPI_Fortran_LIBRARIES})
#  target_link_libraries(merge.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(mir_perf1.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(mir_perf2.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(mmatrix.x ga ${MPI_Fortran_LIBRARIES})
#  target_link_libraries(mulmatpatch.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(nb2test.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(ndim.x ga ${MPI_Fortran_LIBRARIES})
#  target_link_libraries(nga-onesided.x ga ${MPI_Fortran_LIBRARIES})
#  target_link_libraries(nga-patch.x ga ${MPI_Fortran_LIBRARIES})
#  target_link_libraries(nga-periodic.x ga ${MPI_Fortran_LIBRARIES})
#  target_link_libraries(nga-scatter.x ga ${MPI_Fortran_LIBRARIES})
#  target_link_libraries(nga_test.x ga ${MPI_Fortran_LIBRARIES})
#  target_link_libraries(ngatest.x ga ${MPI_Fortran_LIBRARIES})
#  target_link_libraries(nga-util.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(patch2.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(patch_enumf.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(patch.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(perf.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(perfmod.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(perform.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(pg2test.x ga ${MPI_Fortran_LIBRARIES})
#  target_link_libraries(pg2testmatmult.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(pgtest.x ga ${MPI_Fortran_LIBRARIES})
#  target_link_libraries(pgtestmatmult.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(random.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(scan.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(simple_groups_comm.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(simple_groups.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(sparse.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(sprsmatmult.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(stride.x ga ${MPI_Fortran_LIBRARIES})
#  target_link_libraries(testblas.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(testeig.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(test.x ga ${MPI_Fortran_LIBRARIES})
#  target_link_libraries(testmatmult.x ga ${MPI_Fortran_LIBRARIES} ${linalg_lib})
  target_link_libraries(testsolve.x ga ${MPI_Fortran_LIBRARIES})
#  target_link_libraries(testspd.x ga ${MPI_Fortran_LIBRARIES})
  target_link_libraries(types-test.x ga ${MPI_Fortran_LIBRARIES})
endif()
