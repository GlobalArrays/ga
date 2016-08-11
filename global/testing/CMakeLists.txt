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
add_executable (mpi_gatscat_test.x mpi_gatscat_test.c util.c)
add_executable (mtest.x mtest.c util.c)
add_executable (mulmatpatchc.x mulmatpatchc.c util.c)
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
target_link_libraries(big.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(elempatch.x ga ${MPI_C_LIBRARIES} m)
if (LAPACK_FOUND)
target_link_libraries(ga_lu.x ga ${MPI_C_LIBRARIES} m)
endif()
target_link_libraries(ga-mpi.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(gatscat.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(getmem.x ga ${MPI_C_LIBRARIES} m)
#target_link_libraries(ipc.clean.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(lock.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(mpi_gatscat_test.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(mtest.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(mulmatpatchc.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(normc.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(ntestc.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(ntestfc.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(packc.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(patch_enumc.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(perf2.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(print.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(scan_addc.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(scan_copyc.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(simple_groups_commc.x ga ${MPI_C_LIBRARIES} m)
#target_link_libraries(sprsmatvec.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(testc.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(testmatmultc.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(testmult.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(testmultrect.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(testmult.x ga ${MPI_C_LIBRARIES} m)
target_link_libraries(unpackc.x ga ${MPI_C_LIBRARIES} m)
