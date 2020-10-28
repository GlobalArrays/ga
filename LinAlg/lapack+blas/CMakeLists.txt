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

# -------------------------------------------------------------
# LinAlg header installation
# -------------------------------------------------------------

set(LINALG_HEADERS
  xgemm.h
)

# -------------------------------------------------------------
# LINALG library installation
# -------------------------------------------------------------

if (ENABLE_FORTRAN)
    set(FORTRAN_FILES
        gal_cgemm.f
        gal_daxpy.f
        gal_dcabs1.f
        gal_dcopy.f
        gal_ddot.f
        gal_dgemm.f
        gal_dgemv.f
        gal_dger.f
        gal_dgetf2.f
        gal_dgetrf.f
        gal_dgetrs.f
        gal_disnan.f
        gal_dlacpy.f
        gal_dlae2.f
        gal_dlaev2.f
        gal_dlaisnan.f
        gal_dlamch.f
        gal_dlanst.f
        gal_dlansy.f
        gal_dlapy2.f
        gal_dlarfb.f
        gal_dlarf.f
        gal_dlarfg.f
        gal_dlarft.f
        gal_dlartg.f
        gal_dlascl.f
        gal_dlaset.f
        gal_dlasr.f
        gal_dlasrt.f
        gal_dlassq.f
        gal_dlaswp.f
        gal_dlatrd.f
        gal_dnrm2.f
        gal_dorg2l.f
        gal_dorg2r.f
        gal_dorgql.f
        gal_dorgqr.f
        gal_dorgtr.f
        gal_dpotf2.f
        gal_dpotrf.f
        gal_dscal.f
        gal_dsteqr.f
        gal_dsterf.f
        gal_dswap.f
        gal_dsyev.f
        gal_dsygs2.f
        gal_dsygst.f
        gal_dsygv.f
        gal_dsymm.f
        gal_dsymv.f
        gal_dsyr2.f
        gal_dsyr2k.f
        gal_dsyrk.f
        gal_dsytd2.f
        gal_dsytrd.f
        gal_dtrmm.f
        gal_dtrmv.f
        gal_dtrsm.f
        gal_dtrsv.f
        gal_idamax.f
        gal_ieeeck.f
        gal_iladlc.f
        gal_iladlr.f
        gal_ilaenv.f
        gal_iparmq.f
        gal_lsame.f
        gal_sgemm.f
        gal_xerbla.f
        gal_zaxpy.f
        gal_zgemm.f
      )
endif()

add_library(linalg OBJECT
  xgemm.c
  ${FORTRAN_FILES}
)

target_include_directories(linalg BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
