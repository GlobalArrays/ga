#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "globalp.h"
#include "papi.h"
#include "wapi.h"

#define _MAX_PROB_SIZE_ 10000 /* 100x100 */

void FATR ga_lu_solve_alt_(Integer *tran, Integer * g_a, Integer * g_b) {
  pnga_error("ga_lu_solve:scalapack not interfaced",0L);
}

void FATR ga_lu_solve_(char *tran, Integer * g_a, Integer * g_b) {

  Integer dimA1, dimA2, typeA;
  Integer dimB1, dimB2, typeB;
  Integer dimsA[2], dimsB[2], ndim;

  /** check GA info for input arrays */
  pnga_check_handle(g_a, "ga_lu_solve: a");
  pnga_check_handle(g_b, "ga_lu_solve: b");
  pnga_inquire (g_a, &typeA, &ndim, dimsA);
  pnga_inquire (g_b, &typeB, &ndim, dimsB);
  dimA1 = dimsA[0];
  dimA2 = dimsA[1];
  dimB1 = dimsB[0];
  dimB2 = dimsB[1];
  
  if( (dimA1*dimA2 > _MAX_PROB_SIZE_) || (dimB1*dimB2 > _MAX_PROB_SIZE_) )
    pnga_error("ga_lu_solve:Array size too large. Use scalapack for optimum performance. configure --with-scalapack or --with-scalapack-i8 for ga_lu_solve to use Scalapack interface",0L);

  pnga_lu_solve_seq(tran, g_a, g_b);
}

Integer FATR ga_llt_solve_(Integer * g_a, Integer * g_b) {
  pnga_error("ga_llt_solve:scalapack not interfaced",0L);
  return 0;
}

Integer FATR ga_solve_(Integer * g_a, Integer * g_b) {
  pnga_error("ga_solve:scalapack not interfaced",0L);
  return 0;
}

Integer FATR ga_spd_invert_(Integer * g_a) {
  pnga_error("ga_spd_invert:scalapack not interfaced",0L);
  return 0;
}

