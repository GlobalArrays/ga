#include "global.h"
#include "globalp.h"


void FATR ga_lu_solve_alt_(Integer *tran, Integer * g_a, Integer * g_b) {
  ga_error("ga_lu_solve:scalapack not interfaced",0L);
}

void FATR ga_lu_solve_(char *tran, Integer * g_a, Integer * g_b) {
  ga_error("ga_lu_solve:scalapack not interfaced",0L);
}

Integer FATR ga_llt_solve_(Integer * g_a, Integer * g_b) {
  ga_error("ga_llt_solve:scalapack not interfaced",0L);
  return 0;
}

Integer FATR ga_solve_(Integer * g_a, Integer * g_b) {
  ga_error("ga_solve:scalapack not interfaced",0L);
  return 0;
}

Integer FATR ga_spd_invert_(Integer * g_a) {
  ga_error("ga_spd_invert:scalapack not interfaced",0L);
  return 0;
}

