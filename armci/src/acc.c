#include "acc.h"
#include "locks.h"
#include "armcip.h"

/*\ 2-dimensional accumulate
\*/
void armci_acc_2D(int op, void* scale, int proc, void *src_ptr, void *dst_ptr, int bytes, 
		  int cols, int src_stride, int dst_stride)
{
int   rows, lds, ldd;
void (*func)();

      switch (op){
      case ARMCI_ACC_INT:
          rows = bytes/sizeof(int);
          ldd  = dst_stride/sizeof(int);
          lds  = src_stride/sizeof(int);
          func = I_ACCUMULATE_2D;
          break;
      case ARMCI_ACC_DBL:
          rows = bytes/sizeof(double);
          ldd  = dst_stride/sizeof(double);
          lds  = src_stride/sizeof(double);
          func = D_ACCUMULATE_2D;
          break;
      case ARMCI_ACC_DCP:
          rows = bytes/(2*sizeof(double));
          ldd  = dst_stride/(2*sizeof(double));
          lds  = src_stride/(2*sizeof(double));
          func = Z_ACCUMULATE_2D;
          break;
      case ARMCI_ACC_CPL:
          rows = bytes/(2*sizeof(float));
          ldd  = dst_stride/(2*sizeof(float));
          lds  = src_stride/(2*sizeof(float));
          func = C_ACCUMULATE_2D;
          break;
      case ARMCI_ACC_FLT:
          rows = bytes/sizeof(float);
          ldd  = dst_stride/sizeof(float);
          lds  = src_stride/sizeof(float);
          func = I_ACCUMULATE_2D;
          break;
      default: armci_die("ARMCI accumulate: operation not supported",op);
      }

#if !defined(SYSV) && !defined(WIN32)
  if(proc == armci_me)
#endif
  {

      NATIVE_LOCK(proc%NUM_LOCKS);
      func(scale, &rows, &cols, dst_ptr, &ldd, src_ptr, &lds);
      NATIVE_UNLOCK(proc%NUM_LOCKS);
  }
#if !defined(SYSV) && !defined(WIN32)
  else armci_die("nonlocal acc not yet implemented",0);
#endif

}

