#include "acc.h"
#include "locks.h"
#include "armcip.h"

/*\ 2-dimensional accumulate
\*/
void armci_acc_2D(int op, void* scale, int proc, void *src_ptr, void *dst_ptr, int bytes, 
		  int cols, int src_stride, int dst_stride)
{
int   rows;
void (*func)();

      switch (op){
      case ARMCI_ACC_INT:
          rows = bytes/sizeof(int);
          func = I_ACCUMULATE_2D;
          break;
      case ARMCI_ACC_DBL:
          rows = bytes/sizeof(double);
          func = D_ACCUMULATE_2D;
          break;
      case ARMCI_ACC_DCP:
          rows = bytes/(2*sizeof(double));
          func = Z_ACCUMULATE_2D;
          break;
      case ARMCI_ACC_CPL:
          rows = bytes/(2*sizeof(float));
          func = C_ACCUMULATE_2D;
          break;
      case ARMCI_ACC_FLT:
          rows = bytes/sizeof(float);
          func = I_ACCUMULATE_2D;
          break;
      default: armci_die("ARMCI accumulate: operation not supported",op);
      }

#if !defined(SYSV) && !defined(WIN32)
  if(proc == armci_me)
#endif
  {

      NATIVE_LOCK(proc%NUM_LOCKS);
      func(scale, &rows, &cols, src_ptr, &src_stride, dst_ptr, &dst_stride);
      NATIVE_UNLOCK(proc%NUM_LOCKS);
  }
#if !defined(SYSV) && !defined(WIN32)
  else armci_die("nonlocal acc not yet implemented",0);
#endif

}

