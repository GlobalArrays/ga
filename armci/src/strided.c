#include "armcip.h"
#include "copy.h"
#include "acc.h"
#include "memlock.h"
#include <stdio.h>

#if defined(SGI_N32) || defined(SGI)
#   define PTR_ALIGN
#endif

#define ARMCI_OP_2D(op, scale, proc, src, dst, bytes, count, src_stride, dst_stride,lockit)\
if(op == GET || op ==PUT)\
      armci_copy_2D(op, proc, src, dst, bytes, count, src_stride,dst_stride);\
else\
      armci_acc_2D(op, scale, proc, src, dst, bytes, count, src_stride,dst_stride,lockit) 



/*\ 2-dimensional array copy
\*/
void armci_copy_2D(int op, int proc, void *src_ptr, void *dst_ptr, int bytes, 
		  int count, int src_stride, int dst_stride)
{
#if !defined(SYSV) && !defined(WIN32)
  if(proc == armci_me)
#endif
  {
    if(count==1){

       armci_copy(src_ptr, dst_ptr, bytes); 

    }else {

        if(bytes < THRESH){     /* low-latency copy for small data segments */        
          char *ps=(char*)src_ptr;
          char *pd=(char*)dst_ptr;
          int j;

          for (j = 0;  j < count;  j++){
              int i;
              for(i=0;i<bytes;i++) pd[i] = ps[i];
              ps += src_stride;
              pd += dst_stride;
          }

        } else if(    bytes %ALIGN_SIZE  
                   || dst_stride % ALIGN_SIZE
                   || src_stride % ALIGN_SIZE
#ifdef PTR_ALIGN
                   || (unsigned long)src_ptr%ALIGN_SIZE
                   || (unsigned long)dst_ptr%ALIGN_SIZE
#endif
                ){ 

            /* size/address not alligned */
            ByteCopy2D(bytes, count, src_ptr, src_stride, dst_ptr, dst_stride);

        }else { /* segment size aligned -- should be the most efficient copy */

            DCopy2D(bytes/ALIGN_SIZE, count, src_ptr, src_stride/ALIGN_SIZE, 
                                        dst_ptr, dst_stride/ALIGN_SIZE);
        }
      }
  }

#if !defined(SYSV) && !defined(WIN32)
  else {

       if(op==PUT){ 

          UPDATE_FENCE_STATE(proc, PUT, count);

#ifdef LAPI
          SET_COUNTER(ack_cntr,count);
#endif
          if(count==1){
              armci_put(src_ptr, dst_ptr, bytes, proc);
          }else{
              armci_put2D(proc, bytes, count, src_ptr, src_stride,
                                                  dst_ptr, dst_stride);
          }

       }else{

#ifdef LAPI
          SET_COUNTER(get_cntr, count);
#endif
          if(count==1){
              armci_get(src_ptr, dst_ptr, bytes, proc);
          }else{
             armci_get2D(proc, bytes, count, src_ptr, src_stride,
                                            dst_ptr, dst_stride);
          }
       }
  }
#endif

}


#if defined(CRAY_T3E) || defined(FUJITSU)
#ifdef CRAY
#  define DAXPY  SAXPY
#else
#  define DAXPY  daxpy_
#endif

static int ONE=1;
#define THRESH_ACC 32

static void daxpy_2d_(void* alpha, int *rows, int *cols, void *a, int *ald,
               void* b, int *bld)
{
   int c,r;   
   double *A = (double*)a;
   double *B = (double*)b;
   double Alpha = *(double*)alpha;

   if(*rows < THRESH_ACC)
      for(c=0;c<*cols;c++)
         for(r=0;r<*rows;r++)
           A[c* *ald+ r] += Alpha * B[c* *bld+r];
   else for(c=0;c<*cols;c++)
         DAXPY(rows, alpha, B + c* *bld, &ONE, A + c* *ald, &ONE);
}
#endif


/*\ 2-dimensional accumulate
\*/
void armci_acc_2D(int op, void* scale, int proc, void *src_ptr, void *dst_ptr, int bytes, 
		  int cols, int src_stride, int dst_stride, int lockit)
{
int   rows, lds, ldd, span;
void (FATR *func)(void*, int*, int*, void*, int*, void*, int*);

/*
      if((long)src_ptr%ALIGN)armci_die("src not aligned",(long)src_ptr);
      if((long)dst_ptr%ALIGN)armci_die("src not aligned",(long)dst_ptr);
*/

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
          func = F_ACCUMULATE_2D;
          break;
      default: armci_die("ARMCI accumulate: operation not supported",op);
      }

             
      if(lockit){ 
          span = cols*dst_stride;
          ARMCI_LOCKMEM(dst_ptr, span + (char*)dst_ptr, proc);
      }
      func(scale, &rows, &cols, dst_ptr, &ldd, src_ptr, &lds);
      if(lockit)ARMCI_UNLOCKMEM();

}


int armci_iwork[MAX_STRIDE_LEVEL];

/*\ strided accumulate on top of remote memory copy:
 *  copies remote data to local buffer, accumulates, puts it back 
 *  Note: if we are here then remote patch must fit in the ARMCI buffer
\*/
int armci_acc_copy_strided(int optype, void* scale, int proc,
                                  void* src_ptr, int src_stride_arr[],  
		                  void* dst_ptr, int dst_stride_arr[], 
                                  int count[], int stride_levels)
{
    int  *buf_stride_arr = armci_iwork; 
    void *buf_ptr = armci_internal_buffer;
    int  rc, i, span = count[stride_levels];

    /* compute range of data to lock AND stride array for data in buffer*/
    buf_stride_arr[0]=count[0];
    for(i=0; i< stride_levels; i++) {
         span *= dst_stride_arr[i];
         buf_stride_arr[i+1]= buf_stride_arr[i]*count[i+1];
    }

    /* lock region of remote memory */
    ARMCI_LOCKMEM(dst_ptr, span + (char*)dst_ptr, proc);

    /* get remote data to local buffer */
    rc = armci_op_strided(GET, scale, proc, dst_ptr, dst_stride_arr, 
                          buf_ptr, buf_stride_arr, count, stride_levels, 0);
    if(rc) { ARMCI_UNLOCKMEM(); return(rc); }

    /* call local accumulate with lockit=0 (we locked it already) and proc=me */
    rc = armci_op_strided(optype, scale, armci_me, src_ptr, src_stride_arr, 
                          buf_ptr, buf_stride_arr, count, stride_levels, 0);
    if(rc) { ARMCI_UNLOCKMEM(); return(rc); }

    /* put data back from the buffer to remote location */
    rc = armci_op_strided(PUT, scale, proc, buf_ptr, buf_stride_arr, 
                          dst_ptr, dst_stride_arr, count, stride_levels, 0);

    FENCE_NODE(proc); /* make sure put completes before unlocking */
    ARMCI_UNLOCKMEM();    /* release memory lock */

    return(rc);
}
    



/*\ Strided  operation
\*/
int armci_op_strided(int op, void* scale, int proc,void *src_ptr, int src_stride_arr[],  
		     void* dst_ptr, int dst_stride_arr[], 
                     int count[], int stride_levels, int lockit)
{
    char *src = (char*)src_ptr, *dst=(char*)dst_ptr;
    int s2, s3, s4, sn;

#   ifdef ACC_COPY
    if ( ACC(op) && proc!=armci_me) /* copy remote data, accumulate, copy back*/
        return (armci_acc_copy_strided(op,scale, proc, src_ptr, src_stride_arr,
                                       dst_ptr, dst_stride_arr, count, stride_levels));
#   endif


    switch (stride_levels){
    case 0: /* 1D copy */ 

            ARMCI_OP_2D(op, scale, proc, src_ptr, dst_ptr, count[0], 1, 
                        count[0], count[0], lockit); 

            break;
    
    case 1: /* 2D op */
            ARMCI_OP_2D(op, scale, proc, src_ptr, dst_ptr, count[0], count[1], 
                         src_stride_arr[0], dst_stride_arr[0], lockit);
            break;
    
    case 2: /* 3D op */
            for (s2= 0; s2  < count[2]; s2++){ /* 2D copy */
              ARMCI_OP_2D(op, scale, proc, src+s2*src_stride_arr[1], 
                           dst+s2*dst_stride_arr[1], count[0], count[1], 
                       src_stride_arr[0], dst_stride_arr[0], lockit );
            }
            break;

    case 3: /* 4D op */
            for(s3=0; s3< count[3]; s3++){
               src = (char*)src_ptr + src_stride_arr[2]*s3;
               dst = (char*)dst_ptr + dst_stride_arr[2]*s3;
               for (s2= 0; s2  < count[2]; s2++){ /* 3D copy */
                 ARMCI_OP_2D(op, scale, proc, src+s2*src_stride_arr[1],
                       dst+s2*dst_stride_arr[1],
                       count[0], count[1],src_stride_arr[0],dst_stride_arr[0],lockit);
               }
            }
            break;
    
    case 4: /* 5D op */
            for(s4=0; s4< count[4]; s4++){
              for(s3=0; s3< count[3]; s3++){      /* 4D copy */
                 src = (char*)src_ptr + src_stride_arr[2]*s3 + src_stride_arr[3]*s4;
                 dst = (char*)dst_ptr + dst_stride_arr[2]*s3 + dst_stride_arr[3]*s4;
                 for (s2= 0; s2  < count[2]; s2++){ /* 3D copy */
                   ARMCI_OP_2D(op, scale, proc, src+s2*src_stride_arr[1], 
                                dst+s2*dst_stride_arr[1], 
                                count[0], count[1],
                                src_stride_arr[0], dst_stride_arr[0],lockit);
                 }
              }
            }
            break;
    
    default: /* N-dimensional op by recursion */
             for(sn = 0; sn < count[stride_levels]; sn++){
                 int rc;
                 src = (char*)src_ptr + src_stride_arr[stride_levels -1]* sn;
                 dst = (char*)dst_ptr + dst_stride_arr[stride_levels -1]* sn;
                 rc  = armci_op_strided(op, scale, proc, src, src_stride_arr,  
		                          dst, dst_stride_arr, 
                                          count, stride_levels -1, lockit);
                 if(rc) return(rc);
             }
    }

#ifdef LAPI
    if(proc != armci_me){

       if(op == GET){
           CLEAR_COUNTER(get_cntr); /* wait for data arrival */
       }else { 
           CLEAR_COUNTER(ack_cntr); /* data must be copied out*/ 
       }
    }
#endif

    return 0;
}

