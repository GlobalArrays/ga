/* $Id: strided.c,v 1.25 2000-06-08 23:47:47 d3h325 Exp $ */
#include "armcip.h"
#include "copy.h"
#include "acc.h"
#include "memlock.h"
#include <stdio.h>


#define ARMCI_OP_2D(op, scale, proc, src, dst, bytes, count, src_stride, dst_stride,lockit)\
if(op == GET || op ==PUT)\
      armci_copy_2D(op, proc, src, dst, bytes, count, src_stride,dst_stride);\
else\
      armci_acc_2D(op, scale, proc, src, dst, bytes, count, src_stride,dst_stride,lockit) 


int armci_iwork[MAX_STRIDE_LEVEL];

/*\ 2-dimensional array copy
\*/
static void armci_copy_2D(int op, int proc, void *src_ptr, void *dst_ptr, 
                          int bytes, int count, int src_stride, int dst_stride)
{
#ifdef LAPI2
#  define COUNT 1
#else
#  define COUNT count
#endif

    int shmem = SAMECLUSNODE(proc);
    
    if(shmem) {
        
        /* data is in local/shared memory -- can use memcpy */

        if(count==1 && bytes <THRESH1D){
            
            armci_copy(src_ptr, dst_ptr, bytes); 
            
        }else {
            
            if(bytes < THRESH){ /* low-latency copy for small data segments */        
                char *ps=(char*)src_ptr;
                char *pd=(char*)dst_ptr;
                int j;
                
                for (j = 0;  j < count;  j++){
                    int i;
                    for(i=0;i<bytes;i++) pd[i] = ps[i];
                    ps += src_stride;
                    pd += dst_stride;
                }
                
            } else if(bytes %ALIGN_SIZE  
                      || dst_stride % ALIGN_SIZE
                      || src_stride % ALIGN_SIZE
#ifdef PTR_ALIGN
                      || (unsigned long)src_ptr%ALIGN_SIZE
                      || (unsigned long)dst_ptr%ALIGN_SIZE
#endif
                      ){ 

                /* size/address not alligned */
                ByteCopy2D(bytes, count, src_ptr, src_stride, dst_ptr, dst_stride);
                
            }else { /* size aligned -- should be the most efficient copy */
                
                DCopy2D(bytes/ALIGN_SIZE, count,src_ptr, src_stride/ALIGN_SIZE, 
                        dst_ptr, dst_stride/ALIGN_SIZE);
            }
        }
        
    } else {
        
        /* data not in local/shared memory-access through global address space*/
        
        if(op==PUT){ 
            
            UPDATE_FENCE_STATE(proc, PUT, COUNT);
#ifdef LAPI
            SET_COUNTER(ack_cntr,COUNT);
#endif
            if(count==1){
                armci_put(src_ptr, dst_ptr, bytes, proc);
            }else{
                armci_put2D(proc, bytes, count, src_ptr, src_stride,
                            dst_ptr, dst_stride);
            }
            
        }else{
            
#ifdef LAPI
            SET_COUNTER(get_cntr, COUNT);
#endif
            if(count==1){
                armci_get(src_ptr, dst_ptr, bytes, proc);
            }else{
                armci_get2D(proc, bytes, count, src_ptr, src_stride,
                            dst_ptr, dst_stride);
            }
        }
    }
}


#if defined(CRAY) || defined(FUJITSU)
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
void armci_acc_2D(int op, void* scale, int proc, void *src_ptr, void *dst_ptr,
                  int bytes, int cols, int src_stride, int dst_stride, int lockit)
{
int   rows, lds, ldd, span;
void (ATR *func)(void*, int*, int*, void*, int*, void*, int*);

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
      case ARMCI_ACC_LNG:
          rows = bytes/sizeof(long);
          ldd  = dst_stride/sizeof(long);
          lds  = src_stride/sizeof(long);
          func = L_ACCUMULATE_2D;
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
      if(lockit)ARMCI_UNLOCKMEM(proc);

}



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
    if(rc) { ARMCI_UNLOCKMEM(proc); return(rc); }

    /* call local accumulate with lockit=0 (we locked it already) and proc=me */
    rc = armci_op_strided(optype, scale, armci_me, src_ptr, src_stride_arr, 
                          buf_ptr, buf_stride_arr, count, stride_levels, 0);
    if(rc) { ARMCI_UNLOCKMEM(proc); return(rc); }

    /* put data back from the buffer to remote location */
    rc = armci_op_strided(PUT, scale, proc, buf_ptr, buf_stride_arr, 
                          dst_ptr, dst_stride_arr, count, stride_levels, 0);

    FENCE_NODE(proc); /* make sure put completes before unlocking */
    ARMCI_UNLOCKMEM(proc);    /* release memory lock */

    return(rc);
}
    



/*\ Strided  operation
\*/
int armci_op_strided(int op, void* scale, int proc,void *src_ptr, int src_stride_arr[],  
		     void* dst_ptr, int dst_stride_arr[], 
                     int count[], int stride_levels, int lockit)
{
    char *src = (char*)src_ptr, *dst=(char*)dst_ptr;
    int s2, s3;

    int i, j;
    int total_of_2D;
    int index[MAX_STRIDE_LEVEL], unit[MAX_STRIDE_LEVEL];
    
#   if defined(ACC_COPY)
      
#      ifdef ACC_SMP
         if(ACC(op) && !(SAMECLUSNODE(proc)) )
#      else
         if ( ACC(op) && proc!=armci_me)
#      endif
             /* copy remote data, accumulate, copy back*/
             return (armci_acc_copy_strided(op,scale, proc, src_ptr, src_stride_arr,
                                       dst_ptr, dst_stride_arr, count, stride_levels));

         else; /* do it directly through shared/local memory */
#   endif


/*    if(proc!=armci_me) INTR_OFF;*/

    switch (stride_levels) {
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
                              count[0], count[1],src_stride_arr[0],
                              dst_stride_arr[0],lockit);
              }
          }
          break;
          
      default: /* N-dimensional */ 
      {
	  /* stride_levels is not the same as ndim. it is ndim-1
	   * For example a 10x10x10... array, suppose the datatype is byte
	   * the stride_arr is 10, 10x10, 10x10x10 ....
	   */
          index[2] = 0; unit[2] = 1; total_of_2D = count[2];
          for(j=3; j<=stride_levels; j++) {
              index[j] = 0; unit[j] = unit[j-1] * count[j-1];
              total_of_2D *= count[j];
          }

          for(i=0; i<total_of_2D; i++) {
              src = (char *)src_ptr; dst = (char *)dst_ptr;
              for(j=2; j<=stride_levels; j++) {
                  src += index[j] * src_stride_arr[j-1];
                  dst += index[j] * dst_stride_arr[j-1];
                  
                  if(((i+1) % unit[j]) == 0) index[j]++;
                  if(index[j] >= count[j]) index[j] = 0;
              }
              
              ARMCI_OP_2D(op, scale, proc, src, dst, count[0], count[1], 
                          src_stride_arr[0], dst_stride_arr[0], lockit);
          }
          
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

/*    if(proc!=armci_me) INTR_ON;*/
    return 0;
}


int ARMCI_PutS( void *src_ptr,  /* pointer to 1st segment at source*/ 
		int src_stride_arr[], /* array of strides at source */
		void* dst_ptr,        /* pointer to 1st segment at destination*/
		int dst_stride_arr[], /* array of strides at destination */
		int count[],          /* number of segments at each stride levels: count[0]=bytes*/
		int stride_levels,    /* number of stride levels */
                int proc              /* remote process(or) ID */
                )
{
    int rc, direct=1;

    if(src_ptr == NULL || dst_ptr == NULL) return FAIL;
    if(count[0]<0)return FAIL3;
    if(stride_levels <0 || stride_levels > MAX_STRIDE_LEVEL) return FAIL4;
    if(proc<0)return FAIL5;

    ORDER(PUT,proc); /* ensure ordering */

#ifndef QUADRICS
    direct=SAMECLUSNODE(proc);
#endif

    /* use direct protocol for remote access when performance is better */
#   if defined(LAPI) && !defined(LAPI2)
      if(!direct)
         if(stride_levels==0 || count[0]> LONG_PUT_THRESHOLD )direct=1;
#   endif

#ifndef LAPI2
    if(!direct)
       rc = armci_pack_strided(PUT, NULL, proc, src_ptr, src_stride_arr,
                       dst_ptr, dst_stride_arr, count, stride_levels, -1, -1);
    else
#endif
       rc = armci_op_strided( PUT, NULL, proc, src_ptr, src_stride_arr, 
                              dst_ptr, dst_stride_arr, count, stride_levels, 0);

    if(rc) return FAIL6;
    else return 0;

}


int ARMCI_GetS( void *src_ptr,  	/* pointer to 1st segment at source*/ 
		int src_stride_arr[],   /* array of strides at source */
		void* dst_ptr,          /* pointer to 1st segment at destination*/
		int dst_stride_arr[],   /* array of strides at destination */
		int count[],            /* number of segments at each stride levels: count[0]=bytes*/
		int stride_levels,      /* number of stride levels */
                int proc                /* remote process(or) ID */
                )
{
    int rc,direct=1;
    int bypass=0;

    if(src_ptr == NULL || dst_ptr == NULL) return FAIL;
    if(count[0]<0)return FAIL3;
    if(stride_levels <0 || stride_levels > MAX_STRIDE_LEVEL) return FAIL4;
    if(proc<0)return FAIL5;
    
    ORDER(GET,proc); /* ensure ordering */
#ifndef QUADRICS
    direct=SAMECLUSNODE(proc);
#endif

    /* use direct protocol for remote access when performance is better */
#   if defined(LAPI) && !defined(LAPI2)
      if(!direct)
        if( stride_levels==0 || count[0]> LONG_GET_THRESHOLD)direct=1;
        else{
          int i, chunks=1;
          for(i=1, direct=1; i<= stride_levels; i++)
              if((chunks *= count[i]) >MAX_CHUNKS_SHORT_GET){ direct=0; break;}
        }
#   endif

#ifndef LAPI2

    if(!direct){

#if defined(DATA_SERVER) && (defined(SOCKETS) || defined(CLIENT_BUF_BYPASS))
       /* larger strided or 1-D reqests, buffer not used to send data 
        * we can bypass the packetization step and send request directly
        */
        if(count[0]> LONG_GET_THRESHOLD) {
#        ifdef GM
            if(armci_gm_bypass)
                bypass= armci_pin_memory(dst_ptr,dst_stride_arr,count,
                                         stride_levels);
#        endif
            rc = armci_rem_strided(GET, NULL, proc, src_ptr, src_stride_arr,
                                   dst_ptr, dst_stride_arr, count,
                                   stride_levels,bypass);
#        ifdef GM
            if(armci_gm_bypass)
                if(bypass)armci_unpin_memory(dst_ptr,dst_stride_arr,count,
                                             stride_levels);
#        endif
       }else
#endif
         rc = armci_pack_strided(GET, NULL, proc, src_ptr, src_stride_arr,
                       dst_ptr, dst_stride_arr, count, stride_levels,-1,-1);
    }else
#endif
       rc = armci_op_strided(GET, NULL, proc, src_ptr, src_stride_arr, 
                               dst_ptr, dst_stride_arr, count, stride_levels,0);

    if(rc) return FAIL6;
    else return 0;
}




int ARMCI_AccS( int  optype,            /* operation */
                void *scale,            /* scale factor x += scale*y */
                void *src_ptr,          /* pointer to 1st segment at source*/ 
		int src_stride_arr[],   /* array of strides at source */
		void* dst_ptr,          /* pointer to 1st segment at destination*/
		int dst_stride_arr[],   /* array of strides at destination */
		int count[],            /* number of segments at each stride levels: count[0]=bytes*/
		int stride_levels,      /* number of stride levels */
                int proc                /* remote process(or) ID */
                )
{
    int rc, direct=1;

    if(src_ptr == NULL || dst_ptr == NULL) return FAIL;
    if(src_stride_arr == NULL || dst_stride_arr ==NULL) return FAIL2;
    if(count[0]<0)return FAIL3;
    if(stride_levels <0 || stride_levels > MAX_STRIDE_LEVEL) return FAIL4;
    if(proc<0)return FAIL5;

    ORDER(optype,proc); /* ensure ordering */
    direct=SAMECLUSNODE(proc);

#   if defined(ACC_COPY) && !defined(ACC_SMP)
       if(armci_me != proc) direct=0;
#   endif
 
    if(direct)
      rc = armci_op_strided( optype, scale, proc, src_ptr, src_stride_arr, 
                           dst_ptr, dst_stride_arr, count, stride_levels,1);
    else
      rc = armci_pack_strided(optype, scale, proc, src_ptr, src_stride_arr, 
                              dst_ptr,dst_stride_arr,count,stride_levels,-1,-1);

    if(rc) return FAIL6;
    else return 0;
}


void armci_write_strided(void *ptr, int stride_levels, int stride_arr[],
                   int count[], char *buf)
{
    int i, j;
    long idx;    /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    int bvalue[MAX_STRIDE_LEVEL], bunit[MAX_STRIDE_LEVEL];

    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++)
        n1dim *= count[i];

    /* calculate the destination indices */
    bvalue[0] = 0; bvalue[1] = 0; bunit[0] = 1; bunit[1] = 1;
    for(i=2; i<=stride_levels; i++) {
        bvalue[i] = 0;
        bunit[i] = bunit[i-1] * count[i-1];
    }

    for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<=stride_levels; j++) {
            idx += bvalue[j] * stride_arr[j-1];
            if((i+1) % bunit[j] == 0) bvalue[j]++;
            if(bvalue[j] > (count[j]-1)) bvalue[j] = 0;
        }

    memcpy(buf, ((char*)ptr)+idx, count[0]);
    buf += count[0];
    }
}


void armci_read_strided(void *ptr, int stride_levels, int stride_arr[],
                        int count[], char *buf)
{
    int i, j;
    long idx;    /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    int bvalue[MAX_STRIDE_LEVEL], bunit[MAX_STRIDE_LEVEL];

    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++)
        n1dim *= count[i];

    /* calculate the destination indices */
    bvalue[0] = 0; bvalue[1] = 0; bunit[0] = 1; bunit[1] = 1;
    for(i=2; i<=stride_levels; i++) {
        bvalue[i] = 0;
        bunit[i] = bunit[i-1] * count[i-1];
    }

    for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<=stride_levels; j++) {
            idx += bvalue[j] * stride_arr[j-1];
            if((i+1) % bunit[j] == 0) bvalue[j]++;
            if(bvalue[j] > (count[j]-1)) bvalue[j] = 0;
        }

    memcpy(((char*)ptr)+idx, buf, count[0]);
    buf += count[0];
    }
}
