/* $Id: vector.c,v 1.12 2002-01-09 18:56:41 vinod Exp $ */
#include "armcip.h"
#include "copy.h"
#include "acc.h"
#include "memlock.h"
#include <stdio.h>


typedef struct {
    float real;
    float imag;
} complex_t;

typedef struct {
    double real;
    double imag;
} dcomplex_t;


/*
void I_ACCUMULATE(void* scale, int elems, void*src, void* dst)
{
    int j;
    int *a=(int*)dst, *b=(int*)src;
    int alpha = *(int*)scale;

    for(j=0;j<elems;j++) a[j] += alpha*b[j];
}
*/


#define ACCUMULATE( DTYPE, scale, elems, src, dst) {\
    int j;\
    DTYPE *a =(DTYPE *)(dst);\
    DTYPE *b =(DTYPE *)(src);\
    DTYPE alpha = *(DTYPE *)(scale);\
    for(j=0;j<(elems);j++)a[j] += alpha*b[j];\
}
        
#define CPL_ACCUMULATE( DTYPE, scale, elems, src, dst) {\
    int j;\
    DTYPE *a =(DTYPE *)(dst);\
    DTYPE *b =(DTYPE *)(src);\
    DTYPE alpha = *(DTYPE *)(scale);\
    for(j=0;j<(elems);j++){\
        a[j].real += alpha.real*b[j].real - alpha.imag*b[j].imag;\
        a[j].imag += alpha.imag*b[j].real + alpha.real*b[j].imag;\
    }\
}



/*\ compute address range for memory to lock 
\*/
void armci_lockmem_scatter(void *ptr_array[], int len, int bytes, int proc)
{
     int i;
     void *pmin, *pmax;

     pmin=ptr_array[0];
     pmax=ptr_array[0];

     for(i = 0; i< len; i++){
              pmin = MIN(ptr_array[i],pmin);
              pmax = MAX(ptr_array[i],pmax);
     }
     pmax =  bytes-1 + (char*)pmax;
     ARMCI_LOCKMEM(pmin, pmax, proc);

/*    printf("%d: locked %ld-%ld bytes=%d\n",armci_me,pmin,pmax,
     1+(char*)pmax -(char*)pmin);fflush(stdout); */  
}



void armci_scatter_acc(int op, void *scale, armci_giov_t dsc, 
                                            int proc, int lockit)
{
#   define ITERATOR for(i = 0; i< dsc.ptr_array_len; i++)
    int i, elems, size;

      if(lockit)
         armci_lockmem_scatter(dsc.dst_ptr_array, dsc.ptr_array_len, 
                               dsc.bytes, proc); 

      switch (op){
      case ARMCI_ACC_INT:
          size  = sizeof(int);
          elems = dsc.bytes/size;
          if(dsc.bytes%size) armci_die("ARMCI vector accumulate: bytes not consistent with datatype",dsc.bytes);
          ITERATOR{
            ACCUMULATE(int, scale, elems, dsc.src_ptr_array[i], dsc.dst_ptr_array[i])
          }
          break;

      case ARMCI_ACC_LNG:
          size  = sizeof(int);
          elems = dsc.bytes/size;          
          if(dsc.bytes%size) armci_die("ARMCI vector accumulate: bytes not consistent with datatype",dsc.bytes);
          ITERATOR{
            ACCUMULATE(long, scale, elems, dsc.src_ptr_array[i], dsc.dst_ptr_array[i])
          }
          break;

      case ARMCI_ACC_DBL:
          size  = sizeof(double);      
          elems = dsc.bytes/size;
          if(dsc.bytes%size) armci_die("ARMCI vector accumulate: bytes not consistent with datatype",dsc.bytes);
          ITERATOR{
            ACCUMULATE(double, scale, elems, dsc.src_ptr_array[i], dsc.dst_ptr_array[i])
          }
          break;

      case ARMCI_ACC_DCP:
          size  = 2*sizeof(double);       
          elems = dsc.bytes/size;
          if(dsc.bytes%size) armci_die("ARMCI vector accumulate: bytes not consistent with datatype",dsc.bytes);
          ITERATOR{
            CPL_ACCUMULATE(dcomplex_t, scale, elems, dsc.src_ptr_array[i], dsc.dst_ptr_array[i])
          }
          break;

      case ARMCI_ACC_CPL:
          size  = 2*sizeof(float);      
          elems = dsc.bytes/size;
          if(dsc.bytes %size) armci_die("ARMCI vector accumulate: bytes not consistent with datatype",dsc.bytes);
          ITERATOR{
            CPL_ACCUMULATE(complex_t, scale, elems, dsc.src_ptr_array[i], dsc.dst_ptr_array[i])
          }
          break;

      case ARMCI_ACC_FLT:
          size  = sizeof(float);      
          elems = dsc.bytes/size;
          if(dsc.bytes%size) armci_die("ARMCI vector accumulate: bytes not consistent with datatype",dsc.bytes);
          ITERATOR{
            ACCUMULATE(float, scale, elems, dsc.src_ptr_array[i], dsc.dst_ptr_array[i])
          }
          break;
      default: armci_die("ARMCI vector accumulate: operation not supported",op);
      }

      if(lockit) ARMCI_UNLOCKMEM(proc);
}


#ifdef ACC_COPY
#  define PWORKLEN 2048
   static void *pwork[PWORKLEN];  /* work array of pointers */
#endif

int armci_acc_vector(int op,             /* operation code */
                    void *scale,         /* pointer to scale factor in accumulate */
                    armci_giov_t darr[], /* descriptor array */
                    int len,             /* length of descriptor array */
                    int proc             /* remote process(or) ID */
              )
{
    int i;

#if defined(ACC_COPY)
    if(proc == armci_me ){
#endif

       for(i = 0; i< len; i++) armci_scatter_acc(op, scale, darr[i], proc, 1);
#if defined(ACC_COPY)
    }else{

       for(i = 0; i< len; i++){
           armci_giov_t dr =  darr[i];
           int j, rc, nb;

           if(dr.bytes > BUFSIZE/2){
               /* for large segments use strided implementation */
               for(j=0; j< dr.ptr_array_len; j++){
                   rc = armci_acc_copy_strided(op, scale,proc, 
                           dr.src_ptr_array[j], NULL, dr.dst_ptr_array[j],NULL,
                           &dr.bytes, 0);
                   if(rc)return(rc);
               }

           }else{

               armci_giov_t dl;

               /*lock memory:should optimize it to lock only a chunk at a time*/
               armci_lockmem_scatter(dr.dst_ptr_array, dr.ptr_array_len, dr.bytes, proc);

               /* copy as many blocks as possible into the local buffer */
               dl.bytes = dr.bytes;
               nb = MIN(PWORKLEN,BUFSIZE/dr.bytes);

               for(j=0; j< dr.ptr_array_len; j+= nb){
                   int nblocks = MIN(nb, dr.ptr_array_len -j);
                   int k;

                   /* setup vector descriptor for remote memory copy 
                      to bring data into buffer*/

                   dl.ptr_array_len = nblocks;
                   dl.src_ptr_array = dr.dst_ptr_array + j; /* GET destination becomes source for copy */
                   for(k=0; k< nblocks; k++) pwork[k] = k*dl.bytes + (char*)armci_internal_buffer;
                   dl.dst_ptr_array = pwork;

                   /* get data to the local buffer */
                   rc = armci_copy_vector(GET, &dl, 1, proc);
                   if(rc){ ARMCI_UNLOCKMEM(proc); return(rc);}

                   /* update source array for accumulate */
                   dl.src_ptr_array = dr.src_ptr_array +j;

                   /* do scatter accumulate updating copy of data in buffer */
                   armci_scatter_acc(op, scale, dl, armci_me, 0);

                   /* modify descriptor-now source becomes destination for PUT*/
                   dl.dst_ptr_array = dr.dst_ptr_array + j;
                   dl.src_ptr_array = pwork;

                   /* put data back */
                   rc = armci_copy_vector(PUT, &dl, 1, proc);
                   FENCE_NODE(proc);

                   if(rc){ ARMCI_UNLOCKMEM(proc); return(rc);}
               }

               ARMCI_UNLOCKMEM(proc);
           }
       }/*endfor*/
    }
#endif

    return 0;
}




int armci_copy_vector(int op,            /* operation code */
                    armci_giov_t darr[], /* descriptor array */
                    int len,             /* length of descriptor array */
                    int proc             /* remote process(or) ID */
              )
{
    int i,s,shmem= SAMECLUSNODE(proc);

    if(shmem ){ 
      /* local/shared memory copy */

      for(i = 0; i< len; i++){
        for( s=0; s< darr[i].ptr_array_len; s++){
           armci_copy(darr[i].src_ptr_array[s],darr[i].dst_ptr_array[s],darr[i].bytes);
        }
      }

    }else {   

      /* access through global address space */

      /* March 19 - removed FENCE from here - it is in ORDER inside armci.c */

      switch(op){
      case PUT:

        for(i = 0; i< len; i++){

#         ifdef LAPI
                SET_COUNTER(ack_cntr,darr[i].ptr_array_len);
#         endif
          UPDATE_FENCE_STATE(proc, PUT, darr[i].ptr_array_len);
 
          for( s=0; s< darr[i].ptr_array_len; s++){   
              armci_put(darr[i].src_ptr_array[s],darr[i].dst_ptr_array[s],darr[i].bytes, proc);
           }
        }
        break;

      case GET:

        for(i = 0; i< len; i++){

#         ifdef LAPI
                SET_COUNTER(get_cntr,darr[i].ptr_array_len);
#         endif

          for( s=0; s< darr[i].ptr_array_len; s++){   
              armci_get(darr[i].src_ptr_array[s],darr[i].dst_ptr_array[s],darr[i].bytes,proc);
           }
        }
        break;

      default:
          armci_die("armci_copy_vector: wrong optype",op);
      }
   }

#ifdef LAPI
    if(!shmem){

       if(op == GET) CLEAR_COUNTER(get_cntr); /* wait for data arrival */
       if(op == PUT) CLEAR_COUNTER(ack_cntr); /* data must be copied out*/
    }
#endif

   return 0;
}


void armci_vector_to_buf(armci_giov_t darr[], int len, void* buf)
{
int i,s;
char *ptr = (char*)buf; 

      for(i = 0; i< len; i++){
        for( s=0; s< darr[i].ptr_array_len; s++){
          armci_copy(darr[i].src_ptr_array[s],ptr,darr[i].bytes);
          ptr += darr[i].bytes;
        }
      }
}


void armci_vector_from_buf(armci_giov_t darr[], int len, void* buf)
{
int i,s;
char *ptr = (char*)buf;

      for(i = 0; i< len; i++){
        for( s=0; s< darr[i].ptr_array_len; s++){
          armci_copy(ptr, darr[i].dst_ptr_array[s],darr[i].bytes);
          ptr += darr[i].bytes;
        }
      }
}

int ARMCI_PutV( armci_giov_t darr[], /* descriptor array */
                int len,  /* length of descriptor array */
                int proc  /* remote process(or) ID */
              )
{
    int rc, i,direct=1;


    if(len<1) return FAIL;
    for(i=0;i<len;i++){
        if(darr[i].src_ptr_array == NULL || darr[i].dst_ptr_array ==NULL) return FAIL2;
        if(darr[i].bytes<1)return FAIL3;
        if(darr[i].ptr_array_len <1) return FAIL4;
    }

    if(proc<0 || proc >= armci_nproc)return FAIL5;

    ORDER(PUT,proc); /* ensure ordering */
#ifndef QUADRICS
    direct=SAMECLUSNODE(proc);
#endif

    /* use direct protocol for remote access when performance is better */
#   ifdef LAPI
      if(!direct)
          if(len <5 || darr[0].ptr_array_len <5) direct=1;
#   endif


    if(direct)
         rc = armci_copy_vector(PUT, darr, len, proc);
    else{
#if defined(DATA_SERVER) && defined(SOCKETS) && defined(USE_SOCKET_VECTOR_API)  
       rc = armci_rem_vector(PUT, NULL, darr, len, proc, 1);
#else	 
         rc = armci_pack_vector(PUT, NULL, darr, len, proc);
#endif    
    }
    if(rc) return FAIL6;
    else return 0;

}


int ARMCI_GetV( armci_giov_t darr[], /* descriptor array */
                int len,  /* length of descriptor array */
                int proc  /* remote process(or) ID */
              )
{
    int rc, i,direct=1;

    if(len<1) return FAIL;
    for(i=0;i<len;i++){
      if(darr[i].src_ptr_array==NULL ||darr[i].dst_ptr_array==NULL)return FAIL2;
      if(darr[i].bytes<1)return FAIL3;
      if(darr[i].ptr_array_len <1) return FAIL4;
    }

    if(proc<0 || proc >= armci_nproc)return FAIL5;

    ORDER(GET,proc); /* ensure ordering */
#ifndef QUADRICS
    direct=SAMECLUSNODE(proc);
#endif

    /* use direct protocol for remote access when performance is better */
#   ifdef LAPI
      if(!direct)
          if(len <5 || darr[0].ptr_array_len <8) direct=1;
#   endif


    if(direct)
       rc = armci_copy_vector(GET, darr, len, proc);
    else{
#if defined(DATA_SERVER) && defined(SOCKETS) && defined(USE_SOCKET_VECTOR_API)       	
       rc = armci_rem_vector(GET, NULL, darr, len, proc,1);
#else
       rc = armci_pack_vector(GET, NULL, darr, len, proc);
#endif   
    }
    if(rc) return FAIL6;
    else return 0;
}




int ARMCI_AccV( int op,              /* oeration code */
                void *scale,         /*scaling factor for accumulate */
                armci_giov_t darr[], /* descriptor array */
                int len,             /* length of descriptor array */
                int proc             /* remote process(or) ID */
              )
{
    int rc, i,direct=1;

    if(len<1) return FAIL;
    for(i=0;i<len;i++){
      if(darr[i].src_ptr_array==NULL ||darr[i].dst_ptr_array==NULL)return FAIL2;
      if(darr[i].bytes<1)return FAIL3;
      if(darr[i].ptr_array_len <1) return FAIL4;
    }

    if(proc<0 || proc >= armci_nproc)return FAIL5;

    ORDER(op,proc); /* ensure ordering */
    direct=SAMECLUSNODE(proc);

#   if defined(ACC_COPY)
       if(armci_me != proc) direct=0;
#   endif

    if(direct)
         rc = armci_acc_vector( op, scale, darr, len, proc);
    else
         rc = armci_pack_vector(op, scale, darr, len, proc);

    if(rc) return FAIL6;
    else return 0;
}

