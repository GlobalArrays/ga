#include <stdio.h>
#ifdef LAPI
#  include "lapidefs.h"
#endif
#include <mpi.h>
#include "armcip.h"
#include "copy.h"
#include "memlock.h"
#include "shmem.h"
#include "signaltrap.h"


int armci_me, armci_nproc;
static int armci_initialized=0;
int armci_cluster_nodes;
double armci_internal_buffer[BUFSIZE_DBL];

#if defined(SYSV) || defined(WIN32)
#   include "locks.h"
    lockset_t lockid;
#endif


void ARMCI_Cleanup()
{
#if defined(SYSV) || defined(WIN32)
    Delete_All_Regions();
    DeleteLocks(lockid);
#endif
}

void armci_die(char *msg, int code)
{
    fprintf(stdout,"%d:%s: %d\n",armci_me, msg, code); fflush(stdout);
    fprintf(stderr,"%d:%s: %d\n",armci_me, msg, code);
    ARMCI_Cleanup();
    MPI_Abort(MPI_COMM_WORLD,code);
}

void ARMCI_Error(char *msg, int code)
{
    armci_die(msg,code);
}



int ARMCI_Init()
{
    int rc;

    if(armci_initialized)return 0;
    else armci_initialized=1;

    MPI_Comm_size(MPI_COMM_WORLD, &armci_nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &armci_me);
    armci_cluster_nodes = armci_nproc;
#ifdef CRAY
    cmpl_proc=-1;
#endif

#ifdef LAPI
    armci_init_lapi();
#endif


    /* trap signals to cleanup ARMCI system resources in case of crash */
    if(armci_me==0) ARMCI_ParentTrapSignals();
    ARMCI_ChildrenTrapSignals();

#if defined(SYSV) || defined(WIN32)

    /* allocate locks */
    if(armci_me == 0)CreateInitLocks(NUM_LOCKS, &lockid);
    MPI_Bcast(&lockid,sizeof(lockid),MPI_BYTE,0,MPI_COMM_WORLD);
    if(armci_me)InitLocks(NUM_LOCKS, lockid);

#endif

    memlock_table_array = malloc(armci_nproc*sizeof(void*));
    if(!memlock_table_array) armci_die("malloc failed for ARMCI lock array",0); 
    rc = ARMCI_Malloc(memlock_table_array, MAX_SLOTS*sizeof(memlock_t));
    if(rc) armci_die("failed to allocate ARMCI memlock array",rc); 
    bzero(memlock_table_array[armci_me],MAX_SLOTS*sizeof(memlock_t));
    MPI_Barrier(MPI_COMM_WORLD);

    return 0;
}


void ARMCI_Finalize()
{
    MPI_Barrier(MPI_COMM_WORLD);
/*
    ARMCI_Free(memlock_table_array[armci_me]);
*/
    if(armci_me==0) ARMCI_ParentRestoreSignals();
    ARMCI_Cleanup();
    MPI_Barrier(MPI_COMM_WORLD);
}




int ARMCI_PutS( void *src_ptr,  /* pointer to 1st segment at source*/ 
		int src_stride_arr[],   /* array of strides at source */
		void* dst_ptr,          /* pointer to 1st segment at destination*/
		int dst_stride_arr[],   /* array of strides at destination */
		int count[],            /* number of segments at each stride levels: count[0]=bytes*/
		int stride_levels,      /* number of stride levels */
                int proc                /* remote process(or) ID */
                )
{
    int rc;

    if(src_ptr == NULL || dst_ptr == NULL) return FAIL;
    if(count[0]<0)return FAIL3;
    if(stride_levels <0 || stride_levels > MAX_STRIDE_LEVEL) return FAIL4;
    if(proc<0)return FAIL5;

    ORDER(PUT,proc); /* ensure ordering */

#   ifdef REMOTE_OP
      if(armci_me != proc
#      ifdef LAPI
             && stride_levels>0 && count[0]< LONG_PUT_THRESHOLD
#      endif
       )

       rc = armci_pack_strided(PUT, NULL, proc, src_ptr, src_stride_arr,
                       dst_ptr, dst_stride_arr, count, stride_levels, -1, -1);
      else
#   endif

       rc = armci_op_strided( PUT, NULL, proc, src_ptr, src_stride_arr, 
                              dst_ptr, dst_stride_arr, count, stride_levels, 0);

    if(rc) return FAIL6;
    else return 0;

}

int ARMCI_GetS( void *src_ptr,  /* pointer to 1st segment at source*/ 
		int src_stride_arr[],   /* array of strides at source */
		void* dst_ptr,          /* pointer to 1st segment at destination*/
		int dst_stride_arr[],   /* array of strides at destination */
		int count[],            /* number of segments at each stride levels: count[0]=bytes*/
		int stride_levels,      /* number of stride levels */
                int proc                /* remote process(or) ID */
                )
{
    int rc;

    if(src_ptr == NULL || dst_ptr == NULL) return FAIL;
    if(count[0]<0)return FAIL3;
    if(stride_levels <0 || stride_levels > MAX_STRIDE_LEVEL) return FAIL4;
    if(proc<0)return FAIL5;

    ORDER(GET,proc); /* ensure ordering */

#   ifdef REMOTE_OP
      if(armci_me != proc 
#      ifdef LAPI
             && stride_levels>0 && count[0]< LONG_GET_THRESHOLD 
#      endif
       )
       rc = armci_pack_strided(GET, NULL, proc, src_ptr, src_stride_arr,
                       dst_ptr, dst_stride_arr, count, stride_levels,-1,-1);
      else
#   endif
       rc = armci_op_strided(GET, NULL, proc, src_ptr, src_stride_arr, 
                               dst_ptr, dst_stride_arr, count, stride_levels,0);

    if(rc) return FAIL6;
    else return 0;

}



int ARMCI_PutV( armci_giov_t darr[], /* descriptor array */
                int len,  /* length of descriptor array */
                int proc  /* remote process(or) ID */
              )
{
    int rc, i;

    if(len<1) return FAIL;
    for(i=0;i<len;i++){
        if(darr[i].src_ptr_array == NULL || darr[i].dst_ptr_array ==NULL) return FAIL2;
        if(darr[i].bytes<1)return FAIL3;
        if(darr[i].ptr_array_len <1) return FAIL4;
    }

    if(proc<0 || proc >= armci_nproc)return FAIL5;

    rc = armci_copy_vector( PUT, darr, len, proc);

    if(rc) return FAIL6;
    else return 0;

}


int ARMCI_GetV( armci_giov_t darr[], /* descriptor array */
                int len,  /* length of descriptor array */
                int proc  /* remote process(or) ID */
              )
{
    int rc, i;

    if(len<1) return FAIL;
    for(i=0;i<len;i++){
        if(darr[i].src_ptr_array == NULL || darr[i].dst_ptr_array ==NULL) return FAIL2;
        if(darr[i].bytes<1)return FAIL3;
        if(darr[i].ptr_array_len <1) return FAIL4;
    }

    if(proc<0 || proc >= armci_nproc)return FAIL5;

    rc = armci_copy_vector( GET, darr, len, proc);

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
    int rc;

    if(src_ptr == NULL || dst_ptr == NULL) return FAIL;
    if(src_stride_arr == NULL || dst_stride_arr ==NULL) return FAIL2;
    if(count[0]<0)return FAIL3;
    if(stride_levels <0 || stride_levels > MAX_STRIDE_LEVEL) return FAIL4;
    if(proc<0)return FAIL5;

    ORDER(optype,proc); /* ensure ordering */

#   if defined(ACC_COPY) || defined(REMOTE_OP)
    if(armci_me != proc)
       rc = armci_pack_strided(optype, scale, proc, src_ptr, src_stride_arr, 
                               dst_ptr, dst_stride_arr,count,stride_levels,-1,-1);
    else  
#   endif
 
       rc = armci_op_strided( optype, scale, proc, src_ptr, src_stride_arr, 
                           dst_ptr, dst_stride_arr, count, stride_levels,1);

    if(rc) return FAIL6;
    else return 0;

}



int ARMCI_AccV( int op,              /* oeration code */
                void *scale,         /*scaling factor for accumulate */
                armci_giov_t darr[], /* descriptor array */
                int len,  /* length of descriptor array */
                int proc  /* remote process(or) ID */
              )
{
    int rc, i;

    if(len<1) return FAIL;
    for(i=0;i<len;i++){
        if(darr[i].src_ptr_array == NULL || darr[i].dst_ptr_array ==NULL) return FAIL2;
        if(darr[i].bytes<1)return FAIL3;
        if(darr[i].ptr_array_len <1) return FAIL4;
    }

    if(proc<0 || proc >= armci_nproc)return FAIL5;

    rc = armci_acc_vector( op, scale, darr, len, proc);

    if(rc) return FAIL6;
    else return 0;

}
