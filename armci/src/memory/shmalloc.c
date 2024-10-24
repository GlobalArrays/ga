#if HAVE_CONFIG_H
#   include "config.h"
#endif

/* $Id: shmalloc.c,v 1.10 2002-06-20 23:34:17 vinod Exp $ */
#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_STRING_H
#   include <string.h>
#endif
#include "armcip.h"
#include "message.h"
#include "kr_malloc.h"

static long *offset_arr;

void armci_shmalloc_exchange_offsets(context_t *ctx_local) 
{
    void **ptr_arr;
    void *ptr;
    armci_size_t bytes = 128;
    int i;    
    
    ptr_arr    = (void**)malloc(armci_nproc*sizeof(void*));
    offset_arr = (long*) malloc(armci_nproc*sizeof(long));
    if(!ptr_arr || !offset_arr) armci_die("armci_shmalloc_get_offsets: malloc failed", 0);

    /* get memory with same size on all procs */
    ptr = kr_malloc(bytes, ctx_local);
    if(!ptr) armci_die("armci_shmalloc_get_offsets: kr_malloc failed",bytes);
    
    bzero((char*)ptr_arr,armci_nproc*sizeof(void*));
    ptr_arr[armci_me] = ptr;

    /* now combine individual addresses into a single array */
    armci_exchange_address(ptr_arr, armci_nproc);
    
    /* identify offets */
    for (i=0; i<armci_nproc; i++) 
    {
       offset_arr[i] = (long) ((char*)ptr - (char*)ptr_arr[i]);
    }
       
    /* release memory */
    kr_free(ptr, ctx_local);
}

void armci_shmalloc_exchange_address(void **ptr_arr) 
{
    int i;

    /* now combine individual addresses into a single array */
    armci_exchange_address(ptr_arr, armci_nproc);

    /* since shmalloc may not give symmetric addresses (especially on Linux),
     * adjust addresses based on offset calculated during initialization */
    for (i=0; i<armci_nproc; i++) 
    {
       ptr_arr[i] = (char*)ptr_arr[i] + offset_arr[i];
    }
}

#ifdef MSG_COMMS_MPI

extern int ARMCI_Absolute_id(ARMCI_Group *group,int group_rank);

/* group based exchange address */
void armci_shmalloc_exchange_address_grp(void **ptr_arr, ARMCI_Group *group) 
{
    int i, world_rank;
    int grp_nproc;

    ARMCI_Group_size(group, &grp_nproc);
    
    /* now combine individual addresses into a single array */
    armci_exchange_address_grp(ptr_arr, grp_nproc, group);

    /* since shmalloc may not give symmetric addresses (especially on Linux),
     * adjust addresses based on offset calculated during initialization */
    for (i=0; i<grp_nproc; i++) 
    {
       world_rank = ARMCI_Absolute_id(group,i);
       ptr_arr[i] = (char*)ptr_arr[i] + offset_arr[world_rank];
    }
}
#endif

/* get the remote process's pointer */
void* armci_shmalloc_remote_addr(void *ptr, int proc) 
{
    return (void*)((char*)ptr - offset_arr[proc]);
}

