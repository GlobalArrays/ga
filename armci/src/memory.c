#include <stdio.h>
#include <mpi.h>
#include <assert.h>

#if defined(SYSV) || defined(WIN32)
#include "shmem.h"
#endif

#include "armcip.h"



/*\ Collective Memory Allocation
 *  returns array of pointers to blocks of memory allocated by everybody
 *  Note: as the same shared memory region can be mapped at different locations
 *        in each process address space, the array might hold differnt values
 *        on every processors. However, the addresses are legitimate
 *        and can be used in the ARMCI data transfer operations.
 *        ptr_arr[nproc]
\*/
int ARMCI_Malloc(void *ptr_arr[],int bytes)
{
    void *ptr;

#if defined(SYSV) || defined(WIN32)

    long idlist[SHMIDLEN];
    long size=0, offset=0;
    int *out_arr,*inp_arr;
    int root=0, i;

#else

    void **addr;

#endif

    if(armci_nproc == 1) {
      ptr = malloc(bytes);
      assert(ptr);
      ptr_arr[armci_me] = ptr;
      return (0);
    }

#if defined(SYSV) || defined(WIN32)
    
    /* allocate a work arrays */
    out_arr = (int*)malloc(armci_nproc*sizeof(int));
    assert(out_arr);

    inp_arr = (int*)calloc(armci_nproc,sizeof(int)); /* must be zero */
    assert(inp_arr);

    inp_arr[armci_me] = bytes;

    /* combine all memory requests into out_arr  */
    MPI_Allreduce(inp_arr, out_arr, armci_nproc, MPI_LONG,MPI_SUM,MPI_COMM_WORLD);

    /* determine aggregate request size*/
    size =0;
    for(i=0; i< armci_nproc; i++) size += out_arr[i];


    /* process 0 creates shmem region and then others attach to it */
    if(armci_me == 0){
       ptr = Create_Shared_Region(idlist+1,size,idlist);
       assert(ptr);
    }

    MPI_Bcast(idlist,SHMIDLEN,MPI_INT,0,MPI_COMM_WORLD);/* broadcast shmem id*/

    if(armci_me){
        ptr=(double*)Attach_Shared_Region(idlist+1,size,idlist[0]);
        assert(ptr);
    }

    /* construct array of addresses pointing to the memory regions for each process*/
    offset = 0;
    for(i=0; i< armci_nproc; i++){
        ptr_arr[i] = (out_arr[i]) ? ((char*)ptr) + offset : NULL; /* NULL if request size is 0*/
        offset += out_arr[i];
    }

    /*free work arrays */
    free(out_arr);
    free(inp_arr);

#else

    /* on distributed-memory systems just malloc & collect all addresses */
    addr = calloc(armci_nproc,sizeof(void*)); /* must be zero */
    assert(addr);

    ptr = malloc(bytes);
    if(bytes) assert(ptr);

    addr[armci_me] = ptr;

    /* now combine individual addresses into a single array */
    assert(sizeof(long) == sizeof(void*)); /* is it ever false ? */
    MPI_Allreduce(addr, ptr_arr, armci_nproc, MPI_LONG,MPI_SUM,MPI_COMM_WORLD);

    free(addr);

#endif
    return(0);
}


/*\ shared memory is released to shmalloc only on process 0
\*/
int ARMCI_Free(void *ptr)
{
#if defined(SYSV) || defined(WIN32)

    if(armci_nproc == 1)

#endif
    {

	if(!ptr)return 1;
	free(ptr);
    }

#if defined(SYSV) || defined(WIN32)

    else if(armci_me==0) Free_Shmem_Ptr( 0, 0, ptr);

#endif

    ptr = NULL;
    return 0;
}
