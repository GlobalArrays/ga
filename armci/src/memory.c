/* $Id: memory.c,v 1.22 2000-12-11 18:06:18 d3h325 Exp $ */
#include <stdio.h>
#include <assert.h>
#include "armcip.h"
#include "message.h"
 
#define DEBUG_ 0
#define USE_MALLOC 
#define USE_SHMEM_

#if defined(SYSV) || defined(WIN32)
#include "shmem.h"

#if !(defined(LAPI)||defined(QUADRICS)||defined(SERVER_THREAD)) ||defined(USE_SHMEM)
#define RMA_NEEDS_SHMEM  
#endif

void  armci_print_ptr(void **ptr_arr, int bytes, int size, void* myptr, int off)
{
int i;
int nproc = armci_clus_info[armci_clus_me].nslave;

    for(i=0; i< armci_nproc; i++){
      int j;
      if(armci_me ==i){
        printf("%d master =%d nproc=%d off=%d\n",armci_me, 
               armci_master,nproc, off);
        printf("%d:bytes=%d mptr=%p s=%d ",armci_me, bytes, myptr,size);
        for(j = 0; j< armci_nproc; j++)printf(" %p",ptr_arr[j]);
        printf("\n"); fflush(stdout);
      }
      armci_msg_barrier();
   }
}


/*\ master exports its address of shmem region at the beggining of that region
\*/
static void armci_master_exp_attached_ptr(void* ptr)
{
    if(!ptr) armci_die("armci_master_exp_att_ptr: null ptr",0);
    *(volatile void**)ptr = ptr;
}


/*\ Collective Memory Allocation on shared memory systems
\*/
void armci_shmem_malloc(void *ptr_arr[],int bytes)
{
    void *myptr=NULL, *ptr=NULL;
    long idlist[SHMIDLEN];
    long size=0, offset=0;
    long *size_arr;
    void **ptr_ref_arr;
    int  i,cn, len;
    int  nproc = armci_clus_info[armci_clus_me].nslave;

    bzero((char*)ptr_arr,armci_nproc*sizeof(void*));

    /* allocate work arrays */
    size_arr = (long*)calloc(armci_nproc,sizeof(long));
    if(!size_arr)armci_die("armci_malloc:calloc failed",armci_nproc);

    /* allocate arrays for cluster address translations */
#   if defined(DATA_SERVER)
       len = armci_nclus;
#   else
       len = nproc;
#   endif

    ptr_ref_arr = calloc(len,sizeof(void*)); /* must be zero */
    if(!ptr_ref_arr)armci_die("armci_malloc:calloc 2 failed",len);

    /* combine all memory requests into size_arr  */
    size_arr[armci_me] = bytes;
    armci_msg_lgop(size_arr, armci_nproc, "+");

    /* determine aggregate request size on the cluster node */
    for(i=0, size=0; i< nproc; i++) size += size_arr[i+armci_master];

    /* master process creates shmem region and then others attach to it */
    if(armci_me == armci_master ){

       /* can malloc if there is no data server process and has 1 process/node*/
#      ifndef RMA_NEEDS_SHMEM
             if(nproc == 1)
                myptr = malloc(size);
             else
#      endif
                myptr = Create_Shared_Region(idlist+1,size,idlist);
       if(!myptr && size>0 )armci_die("armci_malloc: could not create", size);

       /* place its address at begining of attached region for others to see */
       if(size)armci_master_exp_attached_ptr(myptr);

       if(DEBUG_){
         printf("%d:armci_malloc addr mptr=%p size=%ld\n",armci_me,myptr,size);
         fflush(stdout);
       }
    }

    /* broadcast shmem id to other processes on the same cluster node */
    armci_msg_clus_brdcst(idlist, SHMIDLEN*sizeof(long));

    if(armci_me != armci_master){
       myptr=(double*)Attach_Shared_Region(idlist+1,size,idlist[0]);
       if(!myptr)armci_die("armci_malloc: could not attach", size);

       /* now every process in a SMP node needs to find out its offset
        * w.r.t. master - this offset is ncessary to use memlock table
        */
       if(size) armci_set_mem_offset(myptr);
       if(DEBUG_){
          printf("%d:armci_malloc attached addr mptr=%p ref=%p size=%ld\n",
                 armci_me,myptr, *(void**)myptr,size); fflush(stdout);
       }
    }

#   if defined(DATA_SERVER)

       /* get server reference address for every cluster node to perform
        * remote address translation for global address space */
       if(armci_nclus>1){
          if(armci_me == armci_master){

#            ifdef SERVER_THREAD

               /* data server thread runs on master process */
               ptr_ref_arr[armci_clus_me]=myptr;

#            else
               /* ask data server process to attach to the region and get ptr */
               armci_serv_attach_req(idlist, SHMIDLEN*sizeof(long), size, 
                                     &ptr, sizeof(void*));
               ptr_ref_arr[armci_clus_me]= ptr; /* from server*/

               if(DEBUG_){
                 printf("%d:addresses server=%p myptr=%p\n",armci_me,ptr,myptr);
                 fflush(stdout);
               }
#            endif
          }

          /* exchange ref addr of shared memory region on every cluster node*/
          armci_exchange_address(ptr_ref_arr, armci_nclus);

       }else {

          ptr_ref_arr[armci_master] = myptr;

       }

       /* translate addresses for all cluster nodes */
       for(cn = 0; cn < armci_nclus; cn++){

         int master = armci_clus_info[cn].master;
         offset = 0;

         /* on local cluster node use myptr directly */
         ptr = (armci_clus_me == cn) ? myptr: ptr_ref_arr[cn];

         /* compute addresses pointing to the memory regions on cluster node*/
         for(i=0; i< armci_clus_info[cn].nslave; i++){

           /* NULL if request size is 0*/
           ptr_arr[i+master] = (size_arr[i+master])? ((char*)ptr)+offset : NULL;
           offset += size_arr[i+master];

         }
       }

#   else

      /* compute addresses for local cluster node */
      offset =0;
      for(i=0; i< nproc; i++) {

        ptr_ref_arr[i] = (size_arr[i+armci_master])? ((char*)myptr)+offset : 0L;
        offset += size_arr[i+armci_master];

      }
      
      /* exchange addreses with all other processes */
      ptr_arr[armci_me] = (char*)ptr_ref_arr[armci_me-armci_master]; 
      armci_exchange_address(ptr_arr, armci_nproc);

      /* overwrite entries for local cluster node with ptr_ref_arr */
      bcopy( ptr_ref_arr, ptr_arr + armci_master, nproc*sizeof(void*)); 

      /* armci_print_ptr(ptr_arr, bytes, size, myptr, off);*/

#   endif

    /* free work arrays */
    free(ptr_ref_arr);
    free(size_arr);

    armci_msg_barrier();

}

#else

void armci_shmem_malloc(void* ptr_arr[], int bytes)
{
  armci_die("armci_shmem_malloc should never be called on this system",0);
}

#endif



/*\ Collective Memory Allocation
 *  returns array of pointers to blocks of memory allocated by everybody
 *  Note: as the same shared memory region can be mapped at different locations
 *        in each process address space, the array might hold different values
 *        on every process. However, the addresses are legitimate
 *        and can be used in the ARMCI data transfer operations.
 *        ptr_arr[nproc]
\*/
int ARMCI_Malloc(void *ptr_arr[],int bytes)
{
    void *ptr;

    if(DEBUG_)
       fprintf(stderr,"%d bytes in armci_malloc %d\n",armci_me, bytes);
#ifdef USE_MALLOC
    if(armci_nproc == 1) {
      ptr = malloc(bytes);
      if(bytes) if(!ptr) armci_die("armci_malloc:malloc failed",bytes);
      ptr_arr[armci_me] = ptr;
      return (0);
    }
#endif

    if( ARMCI_Uses_shm() ) armci_shmem_malloc(ptr_arr,bytes);
    else {

      /* on distributed-memory systems just malloc & collect all addresses */
      ptr = malloc(bytes);
      if(bytes) if(!ptr) armci_die("armci_malloc:malloc failed",bytes);

      bzero(ptr_arr,armci_nproc*sizeof(void*));
      ptr_arr[armci_me] = ptr;

      /* now combine individual addresses into a single array */
      armci_exchange_address(ptr_arr, armci_nproc);
    }

    return(0);
}



/*\ shared memory is released to shmalloc only on process 0
 *  with data server malloc cannot be used
\*/
int ARMCI_Free(void *ptr)
{
    if(!ptr)return 1;

#if (defined(SYSV) || defined(WIN32)) && !defined(NO_SHM)
#   ifdef USE_MALLOC
      if(armci_nproc > 1)
#   endif
      if(ARMCI_Uses_shm()){
         if(armci_me==armci_master){
#          ifdef RMA_NEEDS_SHMEM
            Free_Shmem_Ptr(0,0,ptr);
#          else
            if(armci_clus_info[armci_clus_me].nslave>1) Free_Shmem_Ptr(0,0,ptr);
            else free(ptr);
#          endif
         }
         ptr = NULL;
         return 0;
      }
#endif
        free(ptr);
        ptr = NULL;
        return 0;
}


int ARMCI_Uses_shm()
{
    int uses=0;

#if (defined(SYSV) || defined(WIN32)) && !defined(NO_SHM)
#   ifdef RMA_NEEDS_SHMEM
      if(armci_nproc >1) uses= 1; /* always unless serial mode */
#   else
      if(armci_nproc != armci_nclus)uses= 1; /* only when > 1 node used */
#   endif
#endif
    if(DEBUG_) fprintf(stderr,"%d:uses shmem %d\n",armci_me, uses);
    return uses;
}


