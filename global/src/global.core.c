/*$Id: global.core.c,v 1.31 1996-09-18 16:46:17 d3h325 Exp $*/
/*
 * module: global.core.c
 * author: Jarek Nieplocha
 * last major change date: Mon Dec 19 19:03:38 CST 1994
 * description: implements GA primitive operations --
 *              create (regular& irregular) and duplicate, destroy, get, put,
 *              accumulate, scatter, gather, read&increment & synchronization 
 * 
 * DISCLAIMER
 *
 * This material was prepared as an account of work sponsored by an
 * agency of the United States Government.  Neither the United States
 * Government nor the United States Department of Energy, nor Battelle,
 * nor any of their employees, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY,
 * COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT,
 * SOFTWARE, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
 * INFRINGE PRIVATELY OWNED RIGHTS.
 *
 *
 * ACKNOWLEDGMENT
 *
 * This software and its documentation were produced with United States
 * Government support under Contract Number DE-AC06-76RLO-1830 awarded by
 * the United States Department of Energy.  The United States Government
 * retains a paid-up non-exclusive, irrevocable worldwide license to
 * reproduce, prepare derivative works, perform publicly and display
 * publicly by or for the US Government, including the right to
 * distribute to other US Government contractors.
 */

 
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "global.h"
#include "globalp.h"
#include "message.h"
#include "macommon.h"
#include "global.core.h"

#define DEBUG 0
#define USE_MALLOC 1
#define INVALID_MA_HANDLE -1 
#define NEAR_INT(x) (x)< 0.0 ? ceil( (x) - 0.5) : floor((x) + 0.5)



/* need to move these variables outside gaCentralBarrier() so 
 * that SGI compiler will not break (overoptimize) the code 
 */
volatile int local_flag=0;
volatile int local_barrier, local_barrier1;

/*\ central barrier algorithm 
\*/
void gaCentralBarrier()
{
#if (defined(SYSV) && !defined(KSR))
      if(cluster_compute_nodes == 1)return;
      local_flag = 1 - local_flag;
      P(MUTEX);
         (*Barrier) ++;
         local_barrier = *Barrier;
         if(local_barrier==barrier_size) *Barrier  = 0;
      V(MUTEX);
 
      if(local_barrier==barrier_size){
                /*LOCK(Barrier1);*/
                  *Barrier1 = local_flag;
                /* UNLOCK(Barrier1);*/
      }else do{
                /*LOCK(Barrier1);*/
                   local_barrier1 = *Barrier1;
                /*UNLOCK(Barrier1);*/
           }while (local_barrier1 != local_flag);
#endif
}



/*\ SYNCHRONIZE ALL THE PROCESSES
\*/
void ga_sync_()
{
void   ga_wait_server();
       extern int GA_fence_set;

       GA_fence_set=0;
       if (GAme < 0) return;
#ifdef CONVEX
       ga_msg_sync_();
#elif defined(CRAY_T3D) || defined(KSR)
       NATIVEbarrier();
#elif defined(SYSV)
#      ifdef IWAY
             gaCentralBarrier();
#      endif
       if(ClusterMode) ga_wait_server();
       gaCentralBarrier();
#else  
       ga_msg_sync_();
#      if defined(PARAGON) || defined(IWAY)
             ga_wait_server();  /* synchronize data server thread */
#      endif
#      ifdef IWAY
             ga_msg_sync_();
#      endif
#endif
}


Integer GAsizeof(type)    
        Integer type;
{
  switch (type) {
     case MT_F_DBL  : return (sizeof(DoublePrecision));
     case MT_F_INT  : return (sizeof(Integer));
     case MT_F_DCPL : return (sizeof(DoubleComplex));
          default   : return 0; 
  }
}


/*\ FINAL CLEANUP of shmem when terminating
\*/
void ga_clean_resources()
{                  
#ifdef SYSV 
    if(GAinitialized){
#      ifndef KSR
#         if defined(SGIUS) || defined (SPPLOCKS)
             if(GAnproc>1) DeleteLocks(lockID);
#         else
             if(GAnproc>1) SemDel();
#         endif
#      endif
       if(!(USE_MALLOC) || GAnproc >1)(void)Delete_All_Regions(); 
     }
#endif
}


/*\ CHECK GA HANDLE and if it's wrong TERMINATE
 *  Fortran version
\*/
#ifdef CRAY_T3D
void ga_check_handle_(g_a, fstring)
     Integer *g_a;
     _fcd fstring;
#else
void ga_check_handle_(g_a, fstring,slen)
     Integer *g_a;
     int  slen;
     char *fstring;
#endif
{
char  buf[FLEN];

    if( GA_OFFSET + (*g_a) < 0 || GA_OFFSET + (*g_a) >= max_global_array){
#ifdef CRAY_T3D
      f2cstring(_fcdtocp(fstring), _fcdlen(fstring), buf, FLEN);
#else
      f2cstring(fstring, slen, buf, FLEN);
#endif
      fprintf(stderr, " ga_check_handle: %s ", buf);
      ga_error(" invalid global array handle ", (*g_a));
    }
    if( ! (GA[GA_OFFSET + (*g_a)].actv) ){
#ifdef CRAY_T3D
      f2cstring(_fcdtocp(fstring), _fcdlen(fstring), buf, FLEN);
#else
      f2cstring(fstring, slen, buf, FLEN);
#endif
      ga_error(" global array is not active ", (*g_a));
    }
}



/*\ CHECK GA HANDLE and if it's wrong TERMINATE
 *  C version
\*/
void ga_check_handle(g_a, string)
     Integer *g_a;
     char *string;
{
  ga_check_handleM(g_a,string);
}


/*\ signals trapped by all processes
\*/
void gaAllTrapSignals()
{
#ifdef SYSV
void TrapSigInt(), TrapSigChld(),
     TrapSigBus(), TrapSigFpe(),
     TrapSigIll(), TrapSigSegv(),
     TrapSigSys(), TrapSigTrap(),
     TrapSigHup(), TrapSigTerm(),
     TrapSigIot();

     TrapSigBus();
     TrapSigFpe();
     TrapSigIll();
     TrapSigSegv();
     TrapSigSys();
     TrapSigTrap();
     TrapSigTerm();
#ifdef SGI
     TrapSigIot();
#endif
#endif
}


/*\  signals trapped only by parent process
\*/
void gaParentTrapSignals()
{
#ifdef SYSV
void TrapSigChld(), TrapSigInt(), TrapSigHup();
     TrapSigChld();
     TrapSigInt();
     TrapSigHup();
#endif
}

/*\  parent process restores the original handlers 
\*/
void gaParentRestoreSignals()
{
#ifdef SYSV
void RestoreSigChld(), RestoreSigInt(), RestoreSigHup();
     RestoreSigChld();
     RestoreSigInt();
     RestoreSigHup();
#endif
}



/*\ prepare permuted list of processes for remote ops
\*/
#define gaPermuteProcList(nproc)\
{\
  if((nproc) ==1) ProcListPerm[0]=0; \
  else{\
    int _i, iswap, temp;\
    if((nproc) > GAnproc) ga_error("permute_proc: error ", (nproc));\
\
    /* every process generates different random sequence */\
    (void)srand((unsigned)GAme); \
\
    /* initialize list */\
    for(_i=0; _i< (nproc); _i++) ProcListPerm[_i]=_i;\
\
    /* list permutation generated by random swapping */\
    for(_i=0; _i< (nproc); _i++){ \
      iswap = (int)(rand() % (nproc));  \
      temp = ProcListPerm[iswap]; \
      ProcListPerm[iswap] = ProcListPerm[_i]; \
      ProcListPerm[_i] = temp; \
    } \
  }\
}
     

void gaPermuteProcList2(nproc)
    Integer nproc;
{
    int i, iswap, temp;

    if(nproc ==1) {
       ProcListPerm[0]=0;
       return;
    }
    if(nproc > GAnproc) ga_error("permute_proc: nproc error ", nproc);

    /* every process generates different random sequence */
    (void)srand((unsigned)GAme);

    /* initialize list */
    for(i=0; i< nproc; i++) ProcListPerm[i]=i;

    /* list permutation generated by random swapping */
    for(i=0; i< nproc; i++){
      iswap = (int)(rand() % nproc);
      temp = ProcListPerm[iswap];
      ProcListPerm[iswap] = ProcListPerm[i];
      ProcListPerm[i] = temp;
    }
}
 


/*\ INITIALIZE GLOBAL ARRAY STRUCTURES
 *
 *  either ga_initialize_ltd or ga_initialize must be the first 
 *         GA routine called (except ga_uses_ma)
\*/
void ga_initialize_()
{
Integer type, i;
Integer buf_size, bar_size;
long *msg_buf;

    if(GAinitialized) return;

   /* initialize msg buffers */
    MessageSnd = (struct message_struct*)allign_page(MessageSnd);
    MessageRcv = (struct message_struct*)allign_page(MessageRcv);
#   ifdef PARAGON
       /* wire down buffer memory pages */ 
       mcmsg_wire(MessageRcv,MSG_BUF_SIZE+PAGE_SIZE);
       mcmsg_wire(MessageSnd,MSG_BUF_SIZE+PAGE_SIZE);
#   endif
    msg_buf = (long*)MessageRcv->buffer;

    /* zero in pointers in GA array */
    for(i=0;i<MAX_ARRAYS; i++) {
       GA[i].ptr  = (char**)0;
       GA[i].mapc = (int*)0;
    }
   
    ClustInfoInit(); /* learn about process configuration */
    if(ClusterMode){
       /*** current setup works only for one server per cluster ***
        *  .there are two groups of processes: compute and data servers
        *  .in each cluster, compute processes have one of them 
        *   distinguished as cluster master
        *  .cluster master participates in inter-cluster collective ops
        *   and contacts data server to create or destroy arrays
        */ 
       GAnproc = ga_msg_nnodes_() - GA_n_clus;
       GAme = ga_msg_nodeid_();
       GAmaster= cluster_master - GA_clus_id; 

       /* data servers have their message-passing node id negated */
       if(GAme > cluster_master + cluster_compute_nodes -1) GAme = -GAme; 
          else GAme -= GA_clus_id;
    }else{
       GAmaster= 0;
       GAnproc = (Integer)ga_msg_nnodes_();
       GAme = (Integer)ga_msg_nodeid_();
    }
    MPme= ga_msg_nodeid_();
    MPnproc = ga_msg_nnodes_();

    if(GAnproc > MAX_NPROC && MPme==0){
      fprintf(stderr,"current GA setup is for up to %d processors\n",MAX_NPROC);
      fprintf(stderr,"please change MAX_NPROC in globalp.h & recompile\n");
      ga_error("terminating...",0);
    }

    if(DEBUG)
    fprintf(stderr, "mode=%d, me=%d, master=%d, clusters=%d clust_nodes=%d\n",
            ClusterMode, GAme, cluster_master, GA_n_clus, cluster_nodes); 

    gaAllTrapSignals(); /* all processes set up own signal handlers */

#ifdef KSR
    bar_size = KSRbarrier_mem_req();
#else
    bar_size = 2*sizeof(long);
#endif

#ifdef SYSV 

    /*....................... System V IPC stuff  ..........................*/
    buf_size = sizeof(DoublePrecision)*cluster_compute_nodes; /*shmem buffer*/ 

    /* at the end there is shmem counter for ONE server request counter */
    shmSIZE  = bar_size + buf_size+ sizeof(Integer); 

    if(MPme == cluster_master){
         gaParentTrapSignals(); /* set up the remaining signal handlers */

        /* allocate shared memory for communication buffer and barrier  */
        if(GAnproc == 1 && USE_MALLOC){ 
           Barrier  = (int*) malloc((int)shmSIZE);/*use malloc for single proc*/
        }else
           Barrier  = (int*) Create_Shared_Region(msg_buf+1,&shmSIZE,msg_buf);

        /* Now, set up shmem communication buffer */
        shmBUF = (DoublePrecision*)( bar_size + (char *)Barrier);
        NumRecReq = (Integer*)( bar_size + buf_size +(char *)Barrier );
        *NumRecReq= 0;

#       if !defined(KSR)
           if(GAnproc > 1 ){
#             if defined(SGIUS)  || defined (SPPLOCKS)
                 CreateInitLocks(cluster_nodes+1, &lockID); 
#             else
                 /* allocate and intialize semaphores */
                 semaphoreID = SemGet(NUM_SEM);
                 SemInit(ALL_SEMS,1);
                 *((int*)shmBUF) = semaphoreID;
#             endif
           }
#       endif
    }

    /* Broadcast shmem ID to all the processes */

    if(DEBUG) fprintf(stderr,"brdcst GAme=%d\n",GAme);
    type = GA_TYPE_BRD;
    ga_brdcst_clust(type, (char*) msg_buf, SHMID_BUF_SIZE, cluster_master, 
                    ALL_CLUST_GRP);
    if(DEBUG) fprintf(stderr,"GAme=%d\n",GAme);
#   if defined(SGIUS)  || defined (SPPLOCKS)
       ga_brdcst_clust(type, (char*) &lockID, sizeof(long), cluster_master, 
                       ALL_CLUST_GRP);
#   endif
#   if defined(SGIUS)
       ga_brdcst_clust(type, (char*)lock_array,
                      (cluster_nodes+1)*sizeof(ulock_t*), cluster_master,
                       ALL_CLUST_GRP);
#   endif

    if(MPme != cluster_master){
        /* remaining processes atach to the shared memory */
        Barrier  = (int *) Attach_Shared_Region(msg_buf+1,shmSIZE, msg_buf);

        /* Now, set up shmem communication buffer */
        shmBUF = (DoublePrecision*)( bar_size + (char *)Barrier);
        NumRecReq = (Integer*)( bar_size + buf_size +(char *)Barrier );

#       if !defined(KSR)
           if(GAnproc > 1 ){
#             if defined(SGIUS) || defined (SPPLOCKS)
                 InitLocks(cluster_nodes+1, lockID); 
#             else
                 /* read semaphore_id from shmem buffer */
                 semaphoreID = *((int*)shmBUF);
#             endif
           }
#       endif
    }


    /* initialize the barrier for nproc processes  */
#   ifdef KSR
       KSRbarrier_init((int)cluster_compute_nodes, (int)GAme, 6,(char*)Barrier);
#   else
       barrier_size = cluster_compute_nodes;
       Barrier1 = Barrier +1; /*next element */
#   endif
    /*..................................................................*/
#endif


    /* set activity status for all arrays to inactive */
    for(i=0;i<max_global_array;i++)GA[i].actv=0;

    GAinitialized = 1;

    /* Initialize MA-like addressing:
     *    get addressees for the base arrays for DP and INT
     *
     * MA include files: macommon.h, macdecls.h and mafdecls.h
     *
     * DBL_MB, DCPL_MB, and INT_MB are assigned adresses of their counterparts
     *    (of the same name in MA mafdecls.h file) by calling Fortran
     *    ga_ma_base_address_() routine that calls C ga_ma_get_ptr_ to copy
     *    pointers
     */
    {
      static Integer dtype = MT_F_DBL;
      ga_ma_base_address_(&dtype, (Void**)&DBL_MB);
      if(!DBL_MB)ga_error("ga_initialize: wrong dbl pointer ", 1L);
      dtype = MT_F_INT;
      ga_ma_base_address_(&dtype, (Void**)&INT_MB);
      if(!INT_MB)ga_error("ga_initialize: wrong int pointer ", 2L);
      dtype = MT_F_DCPL;
      ga_ma_base_address_(&dtype, (Void**)&DCPL_MB);
      if(!DCPL_MB)ga_error("ga_initialize: wrong dcmpl pointer ", 3L);
    }

    /* selected processes now become data servers */
#ifdef DATA_SERVER
       if(ClusterMode) if(GAme <0) ga_SERVER(0);
#elif defined(IWAY)
    if(ClusterMode) if(GAme <0) ga_server_handler();
#endif

    /* enable interrupts on machines with interrupt receive */
#   if defined(SP1) || defined(SP) || defined (NX)
    {
       long oldmask;
       ga_init_handler(MessageRcv, TOT_MSG_SIZE );
       ga_mask(0L, &oldmask);
    }
#      ifdef SP
          mpc_setintrdelay(1);
#      endif
#   endif

#if defined(CRAY_T3D) && !defined(FLUSHCACHE)
    shmem_set_cache_inv();
#endif

    /* synchronize, and then we are ready to do real work */
#   ifdef KSR
      ga_msg_sync_(); /* barrier not ready yet */
#   else
      ga_sync_();
#   endif
    if(DEBUG)    fprintf(stderr,"ga_init done=%ld\n",GAme);
}




/*\ IS MA USED FOR ALLOCATION OF GA MEMORY ?
\*/ 
logical ga_uses_ma_()
{
#  ifdef SYSV
     return FALSE;
#  else
     return TRUE;
#  endif
}


/*\ IS MEMORY LIMIT SET ?
\*/
logical ga_memory_limited_()
{
   if(GA_memory_limited) return TRUE;
   else                  return FALSE;
}



/*\ RETURNS AMOUNT OF MEMORY on each processor IN ACTIVE GLOBAL ARRAYS 
\*/
Integer  ga_inquire_memory_()
{
Integer i, sum=0;
    for(i=0; i<max_global_array; i++) 
        if(GA[i].actv) sum += GA[i].size; 
    return(sum);
}


/*\ RETURNS AMOUNT OF GA MEMORY AVAILABLE on calling processor 
\*/
Integer ga_memory_avail_()
{
#ifdef SYSV
   return(GA_total_memory); 
#else
   Integer ma_limit = MA_inquire_avail(MT_F_BYTE);

   if ( GA_memory_limited ) return( MIN(GA_total_memory, ma_limit) );
   else return( ma_limit );
#endif
}



/*\ INITIALIZE GLOBAL ARRAY STRUCTURES and SET LIMIT ON GA MEMORY USAGE
 *  
 *  the byte limit is per processor (even for shared memory)
 *  either ga_initialize_ltd or ga_initialize must be the first 
 *         GA routine called (except ga_uses_ma)
 *  ga_initialize_ltd is another version of ga_initialize 
 *         without memory control
 *  mem_limit < 0 means "memory unlimited"
\*/
void ga_initialize_ltd_(mem_limit)
Integer *mem_limit;
{

  GA_total_memory = *mem_limit; 
  if(*mem_limit >= 0) GA_memory_limited = 1; 
  ga_initialize_();
}




/*\ CREATE A GLOBAL ARRAY
\*/
logical ga_create(type, dim1, dim2, array_name, chunk1, chunk2, g_a)
     Integer *type, *dim1, *dim2, *chunk1, *chunk2, *g_a; 
     char *array_name;
     /*
      * array_name    - a unique character string [input]
      * type          - MA type [input]
      * dim1/2        - array(dim1,dim2) as in FORTRAN [input]
      * chunk1/2      - minimum size that dimensions should
      *                 be chunked up into [input]
      *                 setting chunk1=dim1 gives distribution by rows
      *                 setting chunk2=dim2 gives distribution by columns 
      *                 Actual chunk sizes are modified so that they are
      *                 at least the min size and each process has either
      *                 zero or one chunk. 
      *                 chunk1/2 <=1 yields even distribution
      * g_a           - Integer handle for future references [output]
      */
{
int   i, nprocx, nprocy, fchunk1, fchunk2;
static Integer map1[MAX_NPROC], map2[MAX_NPROC];
Integer nblock1, nblock2;

      if(!GAinitialized) ga_error("GA not initialized ", 0);

/*      ga_sync_();*/ /* syncing in ga_create_irreg too */

      if(*type != MT_F_DBL && *type != MT_F_INT &&  *type != MT_F_DCPL)
         ga_error("ga_create: type not yet supported ",  *type);
      else if( *dim1 <= 0 )
         ga_error("ga_create: array dimension1 invalid ",  *dim1);
      else if( *dim2 <= 0)
         ga_error("ga_create: array dimension2 invalid ",  *dim2);

      /*** figure out chunking ***/
      if(*chunk1 <= 1 && *chunk2 <= 1){
        if(*dim1 == 1)      { nprocx =1; nprocy=(int)GAnproc;}
        else if(*dim2 == 1) { nprocy =1; nprocx=(int)GAnproc;}
        else {
           /* nprocx= (int)sqrt((double)GAnproc);*/
           double dproc = ((double)GAnproc*(*dim1))/((double) *dim2);
           nprocx = NEAR_INT(sqrt(dproc)); 
           nprocx = MAX(1, nprocx); /* to avoid division by 0 */
           for(i=nprocx;i>0&& (GAnproc%i);i--);
           nprocx =(int)i; nprocy=(int)GAnproc/nprocx;
        }

        fchunk1 = (int) MAX(1, *dim1/nprocx);
        fchunk2 = (int) MAX(1, *dim2/nprocy);
      }else if(*chunk1 <= 1){
        fchunk1 = (int) MAX(1, (*dim1 * *dim2)/(GAnproc* *chunk2));
        fchunk2 = (int) *chunk2;
      }else if(*chunk2 <= 1){
        fchunk1 = (int) *chunk1;
        fchunk2 = (int) MAX(1, (*dim1 * *dim2)/(GAnproc* *chunk1));
      }else{
        fchunk1 = (int) MAX(1,  *chunk1);
        fchunk2 = (int) MAX(1,  *chunk2);
      }

      fchunk1 = (int)MIN(fchunk1, *dim1);
      fchunk2 = (int)MIN(fchunk2, *dim2);

      /*** chunk size correction for load balancing ***/
      while(((*dim1-1)/fchunk1+1)*((*dim2-1)/fchunk2+1) >GAnproc){
           if(fchunk1 == *dim1 && fchunk2 == *dim2) 
                     ga_error("ga_create: chunking failed !! ", 0L);
           if(fchunk1 < *dim1) fchunk1 ++; 
           if(fchunk2 < *dim2) fchunk2 ++; 
      }

      /* Now build map arrays */
      for(i=0, nblock1=0; i< *dim1; i += fchunk1, nblock1++) map1[nblock1]=i+1;
      for(i=0, nblock2=0; i< *dim2; i += fchunk2, nblock2++) map2[nblock2]=i+1;   
      if(GAme==0&& DEBUG){
         fprintf(stderr,"blocks (%d,%d)\n",nblock1, nblock2);
         fprintf(stderr,"chunks (%d,%d)\n",fchunk1, fchunk2);
         if(GAme==0){
           for (i=0;i<nblock1;i++)fprintf(stderr," %d ",map1[i]);
           for (i=0;i<nblock2;i++)fprintf(stderr," .%d ",map2[i]);
         }
      }

      return( ga_create_irreg(type, dim1, dim2, array_name, map1, &nblock1,
                         map2, &nblock2, g_a) );
}



/*\ CREATE A GLOBAL ARRAY
 *  Fortran version
\*/
#ifdef CRAY_T3D
logical ga_create_(type, dim1, dim2, array_name, chunk1, chunk2, g_a)
     Integer *type, *dim1, *dim2, *chunk1, *chunk2, *g_a;
     _fcd array_name;
#else
logical ga_create_(type, dim1, dim2, array_name, chunk1, chunk2, g_a, slen)
     Integer *type, *dim1, *dim2, *chunk1, *chunk2, *g_a;
     char* array_name;
     int slen;
#endif
{
char buf[FNAM];
#ifdef CRAY_T3D
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

  return(ga_create(type, dim1, dim2, buf, chunk1, chunk2, g_a));
}



/*\ determine how much shared memory needs to be allocated in the cluster
\*/
Integer ga__shmem_size(g_a)
Integer g_a;
{
   Integer clust_node, ganode;
   Integer ilo, ihi, jlo, jhi, nelem;
   Integer item_size=GAsizeofM(GA[GA_OFFSET + g_a].type);

   nelem =0;
   for(clust_node=0; clust_node < cluster_compute_nodes; clust_node++){ 
       ganode = clust_node + GAmaster; 
       ga_distribution_(&g_a, &ganode, &ilo, &ihi, &jlo, &jhi);
       nelem += (ihi-ilo+1)*(jhi-jlo+1);
   }
   return (nelem*item_size);
}



/*\ set up array of pointers to (all or some) blocks of g_a
\*/
void ga__set_ptr_array(g_a, ptr)
Integer g_a;
Void    *ptr;
{
Integer  ga_handle = g_a + GA_OFFSET;

#  ifdef CRAY_T3D
     Integer i, len = sizeof(Void*);
     Integer mtype = GA_TYPE_BRD;
     GA[ga_handle].ptr[GAme] = ptr;

     /* need pointers on all procs to support global addressing */
     for(i=0; i < GAnproc; i++) ga_brdcst_(&mtype, GA[ga_handle].ptr+i,&len,&i);

#  else
     GA[ga_handle].ptr[0] = ptr;

#    ifdef SYSV 
     {
       Integer ilo, ihi, jlo, jhi, nelem, ganode, clust_node;
       Integer item_size=GAsizeofM(GA[ga_handle].type);

       /* determine pointers to memory "held" by procs in the cluster */
       for(clust_node=1; clust_node < cluster_compute_nodes; clust_node++){
           ganode = clust_node-1 + GAmaster; /* previous ganode */
           ga_distribution_(&g_a, &ganode, &ilo, &ihi, &jlo, &jhi);
           nelem = (ihi-ilo+1)*(jhi-jlo+1);
           GA[ga_handle].ptr[clust_node] = 
                          GA[ga_handle].ptr[clust_node-1] + nelem*item_size;
       }
     }
#    endif
#  endif
}


/*\ allocate local/shared memory
\*/
Integer ga__allocate_memory(type, nelem, mem_size, array_name, id, pptr)
Integer type, nelem, mem_size, *id;
Void    **pptr;
char    *array_name;
{
#ifdef SYSV
   long *msg_buf = (long*)MessageRcv->buffer, bytes=(long)mem_size;
   int adjust, diff, item_size;
   char *base;
#else
   Integer handle, index;
#endif

   *id   = INVALID_MA_HANDLE;
   *pptr = (Void*)NULL;

   if(mem_size == 0 ) return (1); /*   0 byte request is OK */ 
   if(mem_size <  0 ) return (0); /* < 0 byte request is not OK */ 

   *id   = 1;

#  ifndef SYSV
         /*............. allocate local memory ...........*/

         if(MA_alloc_get(type, nelem, array_name, &handle, &index)) 
              MA_get_pointer(handle, pptr);
         else handle = INVALID_MA_HANDLE; 
         *id   = handle;
#  else
         /*............. allocate shared memory ..........*/

         mem_size += 15;               /* Worst case alignment error */
         bytes = (long) mem_size;

         if(MPme == cluster_master){
            if(GAnproc == 1 && USE_MALLOC){
               /* for single process, shmem not needed */
               *pptr  = malloc((int)mem_size);
            }else {
               /* cluster master uses Snd buffer */
               msg_buf = (long*)MessageSnd->buffer;
               /* cluster master creates shared memory */
               *pptr = Create_Shared_Region(msg_buf+1,&bytes,msg_buf);
            }
         }

         /* every other compute node in the cluster gets shmem id(s) to attach*/
         ga_brdcst_clust(GA_TYPE_BRD, (char*)msg_buf, SHMID_BUF_SIZE,
                                      cluster_master, CLUST_GRP);

         if(MPme != cluster_master)
                    *pptr =  Attach_Shared_Region(msg_buf+1, bytes, msg_buf);

	 /* need to enforce proper, natural allignment (on size boundary)  */
         base = (type == MT_F_DBL)? (char *) DBL_MB: ((type == MT_F_DCPL)?
                                    (char *) DCPL_MB: (char *) INT_MB);

         item_size = GAsizeofM(type);
         diff = (ABS( base - (char *) *pptr)) % item_size; 
         adjust = (diff > 0) ? item_size - diff : 0;

         *id = adjust;          /* Id kludged to hold adjust */
         *pptr = (Void *) (adjust + (char *) *pptr);

#  endif

   if(DEBUG)fprintf(stderr,"me=%d ga__alloc_mem:ptr=%d id=%d\n",GAme,*pptr,*id);

   /* to return succesfully from here, we must have a non-NULL pointer */
   if(*pptr) return 1; 
   else      return 0;

}

void gai_init_struct(handle)
Integer handle;
{
     if(!GA[handle].ptr){
        int len = MIN(GAnproc, MAX_PTR);
        GA[handle].ptr = (char**)malloc(len*sizeof(char**));
     }
     if(!GA[handle].mapc){
        int len = MAPLEN;
        GA[handle].mapc = (int*)malloc(len*sizeof(int*));
     }
     if(!GA[handle].ptr)ga_error("malloc failed: ptr:",0);
     if(!GA[handle].mapc)ga_error("malloc failed: mapc:",0);
}



#if defined(FIXHEAP) && defined(CRAY_T3D)
int fix_heap=1;
#endif


/*\ CREATE A GLOBAL ARRAY -- IRREGULAR DISTRIBUTION
\*/
logical ga_create_irreg(type, dim1, dim2, array_name, map1, nblock1,
                         map2, nblock2, g_a)
     Integer *type, *dim1, *dim2, *map1, *map2, *nblock1, *nblock2, *g_a;
     char *array_name;
     /*
      * array_name    - a unique character string [input]
      * type          - MA type [input]
      * dim1/2        - array(dim1,dim2) as in FORTRAN [input]
      * nblock1       - no. of blocks first dimension is divided into [input]
      * nblock2       - no. of blocks second dimension is divided into [input]
      * map1          - no. ilo in each block [input]
      * map2          - no. jlo in each block [input]
      * g_a           - Integer handle for future references [output]
      */
{
char     op='*', *ptr = NULL;
Integer  ilo, ihi, jlo, jhi;
Integer  mem_size, nelem, mem_size_proc;
Integer  i, ga_handle, status;

      if(!GAinitialized) ga_error("GA not initialized ", 0);

      ga_sync_();

      GAstat.numcre ++; 

      if(*type != MT_F_DBL && *type != MT_F_INT &&  *type != MT_F_DCPL)
         ga_error("ga_create_irreg: type not yet supported ",  *type);
      else if( *dim1 <= 0 )
         ga_error("ga_create_irreg: array dimension1 invalid ",  *dim1);
      else if( *dim2 <= 0)
         ga_error("ga_create_irreg: array dimension2 invalid ",  *dim2);
      else if(*nblock1 <= 0)
         ga_error("ga_create_irreg: nblock1 <=0  ",  *nblock1);
      else if(*nblock2 <= 0)
         ga_error("ga_create_irreg: nblock2 <=0  ",  *nblock2);
      else if(*nblock1 * *nblock2 > GAnproc)
         ga_error("ga_create_irreg: too many blocks ",*nblock1 * *nblock2);

      if(GAme==0&& DEBUG){
        fprintf(stderr," array:%d map1:\n", *g_a);
        for (i=0;i<*nblock1;i++)fprintf(stderr," %d |",map1[i]);
        fprintf(stderr," \n array:%d map2:\n", *g_a);
        for (i=0;i<*nblock2;i++)fprintf(stderr," %d |",map2[i]);
        fprintf(stderr,"\n\n");
      }

      /*** Get next free global array handle ***/
      ga_handle =-1; i=0;
      do{
          if(!GA[i].actv) ga_handle=i;
          i++;
      }while(i<max_global_array && ga_handle==-1);
      if( ga_handle == -1)
          ga_error("ga_create: too many arrays ", (Integer)max_global_array);
      *g_a = (Integer)ga_handle - GA_OFFSET;

      /*** fill in Global Info Record for g_a ***/
      gai_init_struct(ga_handle);
      GA[ga_handle].type = *type;
      GA[ga_handle].actv = 1;
      strcpy(GA[ga_handle].name, array_name);
      GA[ga_handle].dims[0] = (int)*dim1;
      GA[ga_handle].dims[1] = (int)*dim2;
      GA[ga_handle].nblock[0] = (int) *nblock1;
      GA[ga_handle].nblock[1] = (int) *nblock2;
      GA[ga_handle].scale[0] = (double)*nblock1/(double)*dim1;
      GA[ga_handle].scale[1] = (double)*nblock2/(double)*dim2;

      /* Copy distribution maps, map1 & map2, into mapc:
       * . since nblock1*nblock2<=GAnproc,  mapc[GAnproc+1] suffices
       *   to pack everything into it;
       * . the dimension of block i is given as: MAX(mapc[i+1]-mapc[i],dim1/2)
       */
      for(i=0;i< *nblock1; i++) GA[ga_handle].mapc[i] = (int)map1[i];
      for(i=0;i< *nblock2; i++) GA[ga_handle].mapc[i+ *nblock1] = (int)map2[i];
      GA[ga_handle].mapc[*nblock1 + *nblock2] = -1; /* end of block marker */

      if(GAme ==0 && DEBUG){
         fprintf(stderr,"\nmapc %d elem\n", *nblock1 + *nblock2);
         for(i=0;i<1+*nblock1+ *nblock2;i++)
             fprintf(stderr,"%d,",GA[ga_handle].mapc[i]);
         fprintf(stderr,"\n\n");
      }

      /*** determine which portion of the array I am supposed to hold ***/
      ga_distribution_(g_a, &GAme, &ilo, &ihi, &jlo, &jhi);
      GA[ga_handle].chunk[0] = (int) (ihi-ilo+1);
      GA[ga_handle].chunk[1] = (int) (jhi-jlo+1);
      GA[ga_handle].ilo = ilo;
      GA[ga_handle].jlo = jlo;
      nelem = (ihi-ilo+1)*(jhi-jlo+1);

      /*** Memory Allocation & Initialization of GA Addressing Space ***/
#     ifdef SYSV
            mem_size = ga__shmem_size(*g_a);
            mem_size_proc = mem_size/cluster_compute_nodes;
#     else
            mem_size = mem_size_proc =  nelem * GAsizeofM(*type);  
            GA[ga_handle].id = INVALID_MA_HANDLE;
#     endif

      /* on shmem platforms, we use avg mem_size per processor in cluster */
      GA[ga_handle].size = mem_size_proc;

      /* if requested, enforce limits on memory consumption */
      if(GA_memory_limited) GA_total_memory -= mem_size_proc; 

      if(!GA_memory_limited || GA_total_memory >= 0){
         /*          fprintf(stderr,"%d allocating mem %d \n",GAme,mem_size);*/
          status =  ga__allocate_memory(*type, nelem, mem_size, array_name, 
                                         &(GA[ga_handle].id), &ptr);
      }else
          status = 0;

      if(!status) GA[ga_handle].actv=0; /* no memory allocated */

      ga_igop(GA_TYPE_GOP, &status, 1, &op); /* check if everybody succeded */

      /* determine pointers to individual blocks*/
      if(status) ga__set_ptr_array(*g_a, ptr);

#     if defined(FIXHEAP) && defined(CRAY_T3D)
         /* extend heap so that every address used in GA is valid on 
          * every T3D processor - possible to do only with ga_initialize_ltd 
          */ 

         if(fix_heap && status && GA_memory_limited){ 
           char *maxptr = 0;
           for(i=0; i< GAnproc; i++)
                    if(maxptr < GA[ga_handle].ptr[i]) 
                       maxptr = GA[ga_handle].ptr[i];
           malloc_brk(maxptr + GA_total_memory +  mem_size_proc);
           fix_heap = 0;
         } 
#     endif

#     ifdef SYSV
        /*** in cluster mode, master sends create request to data server ***/
        if(ClusterMode && (MPme == cluster_master)) {
            Integer nbytes = *nblock1 * sizeof(Integer);
            Copy(map1, MessageSnd->buffer + SHMID_BUF_SIZE, nbytes);
            Copy(map2, MessageSnd->buffer + SHMID_BUF_SIZE +nbytes,
                 *nblock2 * sizeof(Integer));
            nbytes = SHMID_BUF_SIZE+ (*nblock1 + *nblock2) *sizeof(Integer);

            /*** send g_a info + shmem id(s) to data server ***/
            ga_snd_req(0,  *dim1, *nblock1, *dim2, *nblock2, nbytes, *type,
                       GA_OP_CRE, GAme, cluster_server);
        }
#     endif

      ga_sync_();

      if(status){
         GAstat.curmem += GA[ga_handle].size;
         GAstat.maxmem  = MAX(GAstat.maxmem, GAstat.curmem);
         return(TRUE);
      }else{
         ga_destroy_(g_a);
         return(FALSE);
      }
}



/*\ CREATE A GLOBAL ARRAY -- IRREGULAR DISTRIBUTION
 *  Fortran version
\*/
#ifdef CRAY_T3D
logical ga_create_irreg_(type, dim1, dim2, array_name, map1, nblock1,
                         map2, nblock2, g_a)
     Integer *type, *dim1, *dim2, *map1, *map2, *nblock1, *nblock2, *g_a;
     _fcd array_name;
#else
logical ga_create_irreg_(type, dim1, dim2, array_name, map1, nblock1,
                         map2, nblock2, g_a, slen)
     Integer *type, *dim1, *dim2, *map1, *map2, *nblock1, *nblock2, *g_a;
     char *array_name;
     int slen;
#endif
{
char buf[FNAM];
#ifdef CRAY_T3D
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif
  return( ga_create_irreg(type, dim1, dim2, buf, map1, nblock1,
                         map2, nblock2, g_a));
}



/*\ DUPLICATE A GLOBAL ARRAY
 *  -- new array g_b will have properties of g_a
\*/
logical ga_duplicate(g_a, g_b, array_name)
     Integer *g_a, *g_b;
     char *array_name;
     /*
      * array_name    - a character string [input]
      * g_a           - Integer handle for reference array [input]
      * g_b           - Integer handle for new array [output]
      */
{
char     op='*', *ptr = NULL, **save_ptr;
Integer  mem_size, mem_size_proc, nelem;
Integer  i, ga_handle, status;
int      *save_mapc;

      ga_sync_();

      GAstat.numcre ++; 

      ga_check_handleM(g_a,"ga_duplicate");       

      /* find a free global_array handle for g_b */
      ga_handle =-1; i=0;
      do{
        if(!GA[i].actv) ga_handle=i;
        i++;
      }while(i<max_global_array && ga_handle==-1);
      if( ga_handle == -1)
          ga_error("ga_duplicate: too many arrays ", (Integer)max_global_array);
      *g_b = (Integer)ga_handle - GA_OFFSET;

      gai_init_struct(ga_handle);

      /*** copy content of the data structure ***/
      save_ptr = GA[ga_handle].ptr;
      save_mapc = GA[ga_handle].mapc;
      GA[ga_handle] = GA[GA_OFFSET + *g_a];
      strcpy(GA[ga_handle].name, array_name);
      GA[ga_handle].ptr = save_ptr;
      GA[ga_handle].mapc = save_mapc;
      for(i=0;i<MAPLEN; i++)GA[ga_handle].mapc[i] = GA[GA_OFFSET+ *g_a].mapc[i];

      /*** Memory Allocation & Initialization of GA Addressing Space ***/
      nelem = GA[ga_handle].chunk[1]*GA[ga_handle].chunk[0];
#     ifdef SYSV
            /* we have to recompute mem_size for the cluster */
            mem_size = ga__shmem_size(*g_a);
            mem_size_proc = mem_size/cluster_compute_nodes;
            if(GA[ga_handle].size != mem_size_proc)
                  ga_error("ga_duplicate: mem_size_proc error ",mem_size_proc);
#     else
            mem_size = mem_size_proc = GA[ga_handle].size; 
            GA[ga_handle].id = INVALID_MA_HANDLE;
#     endif


      /* if requested, enforce limits on memory consumption */
      if(GA_memory_limited) GA_total_memory -= mem_size_proc;

      if(!GA_memory_limited || GA_total_memory >= 0)
          status =  ga__allocate_memory(GA[ga_handle].type, nelem, mem_size, 
                                        array_name, &(GA[ga_handle].id), &ptr);
      else
          status = 0;

      if(!status) GA[ga_handle].actv=0; /* no memory allocated */
      ga_igop(GA_TYPE_GOP, &status, 1, &op); /* check if everybody succeded */

      /* determine pointers to individual blocks*/
      if(status) ga__set_ptr_array(*g_b, ptr);

#     ifdef SYSV
        if( ClusterMode && MPme == cluster_master){
                ga_snd_req(*g_a, 0, 0, 0, 0, SHMID_BUF_SIZE, 0, GA_OP_DUP,
                           GAme, cluster_server);
        }
#     endif

      ga_sync_();

      if(status){
         GAstat.curmem += GA[ga_handle].size;
         GAstat.maxmem  = MAX(GAstat.maxmem, GAstat.curmem);
         return(TRUE);
      }else{ 
         ga_destroy_(g_b);
         return(FALSE);
      }
}


/*\ DUPLICATE A GLOBAL ARRAY
 *  Fortran version
\*/
#ifdef CRAY_T3D
logical ga_duplicate_(g_a, g_b, array_name)
     Integer *g_a, *g_b;
     _fcd array_name;
#else
logical ga_duplicate_(g_a, g_b, array_name, slen)
     Integer *g_a, *g_b;
     char  *array_name;
     int   slen;
#endif
{
char buf[FNAM];
#ifdef CRAY_T3D
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

  return(ga_duplicate(g_a, g_b, buf));
}




/*\ DESTROY A GLOBAL ARRAY
\*/
logical ga_destroy_(g_a)
        Integer *g_a;
{
Integer ga_handle = GA_OFFSET + *g_a;

    ga_sync_();
    GAstat.numdes ++; /*regardless of array status we count this call */

    /* fails if handle is out of range or array not active */
    if(ga_handle < 0 || ga_handle >= max_global_array) return FALSE;
    if(GA[ga_handle].actv==0) return FALSE;       
 
#   ifdef SYSV 
      if(GAnproc == 1 && USE_MALLOC){
            free(GA[ga_handle].ptr[0]-GA[ga_handle].id);   /* Id = adjust */
      }else{
         if(MPme == cluster_master){
            /* Now, deallocate shared memory */
            if(GA[ga_handle].ptr[0]){
                     Free_Shmem_Ptr(GA[ga_handle].id, GA[ga_handle].size,
                                    GA[ga_handle].ptr[0]-GA[ga_handle].id);
               GA[ga_handle].ptr[0]=NULL;
            }
            if(ClusterMode) 
               ga_snd_req(*g_a, 0,0,0,0,0,0, GA_OP_DES, GAme, cluster_server);
         } 
      }
#   else
      if(GA[ga_handle].id != INVALID_MA_HANDLE) MA_free_heap(GA[ga_handle].id);
#   endif

    if(GA_memory_limited) GA_total_memory += GA[ga_handle].size;
    GA[ga_handle].actv = 0;     
    GAstat.curmem -= GA[ga_handle].size;
    return(TRUE);
}

    
     
/*\ TERMINATE GLOBAL ARRAY STRUCTURES
 *
 *  all GA arrays are destroyed & shared memory is dealocated
 *  GA routines (except for ga_initialize) should not be called thereafter 
\*/
void ga_terminate_() 
{
Integer i, handle;

    if(!GAinitialized) return;
    for (i=0;i<max_global_array;i++){
          handle = i - GA_OFFSET ;
          if(GA[i].actv) ga_destroy_(&handle);
          if(GA[i].ptr) free(GA[i].ptr);
          if(GA[i].mapc) free(GA[i].mapc);
    }
    
    ga_sync_();

    GA_total_memory = -1; /* restore "unlimited" memory usage status */
    GA_memory_limited = 0;

#   ifdef SYSV
      if(GAnproc == 1 && USE_MALLOC){
         free((char*)Barrier);
         GAinitialized = 0;
         return;
      }
#   endif

    if(MPme == cluster_master){
       if(ClusterMode) 
             ga_snd_req(0, 0, 0, 0, 0, 0, 0, GA_OP_END, GAme, cluster_server);
         ga_clean_resources();
         gaParentRestoreSignals();
    }
    GAinitialized = 0;
}   

    
/*\ IS ARRAY ACTIVE/INACTIVE
\*/ 
Integer ga_verify_handle_(g_a)
     Integer *g_a;
{
  return (Integer)
    ((*g_a + GA_OFFSET>= 0) && (*g_a + GA_OFFSET< max_global_array) && 
             GA[GA_OFFSET + (*g_a)].actv);
}


/*\ determine if access to <proc> data through shared memory is possible 
\*/
logical gaDirectAccess(proc)
   Integer proc;
{
#ifdef SHMEM
#  ifndef CRAY_T3D
     if(ClusterMode && (ClusterID(proc) != GA_clus_id))
         return(FALSE);
     else
#  endif
   return(TRUE);
#else
   if(proc == GAme) return(TRUE);
   else return(FALSE);
#endif
}


/*\ computes leading dim and pointer to (i,j) element owned by processor proc
\*/
void gaShmemLocation2(proc, g_a, i, j, ptr, ld)
Integer g_a, i, j, proc, *ld;
char **ptr;
{
Integer ilo, ihi, jlo, jhi, offset, proc_place,g_handle=g_a+GA_OFFSET;

      ga_ownsM(g_handle, proc, ilo, ihi, jlo, jhi);
      if(i<ilo || i>ihi || j<jlo || j>jhi)
                           ga_error(" gaShmemLocation: invalid (i,j) ",GAme);

      offset = (i - ilo) + (ihi-ilo+1)*(j-jlo);

      /* find location of the proc in current cluster pointer array */
      if(! ClusterMode) proc_place = proc;
      else{
         proc_place = proc - GAmaster;
         if(proc_place < 0 || proc_place >= cluster_compute_nodes){
              ga_error(" gaShmemLocation: invalid process ",proc);
         }
      }

      *ptr = GA[g_handle].ptr[proc_place] + offset*GAsizeofM(GA[g_handle].type);
      *ld = ihi-ilo+1;
}


#ifdef SHMEM
#   define ProcPlace(proc_place, proc)                                         \
    {                                                                          \
      if(! ClusterMode) proc_place = proc;                                     \
      else{                                                                    \
         proc_place = proc - GAmaster;                                         \
         if(proc_place < 0 || proc_place >= cluster_compute_nodes){            \
              ga_error(" gaShmemLocation: invalid process ",proc);             \
         }                                                                     \
      }                                                                        \
    }
#else
#   define ProcPlace(proc_place, proc) proc_place=0
#endif
  

#define gaShmemLocation(proc, g_a, _i, _j, ptr_loc, ld)                        \
{                                                                              \
Integer _ilo, _ihi, _jlo, _jhi, offset, proc_place, g_handle=(g_a)+GA_OFFSET;  \
                                                                               \
      ga_ownsM(g_handle, (proc), _ilo, _ihi, _jlo, _jhi);                      \
      if((_i)<_ilo || (_i)>_ihi || (_j)<_jlo || (_j)>_jhi){                    \
          sprintf(err_string,"%s: p=%d invalid i/j (%d,%d) >< (%d:%d,%d:%d)",  \
                 "gaShmemLocation", proc, (_i),(_j), _ilo, _ihi, _jlo, _jhi);  \
          ga_error(err_string, g_a );                                          \
      }                                                                        \
      offset = ((_i) - _ilo) + (_ihi-_ilo+1)*((_j)-_jlo);                      \
                                                                               \
      /* find location of the proc in current cluster pointer array */         \
      ProcPlace(proc_place, proc);                                             \
      *(ptr_loc) = GA[g_handle].ptr[proc_place] +                              \
                   offset*GAsizeofM(GA[g_handle].type);                        \
      *(ld) = _ihi-_ilo+1;                                                     \
}


#define ga_check_regionM(g_a, ilo, ihi, jlo, jhi, string){                     \
   if (*(ilo) <= 0 || *(ihi) > GA[GA_OFFSET + *(g_a)].dims[0] ||               \
       *(jlo) <= 0 || *(jhi) > GA[GA_OFFSET + *(g_a)].dims[1] ||               \
       *(ihi) < *(ilo) ||  *(jhi) < *(jlo)){                                   \
       sprintf(err_string,"%s:request(%d:%d,%d:%d) out of range (1:%d,1:%d)",  \
               string, *(ilo), *(ihi), *(jlo), *(jhi),                         \
               GA[GA_OFFSET + *(g_a)].dims[0], GA[GA_OFFSET + *(g_a)].dims[1]);\
       ga_error(err_string, *(g_a));                                           \
   }                                                                           \
}


/*\ local put of a 2-dimensional patch of data into a global array
\*/
void ga_put_local(g_a, ilo, ihi, jlo, jhi, buf, offset, ld, proc)
   Integer g_a, ilo, ihi, jlo, jhi,  ld, offset, proc;
   Void *buf;
{
char     *ptr_glob, *ptr_loc;
Integer  ld_glob, rows, cols, type;

   GA_PUSH_NAME("ga_put_local");

   type = GA[GA_OFFSET + g_a].type;
   rows = ihi - ilo +1;
   cols = jhi - jlo +1;

   if(GAme == proc && !in_handler) GAbytes.putloc += (double)GAsizeofM(type)*rows*cols;
   gaShmemLocation(proc, g_a, ilo, jlo, &ptr_glob, &ld_glob);
   ptr_loc = (char *)buf  + GAsizeofM(type) * offset;

   Copy2DTo(type, proc, &rows, &cols, ptr_loc, &ld, ptr_glob, &ld_glob);    

   GA_POP_NAME;
}



/*\ remote put of a 2-dimensional patch of data into a global array
\*/
void ga_put_remote(g_a, ilo, ihi, jlo, jhi, buf, offset, ld, proc)
   Integer g_a, ilo, ihi, jlo, jhi, ld, offset, proc;
   Void *buf;
{
char     *ptr_src, *ptr_dst;
Integer  type, rows, cols, msglen;

   if(proc<0)ga_error(" ga_put_remote: invalid process ",proc);

   type = GA[GA_OFFSET + g_a].type;
   rows = ihi - ilo +1;
   cols = jhi - jlo +1;

   /* Copy patch [ilo:ihi, jlo:jhi] into MessageBuffer */
   ptr_dst = (char*)MessageSnd->buffer;
   ptr_src = (char *)buf  + GAsizeofM(type)* offset;

   Copy2D(type, &rows, &cols, ptr_src, &ld, ptr_dst, &rows); 

   msglen = rows*cols*GAsizeofM(type);
   ga_snd_req(g_a, ilo,ihi,jlo,jhi, msglen, type, GA_OP_PUT,
              proc, DataServer(proc));
}


/*\ PUT A 2-DIMENSIONAL PATCH OF DATA INTO A GLOBAL ARRAY 
\*/
void ga_put_(g_a, ilo, ihi, jlo, jhi, buf, ld)
   Integer  *g_a,  *ilo, *ihi, *jlo, *jhi,  *ld;
   Void  *buf;
{
Integer  p, np, proc, idx, type=GA[GA_OFFSET + *g_a].type;
Integer  ilop, ihip, jlop, jhip, offset;

#ifdef GA_TRACE
   trace_stime_();
#endif

      GA_PUSH_NAME("ga_put");
      GAstat.numput++;
      GAbytes.puttot += (double)GAsizeofM(type)*(*ihi-*ilo+1)*(*jhi-*jlo+1);

      if(!ga_locate_region_(g_a, ilo, ihi, jlo, jhi, map, &np )){
          sprintf(err_string, "cannot locate region (%d:%d,%d:%d)",
                  *ilo, *ihi, *jlo, *jhi);
          ga_error(err_string, *g_a);
      }

      gaPermuteProcList(np); 
      for(idx=0; idx<np; idx++){
          p = (Integer)ProcListPerm[idx];

          ilop = map[p][0];
          ihip = map[p][1];
          jlop = map[p][2];
          jhip = map[p][3];
          proc = map[p][4];

          if(gaDirectAccess(proc)){

             offset = (jlop - *jlo)* *ld + ilop - *ilo;
             ga_put_local(*g_a, ilop, ihip, jlop, jhip, buf, offset, *ld, proc);

          }else{
            /* number of messages determined by message-buffer size */

            Integer ilo_chunk, ihi_chunk, jlo_chunk, jhi_chunk;
            Integer TmpSize = MSG_BUF_SIZE/GAsizeofM(type);
            Integer ilimit  = MIN(TmpSize, ihip-ilop+1);
            Integer jlimit  = MIN(TmpSize/ilimit, jhip-jlop+1);

            for(jlo_chunk = jlop; jlo_chunk <= jhip; jlo_chunk += jlimit){
               jhi_chunk  = MIN(jhip, jlo_chunk+jlimit-1);
               for( ilo_chunk = ilop; ilo_chunk<= ihip; ilo_chunk += ilimit){

                  ihi_chunk = MIN(ihip, ilo_chunk+ilimit-1);
                  offset = (jlo_chunk - *jlo)* *ld + ilo_chunk - *ilo;
                  ga_put_remote(*g_a, ilo_chunk, ihi_chunk,
                                jlo_chunk, jhi_chunk, buf, offset, *ld, proc);

               }
            }
         }
      }
      GA_POP_NAME;

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_PUT; 
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
}


/*\ local get of a 2-dimensional patch of data into a global array
\*/
void ga_get_local(g_a, ilo, ihi, jlo, jhi, buf, offset, ld, proc)
   Integer g_a, ilo, ihi, jlo, jhi, ld, offset, proc;
   Void *buf;
{
char     *ptr_glob, *ptr_loc;
Integer  ld_glob, rows, cols, type;

   GA_PUSH_NAME("ga_get_local");

   type = GA[GA_OFFSET + g_a].type;
   rows = ihi - ilo +1;
   cols = jhi - jlo +1;
   if(GAme == proc && !in_handler) GAbytes.getloc += (double)GAsizeofM(type)*rows*cols;

   gaShmemLocation(proc, g_a, ilo, jlo, &ptr_glob, &ld_glob);

   ptr_loc = (char *)buf  + GAsizeofM(type) * offset;
   if(proc==GAme) FLUSH_CACHE;

   Copy2DFrom(type, proc, &rows, &cols, ptr_glob, &ld_glob, ptr_loc, &ld);

   GA_POP_NAME;
}



/*\  get a patch of an array from remote processor
\*/
void ga_get_remote(g_a, ilo, ihi, jlo, jhi, buf, offset, ld, proc)
   Integer g_a, ilo, ihi, jlo, jhi, ld, offset, proc;
   Void *buf;
{
char     *ptr_src, *ptr_dst;
Integer  type, rows, cols, len, to, from, msglen, expected_len, need_copy;
msgid_t  msgid_snd, msgid_rcv;

   if(proc<0)ga_error(" get_remote: invalid process ",proc);
   type = GA[GA_OFFSET + g_a].type;
   rows = ihi - ilo +1;
   cols = jhi - jlo +1;
   msglen = rows*cols*GAsizeofM(type);
   to = DataServer(proc);

   expected_len = msglen;

#  ifdef IWAY
     expected_len += MSG_HEADER_SIZE;
     need_copy = 1;
#  else
     /* this stuff is to avoid double buffering if possible */
     need_copy=cols-1; /* true only if cols > 1 */
#  endif

   if(need_copy)
      ptr_src = MessageSnd->buffer;  /* data arrives to the same msg buffer */
   else
      ptr_src = (char *)buf + GAsizeofM(type)* offset;/*arrives to user buffer*/

#  if defined(NX) || defined(SP1) || defined(SP)
      len = expected_len;
      msgid_rcv = ga_msg_ircv(GA_TYPE_GET,  ptr_src, expected_len, to);
      ga_snd_req(g_a, ilo,ihi,jlo,jhi, (Integer)0, type, GA_OP_GET,proc,to);
      ga_msg_wait(msgid_rcv, &from, &len);

#  else
      ga_snd_req(g_a, ilo,ihi,jlo,jhi, (Integer)0, type, GA_OP_GET,proc,to);
      ga_msg_rcv(GA_TYPE_GET, ptr_src, expected_len, &len, to,&from);
#  endif

   if(len != expected_len)ga_error("get_remote:wrong msg length",len); 

   if(need_copy){
      /* Copy patch [ilo:ihi, jlo:jhi] from MessageBuffer */
      ptr_dst = (char *)buf  + GAsizeofM(type)* offset;
      Copy2D(type, &rows, &cols, ptr_src, &rows, ptr_dst, &ld);  
   }

}


/*\ GET A 2-DIMENSIONAL PATCH OF DATA FROM A GLOBAL ARRAY
\*/
void ga_get_(g_a, ilo, ihi, jlo, jhi, buf, ld)
   Integer  *g_a, *ilo, *ihi, *jlo, *jhi,  *ld;
   Void     *buf;
{
Integer p, np, proc, idx, type=GA[GA_OFFSET + *g_a].type;
Integer ilop, ihip, jlop, jhip, offset;

#ifdef GA_TRACE
   trace_stime_();
#endif

      GA_PUSH_NAME("ga_get");
      GAstat.numget++;
      GAbytes.gettot += (double)GAsizeofM(type)*(*ihi-*ilo+1)*(*jhi-*jlo+1);

      if(!ga_locate_region_(g_a, ilo, ihi, jlo, jhi, map, &np )){
          sprintf(err_string, "cannot locate region (%d:%d,%d:%d)",
                  *ilo, *ihi, *jlo, *jhi);
          ga_error(err_string, *g_a);
      }

      gaPermuteProcList(np);

      for(idx=0; idx<np; idx++){
          p = (Integer)ProcListPerm[idx];
          ilop = map[p][0];
          ihip = map[p][1];
          jlop = map[p][2];
          jhip = map[p][3];
          proc = map[p][4]; 

          if(gaDirectAccess(proc)){

             offset = (jlop - *jlo)* *ld + ilop - *ilo;
             ga_get_local(*g_a, ilop, ihip, jlop, jhip, buf, offset, *ld, proc);

          }else{
            /* number of messages determined by message-buffer size */

            Integer ilo_chunk, ihi_chunk, jlo_chunk, jhi_chunk;
            Integer TmpSize;
            Integer ilimit;
            Integer jlimit;
#           ifdef SP
               int i_on;
#           endif
#           if defined(IWAY) && defined(SP1)
               TmpSize = IWAY_MSG_BUF_SIZE/GAsizeofM(type);
#           else
               TmpSize = MSG_BUF_SIZE/GAsizeofM(GA[GA_OFFSET + *g_a].type);
#           endif
            ilimit  = MIN(TmpSize, ihip-ilop+1);
            jlimit  = MIN(TmpSize/ilimit, jhip-jlop+1);

#           if defined(PARAGON)||defined(SP1) || defined(SP)
              /* this limits column chunking to 1 for larger number of rows */
              /**** it is an optimization only -- comment out if fails ******/
              if(ilimit > 2048) jlimit  = 1;
#           endif

#           ifdef SP
              i_on = mpc_queryintr();
              mpc_disableintr();
#           endif
            for(jlo_chunk = jlop; jlo_chunk <= jhip; jlo_chunk += jlimit){
               jhi_chunk  = MIN(jhip, jlo_chunk+jlimit-1);
               for( ilo_chunk = ilop; ilo_chunk<= ihip; ilo_chunk += ilimit){

                  ihi_chunk = MIN(ihip, ilo_chunk+ilimit-1);
                  offset = (jlo_chunk - *jlo)* *ld + ilo_chunk - *ilo;
                  ga_get_remote(*g_a, ilo_chunk, ihi_chunk,
                                jlo_chunk, jhi_chunk, buf, offset, *ld, proc);

               }
            }
#           ifdef SP
              if(i_on) mpc_enableintr();
#           endif

          }
      }
      GA_POP_NAME;

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_GET; 
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
}



/*\ local accumulate 
\*/
void ga_acc_local(g_a, ilo, ihi, jlo, jhi, buf, offset, ld, proc, alpha)
   Integer g_a, ilo, ihi, jlo, jhi, ld, offset, proc;
   DoublePrecision *alpha, *buf;
{
char     *ptr_src, *ptr_dst;
Integer  item_size, ldp, rows, cols, type = GA[GA_OFFSET + g_a].type;
#ifdef CRAY_T3D
#  define LEN_ACC_BUF 100
   DoublePrecision acc_buffer[LEN_ACC_BUF], *pbuffer, *ptr;
   Integer j, elem, words, handle, index;
#endif

   GA_PUSH_NAME("ga_acc_local"); 
   item_size =  GAsizeofM(type);
   gaShmemLocation(proc, g_a, ilo, jlo, &ptr_dst, &ldp);

#  ifdef CRAY_T3D
     elem = ihi - ilo +1;
     words = elem*item_size/8;
     if(proc != GAme){
        ptr = (DoublePrecision *) ptr_dst;
        if(words>LEN_ACC_BUF){
            if(!MA_push_get(MT_F_DBL, words, "ga_acc_temp", &handle, &index))
                ga_error("allocation of ga_acc buffer failed ",GAme);
            MA_get_pointer(handle, &pbuffer);
        }else pbuffer = acc_buffer;

        LOCK(g_a, proc, ptr_dst);
           for (j = 0;  j < jhi-jlo+1;  j++){
              ptr_dst = (char *)ptr  + item_size* j *ldp;
              ptr_src = (char *)buf  + item_size* (j*ld + offset );
   
              CopyElemFrom(ptr_dst, pbuffer, words, proc);
              if(type== MT_F_DBL)
                 dacc_column(alpha, pbuffer, ptr_src, elem );
              else
                 zacc_column(alpha, pbuffer, ptr_src, elem );
              CopyElemTo(pbuffer, ptr_dst, words, proc);
           }
           /* _remote_write_barrier(); Howard's func.*/
        UNLOCK(g_a, proc, ptr_dst);
        if(words>LEN_ACC_BUF) MA_pop_stack(handle);
        GA_POP_NAME;
        return;
     }
    /* cache coherency problem on T3D */
     FLUSH_CACHE;
#  endif
     ptr_src = (char *)buf   + item_size * offset;
     rows = ihi - ilo +1;
     cols = jhi-jlo+1;
     if(GAme == proc && !in_handler) GAbytes.accloc += (double)GAsizeofM(type)*rows*cols;

     if(GAnproc>1) LOCK(g_a, proc, ptr_dst);
       if(type == MT_F_DBL){
          accumulate(alpha, rows, cols, (DoublePrecision*)ptr_dst, ldp, 
                                        (DoublePrecision*)ptr_src, ld );
       }else{
          zaccumulate(alpha, rows, cols, (DoubleComplex*)ptr_dst, ldp, 
                                         (DoubleComplex*)ptr_src, ld );
       }
#      if defined(CRAY_T3D)
        /* flush write buffer before unlocking */
        _memory_barrier();
#      endif
     if(GAnproc>1) UNLOCK(g_a, proc, ptr_dst);
   GA_POP_NAME;
}


/*\ remote accumulate of a 2-dimensional patch of data into a global array
\*/
void ga_acc_remote(g_a, ilo, ihi, jlo, jhi, buf, offset, ld, proc, alpha)
   Integer g_a, ilo, ihi, jlo, jhi, ld, offset, proc;
   DoublePrecision *alpha, *buf;
{
char     *ptr_src, *ptr_dst;
Integer  type, rows, cols, msglen;

   if(proc<0)ga_error(" acc_remote: invalid process ",proc);
   type = GA[GA_OFFSET + g_a].type;
   rows = ihi - ilo +1;
   cols = jhi - jlo +1;

   /* Copy patch [ilo:ihi, jlo:jhi] into MessageBuffer */
   ptr_dst = (char*)MessageSnd->buffer;
   ptr_src = (char *)buf  + GAsizeofM(type)* offset;

   Copy2D(type, &rows, &cols, ptr_src, &ld, ptr_dst, &rows);

   /* append alpha at the end */
   ptr_dst += rows*cols*GAsizeofM(type);
   if(type==MT_F_DBL)*(DoublePrecision*)ptr_dst= *alpha; 
   else *(DoubleComplex*)ptr_dst= *(DoubleComplex*)alpha;

   msglen = rows*cols*GAsizeofM(type) + GAsizeofM(type); /* plus alpha */
   ga_snd_req(g_a, ilo,ihi,jlo,jhi, msglen, type, GA_OP_ACC,
              proc, DataServer(proc));
}


/*\ ACCUMULATE OPERATION ON A 2-DIMENSIONAL PATCH OF GLOBAL ARRAY
 *
 *  g_a += alpha * patch
\*/
void ga_acc_(g_a, ilo, ihi, jlo, jhi, buf, ld, alpha)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *ld;
   DoublePrecision *buf, *alpha;
{
   Integer p, np, proc, idx;
   Integer ilop, ihip, jlop, jhip, offset, type = GA[GA_OFFSET + *g_a].type;

#ifdef GA_TRACE
   trace_stime_();
#endif

   GA_PUSH_NAME("ga_acc");
   GAstat.numacc++;
   GAbytes.acctot += (double)GAsizeofM(type)*(*ihi-*ilo+1)*(*jhi-*jlo+1);

   if(!ga_locate_region_(g_a, ilo, ihi, jlo, jhi, map, &np )){
          sprintf(err_string, "cannot locate region (%d:%d,%d:%d)",
                  *ilo, *ihi, *jlo, *jhi);
          ga_error(err_string, *g_a);
   }

   if (type != MT_F_DBL && type !=  MT_F_DCPL) 
                       ga_error(" ga_acc: type not supported ",*g_a);

   gaPermuteProcList(np); /* prepare permuted list of indices */
   for(idx=0; idx<np; idx++){
       p = (Integer)ProcListPerm[idx];

       ilop = map[p][0];
       ihip = map[p][1];
       jlop = map[p][2];
       jhip = map[p][3];
       proc = map[p][4];

       if(gaDirectAccess(proc)){

          offset = (jlop - *jlo)* *ld + ilop - *ilo;
          ga_acc_local(*g_a, ilop, ihip, jlop, jhip, buf, offset, *ld,
                       proc, alpha);

       }else{
         /* number of messages determined by message-buffer size */
         /* alpha will be appended at the end of message */

         Integer TmpSize = (MSG_BUF_SIZE - GAsizeofM(type))/GAsizeofM(type);
         Integer ilimit  = MIN(TmpSize, ihip-ilop+1);
         Integer jlimit  = MIN(TmpSize/ilimit, jhip-jlop+1);
         Integer ilo_chunk, ihi_chunk, jlo_chunk, jhi_chunk;

         for(jlo_chunk = jlop; jlo_chunk <= jhip; jlo_chunk += jlimit){
            jhi_chunk  = MIN(jhip, jlo_chunk+jlimit-1);
            for( ilo_chunk = ilop; ilo_chunk<= ihip; ilo_chunk += ilimit){

                  ihi_chunk = MIN(ihip, ilo_chunk+ilimit-1);
                  offset = (jlo_chunk - *jlo)* *ld + ilo_chunk - *ilo;
                  ga_acc_remote(*g_a, ilo_chunk, ihi_chunk,
                         jlo_chunk, jhi_chunk, buf, offset, *ld, proc, alpha);

            }
         }
      }
  }

  GA_POP_NAME;

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_ACC; 
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
}



/*\ PROVIDE ACCESS TO A PATCH OF A GLOBAL ARRAY
\*/
void ga_access_(g_a, ilo, ihi, jlo, jhi, index, ld)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *index, *ld;
{
register char *ptr;
Integer  item_size, proc_place, handle = GA_OFFSET + *g_a;


   ga_check_handleM(g_a, "ga_access");
   ga_check_regionM(g_a, ilo, ihi, jlo, jhi, "ga_access");

   item_size = (int) GAsizeofM(GA[GA_OFFSET + *g_a].type);

#  ifdef SHMEM
     proc_place = GAme -  GAmaster;
#  else
     proc_place = 0;
#  endif

   ptr = GA[handle].ptr[proc_place] + item_size * ( (*jlo - GA[handle].jlo )
         *GA[handle].chunk[0] + *ilo - GA[handle].ilo);
   *ld    = GA[handle].chunk[0];  
   FLUSH_CACHE;


   /*
    * return address of the patch  as the distance in bytes
    * from the reference address
    *
    * .in Fortran we need only the index to the type array: dbl_mb or int_mb
    *  that are elements of COMMON in the the mafdecls.h include file
    *
    * .in C we need both the index and the pointer
    *
    */ 
   /* compute index and check if it is correct */
   switch (GA[handle].type){
     case MT_F_DBL: 
        *index = (Integer) (ptr - (char*)DBL_MB);
        if(ptr != ((char*)DBL_MB)+ *index ){ 
               ga_error("ga_access: MA addressing problem dbl - index",handle);
        }
        break;

     case MT_F_DCPL:
        *index = (Integer) (ptr - (char*)DCPL_MB);
        if(ptr != ((char*)DCPL_MB)+ *index ){
               ga_error("ga_access: MA addressing problem dcpl - index",handle);
        }
        break;

     case MT_F_INT:
        *index = (Integer) (ptr - (char*)INT_MB);
        if(ptr != ((char*)INT_MB) + *index) {
               ga_error("ga_access: MA addressing problem int - index",handle);
        }
        break;

     default: ga_error(" ga_access: type not supported ",-1L);

   }

   /* check the allignment */
   if(*index % item_size)
       ga_error(" ga_access: base address misallignment ",(long)index);

   /* adjust index according to the data type */
   *index /= item_size;

   /* adjust index for Fortran addressing */
   (*index) ++ ;
}



/*\ RELEASE ACCESS TO A PATCH OF A GLOBAL ARRAY
\*/
void ga_release_(g_a, ilo, ihi, jlo, jhi)
     Integer *g_a, *ilo, *ihi, *jlo, *jhi;
{}


/*\ RELEASE ACCESS & UPDATE A PATCH OF A GLOBAL ARRAY
\*/
void ga_release_update_(g_a, ilo, ihi, jlo, jhi)
     Integer *g_a, *ilo, *ihi, *jlo, *jhi;
{}



/*\ INQUIRE POPERTIES OF A GLOBAL ARRAY
 *  Fortran version
\*/ 
void ga_inquire_(g_a,  type, dim1, dim2)
      Integer *g_a, *dim1, *dim2, *type;
{
   ga_check_handleM(g_a, "ga_inquire");
   *type       = GA[GA_OFFSET + *g_a].type;
   *dim1       = GA[GA_OFFSET + *g_a].dims[0];
   *dim2       = GA[GA_OFFSET + *g_a].dims[1];
}



/*\ INQUIRE NAME OF A GLOBAL ARRAY
 *  Fortran version
\*/
#ifdef CRAY_T3D
void ga_inquire_name_(g_a, array_name)
      Integer *g_a;
      _fcd    array_name;
{
   c2fstring(GA[GA_OFFSET+ *g_a].name,_fcdtocp(array_name),_fcdlen(array_name));
}
#else
void ga_inquire_name_(g_a, array_name, len)
      Integer *g_a;
      char    *array_name;
      int     len;
{
   c2fstring(GA[GA_OFFSET + *g_a].name, array_name, len);
}
#endif




/*\ INQUIRE NAME OF A GLOBAL ARRAY
 *  C version
\*/
void ga_inquire_name(g_a, array_name)
      Integer *g_a;
      char    *array_name;
{ 
   ga_check_handleM(g_a, "ga_inquire_name");
   strcpy(array_name, GA[GA_OFFSET + *g_a].name);
}



/*\ RETURN COORDINATES OF A GA PATCH ASSOCIATED WITH CALLING PROCESSOR
\*/
void ga_distribution_(g_a, proc, ilo, ihi, jlo, jhi)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *proc;
{
register Integer iproc, jproc, loc, ga_handle;

   ga_check_handleM(g_a, "ga_distribution");

   ga_handle = (GA_OFFSET + *g_a);
 
   if(*proc > GA[ga_handle].nblock[0] * GA[ga_handle].nblock[1] - 1 || *proc<0){
         *ilo = (Integer)0;    *jlo = (Integer)0; 
         *ihi = (Integer)-1;   *jhi = (Integer)-1; 
   }else{
         jproc = (*proc)/GA[ga_handle].nblock[0]; 
         iproc = (*proc)%GA[ga_handle].nblock[0]; 
         loc = iproc;

         *ilo = GA[ga_handle].mapc[loc]; *ihi = GA[ga_handle].mapc[loc+1] -1; 

         /* correction to find the right spot in mapc*/
         loc = jproc + GA[ga_handle].nblock[0];
         *jlo = GA[ga_handle].mapc[loc]; *jhi = GA[ga_handle].mapc[loc+1] -1; 

         if( iproc == GA[ga_handle].nblock[0] -1) *ihi = GA[ga_handle].dims[0];
         if( jproc == GA[ga_handle].nblock[1] -1) *jhi = GA[ga_handle].dims[1];
   }
}



/*\ RETURN COORDINATES OF ARRAY BLOCK HELD BY A PROCESSOR
\*/
void ga_proc_topology_(g_a, proc, pr, pc)
   Integer *g_a, *proc, *pr, *pc;
{
register Integer ga_handle;

   ga_check_handleM(g_a, "ga_proc_topology");

   ga_handle = (GA_OFFSET + *g_a);

   if(*proc > GA[ga_handle].nblock[0] * GA[ga_handle].nblock[1] - 1 || *proc<0){
         *pc = -1; *pr = -1;
   }else{
         *pc = (*proc)/GA[ga_handle].nblock[0];
         *pr = (*proc)%GA[ga_handle].nblock[0];
   }
}



/*\ finds block i that 'elem' belongs to: map[i]>= elem < map[i+1]
\*/
int findblock2(map,n,scale,elem)
    int *map, n, elem;
    double scale;
{
int candidate, found, b;

    candidate = (int)(scale*elem)-1;
    if(candidate<0)candidate =0;
    found = 0;
    if(map[candidate]>elem){ /* search downward */
         b= candidate-1;
         while(b>=0){
            found = (map[b]<=elem);
            if(found)break;
            b--;
         } 
    }else{ /* search upward */
         b= candidate;
         while(b<n-1){ 
            found = (map[b+1]>elem);
            if(found)break;
            b++;
         }
    }
    if(!found)b=n-1;
    return(b);
}

        
#define findblock(map_ij,n,scale,elem, block)\
{\
int candidate, found, b, *map= (map_ij);\
\
    candidate = (int)(scale*(elem));\
    found = 0;\
    if(map[candidate] <= (elem)){ /* search downward */\
         b= candidate;\
         while(b<(n)-1){ \
            found = (map[b+1]>(elem));\
            if(found)break;\
            b++;\
         } \
    }else{ /* search upward */\
         b= candidate-1;\
         while(b>=0){\
            found = (map[b]<=(elem));\
            if(found)break;\
            b--;\
         }\
    }\
    if(!found)b=(n)-1;\
    *(block) = b;\
}



/*\ LOCATE THE OWNER OF THE (i,j) ELEMENT OF A GLOBAL ARRAY
\*/
logical ga_locate_(g_a, i, j, owner)
        Integer *g_a, *i, *j, *owner;
{
int  iproc, jproc; 
Integer ga_handle;

   ga_check_handleM(g_a, "ga_locate");
   ga_handle = GA_OFFSET + *g_a;

   if (*i <= 0 || *i > GA[GA_OFFSET + *g_a].dims[0]  ||
       *j <= 0 || *j > GA[GA_OFFSET + *g_a].dims[1]){
       *owner = -1;
       return( FALSE);
   }

   findblock(GA[ga_handle].mapc,GA[ga_handle].nblock[0], 
             GA[ga_handle].scale[0], (int)*i, &iproc);
   findblock(GA[ga_handle].mapc+GA[ga_handle].nblock[0],
             GA[ga_handle].nblock[1], GA[ga_handle].scale[1], (int)*j,&jproc);

   *owner = (Integer)jproc* GA[ga_handle].nblock[0] + iproc;
   return(TRUE);
}



/*\ LOCATE OWNERS OF THE SPECIFIED PATCH OF A GLOBAL ARRAY
\*/
logical ga_locate_region_(g_a, ilo, ihi, jlo, jhi, map, np )
        Integer *g_a, *ilo, *jlo, *ihi, *jhi, map[][5], *np;
{
int  iprocLT, iprocRB, jprocLT, jprocRB;
Integer  owner, ilop, ihip, jlop, jhip, i,j, ga_handle;

   ga_check_handleM(g_a, "ga_locate_region");

   ga_handle = GA_OFFSET + *g_a;
   if (*ilo <= 0 || *ihi > GA[ga_handle].dims[0] ||
       *jlo <= 0 || *jhi > GA[ga_handle].dims[1] ||
       *ihi < *ilo ||  *jhi < *jlo){
       return(FALSE);
   }

   /* find "processor coordinates" for the left top corner */
   findblock(GA[ga_handle].mapc,GA[ga_handle].nblock[0], 
             GA[ga_handle].scale[0],(int)*ilo, &iprocLT);
   findblock(GA[ga_handle].mapc+GA[ga_handle].nblock[0],
             GA[ga_handle].nblock[1],GA[ga_handle].scale[1],(int)*jlo,&jprocLT);

   /* find "processor coordinates" for the right bottom corner */
   findblock(GA[ga_handle].mapc,GA[ga_handle].nblock[0], 
             GA[ga_handle].scale[0],(int)*ihi, &iprocRB);
   findblock(GA[ga_handle].mapc+GA[ga_handle].nblock[0],
             GA[ga_handle].nblock[1],GA[ga_handle].scale[1],(int)*jhi,&jprocRB);

   *np = 0;
   for(i=iprocLT;i<=iprocRB;i++)
       for(j=jprocLT;j<=jprocRB;j++){
           owner = j* GA[ga_handle].nblock[0] + i;
           ga_ownsM(ga_handle, owner, ilop, ihip, jlop, jhip);
           map[*np][0] = *ilo < ilop ? ilop : *ilo;    
           map[*np][1] = *ihi > ihip ? ihip : *ihi;    
           map[*np][2] = *jlo < jlop ? jlop : *jlo;    
           map[*np][3] = *jhi > jhip ? jhip : *jhi;    
           map[*np][4] = owner;    
           (*np)++;
   }

   return(TRUE);
}
    



void ga_scatter_local(g_a, v, i, j, nv, proc) 
     Integer g_a, *i, *j, nv, proc;
     Void *v;
{
char *ptr_src, *ptr_ref, *ptr_dst;
Integer ldp, item_size;
Integer ilo, ihi, jlo, jhi;
register Integer k, offset;

  if (nv < 1) return;
  GA_PUSH_NAME("ga_scatter_local");

  ga_distribution_(&g_a, &proc, &ilo, &ihi, &jlo, &jhi);

  /* get a address of the first element owned by proc */
  gaShmemLocation(proc, g_a, ilo, jlo, &ptr_ref, &ldp);

  item_size = GAsizeofM(GA[GA_OFFSET + g_a].type);
  if(GAme==proc && !in_handler) GAbytes.scaloc += (double)item_size*nv ;

  for(k=0; k< nv; k++){
     if(i[k] < ilo || i[k] > ihi  || j[k] < jlo || j[k] > jhi){
       sprintf(err_string,"proc=%d invalid i/j=(%d,%d)>< [%d:%d,%d:%d]",
               proc, i[k], j[k], ilo, ihi, jlo, jhi); 
       ga_error(err_string,g_a);
     }

     offset  = (j[k] - jlo)* ldp + i[k] - ilo;
     ptr_dst = ptr_ref + item_size * offset;
     ptr_src = ((char*)v) + k*item_size; 

#ifdef CRAY_T3D
           CopyElemTo(ptr_src, ptr_dst, item_size/8, proc);
#    else
           Copy(ptr_src, ptr_dst, item_size);
#    endif
  }
  GA_POP_NAME;
}



void ga_scatter_remote(g_a, v, i, j, nv, proc) 
     Integer g_a, *i, *j, nv, proc;
     Void *v;
{
register Integer item_size, offset, nbytes, msglen;

  if (nv < 1) return;
  if(proc<0)ga_error(" scatter_remote: invalid process ",proc);

  item_size = GAsizeofM(GA[GA_OFFSET + g_a].type);
  
  nbytes = nv*item_size; 
  Copy(v, MessageSnd->buffer, nbytes);

  offset = nbytes;
  nbytes = nv*sizeof(Integer); 
  Copy((char*)i, MessageSnd->buffer + offset, nbytes);

  offset += nbytes; 
  Copy((char*)j, MessageSnd->buffer + offset, nbytes);
  
  msglen = offset + nbytes;
  ga_snd_req(g_a, nv, 0,0,0, msglen, GA[GA_OFFSET + g_a].type, GA_OP_DST,proc,  DataServer(proc));
}



/*\ SCATTER OPERATION elements of v into the global array
\*/
void ga_scatter_(g_a, v, i, j, nv)
     Integer *g_a, *nv, *i, *j;
     Void *v;
{
register Integer k;
Integer pindex, phandle, item_size;
Integer first, nelem, BufLimit, proc, type=GA[GA_OFFSET + *g_a].type;


  if (*nv < 1) return;

  ga_check_handleM(g_a, "ga_scatter");
  GA_PUSH_NAME("ga_scatter");
  GAstat.numsca++;


  if(!MA_push_get(MT_F_INT,*nv, "ga_scatter--p", &phandle, &pindex))
            ga_error("MA alloc failed ", *g_a);

  /* find proc that owns the (i,j) element; store it in temp: INT_MB[] */
  for(k=0; k< *nv; k++) if(! ga_locate_(g_a, i+k, j+k, INT_MB+pindex+k)){
         sprintf(err_string,"invalid i/j=(%d,%d)", i[k], j[k]);
         ga_error(err_string,*g_a);
  }

  /* determine limit for message size --  v,i, & j will travel together */
  item_size = GAsizeofM(type);
  BufLimit   = MSG_BUF_SIZE/(2*sizeof(Integer)+item_size);
  GAbytes.scatot += (double)item_size**nv ;

  /* Sort the entries by processor */
  ga_sort_scat(nv, (DoublePrecision*)v, i, j, INT_MB+pindex, type );
   
  /* go through the list again executing scatter for each processor */

  first = 0;
  do { 
      proc  = INT_MB[pindex+first]; 
      nelem = 0;

      /* count entries for proc from "first" to last */
      for(k=first; k< *nv; k++){
        if(proc == INT_MB[pindex+k]) nelem++;
        else break;
      }

      /* send request for processor proc */

      if(gaDirectAccess(proc))
        ga_scatter_local(*g_a, ((char*)v)+item_size*first, i+first, 
                         j+first, nelem,proc);
      else{

        /* limit messages to buffer length */

        Integer last = first + nelem -1; 
        Integer range, chunk;
        for(range = first; range <= last; range += BufLimit){
            chunk = MIN(BufLimit, last -range+1); 
            ga_scatter_remote(*g_a, ((char*)v)+item_size*range, 
                              i+range,j+range, chunk, proc);
        }
      }

      first += nelem;
  }while (first< *nv);
  if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);
  GA_POP_NAME;
}
      

/*\ permutes input index list using sort routine used in scatter/gather
\*/
void ga_sort_permut_(g_a, index, i, j, nv)
     Integer *g_a, *nv, *i, *j, *index;
{
register Integer k;
Integer pindex, phandle;
extern void ga_sort_permutation();

  if (*nv < 1) return;

  if(!MA_push_get(MT_F_INT,*nv, "ga_sort_permut--p", &phandle, &pindex))
            ga_error("MA alloc failed ", *g_a);

  /* find proc that owns the (i,j) element; store it in temp: INT_MB[] */
  for(k=0; k< *nv; k++) if(! ga_locate_(g_a, i+k, j+k, INT_MB+pindex+k)){
         sprintf(err_string,"invalid i/j=(%d,%d)", i[k], j[k]);
         ga_error(err_string,*g_a);
  }

  /* Sort the entries by processor */
  ga_sort_permutation(nv, index, INT_MB+pindex);
  if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);
}



void ga_gather_local(g_a, v, i, j, nv, proc) 
     Integer g_a, *i, *j, nv, proc;
     Void *v;
{
char *ptr_src, *ptr_ref, *ptr_dst;
Integer ldp, item_size;
Integer ilo, ihi, jlo, jhi;
register Integer k, offset;

  if (nv < 1) return;
  GA_PUSH_NAME("ga_gather_local");

  if(proc==GAme) FLUSH_CACHE;

  ga_distribution_(&g_a, &proc, &ilo, &ihi, &jlo, &jhi);

  /* get a address of the first element owned by proc */
  gaShmemLocation(proc, g_a, ilo, jlo, &ptr_ref, &ldp);

  item_size = GAsizeofM(GA[GA_OFFSET + g_a].type);
  if(GAme==proc && !in_handler) GAbytes.gatloc += (double)item_size*nv ;

  for(k=0; k< nv; k++){
     if(i[k] < ilo || i[k] > ihi  || j[k] < jlo || j[k] > jhi){
       sprintf(err_string,"k=%d proc=%d invalid i/j=(%d,%d)>< [%d:%d,%d:%d]",
               k, proc, i[k], j[k], ilo, ihi, jlo, jhi);
       printf("&i=%d, &j=%d, &v=%d\n",(long)i+k, (long)j+k,(long) v+k);
       fflush(stdout);
       ga_error(err_string,g_a);
      

     }

     offset  = (j[k] - jlo)* ldp + i[k] - ilo;
     ptr_src = ptr_ref + item_size * offset;
     ptr_dst = ((char*)v) + k*item_size; 

#    ifdef CRAY_T3D
        CopyElemFrom(ptr_src, ptr_dst, item_size/8, proc);
#    else
        Copy(ptr_src, ptr_dst, item_size);
#    endif
  }
  GA_POP_NAME;
}



void ga_gather_remote(g_a, v, i, j, nv, proc) 
     Integer g_a, *i, *j, nv, proc;
     Void *v;
{
register Integer item_size, offset, nbytes;
Integer  len, from, to,  msglen, handle = GA_OFFSET + g_a, expected_len;
msgid_t  msgid;

  if (nv < 1) return;
  if(proc<0)ga_error(" gather_remote: invalid process ",proc);

  item_size = GAsizeofM(GA[GA_OFFSET + g_a].type);
  
  offset = 0;
  nbytes = nv*sizeof(Integer); 
  Copy((char*)i, MessageSnd->buffer + offset, nbytes);

  offset = nbytes;
  Copy((char*)j, MessageSnd->buffer + offset, nbytes);

  msglen = offset + nbytes; 
  nbytes = item_size * nv; /* data to receive */
  to = DataServer(proc);
  expected_len = nbytes;
# ifdef IWAY
     expected_len += MSG_HEADER_SIZE;
# endif

# if defined(NX) || defined(SP1) || defined(SP)
     len = expected_len;
     msgid = ga_msg_ircv(GA_TYPE_DGT, MessageSnd, expected_len, to);
     ga_snd_req(g_a, nv, 0,0,0, msglen, GA[handle].type, GA_OP_DGT, proc, to);
     ga_msg_wait(msgid, &from, &len);

# else
     ga_snd_req(g_a, nv, 0, 0, 0, msglen, GA[handle].type, GA_OP_DGT, proc, to);
     ga_msg_rcv(GA_TYPE_DGT, MessageSnd, expected_len, &len,to,&from);
# endif

     if(len != expected_len) ga_error(" gather_remote: wrong data length",len); 

# ifdef IWAY
     /* this is a redundant copy; in IWAY version needs message header */
     Copy(MessageSnd->buffer, (char*)v, nbytes);
# endif
}



/*\ GATHER OPERATION elements from the global array into v
\*/
void ga_gather_(g_a, v, i, j, nv)
     Integer *g_a, *nv, *i, *j;
     Void *v;
{
register Integer k, nelem;
Integer pindex, phandle, item_size;
Integer first, BufLimit, proc;

  if (*nv < 1) return;

  ga_check_handleM(g_a, "ga_gather");
  GA_PUSH_NAME("ga_gather");
  GAstat.numgat++;


  if(!MA_push_get(MT_F_INT, *nv, "ga_gather--p", &phandle, &pindex))
            ga_error("MA failed ", *g_a);


  /* find proc that owns the (i,j) element; store it in temp: INT_MB[] */
  for(k=0; k< *nv; k++) if(! ga_locate_(g_a, i+k, j+k, INT_MB+pindex+k)){
       sprintf(err_string,"invalid i/j=(%d,%d)", i[k], j[k]);
       ga_error(err_string,*g_a);
  }

  /* Sort the entries by processor */
  ga_sort_gath_(nv, i, j, INT_MB+pindex);
   
  /* determine limit for message size --                               *
   * --  i,j will travel together in the request;  v will be sent back * 
   * --  due to limited buffer space (i,j,v) will occupy the same buf  * 
   *     when server executes ga_gather_local                          */

  item_size = GAsizeofM(GA[GA_OFFSET + *g_a].type);
  GAbytes.gattot += (double)item_size**nv;

  BufLimit   = MSG_BUF_SIZE/(2*sizeof(Integer)+item_size);
  /*BufLimit = MIN( MSG_BUF_SIZE/(2*sizeof(Integer)), MSG_BUF_SIZE/item_size);*/

  /* go through the list again executing gather for each processor */

  first = 0;
  do {
      proc  = INT_MB[pindex+first];
      nelem = 0;

      /* count entries for proc from "first" to last */
      for(k=first; k< *nv; k++){
        if(proc == INT_MB[pindex+k]) nelem++;
        else break;
      }

      /* send request for processor proc */

      if(gaDirectAccess(proc))
        ga_gather_local(*g_a, ((char*)v)+item_size*first, i+first, j+first,
                        nelem,proc);
      else{

        /* limit messages to buffer length */

        Integer last = first + nelem -1;
        Integer range, chunk;
#       ifdef SP
              int i_on = mpc_queryintr();
              mpc_disableintr();
#       endif
        for(range = first; range <= last; range += BufLimit){
            chunk = MIN(BufLimit, last -range+1);
            ga_gather_remote(*g_a, ((char*)v)+item_size*range, i+range, j+range,
                              chunk, proc);
        }
#       ifdef SP
              if(i_on) mpc_enableintr();
#       endif
      }

      first += nelem;
  }while (first< *nv);

  if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);
  GA_POP_NAME;
}
      
           

/*\ local read and increment of an element of a global array
\*/
Integer ga_read_inc_local(g_a, i, j, inc, proc)
        Integer g_a, i, j, inc, proc;
{
Integer *ptr, ldp, value;

   GA_PUSH_NAME("ga_read_inc_local");
   if(GAme == proc && !in_handler)GAbytes.rdiloc += (double)sizeof(Integer);

   /* get a address of the g_a(i,j) element */
   gaShmemLocation(proc, g_a, i, j, (char**)&ptr, &ldp);

#  ifdef CRAY_T3D
        { long lval;
          while ( (lval = shmem_swap((long*)ptr, INVALID, proc) ) == INVALID);
          value = (Integer) lval;
          (void) shmem_swap((long*)ptr, (lval + inc), proc);
        }
#  else
        if(GAnproc>1)LOCK(g_a, proc, ptr);
          value = *ptr;
          (*ptr) += inc;
        if(GAnproc>1)UNLOCK(g_a, proc, ptr);
#  endif
   GA_POP_NAME;
   return(value);
}


/*\ remote read and increment of an element of a global array
\*/
Integer ga_read_inc_remote(g_a, i, j, inc, proc)
        Integer g_a, i, j, inc, proc;
{
Integer len, from, to, value, handle = GA_OFFSET + g_a, bytes=sizeof(value);
msgid_t  msgid;

   if(proc<0)ga_error(" read_inc_remote: invalid process ",proc);

   to = DataServer(proc);
   bytes = MSG_HEADER_SIZE; /* for iway needs header */
#  if defined(NX) || defined(SP1) || defined(SP)
      len = bytes;
      msgid = ga_msg_ircv(GA_TYPE_RDI, MessageSnd, bytes, to);
      ga_snd_req(g_a, i, inc, j, 0, bytes, GA[handle].type, GA_OP_RDI,proc, to);
      ga_msg_wait(msgid, &from, &len);
#  else
      ga_snd_req(g_a, i, inc, j, 0, bytes, GA[handle].type, GA_OP_RDI,proc, to);
      ga_msg_rcv(GA_TYPE_RDI, MessageSnd, bytes, &len,to,&from);
#  endif

   if(len != bytes)
             ga_error("read_inc_remote: wrong data length",len); 
   value = MessageSnd->ilo;
   return(value);
}


/*\ READ AND INCREMENT AN ELEMENT OF A GLOBAL ARRAY
\*/
Integer ga_read_inc_(g_a, i, j, inc)
        Integer *g_a, *i, *j, *inc;
{
Integer  value, proc; 
#ifdef GA_TRACE
       trace_stime_();
#endif

    ga_check_handleM(g_a, "ga_read_inc");
    GAstat.numrdi++;
    GAbytes.rditot += (double)sizeof(Integer);

    if(GA[GA_OFFSET + *g_a].type !=MT_F_INT)
       ga_error(" ga_read_inc: type must be integer ",*g_a);

    GA_PUSH_NAME("ga_read_inc");
    ga_locate_(g_a, i, j, &proc);
    if(gaDirectAccess(proc)){
        value = ga_read_inc_local(*g_a, *i, *j, *inc, proc);
    }else{
#       ifdef SP
              int i_on = mpc_queryintr();
              mpc_disableintr();
#       endif
        value = ga_read_inc_remote(*g_a, *i, *j, *inc, proc);
#       ifdef SP
              if(i_on) mpc_enableintr();
#       endif
    }

#  ifdef GA_TRACE
     trace_etime_();
     op_code = GA_OP_RDI; 
     trace_genrec_(g_a, i, i, j, j, &op_code);
#  endif

   GA_POP_NAME;
   return(value);
}



Integer ga_nodeid_()
{
  return ((Integer)GAme);
}


Integer ga_nnodes_()
{
  return ((Integer)GAnproc);
}



/*********************** other utility routines *************************/

void ga_ma_get_ptr_(ptr, address)
      char **ptr, *address;
{
   *ptr = address; 
}


Integer ga_ma_diff_(ptr1, ptr2)
        char *ptr1, *ptr2;
{
   return((Integer)(ptr2-ptr1));
}


/*************************************************************************/
