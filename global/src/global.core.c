/*$Id: global.core.c,v 1.55 1999-07-13 23:08:05 d3h325 Exp $*/
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
#include "macdecls.h"
#include "global.core.h"

#define DEBUG 0
#define USE_MALLOC 1
#define INVALID_MA_HANDLE -1 
#define NEAR_INT(x) (x)< 0.0 ? ceil( (x) - 0.5) : floor((x) + 0.5)

/*uncomment line below to initialize arrays in ga_create/duplicate */
/*#define GA_CREATE_INDEF yes */

/*uncomment line below to verify consistency of MA in every sync */
/*#define CHECK_MA yes */

/* uncomment line below to verify if MA base address is alligned wrt datatype*/ 
#if !(defined(LINUX) || defined(CRAY_YMP))
#define CHECK_MA_ALGN 1
#endif

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
void FATR ga_sync_()
{
void   ga_wait_server();
       extern int GA_fence_set;
#ifdef CHECK_MA
extern Integer MA_verify_allocator_stuff();
Integer status;
#endif

       GA_fence_set=0;
       if (GAme < 0) return;
#if defined(CONVEX)
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
#      if defined(LAPI)
             ga_wait_cmpl(); /* remote requests must be completed */
#      endif
       ga_msg_sync_();
#      if defined(PARAGON) || defined(IWAY)
             ga_wait_server();  /* synchronize data server thread */
#      endif
#      ifdef IWAY
             ga_msg_sync_();
#      endif
#endif
#ifdef CHECK_MA
       status = MA_verify_allocator_stuff();
#endif

    /* sanity check to verify if request messages have not been lost */
#if defined(SP) || defined(SP1)
       ga_check_req_balance();
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
#if defined(SYSV) 
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
#if defined(CRAY) || defined(WIN32)
void FATR  ga_check_handle_(g_a, fstring)
     Integer *g_a;
     _fcd fstring;
#else
void FATR  ga_check_handle_(g_a, fstring,slen)
     Integer *g_a;
     int  slen;
     char *fstring;
#endif
{
char  buf[FLEN];

    if( GA_OFFSET + (*g_a) < 0 || GA_OFFSET + (*g_a) >= max_global_array){
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(fstring), _fcdlen(fstring), buf, FLEN);
#else
      f2cstring(fstring, slen, buf, FLEN);
#endif
      fprintf(stderr, " ga_check_handle: %s ", buf);
      ga_error(" invalid global array handle ", (*g_a));
    }
    if( ! (GA[GA_OFFSET + (*g_a)].actv) ){
#if defined(CRAY) || defined(WIN32)
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
     TrapSigIot(), TrapSigXcpu();

     TrapSigBus();
     TrapSigFpe();
     TrapSigIll();
     TrapSigSegv();
     TrapSigSys();
     TrapSigTrap();
     TrapSigTerm();
#ifdef SGI
     TrapSigIot();
     TrapSigXcpu();
#endif
#endif
}


/*\  signals trapped only by parent process
\*/
void gaParentTrapSignals()
{
#ifdef SYSV
void TrapSigChld(), TrapSigInt(), TrapSigHup(),TrapSigXcpu();
     TrapSigChld();
     TrapSigInt();
     TrapSigHup();
#ifdef SGI
     TrapSigXcpu();
#endif
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
void FATR  ga_initialize_()
{
Integer  i;
long *msg_buf;
#ifdef SYSV
Integer buf_size, bar_size;
Integer type;
#endif

    if(GAinitialized) return;

   /* initialize msg buffers */
    MessageSnd = (struct message_struct*)allign_page(MessageSnd);
    MessageRcv = (struct message_struct*)allign_page(MessageRcv);
#   ifdef PARAGON
       /* wire down msg buffer pages */ 
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
      fprintf(stderr,"Current GA setup is for up to %d processors\n",MAX_NPROC);
      fprintf(stderr,"Please change MAX_NPROC in globalp.h & recompile\n");
      ga_error("terminating...",0);
    }

    if(DEBUG)
    fprintf(stderr, "mode=%d, me=%d, master=%d, clusters=%d clust_nodes=%d\n",
            ClusterMode, GAme, cluster_master, GA_n_clus, cluster_nodes); 

    gaAllTrapSignals(); /* all processes set up own signal handlers */


#ifdef SYSV 

    /*....................... System V IPC stuff  ..........................*/
#   ifdef KSR
      bar_size = KSRbarrier_mem_req();
#   else
      bar_size = 2*sizeof(long);
#   endif

    buf_size = sizeof(DoublePrecision)*cluster_compute_nodes; /*shmem buffer*/ 

    /* at the end there is shmem counter for ONE server request counter */
    shmSIZE  = bar_size + buf_size+ sizeof(Integer); 

    if(MPme == cluster_master){
        /* assure that GA will not alocate more shared memory than specified */
        if(GA_memory_limited){ 
            unsigned long shmemlimit;
            extern void Set_Shmem_Limit();

            shmemlimit = (unsigned long) cluster_compute_nodes *GA_total_memory;
            shmemlimit += shmSIZE; /* add GA overhead */
            if (shmemlimit < GA_total_memory)
                ga_error("GA panic: shmemlimit problem ",shmemlimit);
            Set_Shmem_Limit(shmemlimit);
        }


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
                (cluster_nodes+RESERVED_LOCKS)*sizeof(ulock_t*), cluster_master,
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


    /* selected processes now become data servers */
#ifdef DATA_SERVER
       gai_setup_cluster();
       if(ClusterMode) if(GAme <0) ga_SERVER(0, MessageRcv);
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
#      if (defined(SP) || defined(SP1)) && !defined(AIX3)
          mpc_setintrdelay(1);
#      endif
#   endif

#ifdef LAPI
    ga_init_lapi();
#endif

#if defined(CRAY_T3D) && !defined(FLUSHCACHE)
    shmem_set_cache_inv();
#endif

    /* synchronize, and then we are ready to do real work */
#   ifdef LAPI
      LAPI_Gfence(lapi_handle);
#   endif
#   ifdef KSR
      ga_msg_sync_(); /* barrier not ready yet */
#   else
      ga_sync_();
#   endif
    if(DEBUG)    fprintf(stderr,"ga_init done=%ld\n",GAme);
}




/*\ IS MA USED FOR ALLOCATION OF GA MEMORY ?
\*/ 
logical FATR ga_uses_ma_()
{
#  ifdef SYSV
     return FALSE;
#  else
     return TRUE;
#  endif
}


/*\ IS MEMORY LIMIT SET ?
\*/
logical FATR ga_memory_limited_()
{
   if(GA_memory_limited) return TRUE;
   else                  return FALSE;
}



/*\ RETURNS AMOUNT OF MEMORY on each processor IN ACTIVE GLOBAL ARRAYS 
\*/
Integer  FATR ga_inquire_memory_()
{
Integer i, sum=0;
    for(i=0; i<max_global_array; i++) 
        if(GA[i].actv) sum += GA[i].size; 
    return(sum);
}


/*\ RETURNS AMOUNT OF GA MEMORY AVAILABLE on calling processor 
\*/
Integer FATR ga_memory_avail_()
{
#ifdef SYSV
   return(GA_total_memory); 
#else
   Integer ma_limit = MA_inquire_avail(MT_F_BYTE);

   if ( GA_memory_limited ) return( MIN(GA_total_memory, ma_limit) );
   else return( ma_limit );
#endif
}




static Integer GA_memory_limit=0;

/*\ (re)set limit on GA memory usage
\*/
void FATR ga_set_memory_limit_(Integer *mem_limit)
{
     if(GA_memory_limited){

         /* if we had the limit set we need to adjust the amount available */
         if (*mem_limit>=0)
             /* adjust the current value by diff between old and new limit */
             GA_total_memory += (*mem_limit - GA_memory_limit);
         else{

             /* negative values reset limit to "unlimited" */
             GA_memory_limited =  0;                
             GA_total_memory= -1;                   
         }

     }else{

          GA_total_memory = GA_memory_limit  = *mem_limit;
          if(*mem_limit >= 0) GA_memory_limited = 1;        
     }
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
void FATR ga_initialize_ltd_(mem_limit)
Integer *mem_limit;
{

  GA_total_memory = GA_memory_limit  = *mem_limit; 
  if(*mem_limit >= 0) GA_memory_limited = 1; 
  ga_initialize_();
}



/*\ Initialize MA-like addressing:
 *  get addressees for the base arrays for double, complex and int types
\*/
static int ma_address_init=0;
void gai_ma_address_init()
{
#ifdef CHECK_MA_ALGN
Integer  off_dbl, off_int, off_dcpl;
#endif
     ma_address_init=1;
     INT_MB = (Integer*)MA_get_mbase(MT_F_INT);
     DBL_MB = (DoublePrecision*)MA_get_mbase(MT_F_DBL);
     DCPL_MB= (DoubleComplex*)MA_get_mbase(MT_F_DCPL);

#   ifdef CHECK_MA_ALGN
        off_dbl = 0 != ((long)DBL_MB)%sizeof(DoublePrecision);
        off_int = 0 != ((long)INT_MB)%sizeof(Integer);
        off_dcpl= 0 != ((long)DCPL_MB)%sizeof(DoublePrecision);

        if(off_dbl)
           ga_error("GA initialize: MA DBL_MB not alligned", (Integer)DBL_MB);

        if(off_int)
           ga_error("GA initialize: INT_MB not alligned", (Integer)INT_MB);

        if(off_dcpl)
          ga_error("GA initialize: DCPL_MB not alligned", (Integer)DCPL_MB);
#   endif

    if(DEBUG)
        printf("%d INT_MB=%d(%x) DBL_MB=%ld(%lx) DCPL_MB=%d(%lx)\n",
                GAme, INT_MB,INT_MB, DBL_MB,DBL_MB, DCPL_MB,DCPL_MB);
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

      /* sync is in ga_create_irreg */

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
#if defined(CRAY) || defined(WIN32)
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
#if defined(CRAY) || defined(WIN32)
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

   GA[ga_handle].ptr[0] = ptr;

#  ifdef SHMEM
     /* true shared memory */
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

#    else
     /* global address space */
     {
       Integer i, len = sizeof(Void*);
       Integer mtype = GA_TYPE_BRD;
       GA[ga_handle].ptr[GAme] = ptr;

       /* need pointers on all procs to support global addressing */
       /* on YMP we need to adjust them w.r.t. local address base */
#ifdef CRAY_YMP_0
       GA[ga_handle].ptr[GAme] = (char*)((int)ptr - (int)DBL_MB);
#endif
       for(i=0; i<GAnproc; i++) ga_brdcst_(&mtype, GA[ga_handle].ptr+i,&len,&i);
#ifdef CRAY_YMP_0
       for(i=0; i<GAnproc; i++)
          GA[ga_handle].ptr[i] = (char*)((int)GA[ga_handle].ptr[i]+(int)DBL_MB);
#endif
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
   char *base;
   int adjust, diff;
   long *msg_buf = (long*)MessageRcv->buffer, bytes=(long)mem_size;
#else
   Integer handle, index;
#endif
   int item_size;

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
         item_size = GAsizeofM(type);
#ifdef JEFFS_TEST
         if( ((int) *pptr)%item_size){
/*           fprintf(stderr,"%d:GA: MA allocated nonalligned memory(%d): %d mod=%d\n",GAme, item_size, ((int) *pptr)%item_size); */
           ga_error("GA: MA allocated nonalligned memory",*pptr);      
         }
#endif
         
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
#ifdef _CRAYMPP
Integer ncols,maxcols=0,*ptr_Integer,j,nlocks;
long **ptr_ptr_long,*ptr_long;
char opadd='+';
int heap_status;
#endif


      if(!GAinitialized) ga_error("GA not initialized ", 0);
      if(!ma_address_init) gai_ma_address_init();

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

#ifdef _CRAYMPP
      for(i=0;i<GAnproc;i++){
          ga_distribution_(g_a,&i,&ilo,&ihi,&jlo,&jhi);
          ncols = jhi-jlo+1;
          if(ncols > maxcols) maxcols = ncols;
      }
      GA[ga_handle].lock = 1;
#endif


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

#ifdef _CRAYMPP

/* construct lock arrays for columns of local array */

      ncols = maxcols;

      nlocks = (ncols + COLS_PER_LOCK - 1)/COLS_PER_LOCK;
      for(i=0;i<MAX_NPROC;i++) GA[ga_handle].newlock[i] = 0;

      ptr_long = (long *)shmalloc(nlocks*sizeof(long));
      if(ptr_long==NULL) 
         ga_error("ga_create_irreg:malloc failure for ptr_long",0);    
      GA[ga_handle].newlock[GAme]=ptr_long;

      for(i=0;i<nlocks;i++) GA[ga_handle].newlock[GAme][i] = 1;

      ptr_Integer = (Integer *)malloc(nlocks*sizeof(Integer));
      if(ptr_Integer==NULL) 
               ga_error("ga_create_irreg: malloc failure for ptr_Integer",0);
      GA[ga_handle].lock_list=ptr_Integer;

      for(i=0;i<nlocks;i++) GA[ga_handle].lock_list[i]=0;

      /*learn where my fellow pes malloced their arrays -this avoids shmalloc */      ga_igop(GA_TYPE_SYN, (Integer *)GA[ga_handle].newlock, GAnproc, &opadd);

#endif


      ga_sync_();

#     ifdef GA_CREATE_INDEF
      if(status){
         Integer one = 1;
         if (GAme == 0) fprintf(stderr,"Initializing GA array%ld\n",*g_a);
         if(*type == MT_F_DBL){ 
             double bad = DBL_MAX;
             ga_fill_patch_(g_a, &one, dim1, &one, dim2, (Void *) &bad);
         } else if (*type == MT_F_INT) {
             Integer bad = (Integer) INT_MAX;
             ga_fill_patch_(g_a, &one, dim1, &one, dim2, (Void *) &bad);
         } else if (*type == MT_F_DCPL) {
             DoubleComplex bad = {DBL_MAX, DBL_MAX};
             ga_fill_patch_(g_a, &one, dim1, &one, dim2, (Void *) &bad);
         } else {
             ga_error("ga_create_irreg: type not yet supported ",  *type);
         }
      }
#     endif

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
#if defined(CRAY) || defined(WIN32)
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
#if defined(CRAY) || defined(WIN32)
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
#ifdef _CRAYMPP
char opadd='+';
Integer ncols,maxcols=0,*ptr_Integer,nlocks;
Integer ilo,ihi,jlo,jhi;
long **ptr_ptr_long,*ptr_long;
#endif


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

#ifdef _CRAYMPP

      /* construct lock arrays for columns of local array */

      for(i=0;i<GAnproc;i++){
          ga_distribution_(g_a,&i,&ilo,&ihi,&jlo,&jhi);
          ncols = jhi-jlo+1;
          if(ncols > maxcols) maxcols = ncols;
      }

      nlocks = (maxcols + COLS_PER_LOCK - 1)/COLS_PER_LOCK;
      for(i=0;i<MAX_NPROC;i++) GA[ga_handle].newlock[i] = 0;

      ptr_long = (long *)shmalloc(nlocks*sizeof(long));
      if(ptr_long==NULL) ga_error("ga_duplicate:malloc failure for ptr_long",0);
      GA[ga_handle].newlock[GAme]=ptr_long;

      for(i=0;i<nlocks;i++) GA[ga_handle].newlock[GAme][i]  = 1;

      ptr_Integer = (Integer *)malloc(nlocks*sizeof(Integer));
      if(ptr_Integer==NULL)
         ga_error("ga_duplicate: malloc failure for ptr_Integer",0);
      GA[ga_handle].lock_list=ptr_Integer;

      for(i=0;i<nlocks;i++) GA[ga_handle].lock_list[i] = 0;

     /*learn where my fellow pes malloced their arrays - this avoids shmalloc */
      ga_igop(GA_TYPE_SYN, (Integer *)GA[ga_handle].newlock, GAnproc, &opadd);

#endif


      ga_sync_();

#     ifdef GA_CREATE_INDEF
      if(status){
         Integer one = 1; 
         Integer dim1 =GA[ga_handle].dims[1], dim2=GA[ga_handle].dims[2];
         if(GAme==0)fprintf(stderr,"duplicate:initializing GA array%ld\n",*g_b);
         if(GA[ga_handle].type == MT_F_DBL) {
             DoublePrecision bad = DBL_MAX;
             ga_fill_patch_(g_b, &one, &dim1, &one, &dim2,  &bad);
         } else if (GA[ga_handle].type == MT_F_INT) {
             Integer bad = (Integer) INT_MAX;
             ga_fill_patch_(g_b, &one, &dim1, &one, &dim2,  &bad);
         } else if (GA[ga_handle].type == MT_F_DCPL) {
             DoubleComplex bad = {DBL_MAX, DBL_MAX};
             ga_fill_patch_(g_b, &one, &dim1, &one, &dim2,  &bad);
         } else {
             ga_error("ga_duplicate: type not supported ",GA[ga_handle].type);
         }
      }
#     endif


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
#if defined(CRAY) || defined(WIN32)
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
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

  return(ga_duplicate(g_a, g_b, buf));
}




/*\ DESTROY A GLOBAL ARRAY
\*/
logical FATR ga_destroy_(Integer *g_a)
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

#ifdef _CRAYMPP

    free(GA[ga_handle].lock_list);
    shfree(GA[ga_handle].newlock[GAme]);

#endif

    return(TRUE);
}

    
     
/*\ TERMINATE GLOBAL ARRAY STRUCTURES
 *
 *  all GA arrays are destroyed & shared memory is dealocated
 *  GA routines (except for ga_initialize) should not be called thereafter 
\*/
void FATR  ga_terminate_() 
{
Integer i, handle;
extern double t_dgop, n_dgop, s_dgop;


    if(!GAinitialized) return;
    for (i=0;i<max_global_array;i++){
          handle = i - GA_OFFSET ;
          if(GA[i].actv) ga_destroy_(&handle);
          if(GA[i].ptr) free(GA[i].ptr);
          if(GA[i].mapc) free(GA[i].mapc);
    }
    
#ifdef TIME_DGOP
    ga_sync_();
    fprintf(stderr,"%d: t_buf=%f t_fence=%f\n",GAme,t_buf,t_fence);
    fprintf(stderr,"%d: t_dgop=%f n_dgop=%f s_dgop=%f avg_time=%f avg_sz=%f\n",
           GAme,t_dgop,n_dgop, s_dgop, t_dgop/n_dgop, s_dgop/n_dgop);
    sleep(5);
#endif
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
Integer FATR ga_verify_handle_(g_a)
     Integer *g_a;
{
  return (Integer)
    ((*g_a + GA_OFFSET>= 0) && (*g_a + GA_OFFSET< max_global_array) && 
             GA[GA_OFFSET + (*g_a)].actv);
}


/*\ determine if access to "proc" data through shared/global memory is possible 
\*/
logical gaDirectAccess(proc, op)
   Integer proc;
   int op;
{
#ifdef SHMEM
#  ifdef SYSV
     if(ClusterMode && (ClusterID(proc) != GA_clus_id)) return(FALSE);
#  elif defined (LAPI)
     if((proc != GAme) && ((op==GA_OP_ACC) || (op==GA_OP_SCT) )) return (FALSE);
#  endif
#else
   if(proc != GAme) return(FALSE);
#endif
   return(TRUE);
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
void ga_put_local(Integer g_a, Integer ilo, Integer ihi,
                               Integer jlo, Integer jhi,
                  void* buf, Integer offset, Integer ld, Integer proc)
{
char     *ptr_glob, *ptr_loc;
Integer  ld_glob, rows, cols, type;

#ifndef LAPI
   GA_PUSH_NAME("ga_put_local");
#endif

   type = GA[GA_OFFSET + g_a].type;
   rows = ihi - ilo +1;
   cols = jhi - jlo +1;

   gaShmemLocation(proc, g_a, ilo, jlo, &ptr_glob, &ld_glob);
   ptr_loc = (char *)buf  + GAsizeofM(type) * offset;

   Copy2DTo(type, proc, &rows, &cols, ptr_loc, &ld, ptr_glob, &ld_glob);    

#ifdef LAPI
   if(GAme != proc) {
              UPDATE_FENCE_STATE(proc, GA_OP_PUT, cols);
              SET_COUNTER(ack_cntr, cols);
   }
#endif

#ifndef LAPI
   GA_POP_NAME;
#endif
}



/*\ remote put of a 2-dimensional patch of data into a global array
\*/
void ga_put_remote(g_a, ilo, ihi, jlo, jhi, buf, offset, ld, proc)
   Integer g_a, ilo, ihi, jlo, jhi, ld, offset, proc;
   Void *buf;
{
char     *ptr_src;
Integer  type;

   if(proc<0)ga_error(" ga_put_remote: invalid process ",proc);

   /* prepare request data */
   type = GA[GA_OFFSET + g_a].type;
   ptr_src = (char *)buf  + GAsizeofM(type)* offset;

#ifdef LAPI
   CLEAR_COUNTER(buf_cntr); /* make sure the buffer is available */
#endif
   ga_snd_req2D(g_a, ilo,ihi,jlo,jhi, type, GA_OP_PUT, ptr_src, ld,
                proc, DataServer(proc));
}



/*\ PUT A 2-DIMENSIONAL PATCH OF DATA INTO A GLOBAL ARRAY 
\*/
void FATR  ga_put_(g_a, ilo, ihi, jlo, jhi, buf, ld)
   Integer  *g_a,  *ilo, *ihi, *jlo, *jhi,  *ld;
   Void  *buf;
{
Integer  p, np, proc, idx, type=GA[GA_OFFSET + *g_a].type;
Integer  ilop, ihip, jlop, jhip, offset, size, rows, cols;
logical  localop;

#ifdef GA_TRACE
   trace_stime_();
#endif

      GA_PUSH_NAME("ga_put");

      size = GAsizeofM(type);
      GAstat.numput++;

      GAbytes.puttot += (double)size*(*ihi-*ilo+1)*(*jhi-*jlo+1);

      if(!ga_locate_region_(g_a, ilo, ihi, jlo, jhi, map, &np )){
          sprintf(err_string, "cannot locate region (%d:%d,%d:%d)",
                  *ilo, *ihi, *jlo, *jhi);
          ga_error(err_string, *g_a);
      }

      if(np > 1 || map[0][4] != GAme) INTR_OFF; /* disable interrupts */

      gaPermuteProcList(np); 
      for(idx=0; idx<np; idx++){
          p = (Integer)ProcListPerm[idx];

          ilop = map[p][0];
          ihip = map[p][1];
          jlop = map[p][2];
          jhip = map[p][3];
          proc = map[p][4];

          
          rows = ihip-ilop+1;
          cols = jhip-jlop+1;
             
          if(proc == GAme){

             localop =1;
             GAbytes.putloc += (double)size*rows*cols;

          }else{
             /* can we access data on remote process directly? */
             localop = gaDirectAccess(proc, GA_OP_PUT);

             /* might still want to go with remote access protocol */
#            ifdef LAPI
               if((cols > 1) && (rows*size < LONG_PUT_THRESHOLD)) localop = 0;
#            endif

             FENCE_NODE(proc);
          }

          if(localop){

             offset = (jlop - *jlo)* *ld + ilop - *ilo;
             ga_put_local(*g_a, ilop, ihip, jlop, jhip, buf, offset, *ld, proc);

          }else{
            /* number of messages determined by message-buffer size */
            Integer TmpSize, ilimit, jlimit;
            Integer ilo_chunk, ihi_chunk, jlo_chunk, jhi_chunk, chunks=0;

#           ifdef LAPI
              /* small requests packetized to fit in AM header */
              if(cols*rows*size < SHORT_PUT_THRESHOLD)
                    TmpSize = lapi_max_uhdr_data_sz/size;
              else
#           endif
                    TmpSize = MSG_BUF_SIZE/size;

            ilimit  = MIN(TmpSize, rows);
            jlimit  = MIN(TmpSize/ilimit, cols);

            for(jlo_chunk = jlop; jlo_chunk <= jhip; jlo_chunk += jlimit){
               jhi_chunk  = MIN(jhip, jlo_chunk+jlimit-1);
               for( ilo_chunk = ilop; ilo_chunk<= ihip; ilo_chunk += ilimit){

                  ihi_chunk = MIN(ihip, ilo_chunk+ilimit-1);
                  offset = (jlo_chunk - *jlo)* *ld + ilo_chunk - *ilo;
                  ga_put_remote(*g_a, ilo_chunk, ihi_chunk,
                                jlo_chunk, jhi_chunk, buf, offset, *ld, proc);
                  chunks++;
               }
            }
            UPDATE_FENCE_STATE(proc, GA_OP_PUT, chunks);
         }
      }

      GA_POP_NAME;

      if(np > 1 || map[0][4] != GAme){ 

#       ifdef LAPI
          CLEAR_COUNTER(ack_cntr);
#       endif
        INTR_ON;

      }

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_PUT; 
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
}


/*\ local get of a 2-dimensional patch of data into a global array
\*/
void ga_get_local(Integer g_a, Integer ilo, Integer ihi, Integer jlo,
                  Integer jhi, void* buf, Integer offset, Integer ld,
                  Integer proc)
{
char     *ptr_glob, *ptr_loc;
Integer  ld_glob, rows, cols, type;

#ifndef LAPI
   GA_PUSH_NAME("ga_get_local");
#endif

   type = GA[GA_OFFSET + g_a].type;
   rows = ihi - ilo +1;
   cols = jhi - jlo +1;


   gaShmemLocation(proc, g_a, ilo, jlo, &ptr_glob, &ld_glob);

   ptr_loc = (char *)buf  + GAsizeofM(type) * offset;
   if(proc==GAme) FLUSH_CACHE;

   Copy2DFrom(type, proc, &rows, &cols, ptr_glob, &ld_glob, ptr_loc, &ld);

#ifndef LAPI
   GA_POP_NAME;
#endif
}



/*\  get an array patch from remote processor
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

#  if defined(IWAY) || defined(LAPI)
     /* for LAPI data always arrives in msg buffer in remote protocol */ 
#    ifdef IWAY
        expected_len += MSG_HEADER_SIZE;
#    endif
     need_copy = 1;
#  else
     /* this stuff is to avoid data copying if possible */
     need_copy=cols-1; /* true only if cols > 1 */
#  endif

/*   t = tcgtime_();*/

   if(need_copy)
      ptr_src = MessageSnd->buffer;  /* data arrives to the same msg buffer */
   else
      ptr_src = (char *)buf + GAsizeofM(type)* offset;/*arrives to user buffer*/

   /* expected_len is overwritten if the MP layer is able to determine length*/
   len = expected_len;
#  if defined(NX) || defined(SP1) || defined(SP)
      msgid_rcv = ga_msg_ircv(GA_TYPE_GET,  ptr_src, expected_len, to);
      ga_snd_req(g_a, ilo,ihi,jlo,jhi, (Integer)0, type, GA_OP_GET,proc,to);
      ga_msg_wait(msgid_rcv, &from, &len);

#  elif(LAPI)
      CLEAR_COUNTER(buf_cntr); /* make sure the buffer is available */
      SET_COUNTER(buf_cntr,1); /* expect data to arrive into the same buffer */
      ga_snd_req(g_a, ilo,ihi,jlo,jhi, (Integer)0, type, GA_OP_GET,proc,to);
      CLEAR_COUNTER(buf_cntr); /* wait for data to arrive */

#  else
      ga_snd_req(g_a, ilo,ihi,jlo,jhi, (Integer)0, type, GA_OP_GET,proc,to);
      ga_msg_rcv(GA_TYPE_GET, ptr_src, expected_len, &len, to,&from);
#  endif

/*   t = tcgtime_() -t;*/
/*   printf("%d p=%d [%d:%d,%d:%d] t=%lf\n",GAme,proc,ilo,ihi,jlo,jhi,t);*/

   if(len != expected_len)ga_error("ga_get_remote:wrong msg length",len); 

   if(need_copy){
      /* Copy patch [ilo:ihi, jlo:jhi] from MessageBuffer */
      ptr_dst = (char *)buf  + GAsizeofM(type)* offset;
      Copy2D(type, &rows, &cols, ptr_src, &rows, ptr_dst, &ld);  
   }

}


/*\ GET A 2-DIMENSIONAL PATCH OF DATA FROM A GLOBAL ARRAY
\*/
void FATR  ga_get_(g_a, ilo, ihi, jlo, jhi, buf, ld)
   Integer  *g_a, *ilo, *ihi, *jlo, *jhi,  *ld;
   Void     *buf;
{
Integer p, np, proc, idx, type=GA[GA_OFFSET + *g_a].type;
Integer ilop, ihip, jlop, jhip, offset, localop, size, rows,cols;

#ifdef GA_TRACE
   trace_stime_();
#endif

      GA_PUSH_NAME("ga_get");
      GAstat.numget++;
      size = GAsizeofM(type);
      GAbytes.gettot += (double)size*(*ihi-*ilo+1)*(*jhi-*jlo+1);

      if(!ga_locate_region_(g_a, ilo, ihi, jlo, jhi, map, &np )){
          sprintf(err_string, "cannot locate region (%d:%d,%d:%d)",
                  *ilo, *ihi, *jlo, *jhi);
          ga_error(err_string, *g_a);
      }

      if(np > 1 || map[0][4] != GAme) INTR_OFF; /* disable interrupts */

      gaPermuteProcList(np);

      for(idx=0; idx<np; idx++){
          p = (Integer)ProcListPerm[idx];
          ilop = map[p][0];
          ihip = map[p][1];
          jlop = map[p][2];
          jhip = map[p][3];
          proc = map[p][4]; 


          rows = ihip-ilop+1;
          cols = jhip-jlop+1;

          if(proc == GAme){

             localop =1;
             GAbytes.getloc += (double)size*rows*cols;

          }else{

             /* can we access data on remote process directly? */
             localop = gaDirectAccess(proc, GA_OP_GET);

             /* might still want to go with remote access protocol */
#            ifdef LAPI
               if((cols > 10) && (rows*size < LONG_GET_THRESHOLD)) localop = 0;
#            endif

             FENCE_NODE(proc);
          }

          if(localop){

             offset = (jlop - *jlo)* *ld + ilop - *ilo;
             ga_get_local(*g_a, ilop, ihip, jlop, jhip, buf, offset, *ld, proc);

          }else{
            /* number of messages determined by message-buffer size */

            Integer ilo_chunk, ihi_chunk, jlo_chunk, jhi_chunk;
            Integer TmpSize;
            Integer ilimit;
            Integer jlimit;

#           if defined(IWAY) && defined(SP1)
               TmpSize = IWAY_MSG_BUF_SIZE/size;
#           else
               TmpSize = MSG_BUF_SIZE/size;
#           endif
            ilimit  = MIN(TmpSize, ihip-ilop+1);
            jlimit  = MIN(TmpSize/ilimit, jhip-jlop+1);

#           if defined(PARAGON)||defined(SP1) || defined(SP)
              /* this limits column chunking to 1 for larger number of rows */
              /* saves one memory copy */
              /**** it is an optimization only -- comment out if fails ******/
              if(ilimit > 2048) jlimit  = 1;
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
          }
      }

      GA_POP_NAME;
#ifdef LAPI
      /* wait for non-blocking copies to complete */
      CLEAR_COUNTER(get_cntr);
#endif
      if(np > 1 || map[0][4] != GAme) INTR_ON;

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_GET; 
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
}

#ifdef _CRAYMPP

/****** Howard's version of ga_acc_1d_local ***********************/

/*\local accumulate using intermediate buffer in local memory  (fine grain)
\*/
void ga_acc_1d_local_fg(Integer type, void* alpha, Integer rows, Integer cols,
                void* pglobal, Integer ldg, void *plocal, Integer ldl,
                void* buf, Integer buflen, Integer proc, Integer *lock_list,
                Integer jfirst, Integer jlast,global_array_t *ga_ptr)
{
Integer item_size=GAsizeofM(type), elem, words,j,istart,iend,jj,jlock,jmax,jmin;
char *ptr_dst, *ptr_src;

Integer ncols=cols,adj_item_size;

adj_item_size = item_size/sizeof(Integer);

 while(ncols > 0) {

   for (j = jfirst;  j <= jlast;  j++){

/* try to get the lock for this column */

       jlock = j>>LOG2_COLS_PER_LOCK;

/* have I done this block of columns ? */

       if(lock_list[jlock] == 0) continue;

       if(shmem_swap(&ga_ptr->newlock[proc][jlock],INVALID,proc) ==INVALID) continue;

/* have the lock - now work fast! */

       jmax =  MIN(j+COLS_PER_LOCK-j%COLS_PER_LOCK-1,jlast);
       jmin = MAX(jfirst,j-j%COLS_PER_LOCK);

       if(proc != GAme) {

         for(jj=jmin;jj<=jmax;jj++){

            for( istart = 0;  istart < rows;  istart += buflen){

                 iend   = MIN(istart + buflen, rows);
                 elem   = iend - istart;
                 words  = elem*adj_item_size;

                 ptr_dst = (char*)pglobal + item_size*(istart+ (jj-jfirst)*ldg);
                 ptr_src = (char*)plocal + item_size* (istart+ (jj-jfirst)*ldl);

                 CopyElemFrom(ptr_dst, buf, words, proc);
                 switch (type){
                     case MT_F_DBL:
                        dacc_column((DoublePrecision*)alpha,(DoublePrecision*)buf, 
                                    (DoublePrecision *)ptr_src, elem ); break;
                     case MT_F_DCPL:
                        zacc_column((DoubleComplex *)alpha, (DoubleComplex*)buf,
                                    (DoubleComplex *)ptr_src, elem ); break;
                     case MT_F_INT:
                        iacc_column((Integer *)alpha, (Integer *)buf, 
                                    (Integer *)ptr_src, elem ); break;
                  }
                 CopyElemTo(buf, ptr_dst, words, proc);
            }                     /* end loop over column element blocks */

         }                        /* end loop  over columns in lock block */

       }else{

           ptr_dst = (char *)pglobal  + item_size* ((jmin-jfirst) *ldg);
           ptr_src = (char *)plocal   + item_size* ((jmin-jfirst) *ldl);

           switch(type) {

               case(MT_F_DBL):

                  accumulate((DoublePrecision *)alpha, rows, jmax-jmin+1, 
                             (DoublePrecision*)ptr_dst, ldg,
                             (DoublePrecision*)ptr_src, ldl );
                   break;

                case(MT_F_DCPL):

                   zaccumulate((DoubleComplex *)alpha, rows, jmax-jmin+1, 
                               (DoubleComplex*)ptr_dst, ldg,
                               (DoubleComplex*)ptr_src, ldl );
                   break;

                case(MT_F_INT):

                    iaccumulate((Integer *)alpha, rows, jmax-jmin+1, 
                                (Integer*)ptr_dst, ldg, (Integer*)ptr_src, ldl);
                    break;

                default:

                   ga_error(" acc_local: unknown data type",type);
                   break;
            }

        }


      ncols -= jmax-jmin+1;    /* decrement the ncols value */

      lock_list[jlock] = 0; /* set the lock_list element to zero again */

      shmem_quiet(); /* fence */

      /* unlock the processor for others */
      shmem_swap(&ga_ptr->newlock[proc][jlock],1,proc);

   }  /* end loop over columns */

 }  /*  end while loop  over columns*/

}

#endif
/*****************************/


#if defined(SHMEM) && !defined(SYSV) && !defined(LAPI)

/*\local accumulate using intermediate buffer in local memory
\*/
void ga_acc_1d_local(Integer type, void* alpha, Integer rows, Integer cols,
                void* pglobal, Integer ldg, void *plocal, Integer ldl,
                void* buf, Integer buflen, Integer proc)
{
Integer item_size = GAsizeofM(type), elem, words, j, istart, iend;
char *ptr_dst, *ptr_src;

     for (j = 0;  j < cols;  j++)
       for( istart = 0;  istart < rows;  istart += buflen){
          iend   = MIN(istart + buflen, rows);
          elem   = iend - istart;
          words  = elem*item_size/sizeof(Integer);

          ptr_dst = (char *)pglobal  + item_size* (istart + j *ldg);
          ptr_src = (char *)plocal   + item_size* (istart + j *ldl);

          CopyElemFrom(ptr_dst, buf, words, proc);
          switch (type){
                 case MT_F_DBL:
                      dacc_column((DoublePrecision*)alpha,(DoublePrecision*)buf,
                                  (DoublePrecision *)ptr_src, elem ); break;
                 case MT_F_DCPL:
                      zacc_column((DoubleComplex *)alpha, (DoubleComplex*)buf,
                                  (DoubleComplex *)ptr_src, elem ); break;
                 case MT_F_INT:
                      iacc_column((Integer *)alpha, (Integer *)buf,
                                  (Integer *)ptr_src, elem ); break;
          }
          CopyElemTo(buf, ptr_dst, words, proc);
       }
}
#endif

/*\ local accumulate
\*/
void ga_acc_local(g_a, ilo, ihi, jlo, jhi, buf, offset, ld, proc, alpha)
   Integer g_a, ilo, ihi, jlo, jhi, ld, offset, proc;
   void *alpha, *buf;
{
char     *ptr_src, *ptr_dst;
Integer  item_size, ldp, rows, cols, type = GA[GA_OFFSET + g_a].type;
int      work_done=0;

#if defined(SHMEM) && !defined(SYSV)
#  define LEN_DBL_BUF 500
#  define LEN_BUF (LEN_DBL_BUF * sizeof(DoublePrecision))
   DoublePrecision acc_buffer[LEN_DBL_BUF], *pbuffer;
   Integer buflen,  bytes, handle, index;
#ifdef _CRAYMPP
   Integer jstop,jstart,jstop_b,jstart_b;
   Integer jlo_proc,loc,jproc,j,wait_cycles=0;
   Integer *ptr_lock_list;
   long swaperand=INVALID,tswap;
   global_array_t *ga_ptr;
#endif
#endif

#ifdef LAPI
   extern int kevin_ok; /* used to indicate if some thread holds acc lock */
#else
   GA_PUSH_NAME("ga_acc_local"); /* not thread safe */
#endif

   item_size =  GAsizeofM(type);
   gaShmemLocation(proc, g_a, ilo, jlo, &ptr_dst, &ldp);
   ptr_src = (char *)buf   + item_size * offset;
   rows = ihi - ilo +1;
   cols = jhi - jlo +1;

#if defined(SHMEM) && !defined(SYSV) && !defined(LAPI) && !defined(CRAY_YMP)

     bytes = rows*item_size;

     if(proc != GAme){

        buflen  = MIN(bytes,LEN_BUF)/item_size;
        pbuffer = acc_buffer;

        /* for larger patches try to get more buffer space from MA */
        if(bytes > LEN_BUF){

            Integer avail = MA_inquire_avail(type);
            if(avail * item_size * 0.9 > LEN_BUF) {

               buflen = (Integer) (avail *0.9); /* leave some */
               if(!MA_push_get(type, buflen, "ga_acc_buf", &handle, &index))
                        ga_error("allocation of ga_acc buffer failed ",GAme);
               MA_get_pointer(handle, &pbuffer);
            }
          }
       }
#      ifdef _CRAYMPP
         /* we have a fine-grain locking from Howard Pritchard */
         ga_ptr = &GA[GA_OFFSET + g_a];
         jproc = proc/ga_ptr->nblock[0];
         loc = jproc + ga_ptr->nblock[0];
         jlo_proc = ga_ptr->mapc[loc];
         jstart = jlo - jlo_proc;
         jstart_b = jstart>>LOG2_COLS_PER_LOCK;
         jstop = jhi - jlo_proc;
         jstop_b = jstop>>LOG2_COLS_PER_LOCK;

         /*  set the column work list to 1 */
         ptr_lock_list = ga_ptr->lock_list;
         for(j=jstart_b;j<=jstop_b;j++) ptr_lock_list[j] = 1;

         ga_acc_1d_local_fg(type, alpha, rows, cols, ptr_dst,
                            ldp, ptr_src, ld,pbuffer, buflen,
                            proc, ptr_lock_list, jstart,jstop,ga_ptr);
         work_done =1;

#      else
         if(proc != GAme){
         
            LOCK(g_a, proc, ptr_dst);

            ga_acc_1d_local(type, alpha, rows, cols, ptr_dst, ldp, ptr_src, ld,
                                                      pbuffer, buflen, proc);
#           ifdef CRAY_T3E
              shmem_quiet();
#           endif

            UNLOCK(g_a, proc, ptr_dst);

            work_done =1;
         }
#      endif

       if((bytes >LEN_BUF)&&(proc != GAme)) MA_pop_stack(handle);

       if(proc == GAme) FLUSH_CACHE; /* cache coherency problem on T3D/YMP */

#  endif

   if(! work_done) {

     if(GAnproc>1) LOCK(g_a, proc, ptr_dst);

#      ifdef LAPI
         kevin_ok = 0; /* signal other threads that lock is taken */
#      endif

       if(type==MT_F_DBL){
          accumulate(alpha, rows, cols, (DoublePrecision*)ptr_dst, ldp,
                                        (DoublePrecision*)ptr_src, ld );
       }else if(type==MT_F_DCPL){
          zaccumulate(alpha, rows, cols, (DoubleComplex*)ptr_dst, ldp,
                                         (DoubleComplex*)ptr_src, ld );
       }else{
          iaccumulate(alpha, rows, cols, (Integer*)ptr_dst, ldp,
                                         (Integer*)ptr_src, ld );
       }

     if(GAnproc>1) UNLOCK(g_a, proc, ptr_dst);

#    ifdef LAPI
         kevin_ok = 1; /* signal other threads that lock is available */
#    endif

   }

#ifndef LAPI
   GA_POP_NAME; /* not thread safe */
#endif

}


/*\ remote accumulate for a 2-dimensional array patch 
\*/
void ga_acc_remote(g_a, ilo, ihi, jlo, jhi, buf, offset, ld, proc, alpha)
   Integer g_a, ilo, ihi, jlo, jhi, ld, offset, proc;
   void *alpha, *buf;
{
char     *ptr_src;
Integer  type;

   if(proc<0)ga_error(" acc_remote: invalid process ",proc);

   /* prepare request data */
   type = GA[GA_OFFSET + g_a].type;
   ptr_src = (char *)buf  + GAsizeofM(type)* offset;

/*   fprintf(stderr,"buf_cntr=(%d,%d) ptr=%x to=%d patch=[%d:%d,%d:%d]\n",buf_cntr.cntr, buf_cntr.val, &buf_cntr.cntr, proc, ilo, ihi, jlo, jhi);*/
#  ifdef LAPI
      CLEAR_COUNTER(buf_cntr);
#  endif

   if(type==MT_F_DBL)
     *(DoublePrecision*)MessageSnd->alpha= *(DoublePrecision*)alpha;
   else if(type==MT_F_DCPL)
     *(DoubleComplex*)MessageSnd->alpha= *(DoubleComplex*)alpha;
   else *(Integer*)MessageSnd->alpha= *(Integer*)alpha;

   ga_snd_req2D(g_a, ilo,ihi,jlo,jhi, type, GA_OP_ACC, ptr_src, ld,
                proc, DataServer(proc));
}




/*\ ACCUMULATE OPERATION FOR A 2-DIMENSIONAL PATCH OF GLOBAL ARRAY
 *
 *  g_a += alpha * patch
\*/
void FATR  ga_acc_(g_a, ilo, ihi, jlo, jhi, buf, ld, alpha)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *ld;
   void *buf, *alpha;
{
   Integer ilop, ihip, jlop, jhip, offset, type = GA[GA_OFFSET + *g_a].type;
   Integer p, np, proc, idx, rows, cols, size = GAsizeofM(type);

#ifdef GA_TRACE
   trace_stime_();
#endif

   GA_PUSH_NAME("ga_acc");
   GAstat.numacc++;
   GAbytes.acctot += (double)size*(*ihi-*ilo+1)*(*jhi-*jlo+1);


   if(!ga_locate_region_(g_a, ilo, ihi, jlo, jhi, map, &np )){
          sprintf(err_string, "cannot locate region (%d:%d,%d:%d)",
                  *ilo, *ihi, *jlo, *jhi);
          ga_error(err_string, *g_a);
   }

   gaPermuteProcList(np); /* prepare permuted list of indices */

   if(np > 1 || map[0][4] != GAme) INTR_OFF;

   for(idx=0; idx<np; idx++){
       p = (Integer)ProcListPerm[idx];

       ilop = map[p][0];
       ihip = map[p][1];
       jlop = map[p][2];
       jhip = map[p][3];
       proc = map[p][4];

       if(proc == GAme){

             rows = ihip-ilop+1;
             cols = jhip-jlop+1;
             GAbytes.accloc += (double)size*rows*cols;

        }


       if(proc != GAme && PENDING_OPER(proc) != GA_OP_ACC) FENCE_NODE(proc);

       if(gaDirectAccess(proc, GA_OP_ACC)){

          offset = (jlop - *jlo)* *ld + ilop - *ilo;
          ga_acc_local(*g_a, ilop, ihip, jlop, jhip, buf, offset, *ld,
                       proc, alpha);

       }else{

         /* number of messages determined by message-buffer size */
         Integer TmpSize, ilimit, jlimit;
         Integer ilo_chunk, ihi_chunk, jlo_chunk, jhi_chunk, chunks=0;

#        ifdef LAPI
           Integer   bytes = (ihip - ilop +1)*(jhip - jlop +1)*GAsizeofM(type);
           if(bytes<SHORT_ACC_THRESHOLD) 
                   TmpSize = lapi_max_uhdr_data_sz/size; 
           else
#        endif
                   TmpSize = MSG_BUF_SIZE/size;

         ilimit  = MIN(TmpSize, ihip-ilop+1);
         jlimit  = MIN(TmpSize/ilimit, jhip-jlop+1);

         for(jlo_chunk = jlop; jlo_chunk <= jhip; jlo_chunk += jlimit){
            jhi_chunk  = MIN(jhip, jlo_chunk+jlimit-1);
            for( ilo_chunk = ilop; ilo_chunk<= ihip; ilo_chunk += ilimit){

                  ihi_chunk = MIN(ihip, ilo_chunk+ilimit-1);
                  offset = (jlo_chunk - *jlo)* *ld + ilo_chunk - *ilo;

                  ga_acc_remote(*g_a, ilo_chunk, ihi_chunk,
                         jlo_chunk, jhi_chunk, buf, offset, *ld, proc, alpha);

                  chunks++;
            }
         }
         UPDATE_FENCE_STATE(proc, GA_OP_ACC, chunks);
      }
  }

   if(np > 1 || map[0][4] != GAme) INTR_ON;

  GA_POP_NAME;

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_ACC; 
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
}



/*\ PROVIDE ACCESS TO A PATCH OF A GLOBAL ARRAY
\*/
void FATR  ga_access_(g_a, ilo, ihi, jlo, jhi, index, ld)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *index, *ld;
{
register char *ptr;
Integer  item_size, proc_place, handle = GA_OFFSET + *g_a;

Integer ow;

   if(!ga_locate_(g_a,ilo, jlo, &ow))ga_error("ga_access:locate top failed",0);
   if(ow != GAme) ga_error("ga_access: cannot access top of the patch",0);
   if(!ga_locate_(g_a,ihi, jhi, &ow))ga_error("ga_access:locate bot failed",0);
   if(ow != GAme) ga_error("ga_access: cannot access bottom of the patch",0);

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



   }

   /* check the allignment */
   if(*index % item_size){
       ga_error(" ga_access: base address misallignment ",(long)index);
   }

   /* adjust index according to the data type */
   *index /= item_size;

   /* adjust index for Fortran addressing */
   (*index) ++ ;
}



/*\ RELEASE ACCESS TO A PATCH OF A GLOBAL ARRAY
\*/
void FATR  ga_release_(g_a, ilo, ihi, jlo, jhi)
     Integer *g_a, *ilo, *ihi, *jlo, *jhi;
{}


/*\ RELEASE ACCESS & UPDATE A PATCH OF A GLOBAL ARRAY
\*/
void FATR  ga_release_update_(g_a, ilo, ihi, jlo, jhi)
     Integer *g_a, *ilo, *ihi, *jlo, *jhi;
{}



/*\ INQUIRE POPERTIES OF A GLOBAL ARRAY
 *  Fortran version
\*/ 
void FATR  ga_inquire_(g_a,  type, dim1, dim2)
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
#if defined(CRAY) || defined(WIN32)
void FATR  ga_inquire_name_(g_a, array_name)
      Integer *g_a;
      _fcd    array_name;
{
   c2fstring(GA[GA_OFFSET+ *g_a].name,_fcdtocp(array_name),_fcdlen(array_name));
}
#else
void FATR  ga_inquire_name_(g_a, array_name, len)
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
void ga_inquire_name(Integer *g_a, char** array_name)
{ 
   ga_check_handleM(g_a, "ga_inquire_name");
   *array_name = GA[GA_OFFSET + *g_a].name;
}



/*\ RETURN COORDINATES OF A GA PATCH ASSOCIATED WITH CALLING PROCESSOR
\*/
void FATR  ga_distribution_(g_a, proc, ilo, ihi, jlo, jhi)
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
void FATR  ga_proc_topology_(g_a, proc, pr, pc)
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
logical FATR ga_locate_(g_a, i, j, owner)
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
logical FATR ga_locate_region_(g_a, ilo, ihi, jlo, jhi, map, np )
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

#ifndef LAPI
  GA_PUSH_NAME("ga_scatter_local");
#endif

  ga_distribution_(&g_a, &proc, &ilo, &ihi, &jlo, &jhi);

  /* get a address of the first element owned by proc */
  gaShmemLocation(proc, g_a, ilo, jlo, &ptr_ref, &ldp);

  item_size = GAsizeofM(GA[GA_OFFSET + g_a].type);

  for(k=0; k< nv; k++){
     if(i[k] < ilo || i[k] > ihi  || j[k] < jlo || j[k] > jhi){
       sprintf(err_string,"proc=%d invalid i/j=(%d,%d)>< [%d:%d,%d:%d]",
               proc, i[k], j[k], ilo, ihi, jlo, jhi); 
       ga_error(err_string,g_a);
     }

     offset  = (j[k] - jlo)* ldp + i[k] - ilo;
     ptr_dst = ptr_ref + item_size * offset;
     ptr_src = ((char*)v) + k*item_size; 

#    if defined(SHMEM) && !defined(SYSV)
           CopyElemTo(ptr_src, ptr_dst, item_size/sizeof(Integer), proc);
#    else
           Copy(ptr_src, ptr_dst, item_size);
#    endif

  }

# ifdef LAPI
  if(proc!=GAme){
           UPDATE_FENCE_STATE(proc, GA_OP_SCT, nv);
           SET_COUNTER(ack_cntr, nv);
  }
# endif

#ifndef LAPI
  GA_POP_NAME;
#endif
}



void ga_scatter_remote(g_a, v, i, j, nv, proc) 
     Integer g_a, *i, *j, nv, proc;
     Void *v;
{
register Integer item_size, offset, nbytes, msglen;

  if (nv < 1) return;
  if(proc<0)ga_error(" scatter_remote: invalid process ",proc);

  item_size = GAsizeofM(GA[GA_OFFSET + g_a].type);
  
#ifdef LAPI
  CLEAR_COUNTER(buf_cntr); /* wait for buffer */
#endif

  nbytes = nv*item_size; 
  Copy(v, MessageSnd->buffer, nbytes);

  offset = nbytes;
  nbytes = nv*sizeof(Integer); 
  Copy((char*)i, MessageSnd->buffer + offset, nbytes);

  offset += nbytes; 
  Copy((char*)j, MessageSnd->buffer + offset, nbytes);
  
  msglen = offset + nbytes;
  ga_snd_req(g_a, nv, 0,0,0, msglen, GA[GA_OFFSET + g_a].type, GA_OP_SCT,proc,  DataServer(proc));
}



/*\ SCATTER OPERATION elements of v into the global array
\*/
void FATR  ga_scatter_(g_a, v, i, j, nv)
     Integer *g_a, *nv, *i, *j;
     Void *v;
{
register Integer k;
Integer pindex, phandle, item_size, localop;
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

      if(proc == GAme){
             localop =1;
             GAbytes.scaloc += (double)item_size* nelem ;
      }else{

             /* can we access data on remote process directly? */
             localop = gaDirectAccess(proc, GA_OP_SCT);

             /* might still want to go with remote access protocol */
#            ifdef LAPI
               if(nelem > 10 ) localop = 0;
#            endif

             FENCE_NODE(proc);
      }

      if(localop)
        ga_scatter_local(*g_a, ((char*)v)+item_size*first, i+first, 
                         j+first, nelem,proc);
      else{

        /* limit messages to buffer length */

        Integer last = first + nelem -1; 
        Integer range, chunk, num=0;
        for(range = first; range <= last; range += BufLimit){
            chunk = MIN(BufLimit, last -range+1); 
            ga_scatter_remote(*g_a, ((char*)v)+item_size*range, 
                              i+range,j+range, chunk, proc);
            num++;
        }
        UPDATE_FENCE_STATE(proc, GA_OP_SCT, num);
      }

      first += nelem;
  }while (first< *nv);
  if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);

# ifdef LAPI
    CLEAR_COUNTER(ack_cntr);
# endif

  GA_POP_NAME;
}
      

/*\ permutes input index list using sort routine used in scatter/gather
\*/
void FATR  ga_sort_permut_(g_a, index, i, j, nv)
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

#ifndef LAPI
  GA_PUSH_NAME("ga_gather_local");
#endif

  if(proc==GAme) FLUSH_CACHE;

  ga_distribution_(&g_a, &proc, &ilo, &ihi, &jlo, &jhi);

  /* get a address of the first element owned by proc */
  gaShmemLocation(proc, g_a, ilo, jlo, &ptr_ref, &ldp);

  item_size = GAsizeofM(GA[GA_OFFSET + g_a].type);

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

#    if defined(SHMEM) && !defined(SYSV)
        CopyElemFrom(ptr_src, ptr_dst, item_size/sizeof(Integer), proc);
#    else
        Copy(ptr_src, ptr_dst, item_size);
#    endif
  }
#if defined(LAPI)
    if(proc != GAme)LAPI_Fence(lapi_handle);
#endif
#ifndef LAPI
  GA_POP_NAME;
#endif
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

  len = expected_len;
# if defined(NX) || defined(SP1) || defined(SP)
     msgid = ga_msg_ircv(GA_TYPE_GAT, MessageSnd, expected_len, to);
     ga_snd_req(g_a, nv, 0,0,0, msglen, GA[handle].type, GA_OP_GAT, proc, to);
     ga_msg_wait(msgid, &from, &len);
#  elif(LAPI)
     CLEAR_COUNTER(buf_cntr); /* make sure the buffer is available */
     SET_COUNTER(buf_cntr,1); /* expect data to arrive into the same buffer */
     ga_snd_req(g_a, nv, 0,0,0, msglen, GA[handle].type, GA_OP_GAT, proc, to);
     CLEAR_COUNTER(buf_cntr); /* wait for data to arrive */
# else
     ga_snd_req(g_a, nv, 0, 0, 0, msglen, GA[handle].type, GA_OP_GAT, proc, to);
     ga_msg_rcv(GA_TYPE_GAT, MessageSnd, expected_len, &len,to,&from);
# endif

     if(len != expected_len) ga_error(" gather_remote: wrong data length",len); 

# ifdef IWAY
     /* this is a redundant copy; in IWAY version needs message header */
     Copy(MessageSnd->buffer, (char*)v, nbytes);
# endif
}



/*\ GATHER OPERATION elements from the global array into v
\*/
void FATR  ga_gather_(g_a, v, i, j, nv)
     Integer *g_a, *nv, *i, *j;
     Void *v;
{
register Integer k, nelem;
Integer pindex, phandle, item_size;
Integer first, BufLimit, proc, localop;

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

      if(proc == GAme){

             localop =1;
             GAbytes.gatloc += (double)item_size* nelem ;

      }else{

             /* can we access data on remote process directly? */
             localop = gaDirectAccess(proc, GA_OP_GAT);

             /* might still want to go with remote access protocol */
#            ifdef LAPI
               if(nelem > 10 ) localop = 0;
#            endif

             FENCE_NODE(proc);
      }


      if(localop)
        ga_gather_local(*g_a, ((char*)v)+item_size*first, i+first, j+first,
                        nelem,proc);
      else{

        /* send request for processor proc */

        /* limit messages to buffer length */
        Integer last = first + nelem -1;
        Integer range, chunk;
        INTR_OFF;
        for(range = first; range <= last; range += BufLimit){
            chunk = MIN(BufLimit, last -range+1);
            ga_gather_remote(*g_a, ((char*)v)+item_size*range, i+range, j+range,
                              chunk, proc);
        }
        INTR_ON;
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
Integer *ptr, ldp, value, lval;

#ifndef LAPI
   GA_PUSH_NAME("ga_read_inc_local");
#endif

   /* get a address of the g_a(i,j) element */
   gaShmemLocation(proc, g_a, i, j, (char**)&ptr, &ldp);

#  ifdef _CRAYMPP
        { long lval;
          while ( (lval = shmem_swap((long*)ptr, INVALID, proc) ) == INVALID);
          value = (Integer) lval;
          (void) shmem_swap((long*)ptr, (lval + inc), proc);
        }
#  elif defined(LAPI)
   {      int rc, local;
          lapi_cntr_t req_id;
          if( rc = LAPI_Setcntr(lapi_handle,&req_id,0))
                ga_error("setcntr failed",(Integer)rc);
          if( rc= LAPI_Rmw(lapi_handle, FETCH_AND_ADD, (int)proc, (int*)ptr,
                          &inc,&local,&req_id))ga_error("rmw fail",(Integer)rc);
          if( rc= LAPI_Waitcntr(lapi_handle, &req_id, 1, NULL))
                ga_error("wait failed",(Integer)rc);
          value = local;
   }
#  else
        if(GAnproc>1)LOCK(g_a, proc, ptr);

#         if defined(SHMEM) && !defined(SYSV)
             CopyElemFrom(ptr, &value, 1, proc);
             lval = value + inc;
             CopyElemTo(&lval,ptr,1, proc);
#         else
             value = *ptr;
             lval = value +inc;
             (*ptr) = lval;
#         endif

        if(GAnproc>1)UNLOCK(g_a, proc, ptr);
#  endif

#ifndef LAPI
   GA_POP_NAME;
#endif

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
Integer FATR ga_read_inc_(g_a, i, j, inc)
        Integer *g_a, *i, *j, *inc;
{
Integer  value, proc; 
#ifdef GA_TRACE
       trace_stime_();
#endif

    ga_check_handleM(g_a, "ga_read_inc");
    GA_PUSH_NAME("ga_read_inc");

    GAstat.numrdi++;
    GAbytes.rditot += (double)sizeof(Integer);

    if(GA[GA_OFFSET + *g_a].type !=MT_F_INT)
       ga_error("type must be integer ",*g_a);

    ga_locate_(g_a, i, j, &proc);
    if(gaDirectAccess(proc, GA_OP_RDI)){
        value = ga_read_inc_local(*g_a, *i, *j, *inc, proc);
    }else{
        INTR_OFF;
        value = ga_read_inc_remote(*g_a, *i, *j, *inc, proc);
        INTR_ON;
    }

   if(GAme == proc)GAbytes.rdiloc += (double)sizeof(Integer);

#  ifdef GA_TRACE
     trace_etime_();
     op_code = GA_OP_RDI; 
     trace_genrec_(g_a, i, i, j, j, &op_code);
#  endif

   GA_POP_NAME;
   return(value);
}



Integer FATR ga_nodeid_()
{
  return ((Integer)GAme);
}


Integer FATR ga_nnodes_()
{
  return ((Integer)GAnproc);
}


/*\ COMPARE DISTRIBUTIONS of two global arrays
\*/
logical FATR ga_compare_distr_(g_a, g_b)
     Integer *g_a, *g_b;
{
int h_a =*g_a + GA_OFFSET;
int h_b =*g_b + GA_OFFSET;
int i;

   GA_PUSH_NAME("ga_compare_distr");
   ga_check_handleM(g_a, "distribution a");
   ga_check_handleM(g_b, "distribution b");

   GA_POP_NAME;

   if(GA[h_a].dims[0] == GA[h_b].dims[0]) return FALSE;
   if(GA[h_a].dims[1] == GA[h_b].dims[1]) return FALSE;
   for(i=0; i <MAPLEN; i++){
      if(GA[h_a].mapc[i] != GA[h_b].mapc[i]) return FALSE;
      if(GA[h_a].mapc[i] == -1) break;
   }
   return TRUE;
}


/*********************** other utility routines *************************/

void FATR  ga_ma_get_ptr_(ptr, address)
      char **ptr, *address;
{
   *ptr = address; 
}


Integer FATR ga_ma_diff_(ptr1, ptr2)
        char *ptr1, *ptr2;
{
   return((Integer)(ptr2-ptr1));
}


/*************************************************************************/

/*\ returns true/false depending on validity of the handle
\*/
logical FATR ga_valid_handle_(Integer *g_a)
{
   if(GA_OFFSET+ (*g_a) < 0 || GA_OFFSET+(*g_a) >= max_global_array ||
      ! (GA[GA_OFFSET+(*g_a)].actv) ) return FALSE;
   else return TRUE;
}
