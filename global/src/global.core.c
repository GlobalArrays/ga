/*
 * module: global.core.c
 * author: Jarek Nieplocha
 * date: Mon Dec 19 19:03:38 CST 1994
 * last modification:
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
#include "global.core.h"
#include "message.h"


#define DEBUG 0
#define USE_MALLOC 1
#define INVALID_MA_HANDLE -1 

void f2cstring();
void c2fstring();


/*\ central barrier algorithm 
\*/
void _central_barrier()
{
#if (defined(SYSV) && !defined(KSR))
      static int local_flag=0;
      volatile int local_barrier, local_barrier1;

      if(cluster_compute_nodes == 1)return;
      local_flag = 1 - local_flag;
      P(MUTEX);
         (*barrier) ++;
         local_barrier = *barrier;
         if(local_barrier==barrier_size) *barrier  = 0;
      V(MUTEX);
 
      if(local_barrier==barrier_size){
                /*LOCK(barrier1);*/
                  *barrier1 = local_flag;
                /* UNLOCK(barrier1);*/
      }else do{
                /*LOCK(barrier1);*/
                   local_barrier1 = *barrier1;
                /*UNLOCK(barrier1);*/
           }while (local_barrier1 != local_flag);
#endif
}



/*\ SYNCHRONIZE ALL THE PROCESSES
\*/
void ga_sync_()
{
void   ga_wait_server();
       if (GAme < 0) return;

#if    defined(SYSV) && !defined(KSR) 
       if(ClusterMode) ga_wait_server();
       _central_barrier();
#elif  defined(CRAY_T3D)
       barrier();
#elif  defined(KSR)
       KSRbarrier();
#else  
       { Integer stype = GA_TYPE_SYN; synch_(&stype); } /* TCGMSG */
#      ifdef PARAGON
             ga_wait_server();  /* synchronize data server thread */
#      endif
#endif
}


void clean_all()
{                  
#ifdef SYSV 
    if(GAinitialized){
#      ifndef KSR
          if(GAnproc>1) SemDel();
#      endif
       if(!(USE_MALLOC) || GAnproc >1)Delete_All_Regions(); 
     }
#endif
}


/*\ CHECK GA HANDLE AND IF IT'S WRONG TERMINATE
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



/*\ CHECK GA HANDLE AND IF IT'S WRONG TERMINATE
 *  C version
\*/
void ga_check_handle(g_a, string)
     Integer *g_a;
     char *string;
{
  ga_check_handleM(g_a,string);
}

void _all_trap_signals()
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
/*     TrapSigIot();*/
#endif
}


void _parent_trap_signals()
{
#ifdef SYSV
void TrapSigChld(), TrapSigInt(), TrapSigHup();
     TrapSigChld();
     TrapSigInt();
     TrapSigHup();
#endif
}



/*\ prepare permuted list of processes for remote ops
\*/
void _permute_proc(nproc)
    Integer nproc;
{
#if defined(SUN) || defined(KSR)
    int random();
    int rand();
#else
    long random();
    long rand();
#endif
    int i, iswap, temp;
    void srandom(), srand();
    if(nproc > GAnproc) ga_error("permute_proc: nproc error ", nproc);

    (void)srand((unsigned)GAme); 

    for(i=0; i< nproc; i++) ProcListPerm[i]=i;
    for(i=0; i< nproc; i++){
      iswap = (int)(rand() % nproc); 
      temp = ProcListPerm[iswap];
      ProcListPerm[iswap] = ProcListPerm[i];
      ProcListPerm[i] = temp;
    }
}
      
 

/*\ INITIALIZE GLOBAL ARRAY STRUCTURES
 *  must be the first GA routine called
\*/
void ga_initialize_()
{
Integer nnodes_(), nodeid_(),type,i;
int  buf_size, bar_size;
Integer *msg_buf = (Integer*)MessageRcv->buffer;

    if(GAinitialized) return;

    ClustInfoInit();
    if(ClusterMode){
       /*** current setup works only for one server per cluster ****/
       GAnproc = (Integer)nnodes_() - num_clusters;
       GAme = (Integer)nodeid_();
       GAmaster= cluster_master - cluster_id; 
       /* data servers have their tcgmsg nodeid negated */
       if(GAme > cluster_master + cluster_compute_nodes -1) GAme = -GAme; 
          else GAme -= cluster_id;
    }else{
       GAmaster= 0;
       GAnproc = (Integer)nnodes_();
       GAme = (Integer)nodeid_();
    }

    if(DEBUG)
    fprintf(stderr, "mode=%d, me=%d, master=%d, clusters=%d clust_nodes=%d\n",
            ClusterMode, GAme, cluster_master, num_clusters, cluster_nodes); 
#ifdef SYSV 
    _all_trap_signals(); /* all processes set up own signal handlers */
#endif


#ifdef KSR
    bar_size = KSRbarrier_mem_req();
#else
    bar_size = 2*sizeof(long);
#endif

#ifdef SYSV 

    /*........................ System V IPC stuff  ...................*/
    buf_size = sizeof(DoublePrecision)*cluster_compute_nodes; 

    /* at the end there is shmem counter for ONE server request counter */
    shmSIZE  = bar_size + buf_size+ sizeof(Integer); 

    if(nodeid_() == cluster_master){
        /* set up the remaining signal handlers */
         _parent_trap_signals();

        /* allocate shared memory for communication buffer and barrier  */
        if(GAnproc == 1 && USE_MALLOC){ 
           barrier  = (int*) malloc(shmSIZE); /* use malloc for single proc */
        }else
           barrier  = (int*) Create_Shared_Region(msg_buf+1,&shmSIZE,msg_buf);

        /* Now, set up shmem communication buffer */
        shmBUF = (DoublePrecision*)( bar_size + (char *)barrier);
        NumRecReq = (Integer*)( bar_size + buf_size +(char *)barrier );
        *NumRecReq= 0;

#       ifndef KSR
           if(GAnproc > 1 ){
              /* allocate and intialize semaphore */
              semaphoreID = SemGet(NUM_SEM);
              SemInit(ALL_SEMS,1);
              *((int*)shmBUF) = semaphoreID;
           }
#       endif
    }

    /* Broadcast shmem ID to all the processes:
     * use message-passing here because there is no other way to communicate
     * between processes until the shared memory communication buffer
     * is established and the barrier is initialized.
     * Somebody else has to fork processes before GA is initialized.
     *
     */

    if(DEBUG) fprintf(stderr,"brdcst GAme=%d\n",GAme);
    type = GA_TYPE_SYN;
    ga_brdcst_clust(type, msg_buf, SHMID_BUF_SIZE, cluster_master, 
                    ALL_CLUST_GRP);

    if(DEBUG) fprintf(stderr,"GAme=%d\n",GAme);

    if(nodeid_() != cluster_master){
        /* remaining processors atach to the shared memory */
        barrier  = (int *) Attach_Shared_Region(msg_buf+1,shmSIZE, msg_buf);

        /* Now, set up shmem communication buffer */
        shmBUF = (DoublePrecision*)( bar_size + (char *)barrier);
        NumRecReq = (Integer*)( bar_size + buf_size +(char *)barrier );

#       ifndef KSR
               /* read semaphore_id from shmem buffer */
               semaphoreID = *((int*)shmBUF);
#       endif
    }


    /* initialize the barrier for nproc processes  */
#   ifdef KSR
       KSRbarrier_init((int)cluster_compute_nodes, (int)GAme, 6,(char*)barrier);
#   else
       barrier_size = cluster_compute_nodes;
       barrier1 = barrier +1; /*next element */
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
     *    are used. However, the use of MA library is NOT required.
     *
     * dbl_mb and int_mb are assigned adresses of their counterparts
     *    (of the same name in MA mafdecls.h file) by calling Fortran
     *    ma__base_address_() routine that calls C ma__get_ptr_ to copy
     *    pointers
     */
    {
    static Integer dtype = MT_F_DBL;
    ma__base_address_(&dtype,&DBL_MB);
    if(!DBL_MB)ga_error("ga_initialize: wrong dbl pointer ", 1L);
    dtype = MT_F_INT;
    ma__base_address_(&dtype,&INT_MB);
    if(!INT_MB)ga_error("ga_initialize: wrong int pointer ", 2L);
    }

    if(ClusterMode)
       if(GAme <0) ga_SERVER(0);
#   if defined(SP1) || defined (NX)
    {
       long oldmask;
       ga_init_handler(MessageRcv, TOT_MSG_SIZE );
       ga_mask(0L, &oldmask);
    }
#   endif


    ga_sync_();
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
register int   i,nprocx,nprocy, fchunk1, fchunk2;
static Integer map1[max_nproc], map2[max_nproc];
Integer nblock1, nblock2;
logical ga_create_irreg();
double sqrt();

      ga_sync_();

      if(*type != MT_F_DBL && *type != MT_F_INT)
         ga_error("ga_create: type not yet supported ",  *type);
      else if( *dim1 <= 0 )
         ga_error("ga_create: array dimension1 invalid ",  *dim1);
      else if( *dim2 <= 0)
         ga_error("ga_create: array dimension2 invalid ",  *dim2);

      /* figure out chunking */
      if(*chunk1 <= 1 && *chunk2 <= 1){
        if(*dim1 == 1)      { nprocx =1; nprocy=GAnproc;}
        else if(*dim2 == 1) { nprocy =1; nprocx=GAnproc;}
        else {
           nprocx= (int)sqrt((double)GAnproc);
           for(i=nprocx;i>0&& (GAnproc%i);i--);
           nprocx =i; nprocy=GAnproc/nprocx;
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

      fchunk1 = MIN(fchunk1, *dim1);
      fchunk2 = MIN(fchunk2, *dim2);

      /* chunk size correction for load balancing */
      while(((*dim1-1)/fchunk1+1)*((*dim2-1)/fchunk2+1) >GAnproc){
           if(fchunk1 == *dim1 && fchunk2 == *dim2) 
                     ga_error("ga_create: chunking failed !! ", 0L);
           if(fchunk1 < *dim1) fchunk1 ++; 
           if(fchunk2 < *dim2) fchunk2 ++; 
      }

      /* Now build map arrays */
      for(i=0, nblock1=0; i< *dim1; i += fchunk1, nblock1++)
               map1[nblock1]=i+1;   
      for(i=0, nblock2=0; i< *dim2; i += fchunk2, nblock2++)
               map2[nblock2]=i+1;   

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


Integer ga__memsize_clust(g_a)
Integer g_a;
{
   Integer clust_node, ganode;
   Integer ilo, ihi, jlo, jhi, nelem;
   Integer item_size=GAsizeof(GA[GA_OFFSET + g_a].type);

   nelem =0;
   for(clust_node=0; clust_node < cluster_compute_nodes; clust_node++){ 
       ganode = clust_node + GAmaster; 
       ga_distribution_(&g_a, &ganode, &ilo, &ihi, &jlo, &jhi);
       nelem += (ihi-ilo+1)*(jhi-jlo+1);
   }
   return (nelem*item_size);
}


void ga__setptr_clust(g_a)
Integer g_a;
{
   Integer clust_node, ganode;
   Integer ilo, ihi, jlo, jhi, nelem;
   Integer item_size=GAsizeof(GA[GA_OFFSET + g_a].type);

   for(clust_node=1; clust_node < cluster_compute_nodes; clust_node++){
       ganode = clust_node-1 + GAmaster; /* previous ganode */
       ga_distribution_(&g_a, &ganode, &ilo, &ihi, &jlo, &jhi);
       nelem = (ihi-ilo+1)*(jhi-jlo+1);
       GA[GA_OFFSET + g_a].ptr[clust_node] = 
             GA[GA_OFFSET + g_a].ptr[clust_node-1] + nelem*item_size;
   }
}


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
#ifdef SYSV
   Integer *msg_buf = (Integer*)MessageRcv->buffer;
#else
   Integer handle=INVALID_MA_HANDLE, index;
   Integer len = sizeof(char*), nelem, item_size;
   char    op='*', *ptr = NULL;
#endif

Integer  ilo, ihi, jlo, jhi;
Integer  id=1, mtype = GA_TYPE_SYN, mem_size;
Integer  i, ga_handle;
double   sqrt();

      ga_sync_();

      if(*type != MT_F_DBL && *type != MT_F_INT)
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
        fprintf(stderr,"map1\n");
        for (i=0;i<*nblock1;i++)fprintf(stderr," %d ",map1[i]);
        fprintf(stderr,"\nmap2\n");
        for (i=0;i<*nblock2;i++)fprintf(stderr," .%d ",map2[i]);
        fprintf(stderr,"\n\n");
      }

      /* Get next free global_array handle */
      ga_handle =-1; i=0;
      do{
          if(!GA[i].actv) ga_handle=i;
          i++;
      }while(i<max_global_array && ga_handle==-1);
      if( ga_handle == -1)
        ga_error("ga_create: too many arrays ", (Integer)max_global_array);
      *g_a = (Integer)ga_handle - GA_OFFSET;

      /* fill in info record for g_a */
      GA[ga_handle].type = *type;
      GA[ga_handle].actv = 1;
      strcpy(GA[ga_handle].name, array_name);
      GA[ga_handle].dims[0] = *dim1;
      GA[ga_handle].dims[1] = *dim2;
      GA[ga_handle].nblock[0] = (int) *nblock1;
      GA[ga_handle].nblock[1] = (int) *nblock2;


      /* Copy distribution maps, map1 & map2, into mapc:
       * . since nblock1*nblock2<=GAnproc,  mapc[GAnproc+1] suffices
       *   to pack everything into it;
       * . the dimension of block i is given as: MAX(mapc[i+1]-mapc[i],dim1/2)
       *
       */
      for(i=0;i< *nblock1; i++) GA[ga_handle].mapc[i] = map1[i];
      for(i=0;i< *nblock2; i++) GA[ga_handle].mapc[i+ *nblock1] = map2[i];
      GA[ga_handle].mapc[*nblock1 + *nblock2] = -1; /* end of block marker */

      if(GAme==0&& DEBUG){
         fprintf(stderr,"\nmapc %d elem\n", *nblock1 + *nblock2);
         for(i=0;i<1+*nblock1+ *nblock2;i++)
             fprintf(stderr,"%d,",GA[ga_handle].mapc[i]);
         fprintf(stderr,"\n\n");
      }

      ga_distribution_(g_a, &GAme, &ilo, &ihi, &jlo, &jhi);
      GA[ga_handle].chunk[0] = ihi-ilo+1;
      GA[ga_handle].chunk[1] = jhi-jlo+1;
      GA[ga_handle].ilo = ilo;
      GA[ga_handle].jlo = jlo;


      /************************ Allocate Memory ***************************/
#     ifndef SYSV 
         nelem = (ihi-ilo+1)*(jhi-jlo+1);
         item_size = GAsizeof(*type);
         mem_size = nelem * item_size;
   
         if(nelem){
            if(!MA_alloc_get(*type, nelem, array_name, &handle, &index)) id =0;
            if(id)MA_get_pointer(handle, &ptr);
         }

         GA[ga_handle].ptr[GAme] = ptr;
#        ifdef CRAY_T3D
            for(i=0; i < GAnproc; i++) 
                         ga_brdcst_(&mtype, GA[ga_handle].ptr+i, &len, &i);
#        endif
         GA[ga_handle].id   = handle;
         ga_igop(mtype, &id, 1, &op); /*check if everybody succeded */

#     else
      /*......................... allocate shared memory .................*/
         mem_size = ga__memsize_clust(*g_a);
         if(nodeid_() == cluster_master){
            if(GAnproc == 1 && USE_MALLOC){
               GA[ga_handle].ptr[0]  = malloc(mem_size);
            }else {
               msg_buf = (Integer*)MessageSnd->buffer;
               GA[ga_handle].ptr[0] = Create_Shared_Region
                                        (msg_buf+1,&mem_size,msg_buf);

              /* send all g_a info + shm ids to data server */
              if(ClusterMode) {
                Integer nbytes = *nblock1 * sizeof(Integer);
                Copy(map1, MessageSnd->buffer + SHMID_BUF_SIZE, nbytes);
                Copy(map2, MessageSnd->buffer + SHMID_BUF_SIZE +nbytes, 
                     *nblock2 * sizeof(Integer));
                nbytes = SHMID_BUF_SIZE+ (*nblock1 + *nblock2) *sizeof(Integer);

                ga_snd_req(0,  *dim1, *nblock1, *dim2, *nblock2, nbytes, *type,
                           GA_OP_CRE, GAme, DataServer(GAme));
              }
           }
         }


         ga_brdcst_clust(mtype, msg_buf, SHMID_BUF_SIZE, cluster_master, 
                         CLUST_GRP);

         if(nodeid_() != cluster_master)
            GA[ga_handle].ptr[0] = Attach_Shared_Region(msg_buf+1,mem_size, msg_buf);

         ga__setptr_clust(*g_a); /* determine pointers to individual blocks */
         GA[ga_handle].id   = id;
        /* ............................................................... */
#     endif
      GA[ga_handle].size = mem_size;

      ga_sync_();

      if(id) return(TRUE);
      else{
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
 *  -- new array g_b will have the properties of g_a
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
#ifdef SYSV 
   Integer *msg_buf = (Integer*)MessageRcv->buffer;
#else
   Integer  ilo, ihi, jlo, jhi, nelem;
   Integer index, handle=INVALID_MA_HANDLE;
   Integer len = sizeof(char*);
   char    op='*', *ptr = NULL;
#endif

Integer  id=1, mtype = GA_TYPE_SYN, mem_size;
Integer  i, ga_handle;

      ga_sync_();

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

      GA[ga_handle] = GA[GA_OFFSET + *g_a];
      strcpy(GA[ga_handle].name, array_name);

      mem_size  = GA[ga_handle].size;

      /************************ Allocate Memory ***************************/
#     ifndef SYSV 
         nelem = GA[ga_handle].chunk[1]*GA[ga_handle].chunk[0];
         if(nelem>0){
            if(!MA_alloc_get(GA[ga_handle].type, nelem, array_name, &handle, &index))
                                                                         id =0;
            if(id)MA_get_pointer(handle, &ptr);

            GA[ga_handle].ptr[GAme] = ptr;
         }
         GA[ga_handle].id   = handle;
  
#        ifdef CRAY_T3D
            for(i=0;i<GAnproc;i++) 
                      ga_brdcst_(&mtype, GA[ga_handle].ptr+i, &len, &i);
#        endif
  
         ga_igop(mtype, &id, 1, &op); /*check if everybody succeded */

#     else
      /*......................... Allocate Shared Memory .................*/
         if(nodeid_()==cluster_master){
            if(GAnproc == 1 && USE_MALLOC){
               GA[ga_handle].ptr[0]  = malloc(mem_size);
               id =1;
            }else {
               msg_buf = (Integer*)MessageSnd->buffer;
               GA[ga_handle].ptr[0] = Create_Shared_Region
                                        (msg_buf+1,&mem_size,msg_buf);

              /* send all g_a handle + shm ids to data server */
              if(ClusterMode) {
                ga_snd_req(*g_a, 0, 0, 0, 0, SHMID_BUF_SIZE, 0, GA_OP_DUP,
                           GAme, DataServer(GAme));
              }
            }
         }


         ga_brdcst_clust(mtype, msg_buf, SHMID_BUF_SIZE, cluster_master,
                         CLUST_GRP);

         if(nodeid_()!=cluster_master)
            GA[ga_handle].ptr[0] = Attach_Shared_Region(msg_buf+1,mem_size, msg_buf);

         ga__setptr_clust(*g_b); /* determine pointers to individual blocks */
         GA[ga_handle].id   = id;
         /* ............................................................... */
#     endif

      ga_sync_();

      if(id) return(TRUE);
      else{ 
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
void Free_Shmem_Ptr();

    ga_sync_();
    ga_check_handleM(g_a,"ga_destroy");       
 
#   ifdef SYSV 
      if(GAnproc == 1 && USE_MALLOC){
         free(GA[GA_OFFSET + *g_a].ptr[0]);
      }else{
         if(nodeid_() == cluster_master){
            /* Now, deallocate shared memory */
            Free_Shmem_Ptr(GA[GA_OFFSET + *g_a].id,GA[GA_OFFSET + *g_a].size,GA[GA_OFFSET + *g_a].ptr[0]);

            if(ClusterMode) 
               ga_snd_req(*g_a, 0,0,0,0,0,0, GA_OP_DES, GAme, DataServer(GAme));
         } 
      }
      GA[GA_OFFSET + *g_a].ptr[0] = NULL;
#   else
      if(GA[GA_OFFSET + *g_a].id != INVALID_MA_HANDLE) 
                                    MA_free_heap(GA[GA_OFFSET + *g_a].id);
#   endif

    GA[GA_OFFSET + *g_a].actv = 0;     
    return(TRUE);
}

    
     
/*\ TERMINATE GLOBAL ARRAY STRUCTURES
 *
 *  all GA arrays are destroyed & shared memory is dealocated
 *  GA routines (except for ga_initialize) shouldn"t be called thereafter 
\*/
void ga_terminate_() 
{
Integer i, handle;
    if(!GAinitialized) return;
    for (i=0;i<max_global_array;i++)
          if(GA[i].actv){
            handle = i - GA_OFFSET ;
            ga_destroy_(&handle);
          }
    
    ga_sync_();

#   ifdef SYSV
      if(GAnproc == 1 && USE_MALLOC){
         free(barrier);
         GAinitialized = 0;
         return;
      }
#   endif

    if(nodeid_() == cluster_master){
       if(ClusterMode) 
             ga_snd_req(0, 0, 0, 0, 0, 0, 0, GA_OP_END, GAme, DataServer(GAme));
         clean_all();
    }
    GAinitialized = 0;
}   

    
/*\ Return Integer 1/0 if array is active/inactive
\*/ 
Integer ga_verify_handle_(g_a)
     Integer *g_a;
{
  return (Integer)
    ((*g_a + GA_OFFSET>= 0) && (*g_a + GA_OFFSET< max_global_array) && 
             GA[GA_OFFSET + (*g_a)].actv);
}


logical LocalComm(proc)
   Integer proc;
{
#ifdef SHMEM
#  ifndef CRAY_T3D
     Integer ClusterID();
     if(ClusterMode && (ClusterID(proc) != cluster_id))
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
shmem_location(proc, g_a, i, j, ptr, ld)
Integer g_a, i, j, proc, *ld;
char **ptr;
{
Integer ilo, ihi, jlo, jhi, offset, proc_place;

      ga_distribution_(&g_a, &proc, &ilo, &ihi, &jlo, &jhi);
      if(i<ilo || i>ihi || j<jlo || j>jhi)
                           ga_error(" shmem_location: invalid (i,j) ",GAme);

      offset = (i - ilo) + (ihi-ilo+1)*(j-jlo);

      /* find location of the proc in current cluster pointer array */
      if(! ClusterMode) proc_place = proc;
      else{
         proc_place = proc - GAmaster;
         if(proc_place < 0 || proc_place >= cluster_compute_nodes){
              ga_error(" shmem_location: invalid process ",proc);
         }
      }

      *ptr = GA[GA_OFFSET + g_a].ptr[proc_place] + offset*GAsizeof(GA[GA_OFFSET + g_a].type);
      *ld = ihi-ilo+1;
}

  

/*\ local put of a 2-dimensional patch of data into a global array
\*/
void ga_put_local(g_a, ilo, ihi, jlo, jhi, buf, offset, ld, proc)
   Integer g_a, ilo, ihi, jlo, jhi, *buf, ld, offset, proc;
{
char     *ptr_src, *ptr_dst, *ptr;
Integer  j,  item_size, nbytes, ldp, elem;

   item_size = (int) GAsizeof(GA[GA_OFFSET + g_a].type);
   elem = ihi - ilo +1;
   nbytes = item_size * elem;

   shmem_location(proc, g_a, ilo, jlo, &ptr, &ldp);
   for (j = 0;  j < jhi-jlo+1;  j++){
        ptr_src = (char *)buf  + item_size* (j*ld + offset );
        ptr_dst = (char *)ptr  + item_size* j *ldp;
#       ifdef CRAY_T3D
              if(proc==GAme) Copy(ptr_src, ptr_dst, nbytes);
              else CopyTo(ptr_src, ptr_dst, elem, proc);
#       else
              CopyTo(ptr_src, ptr_dst, nbytes);
#       endif
   }
}



/*\ remote put of a 2-dimensional patch of data into a global array
\*/
void ga_put_remote(g_a, ilo, ihi, jlo, jhi, buf, offset, ld, proc)
   Integer g_a, ilo, ihi, jlo, jhi, *buf, ld, offset, proc;
{
char     *ptr_src, *ptr_dst;
register Integer  j,  item_size, nbytes, elem, msglen;

   if(proc<0)ga_error(" put_remote: invalid process ",proc);

   item_size =  GAsizeof(GA[GA_OFFSET + g_a].type);
   elem = ihi - ilo +1;
   nbytes = item_size * elem;

   /* Copy patch [ilo:ihi, jlo:jhi] into MessageBuffer */
   ptr_dst = (char*)MessageSnd->buffer;
   for (j = 0;  j < jhi-jlo+1;  j++){
        ptr_src = (char *)buf  + item_size* (j*ld + offset );
        Copy(ptr_src, ptr_dst, nbytes);
        ptr_dst += nbytes; 
   }

   msglen = nbytes*(jhi-jlo+1);
   ga_snd_req(g_a, ilo,ihi,jlo,jhi, msglen, GA[GA_OFFSET + g_a].type, GA_OP_PUT,
              proc, DataServer(proc));
}


/*\ PUT A 2-DIMENSIONAL PATCH OF DATA INTO A GLOBAL ARRAY 
\*/
void ga_put_(g_a, ilo, ihi, jlo, jhi, buf, ld)
   Integer  *g_a,  *ilo, *ihi, *jlo, *jhi, *buf, *ld;
{
Integer  p, np, proc, idx;
Integer  ilop, ihip, jlop, jhip, offset;

#ifdef GA_TRACE
   trace_stime_();
#endif

   ga_check_handleM(g_a, "ga_put");
   if (*ilo <= 0 || *ihi > GA[GA_OFFSET + *g_a].dims[0] ||
       *jlo <= 0 || *jhi > GA[GA_OFFSET + *g_a].dims[1]) 
       ga_error(" ga_put: indices out of range ", *g_a);

      if(! ga_locate_region_(g_a, ilo, ihi, jlo, jhi, map, &np ))
                        ga_error("ga_put: error locate region ", GAme);

      _permute_proc(np); 
      for(idx=0; idx<np; idx++){
          p = (Integer)ProcListPerm[idx];

          ilop = map[p][0];
          ihip = map[p][1];
          jlop = map[p][2];
          jhip = map[p][3];
          proc = map[p][4];

          if(LocalComm(proc)){

             offset = (jlop - *jlo)* *ld + ilop - *ilo;
             ga_put_local(*g_a, ilop, ihip, jlop, jhip, buf, offset, *ld, proc);

          }else{
            /* number of messages determined by message-buffer size */

            Integer ilo_chunk, ihi_chunk, jlo_chunk, jhi_chunk;
            Integer TmpSize = MSG_BUF_SIZE/GAsizeof(GA[GA_OFFSET + *g_a].type);
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

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_PUT; 
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
}


/*\ local get of a 2-dimensional patch of data into a global array
\*/
void ga_get_local(g_a, ilo, ihi, jlo, jhi, buf, offset, ld, proc)
   Integer g_a, ilo, ihi, jlo, jhi, *buf, ld, offset, proc;
{
char     *ptr_src, *ptr_dst, *ptr;
Integer  j,  item_size, nbytes, ldp, elem;

   item_size = (int) GAsizeof(GA[GA_OFFSET + g_a].type);
   elem = ihi - ilo +1;
   nbytes = item_size * elem;

   shmem_location(proc, g_a, ilo, jlo, &ptr, &ldp);

   /* cache coherency problem on T3D */
   if(proc==GAme) FLUSH_CACHE; 

   for (j = 0;  j < jhi-jlo+1;  j++){
       ptr_dst = (char *)buf  + item_size* (j*ld + offset );
       ptr_src = (char*)ptr + item_size* j *ldp;
#      ifdef CRAY_T3D
         if(proc==GAme) Copy(ptr_src, ptr_dst, nbytes);
         else CopyFrom(ptr_src, ptr_dst, elem, proc);
#      else
          CopyFrom(ptr_src, ptr_dst, nbytes);
#      endif
   }
}


/*\  get a patch of an array from remote processor
\*/
void ga_get_remote(g_a, ilo, ihi, jlo, jhi, buf, offset, ld, proc)
   Integer g_a, ilo, ihi, jlo, jhi, *buf, ld, offset, proc;
{
char     *ptr_src, *ptr_dst;
Integer  j,  item_size, nbytes, elem, to, from, len, msglen=0;

   if(proc<0)ga_error(" get_remote: invalid process ",proc);
   item_size =  GAsizeof(GA[GA_OFFSET + g_a].type);
   elem = ihi - ilo +1;
   nbytes = item_size * elem;
   msglen = nbytes*(jhi-jlo+1);

   to = DataServer(proc);
#  if defined(NX) || defined(SP1)
      ga_rcv_msg(GA_TYPE_GET, MessageSnd->buffer, msglen, &len, to,&from,ASYNC);
      ga_snd_req(g_a, ilo,ihi,jlo,jhi, msglen, GA[GA_OFFSET + g_a].type, GA_OP_GET,proc,to);
      waitcom_(&to); /* TCGMSG */
#  else
      ga_snd_req(g_a, ilo,ihi,jlo,jhi, msglen, GA[GA_OFFSET + g_a].type, GA_OP_GET,proc,to);
      ga_rcv_msg(GA_TYPE_GET, MessageSnd->buffer, msglen, &len, to, &from,SYNC);
#  endif


   /* Copy patch [ilo:ihi, jlo:jhi] from MessageBuffer */
   ptr_src = (char*)MessageSnd->buffer;
   for (j = 0;  j < jhi-jlo+1;  j++){
        ptr_dst = (char *)buf  + item_size* (j*ld + offset );
        Copy(ptr_src, ptr_dst, nbytes);
        ptr_src += nbytes;
   }

}

/*\ GET A 2-DIMENSIONAL PATCH OF DATA FROM A GLOBAL ARRAY
\*/
void ga_get_(g_a, ilo, ihi, jlo, jhi, buf, ld)
   Integer  *g_a, *ilo, *ihi, *jlo, *jhi, *buf, *ld;
{
Integer p, np, proc, idx;
Integer ilop, ihip, jlop, jhip, offset;

#ifdef GA_TRACE
   trace_stime_();
#endif

   ga_check_handleM(g_a, "ga_get");
   if (*ilo <= 0 || *ihi > GA[GA_OFFSET + *g_a].dims[0] ||
       *jlo <= 0 || *jhi > GA[GA_OFFSET + *g_a].dims[1]){
       fprintf(stderr,"me=%d, %d %d %d %d\n", GAme, *ilo, *ihi, *jlo, *jhi); 
       ga_error(" ga_get: indices out of range ", *g_a);
   }

      if(! ga_locate_region_(g_a, ilo, ihi, jlo, jhi, map, &np ))
                        ga_error("ga_get: error locate region ", GAme);

      _permute_proc(np); 
      for(idx=0; idx<np; idx++){
          p = (Integer)ProcListPerm[idx];
          ilop = map[p][0];
          ihip = map[p][1];
          jlop = map[p][2];
          jhip = map[p][3];
          proc = map[p][4]; 

          if(LocalComm(proc)){

             offset = (jlop - *jlo)* *ld + ilop - *ilo;
             ga_get_local(*g_a, ilop, ihip, jlop, jhip, buf, offset, *ld, proc);

          }else{
            /* number of messages determined by message-buffer size */

            Integer ilo_chunk, ihi_chunk, jlo_chunk, jhi_chunk;
            Integer TmpSize = MSG_BUF_SIZE/GAsizeof(GA[GA_OFFSET + *g_a].type);
            Integer ilimit  = MIN(TmpSize, ihip-ilop+1);
            Integer jlimit  = MIN(TmpSize/ilimit, jhip-jlop+1);

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

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_GET; 
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
}



void accumulate(alpha, rows, cols, a, ald, b, bld)
Integer rows, cols, ald, bld;
DoublePrecision alpha, *a, *b;
{
Integer r, c;
#ifdef KSR
   void Accum();
#endif

   /* a and b are Fortran arrays! */
   for(c=0;c<cols;c++)
#     ifdef KSR
           Accum(alpha, b + c*bld, a + c*ald, rows);
#     else
           for(r=0;r<rows;r++)
                *(a +c*ald + r) += alpha * *(b + c*bld +r);
#     endif
}


void acc_column(alpha, a, b,n)
Integer n;
DoublePrecision alpha, *a, *b;
{
  int i;
  for (i=0;i<n;i++) a[i] += alpha* b[i];
}


/*\ local accumulate 
\*/
void ga_acc_local(g_a, ilo, ihi, jlo, jhi, buf, offset, ld, proc, alpha)
   Integer g_a, ilo, ihi, jlo, jhi, ld, offset, proc;
   DoublePrecision alpha, *buf;
{
char     *ptr_src, *ptr_dst;
Integer  item_size, ldp;
#ifdef CRAY_T3D
DoublePrecision *pbuffer, *ptr;
Integer j, elem, handle, index;
#endif

   item_size =  GAsizeof(GA[GA_OFFSET + g_a].type);
   shmem_location(proc, g_a, ilo, jlo, &ptr_dst, &ldp);

#  ifdef CRAY_T3D
     if(proc != GAme){
        ptr = (DoublePrecision *) ptr_dst;
        elem = ihi - ilo +1;
        if(!MA_push_get(MT_F_DBL,elem, "ga_acc_temp", &handle, &index))
            ga_error("allocation of ga_acc buffer failed ",GAme);
        MA_get_pointer(handle, &pbuffer);

        LOCK(g_a, proc, ptr_dst);
           for (j = 0;  j < jhi-jlo+1;  j++){
              ptr_dst = (char *)ptr  + item_size* j *ldp;
              ptr_src = (char *)buf  + item_size* (j*ld + offset );
   
              CopyFrom(ptr_dst, pbuffer, elem, proc);
/*              accumulate(alpha, elem, 1, pbuffer, elem, ptr_src, ld );*/
              acc_column(alpha, pbuffer, ptr_src, elem );
              CopyTo(pbuffer, ptr_dst, elem, proc);
           }
        UNLOCK(g_a, proc, ptr_dst);
        MA_pop_stack(handle);
        return;
     }
#  endif
     FLUSH_CACHE; /* addresses cache coherency problem on T3D */
     ptr_src = (char *)buf   + item_size * offset;
     if(GAnproc>1) LOCK(g_a, proc, ptr_dst);
       accumulate(alpha, ihi - ilo +1, jhi-jlo+1, ptr_dst, ldp, ptr_src, ld );
     if(GAnproc>1) UNLOCK(g_a, proc, ptr_dst);
}


/*\ remote accumulate of a 2-dimensional patch of data into a global array
\*/
void ga_acc_remote(g_a, ilo, ihi, jlo, jhi, buf, offset, ld, proc, alpha)
   Integer g_a, ilo, ihi, jlo, jhi, ld, offset, proc;
   DoublePrecision alpha, *buf;
{
char     *ptr_src, *ptr_dst;
register Integer  j,  item_size, nbytes, elem, msglen;

   if(proc<0)ga_error(" acc_remote: invalid process ",proc);
   item_size =  GAsizeof(GA[GA_OFFSET + g_a].type);
   elem = ihi - ilo +1;
   nbytes = item_size * elem;

   /* Copy patch [ilo:ihi, jlo:jhi] into MessageBuffer */
   ptr_dst = (char*)MessageSnd->buffer;
   for (j = 0;  j < jhi-jlo+1;  j++){
        ptr_src = (char *)buf  + item_size* (j*ld + offset );
        Copy(ptr_src, ptr_dst, nbytes);
        ptr_dst += nbytes;
   }

   /* append alpha at the end */
   *(DoublePrecision*)ptr_dst = alpha; 

   msglen = nbytes*(jhi-jlo+1)+item_size;  /* plus alpha */
   ga_snd_req(g_a, ilo,ihi,jlo,jhi, msglen, GA[GA_OFFSET + g_a].type, GA_OP_ACC, 
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
   Integer ilop, ihip, jlop, jhip, offset;

#ifdef GA_TRACE
   trace_stime_();
#endif

   ga_check_handleM(g_a, "ga_acc");
   if (*ilo <= 0 || *ihi > GA[GA_OFFSET + *g_a].dims[0] ||
       *jlo <= 0 || *jhi > GA[GA_OFFSET + *g_a].dims[1])
       ga_error(" ga_acc: indices out of range ", *g_a);
   if (GA[GA_OFFSET + *g_a].type != MT_F_DBL) 
                       ga_error(" ga_acc: type not supported ",*g_a);
   if(! ga_locate_region_(g_a, ilo, ihi, jlo, jhi, map, &np ))
                       ga_error("ga_acc: error locate region ", GAme);

   _permute_proc(np); /* prepare permuted list of indices */
   for(idx=0; idx<np; idx++){
       p = (Integer)ProcListPerm[idx];

       ilop = map[p][0];
       ihip = map[p][1];
       jlop = map[p][2];
       jhip = map[p][3];
       proc = map[p][4];

       if(LocalComm(proc)){

          offset = (jlop - *jlo)* *ld + ilop - *ilo;
          ga_acc_local(*g_a, ilop, ihip, jlop, jhip, buf, offset, *ld,
                       proc, *alpha);

       }else{
         /* number of messages determined by message-buffer size */
         /* alpha will be appended at the end of message */

         Integer TmpSize = (MSG_BUF_SIZE - GAsizeof(GA[GA_OFFSET + *g_a].type))
                         /  GAsizeof(GA[GA_OFFSET + *g_a].type);
         Integer ilimit  = MIN(TmpSize, ihip-ilop+1);
         Integer jlimit  = MIN(TmpSize/ilimit, jhip-jlop+1);
         Integer ilo_chunk, ihi_chunk, jlo_chunk, jhi_chunk;

         for(jlo_chunk = jlop; jlo_chunk <= jhip; jlo_chunk += jlimit){
            jhi_chunk  = MIN(jhip, jlo_chunk+jlimit-1);
            for( ilo_chunk = ilop; ilo_chunk<= ihip; ilo_chunk += ilimit){

                  ihi_chunk = MIN(ihip, ilo_chunk+ilimit-1);
                  offset = (jlo_chunk - *jlo)* *ld + ilo_chunk - *ilo;
                  ga_acc_remote(*g_a, ilo_chunk, ihi_chunk,
                         jlo_chunk, jhi_chunk, buf, offset, *ld, proc, *alpha);

            }
         }
      }
  }

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
Integer  item_size, proc_place;


   ga_check_handleM(g_a, "ga_access");

   if (*ilo <= 0 || *ihi > GA[GA_OFFSET + *g_a].dims[0] ||
       *jlo <= 0 || *jhi > GA[GA_OFFSET + *g_a].dims[1] ||
       *ihi < *ilo ||  *jhi < *jlo){
       fprintf(stderr," %d-> %d  %d \n",GAme, *ilo, *ihi); 
       fprintf(stderr," %d-> %d  %d \n",GAme, *jlo, *jhi); 
       ga_error(" ga_access: indices out of range ", *g_a);
   }

   item_size = (int) GAsizeof(GA[GA_OFFSET + *g_a].type);

   proc_place = GAme -  GAmaster;

   ptr = GA[GA_OFFSET + *g_a].ptr[proc_place] + item_size * ( (*jlo - GA[GA_OFFSET + *g_a].jlo )
         *GA[GA_OFFSET + *g_a].chunk[0] + *ilo - GA[GA_OFFSET + *g_a].ilo);
   *ld    = GA[GA_OFFSET + *g_a].chunk[0];  
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
   if(GA[GA_OFFSET + *g_a].type == MT_F_DBL){
      *index = (Integer) (ptr - (char*)DBL_MB);
   }
   else if(GA[GA_OFFSET + *g_a].type == MT_F_INT){
      *index = (Integer) (ptr - (char*)INT_MB);
   }
   else ga_error(" ga_access: type not supported ",-1L);

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
   c2fstring( GA[GA_OFFSET + *g_a].name , _fcdtocp(array_name), _fcdlen(array_name));
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
   strcpy(array_name, GA[GA_OFFSET + *g_a].name);
}



/*\ RETURN COORDINATES OF A GA PATCH ASSOCIATED WITH CALLING PROCESSOR
\*/
void ga_distribution_(g_a, proc, ilo, ihi, jlo, jhi)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *proc;
{
register int  iproc, jproc, loc;

   ga_check_handleM(g_a, "ga_distribution");
 
   if(*proc > GA[GA_OFFSET + *g_a].nblock[0] * GA[GA_OFFSET + *g_a].nblock[1] - 1 || *proc < 0){

         *ilo = (Integer)0;    *jlo = (Integer)0; 
         *ihi = (Integer)-1;   *jhi = (Integer)-1; 
   }else{
         jproc =  *proc/GA[GA_OFFSET + *g_a].nblock[0] ; 
         iproc = (*proc)%GA[GA_OFFSET + *g_a].nblock[0]; 


         loc = iproc;
         *ilo = GA[GA_OFFSET + *g_a].mapc[loc]; *ihi = GA[GA_OFFSET + *g_a].mapc[loc+1] -1; 

         /* correction to find the right spot in mapc*/
         loc = jproc + GA[GA_OFFSET + *g_a].nblock[0];
         *jlo = GA[GA_OFFSET + *g_a].mapc[loc]; *jhi = GA[GA_OFFSET + *g_a].mapc[loc+1] -1; 

         if( iproc == GA[GA_OFFSET + *g_a].nblock[0] -1) *ihi = GA[GA_OFFSET + *g_a].dims[0];
         if( jproc == GA[GA_OFFSET + *g_a].nblock[1] -1) *jhi = GA[GA_OFFSET + *g_a].dims[1];
/*
         fprintf(stderr,"%d-> %d-%d  %d-%d\n", *proc, *ilo, *ihi, *jlo, *jhi);
*/
   }
}




/*\ finds block i that 'elem' belongs to: map[i]>= elem < map[i+1]
\*/
static int findblock(map,n,dim,elem)
    int *map, n, dim, elem;
{
int candidate, found, b;
double scale= ((double)n)/(double)dim;

    candidate = (int)(scale*elem)-1;
    if(candidate<0)candidate =0;
    found = 0;
    if(map[candidate]>elem){
         b= candidate-1;
         while(b>=0){
            found = (map[b]<=elem);
            if(found)break;
            b--;
         } 
    }else{
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

        


/*\ LOCATE THE OWNER OF THE (i,j) ELEMENT OF A GLOBAL ARRAY
\*/
logical ga_locate_(g_a, i, j, owner)
        Integer *g_a, *i, *j, *owner;
{
int  iproc, jproc;

   ga_check_handleM(g_a, "ga_locate");

   if (*i <= 0 || *i > GA[GA_OFFSET + *g_a].dims[0]  ||
       *j <= 0 || *j > GA[GA_OFFSET + *g_a].dims[1]){
       *owner = -1;
       return( FALSE);
   }

   iproc = findblock(GA[GA_OFFSET + *g_a].mapc,GA[GA_OFFSET + *g_a].nblock[0], GA[GA_OFFSET + *g_a].dims[0], *i);
   jproc = findblock(GA[GA_OFFSET + *g_a].mapc+GA[GA_OFFSET + *g_a].nblock[0],GA[GA_OFFSET + *g_a].nblock[1],
            GA[GA_OFFSET + *g_a].dims[1], *j);

   *owner = jproc* GA[GA_OFFSET + *g_a].nblock[0] + iproc;
   return(TRUE);
}



/*\ LOCATE OWNERS OF THE SPECIFIED PATCH OF A GLOBAL ARRAY
\*/
logical ga_locate_region_(g_a, ilo, ihi, jlo, jhi, map, np )
        Integer *g_a, *ilo, *jlo, *ihi, *jhi, map[][5], *np;
{
Integer  iprocLT, iprocRB, jprocLT, jprocRB;
Integer  owner, ilop, ihip, jlop, jhip, i,j;

   ga_check_handleM(g_a, "ga_locate_region");

   if (*ilo <= 0 || *ihi > GA[GA_OFFSET + *g_a].dims[0] ||
       *jlo <= 0 || *jhi > GA[GA_OFFSET + *g_a].dims[1] ||
       *ihi < *ilo ||  *jhi < *jlo){
       fprintf(stderr," me %d-> %d  %d \n",GAme, *ilo, *ihi);
       fprintf(stderr," me %d-> %d  %d \n",GAme, *jlo, *jhi);
       ga_error(" ga_locate_region: indices out of range ", *g_a);
   }

   /* find "processor coordinates" for the left top corner */
   iprocLT = findblock(GA[GA_OFFSET + *g_a].mapc,GA[GA_OFFSET + *g_a].nblock[0], GA[GA_OFFSET + *g_a].dims[0],*ilo);
   jprocLT = findblock(GA[GA_OFFSET + *g_a].mapc+GA[GA_OFFSET + *g_a].nblock[0],GA[GA_OFFSET + *g_a].nblock[1],
             GA[GA_OFFSET + *g_a].dims[1], *jlo);

   /* find "processor coordinates" for the right bottom corner */
   iprocRB = findblock(GA[GA_OFFSET + *g_a].mapc,GA[GA_OFFSET + *g_a].nblock[0], GA[GA_OFFSET + *g_a].dims[0],*ihi);
   jprocRB = findblock(GA[GA_OFFSET + *g_a].mapc+GA[GA_OFFSET + *g_a].nblock[0],GA[GA_OFFSET + *g_a].nblock[1],
             GA[GA_OFFSET + *g_a].dims[1], *jhi);

   *np = 0;
   for(i=iprocLT;i<=iprocRB;i++)
       for(j=jprocLT;j<=jprocRB;j++){
           owner = j* GA[GA_OFFSET + *g_a].nblock[0] + i;
           ga_distribution_(g_a, &owner, &ilop, &ihip, &jlop, &jhip);
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
     char    *v;
{
char *ptr_src, *ptr_ref, *ptr_dst;
Integer ldp, item_size;
Integer ilo, ihi, jlo, jhi;
register Integer k, offset;

  if (nv < 1) return;
  ga_check_handleM(&g_a, "ga_scatter");
  
  ga_distribution_(&g_a, &proc, &ilo, &ihi, &jlo, &jhi);

  /* get a address of the first element owned by proc */
  shmem_location(proc, g_a, ilo, jlo, &ptr_ref, &ldp);

  item_size = GAsizeof(GA[GA_OFFSET + g_a].type);

  for(k=0; k< nv; k++){
     if (i[k] < ilo || i[k] > ihi  || j[k] < jlo || j[k] > jhi){
        fprintf(stderr," %d ga_scatter_loc: invalid i=%d j=%d [%d:%d,%d:%d]\n",
               proc, i[k], j[k], ilo, ihi, jlo, jhi); 
        ga_error("exiting..",GAme);
     }

        offset  = (j[k] - jlo)* ldp + i[k] - ilo;
        ptr_dst = ptr_ref + item_size * offset;
        ptr_src = v + k*item_size; 

#       ifdef CRAY_T3D
           if(proc==GAme) Copy(ptr_src, ptr_dst, item_size);
           else CopyTo(ptr_src, ptr_dst, 1, proc);
#       else
           Copy(ptr_src, ptr_dst, item_size);
#       endif
  }
}



void ga_scatter_remote(g_a, v, i, j, nv, proc) 
     Integer g_a, *i, *j, nv, proc;
     char    *v;
{
register Integer item_size, offset, nbytes, msglen;

  if (nv < 1) return;
   if(proc<0)ga_error(" scatter_remote: invalid process ",proc);

  item_size = GAsizeof(GA[GA_OFFSET + g_a].type);
  
  nbytes = nv*item_size; 
  Copy(v, MessageSnd->buffer, nbytes);

  offset = nbytes;
  nbytes = nv*sizeof(Integer); 
  Copy(i, MessageSnd->buffer + offset, nbytes);

  offset += nbytes; 
  Copy(j, MessageSnd->buffer + offset, nbytes);
  
  msglen = offset + nbytes;
  ga_snd_req(g_a, nv, 0,0,0, msglen, GA[GA_OFFSET + g_a].type, GA_OP_DST,proc,  DataServer(proc));
}



/*\ SCATTER OPERATION elements of v into the global array
\*/
void ga_dscatter_(g_a, v, i, j, nv)
     Integer *g_a, *nv, *i, *j;
     char *v;
{
register Integer k;
Integer pindex, phandle, item_size;
Integer first, nelem, BufLimit, proc;

  if (*nv < 1) return;

  ga_check_handleM(g_a, "ga_scatter");

  if(!MA_push_get(MT_F_INT,*nv, "p_scatter", &phandle, &pindex))
            ga_error(" ga_scatter: MA failed ", 0L);

  /* find proc that owns the (i,j) element; store it in temp: INT_MB[] */
  for(k=0; k< *nv; k++){
      if(! ga_locate_(g_a, i+k, j+k, INT_MB+pindex+k))
         ga_error("ga_dscatter: invalid i/j",i[k]*100000 + j[k]);
  }

  /* determine limit for message size --  v,i,j will travel together */
  item_size = GAsizeof(GA[GA_OFFSET + *g_a].type);
  BufLimit   = MSG_BUF_SIZE/(2*sizeof(Integer)+item_size);

  /* Sort the entries by processor */
  if(GA[GA_OFFSET + *g_a].type ==MT_F_DBL){
     ga_sort_scat_dbl_(nv, v, i, j, INT_MB+pindex);
  }else
     ga_sort_scat_int_(nv, v, i, j, INT_MB+pindex);
   
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

      if(LocalComm(proc))
        ga_scatter_local(*g_a, v+item_size*first, i+first, j+first, nelem,proc);
      else{

        /* limit messages to buffer length */

        Integer last = first + nelem -1; 
        Integer range, chunk;
        for(range = first; range <= last; range += BufLimit){
            chunk = MIN(BufLimit, last -range+1); 
            ga_scatter_remote(*g_a, v+item_size*range, i+range, j+range, 
                              chunk, proc);
        }
      }

      first += nelem;
  }while (first< *nv);
  if(! MA_pop_stack(phandle)) ga_error("ga_scatter: pop stack failed!",phandle);
}
      



void ga_gather_local(g_a, v, i, j, nv, proc) 
     Integer g_a, *i, *j, nv, proc;
     char    *v;
{
char *ptr_src, *ptr_ref, *ptr_dst;
Integer ldp, item_size;
Integer ilo, ihi, jlo, jhi;
register Integer k, offset;

  if (nv < 1) return;
  ga_check_handleM(&g_a, "ga_gather_local");
  
  ga_distribution_(&g_a, &proc, &ilo, &ihi, &jlo, &jhi);

  /* get a address of the first element owned by proc */
  shmem_location(proc, g_a, ilo, jlo, &ptr_ref, &ldp);

  item_size = GAsizeof(GA[GA_OFFSET + g_a].type);

  for(k=0; k< nv; k++){
     if (i[k] < ilo || i[k] > ihi  || j[k] < jlo || j[k] > jhi){
        fprintf(stderr,"ga_gather_local: k=%d n=%d (%d,%d)\n",k,nv,i[k],j[k]);
        ga_error("ga_gather_local: invalid i/j",i[k]*100000 + j[k]);
     }

     offset  = (j[k] - jlo)* ldp + i[k] - ilo;
     ptr_src = ptr_ref + item_size * offset;
     ptr_dst = v + k*item_size; 

#    ifdef CRAY_T3D
        if(proc==GAme){
              FLUSH_CACHE_LINE(ptr_dst);
              Copy(ptr_src, ptr_dst, item_size);
        }else CopyTo(ptr_src, ptr_dst, 1, proc);
#    else
        Copy(ptr_src, ptr_dst, item_size);
#    endif
  }
}



void ga_gather_remote(g_a, v, i, j, nv, proc) 
     Integer g_a, *i, *j, nv, proc;
     char    *v;
{
register Integer item_size, offset, nbytes;
Integer  len, from, to,  msglen;

  if (nv < 1) return;
  if(proc<0)ga_error(" gather_remote: invalid process ",proc);

  item_size = GAsizeof(GA[GA_OFFSET + g_a].type);
  
  offset = 0;
  nbytes = nv*sizeof(Integer); 
  Copy(i, MessageSnd->buffer + offset, nbytes);

  offset = nbytes;
  Copy(j, MessageSnd->buffer + offset, nbytes);

  msglen = offset + nbytes; 
  to = DataServer(proc);
# if defined(NX) || defined(SP1)
     ga_rcv_msg(GA_TYPE_DGT, v, item_size * nv, &len, to, &from, ASYNC);
     ga_snd_req(g_a, nv, 0,0,0, msglen, GA[GA_OFFSET + g_a].type, GA_OP_DGT, proc, to);
     waitcom_(&to);
# else
     ga_snd_req(g_a, nv, 0, 0, 0, msglen, GA[GA_OFFSET + g_a].type, GA_OP_DGT, proc, to);
     ga_rcv_msg(GA_TYPE_DGT, v, item_size * nv, &len, to, &from, SYNC);
# endif
}



/*\ GATHER OPERATION elements from the global array into v
\*/
void ga_dgather_(g_a, v, i, j, nv)
     Integer *g_a, *nv, *i, *j;
     char *v;
{
register int k;
Integer pindex, phandle, item_size;
Integer first, nelem, BufLimit, proc;

  if (*nv < 1) return;

  ga_check_handleM(g_a, "ga_gather");

  if(!MA_push_get(MT_F_INT, *nv, "p_gather", &phandle, &pindex))
            ga_error(" ga_gather: MA failed ", 0L);

  /* find proc that owns the (i,j) element; store it in temp: INT_MB[] */
  for(k=0; k< *nv; k++){
      if(! ga_locate_(g_a, i+k, j+k, INT_MB+pindex+k))
         ga_error(" ga_gather: invalid i/j",i[k]*100000 + j[k]);
  }

  /* Sort the entries by processor */
  ga_sort_gath_(nv, i, j, INT_MB+pindex);
   
  /* determine limit for message size --                               *
   * --  i,j will travel together in the request;  v will be sent back * 
   * --  due to limited buffer space (i,j,v) will occupy the same buf  * 
   *     when server executes ga_dgather_local                         */

  item_size = GAsizeof(GA[GA_OFFSET + *g_a].type);
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

      if(LocalComm(proc))
        ga_gather_local(*g_a, v+item_size*first, i+first, j+first, nelem,proc);
      else{

        /* limit messages to buffer length */

        Integer last = first + nelem -1;
        Integer range, chunk;
        for(range = first; range <= last; range += BufLimit){
            chunk = MIN(BufLimit, last -range+1);
            ga_gather_remote(*g_a, v+item_size*range, i+range, j+range, 
                              chunk, proc);
        }
      }

      first += nelem;
  }while (first< *nv);

  if(! MA_pop_stack(phandle)) ga_error("ga_gather: pop stack failed!",phandle);
}
      
           

/*\ local read and increment of an element of a global array
\*/
Integer ga_read_inc_local(g_a, i, j, inc, proc)
        Integer g_a, i, j, inc;
{
Integer *ptr, ldp, value;

   /* get a address of the g_a(i,j) element */
   shmem_location(proc, g_a, i, j, &ptr, &ldp);

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
   return(value);
}


/*\ remote read and increment of an element of a global array
\*/
Integer ga_read_inc_remote(g_a, i, j, inc, proc)
        Integer g_a, i, j, inc;
{
Integer len, from, to, value;

   if(proc<0)ga_error(" read_inc_remote: invalid process ",proc);

   to = DataServer(proc);
#  if defined(NX) || defined(SP1)
      ga_rcv_msg(GA_TYPE_RDI, &value, sizeof(value), &len, to, &from, ASYNC);
      ga_snd_req(g_a, i, inc, j, 0, sizeof(value), GA[GA_OFFSET + g_a].type, GA_OP_RDI,proc,  to);
      waitcom_(&to); /* TCGMSG */
#  else
      ga_snd_req(g_a, i, inc, j, 0, sizeof(value), GA[GA_OFFSET + g_a].type, GA_OP_RDI,proc,  to);
      ga_rcv_msg(GA_TYPE_RDI, &value, sizeof(value), &len, to, &from, SYNC);
#  endif
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
    if(GA[GA_OFFSET + *g_a].type !=MT_F_INT)ga_error(" ga_read_inc: must be integer ",*g_a);

    ga_locate_(g_a, i, j, &proc);
    if(LocalComm(proc)){
        value = ga_read_inc_local(*g_a, *i, *j, *inc, proc);
    }else{
        value = ga_read_inc_remote(*g_a, *i, *j, *inc, proc);
    }

#  ifdef GA_TRACE
     trace_etime_();
     op_code = GA_OP_RDI; 
     trace_genrec_(g_a, i, i, j, j, &op_code);
#  endif

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

void ma__get_ptr_(ptr, address)
      char **ptr, *address;
{
   *ptr = address; 
}


Integer ma__diff_(ptr1, ptr2)
        char *ptr1, *ptr2;
{
   return((Integer)(ptr2-ptr1));
}


Integer GAsizeof(type)    
        Integer type;
{
   return(ma__sizeof_(&type));
}


/*************************************************************************/
