/*
 * module: global.shm.c
 * author: Jarek Nieplocha
 * last modification: Fri Jun 24 11:02:04 PDT 1994
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
#include "global.shmem.h"

#define USE_MALLOC 1

void f2cstring();
void c2fstring();
#if !(defined(SGI)||defined(IBM))
int  fprintf();
#endif
#if defined(SUN)
void fflush();
#endif
#define SHM
char *malloc();


/*\ SYNCHRONIZE ALL THE PROCESSES  
 *  central barrier algorithm
\*/
void ga_sync_()
{
#ifndef KSR
static int local_flag=0;
volatile int local_barrier, local_barrier1;

     if(nproc == 1)return;
     local_flag = 1 - local_flag;
     LOCK(barrier);
         (*barrier) ++;
         local_barrier = *barrier;
         if(local_barrier==barrier_size) *barrier  = 0;
     UNLOCK(barrier);
 
     if(local_barrier==barrier_size){
                /*LOCK(barrier1);*/
                  *barrier1 = local_flag;
                /* UNLOCK(barrier1);*/
     }else do{
                /*LOCK(barrier1);*/
                   local_barrier1 = *barrier1;
                /*UNLOCK(barrier1);*/
           }while (local_barrier1 != local_flag);
#else
     KSRbarrier();
#endif
}



void clean_all()
{                  
int i;
    if(GAinitialized){
#ifndef KSR
       if(nproc>1) SemDel();
#endif
       if(nproc>1 || !(USE_MALLOC))Delete_All_Regions(); 
     }
}

Integer ga_verify_handle_(g_a)
     Integer *g_a;
/*
  Return Integer 1/0 if array is active/inactive
*/  
{
  return (Integer) 
    ((*g_a >= 0) &&
    (*g_a < max_global_array) &&
    GA[(*g_a)].actv);
}  

/*\ CHECK GA HANDLE AND IF IT'S WRONG TERMINATE
 *  Fortran version
\*/
void ga_check_handle_(g_a, fstring,slen)
     Integer *g_a, slen;
     char *fstring;
{
char  buf[FLEN];

    if( (*g_a) < 0 || (*g_a) >= max_global_array){
      f2cstring(fstring ,slen, buf, FLEN);
      fprintf(stderr, " ga_check_handle: %s ", buf);
      fflush(stderr);
      ga_error(" invalid global array handle ", (*g_a));
    }
    if( ! (GA[(*g_a)].actv) ){
      f2cstring(fstring ,slen, buf, FLEN);
      fprintf(stderr, " ga_check_handle: %s ", buf);
      fflush(stderr);
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


/*\ INITIALIZE GLOBAL ARRAY STRUCTURES
 *  must be the first GA routine called
\*/
void ga_initialize_()
{
long id,  originator = 0L;
long nnodes_(), nodeid_(),type,i;
void TrapSigInt(), TrapSigChld(),
     TrapSigBus(), TrapSigFpe(),
     TrapSigIll(), TrapSigSegv(),
     TrapSigSys(), TrapSigTrap(),
     TrapSigHup(), TrapSigTerm();
int  buf_size, bar_size;
long msg_buf[MAX_REG+2], len = sizeof(msg_buf);


    nproc = (Integer)nnodes_();
    if (nproc==0) nproc=1;
    me = (Integer)nodeid_();

    buf_size = 2*sizeof(DoublePrecision)*nproc;

    if(GAinitialized) return;

#ifdef KSR
    bar_size = KSRbarrier_mem_req();
#else
    bar_size = 2*sizeof(long);
#endif

    shmSIZE  = bar_size + buf_size;

    /* set up single process signal handlers */
    TrapSigBus();
    TrapSigFpe();
    TrapSigIll();
    TrapSigSegv();
    TrapSigSys();
    TrapSigTrap();
    TrapSigTerm();

    if(!me){

        /* set up the remaining signal handlers */
        TrapSigChld();
        TrapSigInt();
        TrapSigHup();

        /* allocate shared memory for communication buffer and barrier  */
        if(nproc == 1 && USE_MALLOC){ 
           barrier  = (int*) malloc(shmSIZE); /* use malloc for single proc */
        }else
           barrier  = (int*) Create_Shared_Region(msg_buf+1,&shmSIZE,msg_buf);

        /* Now, set up shmem communication buffer */
        shmBUF = (DoublePrecision*)( bar_size + (char *)barrier);

#ifndef KSR
        if(nproc > 1 ){
           /* allocate and intialize semaphore */
           semaphoreID = SemGet(1);
           SemInit(MUTEX,1);
           *((int*)shmBUF) = semaphoreID;
        }
#endif
    }

    /* Broadcast shmem ID to all the processes:
     * use message-passing here because there is no other way to communicate
     * between processes until the shared memory communication buffer
     * is established and the barrier is initialized.
     * Somebody else has to fork processes before GA is initialized.
     *
     */

    type = GA_TYPE_SYN;
    ga_brdcst_(&type,  msg_buf, &len, &originator );

    if(me){
        /* remaining processors atach to the shared memory */
        barrier  = (int *) Attach_Shared_Region(msg_buf+1,shmSIZE, msg_buf);

        /* Now, set up shmem communication buffer */
        shmBUF = (DoublePrecision*)( bar_size + (char *)barrier);

#ifndef KSR
        /* read semaphore_id from shmem buffer */
        semaphoreID = *((int*)shmBUF);
#endif
    }

    shmID  = id;  /* save the shared memory ID */

    /* initialize the barrier for nproc processes  */
#ifdef KSR
    KSRbarrier_init((int)nproc, (int)me, 6, (char*)barrier);
#else
    barrier_size = nproc;
    barrier1 = barrier +1; /*next element */
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
    ga_sync_();
}



/*\ CREATE A GLOBAL ARRAY
\*/
logical ga_create_(type, dim1, dim2, array_name, chunk1, chunk2, g_a, slen)
     Integer *type, *dim1, *dim2, *chunk1, *chunk2, *g_a, slen; 
     char *array_name;
     /*
      array_name    - a unique character string [input]
      type          - MA type [input]
      dim1/2        - array(dim1,dim2) as in FORTRAN [input]
      chunk1/2      - minimum size that dimensions should
                      be chunked up into [input]
                      setting chunk1=dim1 gives distribution by rows
                      setting chunk2=dim2 gives distribution by columns 
                      Actual chunk sizes are modified so that they are
                      at least the min size and each process has either
                      zero or one chunk. 
                      chunk1/2 <=1 yields even distribution
      g_a           - Integer handle for future references [output]
      */
{
char  name[FNAM];
register int   i,nprocx,nprocy, fchunk1, fchunk2;
static Integer map1[max_nproc], map2[max_nproc];
Integer nblock1, nblock2;
logical ga_create_irreg_();
double sqrt();

      ga_sync_();

      f2cstring(array_name,(int)slen, name, FNAM);
      if(*type != MT_F_DBL && *type != MT_F_INT)
         ga_error("ga_create: type not yet supported ",  *type);
      else if( *dim1 <= 0 )
         ga_error("ga_create: array dimension1 invalid ",  *dim1);
      else if( *dim2 <= 0)
         ga_error("ga_create: array dimension2 invalid ",  *dim2);


      /* figure out chunking */
      if(*chunk1 <= 1 && *chunk2 <= 1){
        if(*dim1 == 1)      { nprocx =1; nprocy=nproc;}
        else if(*dim2 == 1) { nprocy =1; nprocx=nproc;}
        else {
           nprocx= (int)sqrt((double)nproc);
           for(i=nprocx;i>0&& (nproc%i);i--);
           nprocx =i; nprocy=nproc/nprocx;
        }

        fchunk1 = (int) MAX(1, *dim1/nprocx);
        fchunk2 = (int) MAX(1, *dim2/nprocy);
      }else if(*chunk1 <= 1){
        fchunk1 = (int) MAX(1, (*dim1 * *dim2)/(nproc* *chunk2));
        fchunk2 = (int) *chunk2;
      }else if(*chunk2 <= 1){
        fchunk1 = (int) *chunk1;
        fchunk2 = (int) MAX(1, (*dim1 * *dim2)/(nproc* *chunk1));
      }else{
        fchunk1 = (int) MAX(1,  *chunk1);
        fchunk2 = (int) MAX(1,  *chunk2);
      }

      fchunk1 = MIN(fchunk1, *dim1);
      fchunk2 = MIN(fchunk2, *dim2);

      /* chunk size correction for load balancing */
      while(((*dim1-1)/fchunk1+1)*((*dim2-1)/fchunk2+1) >nproc){
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

/*
      fprintf(stderr,"blocks (%d,%d)\n",nblock1, nblock2);
      fprintf(stderr,"chunks (%d,%d)\n",fchunk1, fchunk2);
      if(me==0){
        for (i=0;i<nblock1;i++)fprintf(stderr," %d ",map1[i]);
        for (i=0;i<nblock2;i++)fprintf(stderr," .%d ",map2[i]);
      }
*/
      return( ga_create_irreg_(type, dim1, dim2, array_name, map1, &nblock1,
                         map2, &nblock2, g_a, slen) );

}


/*\ CREATE A GLOBAL ARRAY -- IRREGULAR DISTRIBUTION
\*/
logical ga_create_irreg_(type, dim1, dim2, array_name, map1, nblock1,
                         map2, nblock2, g_a, slen)
     Integer *type, *dim1, *dim2, *map1, *map2, *nblock1, *nblock2, *g_a, slen;
     char *array_name;
     /*
      array_name    - a unique character string [input]
      type          - MA type [input]
      dim1/2        - array(dim1,dim2) as in FORTRAN [input]
      nblock1       - no. of blocks first dimension is divided into [input]
      nblock2       - no. of blocks second dimension is divided into [input]
         map1       - no. ilo in each block [input]
         map2       - no. jlo in each block [input]
      g_a           - Integer handle for future references [output]
      */
{
long  id, shm_size, originator = 0L, mtype ;
char  name[FNAM];
register int   i, k;
double sqrt();
long msg_buf[MAX_REG+2], len = sizeof(msg_buf);
#define DEBUG 0

      ga_sync_();

      f2cstring(array_name,(int)slen, name, FNAM);
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
      else if(*nblock1 * *nblock2 > nproc)
         ga_error("ga_create_irreg: too many blocks ",*nblock1 * *nblock2);

      if(me==0&& DEBUG){
        fprintf(stderr,"map1\n");
        for (i=0;i<*nblock1;i++)fprintf(stderr," %d ",map1[i]);
        fprintf(stderr,"\nmap2\n");
        for (i=0;i<*nblock2;i++)fprintf(stderr," .%d ",map2[i]);
        fprintf(stderr,"\n\n");
      }

      /* Get next free global_array handle */
      k =-1; i=0;
      do{
        if(!GA[i].actv) k=i;
        i++;
      }while(i<max_global_array && k==-1);
      *g_a = (Integer)k;
      if( *g_a == -1)
        ga_error("ga_create: too many arrays ", (Integer)max_global_array);

      /* Processor 0 allocates shared memory */
      shm_size = MAsizeof(*type);
      shm_size *= *dim1* *dim2;
      if(!me){
         if(nproc == 1 && USE_MALLOC){
            GA[*g_a].ptr  = malloc(shm_size);
         }else
            GA[*g_a].ptr  = Create_Shared_Region(msg_buf+1,&shm_size, msg_buf);
      }

      mtype = GA_TYPE_SYN;
      ga_brdcst_(&mtype,  msg_buf, &len, &originator );

      if(me){
         GA[*g_a].ptr  = Attach_Shared_Region(msg_buf+1,shm_size, msg_buf);
      }

      /* complete initialization record for g_a */
      GA[*g_a].type = *type;
      GA[*g_a].actv = 1;
      GA[*g_a].id   = id;
      GA[*g_a].size = shm_size;
      strcpy(GA[*g_a].name, name);
      GA[*g_a].dims[0] = *dim1;
      GA[*g_a].dims[1] = *dim2;
      GA[*g_a].nblock1 = (int) *nblock1;
      GA[*g_a].nblock2 = (int) *nblock2;

      /* Copy distribution maps, map1 & map2, into mapc:
       * . since nblock1*nblock2<=nproc,  mapc[nproc+1] suffices
       *   to pack everything into it;
       * . the dimension of block i is given as: MAX(mapc[i+1]-mapc[i],dim1/2)
       *
       */
      for(i=0;i< *nblock1; i++) GA[*g_a].mapc[i] = map1[i];
      for(i=0;i< *nblock2; i++) GA[*g_a].mapc[i+ *nblock1] = map2[i];
      GA[*g_a].mapc[*nblock1 + *nblock2] = -1; /* end of block marker */
/*
      if(me==0){
      fprintf(stderr,"\nmapc %d elem\n", *nblock1 + *nblock2);
      for(i=0;i<1+ *nblock1 + *nblock2 ; i++) fprintf(stderr," %d, ",GA[*g_a].mapc[i] );
      fprintf(stderr,"\n\n");
      fprintf(stderr,"\n\n");
      }
*/
      ga_sync_();

      return(TRUE);
}



/*\ DUPLICATE A GLOBAL ARRAY
 *  -- new array g_b will have the properties of g_a
\*/
void ga_duplicate_(g_a, g_b, array_name, slen)
     Integer *g_a, *g_b, slen;
     char *array_name;
     /*
      array_name    - a character string [input]
      g_a           - Integer handle for reference array [input]
      g_b           - Integer handle for new array [output]
      */
{
long  id, shm_size, originator = 0L, mtype = GA_TYPE_SYN ;
char  name[FNAM];
register int   i, k;
long msg_buf[MAX_REG+2], len = sizeof(msg_buf);

      ga_sync_();

      ga_check_handleM(g_a,"ga_duplicate");       
      f2cstring(array_name,(int)slen, name, FNAM);

      /* find a free global_array handle for g_b */
      k =-1; i=0;
      do{
        if(!GA[i].actv) k=i;
        i++;
      }while(i<max_global_array && k==-1);
      *g_b = (Integer)k;
      if( *g_b == -1)
        ga_error("ga_duplicate: too many arrays ", (Integer)max_global_array);

      GA[*g_b] = GA[*g_a];
      strcpy(GA[*g_b].name, name);

      shm_size = GA[*g_b].size;

      /* Processor 0 allocates shared memory */
      if(!me){
         if(nproc == 1 && USE_MALLOC) GA[*g_b].ptr  = malloc(shm_size);
         else GA[*g_b].ptr = Create_Shared_Region(msg_buf+1,&shm_size, msg_buf);
      }

      ga_brdcst_(&mtype,  msg_buf, &len, &originator );

      if(me) GA[*g_b].ptr  = Attach_Shared_Region(msg_buf+1,shm_size, msg_buf);

      ga_sync_();
}



/*\ DESTROY A GLOBAL ARRAY
\*/
logical ga_destroy_(g_a)
        Integer *g_a;
{
register long failed=0;
void Free_Shmem_Ptr();

    ga_sync_();
    ga_check_handleM(g_a,"ga_destroy");       
 
    if(nproc == 1 && USE_MALLOC){
       free(GA[*g_a].ptr);
    }else{
       /* Now, deallocate shared memory */
       if(!me){
          Free_Shmem_Ptr(GA[*g_a].id,GA[*g_a].size,GA[*g_a].ptr);
       } 
    }

    GA[*g_a].actv = 0;     
    GA[*g_a].ptr = NULL;
    return(failed ? FALSE : TRUE);
}

    
     
/*\ TERMINATE GLOBAL ARRAY STRUCTURES
 *
 *  all GA arrays are destroyed & shared memory is dealocated
 *  GA routines (except for ga_initialize) shouldn"t be called thereafter 
\*/
void ga_terminate_() 
{
Integer i;
    if(!GAinitialized) return;
    for (i=0;i<max_global_array;i++)
          if(GA[i].actv) ga_destroy_(&i);
    
    ga_sync_();
    if(nproc == 1 && USE_MALLOC){
       free(barrier);
       GAinitialized = 0;
       return;
    }

     if(!me) clean_all();
     GAinitialized = 0;
    
}   

    

/*\ PUT A 2-DIMENSIONAL PATCH OF DATA INTO A GLOBAL ARRAY 
\*/
void ga_put_(g_a, ilo, ihi, jlo, jhi, buf, ld)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *buf, *ld;
{
register char *ptr_src, *ptr_dst;
register int  j,jsrc, item_size, nbytes;

#ifdef GA_TRACE
   trace_stime_();
#endif

   ga_check_handleM(g_a, "ga_put");
   if (*ilo <= 0 || *ihi > GA[*g_a].dims[0] ||
       *jlo <= 0 || *jhi > GA[*g_a].dims[1]) 
       ga_error(" ga_put: indices out of range ", *g_a);

   item_size = (int) MAsizeof(GA[*g_a].type);
   nbytes = item_size * (*ihi - *ilo +1);


   for (j = *jlo-1,jsrc =0; j < *jhi; j++, jsrc ++){
     ptr_dst = GA[*g_a].ptr + item_size*( j*GA[*g_a].dims[0] + *ilo -1);
     ptr_src = (char *)buf  + item_size*( jsrc* *ld );
     CopyTo(ptr_src, ptr_dst, nbytes); 
   }

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_PUT; 
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
}



/*\ GET A 2-DIMENSIONAL PATCH OF DATA FROM A GLOBAL ARRAY
\*/
void ga_get_(g_a, ilo, ihi, jlo, jhi, buf, ld)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *buf, *ld;
{
register char *ptr_src, *ptr_dst;
register int  j,jdst, item_size, nbytes;

#ifdef GA_TRACE
   trace_stime_();
#endif

   ga_check_handleM(g_a, "ga_get");
   if (*ilo <= 0 || *ihi > GA[*g_a].dims[0] ||
       *jlo <= 0 || *jhi > GA[*g_a].dims[1]){
       fprintf(stderr,"me=%d, %d %d %d %d\n", me, *ilo, *ihi, *jlo, *jhi); 
       ga_error(" ga_get: indices out of range ", *g_a);
   }

   item_size = (int) MAsizeof(GA[*g_a].type);
   nbytes = item_size * (*ihi - *ilo +1);


   for (j = *jlo-1, jdst =0; j < *jhi; j++, jdst++){
     ptr_src = GA[*g_a].ptr + item_size*( j*GA[*g_a].dims[0] + *ilo -1);
     ptr_dst = (char *)buf  + item_size*( jdst* *ld );
     CopyFrom(ptr_src, ptr_dst, nbytes); 
   }

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_GET; 
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
}



/*\ ACCUMULATE OPERATION ON A 2-DIMENSIONAL PATCH OF GLOBAL ARRAY
 *
 *  g_a += alpha * patch
\*/
void ga_acc_(g_a, ilo, ihi, jlo, jhi, buf, ld, alpha)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *ld;
   DoublePrecision *buf, *alpha;
{
register DoublePrecision *ptr_src, *ptr_dst;
register int  i,j, jsrc, nelem;
#ifdef KSR
void Accum();
#endif

#ifdef GA_TRACE
   trace_stime_();
#endif

   ga_check_handleM(g_a, "ga_acc");
   if (*ilo <= 0 || *ihi > GA[*g_a].dims[0] ||
       *jlo <= 0 || *jhi > GA[*g_a].dims[1])
       ga_error(" ga_acc: indices out of range ", *g_a);
   if (GA[*g_a].type != MT_F_DBL) ga_error(" ga_acc: type not supported ",*g_a);

#ifndef KSR
   if(nproc>1) LOCK(0);
#endif
   nelem = (*ihi - *ilo +1); 
   for (j = *jlo-1, jsrc=0; j < *jhi; j++, jsrc++){
     ptr_src = buf          + ( jsrc* *ld );
     ptr_dst = ((DoublePrecision *)GA[*g_a].ptr) +(j*GA[*g_a].dims[0] + *ilo-1);

#ifdef KSR
     if(nproc>1) LOCK(ptr_dst);
#endif

#ifdef KSR     
     Accum(*alpha, ptr_src, ptr_dst, nelem);
#else
     for (i = 0; i< nelem; i++) *(ptr_dst +i) += *alpha*  *(ptr_src +i);
#endif

#ifdef KSR     
     if(nproc>1) UNLOCK(ptr_dst);
#endif
   }
#ifndef KSR
   if(nproc>1) UNLOCK(0);
#endif

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
register int  item_size;


   ga_check_handleM(g_a, "ga_access");

   if (*ilo <= 0 || *ihi > GA[*g_a].dims[0] ||
       *jlo <= 0 || *jhi > GA[*g_a].dims[1] ||
       *ihi < *ilo ||  *jhi < *jlo){
       fprintf(stderr," %d-> %d  %d \n",me, *ilo, *ihi); 
       fprintf(stderr," %d-> %d  %d \n",me, *jlo, *jhi); 
       ga_error(" ga_access: indices out of range ", *g_a);
   }

   item_size = (int) MAsizeof(GA[*g_a].type);
   ptr = GA[*g_a].ptr + item_size*((*jlo -1) *GA[*g_a].dims[0] + *ilo -1);

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
   if(GA[*g_a].type == MT_F_DBL){
      *index = (Integer) (ptr - (char*)DBL_MB);
   }
   else if(GA[*g_a].type == MT_F_INT){
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

   /* specify the leading dimension of the patch */
   *ld    = GA[*g_a].dims[0];  
  
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
   *type       = GA[*g_a].type;
   *dim1       = GA[*g_a].dims[0];
   *dim2       = GA[*g_a].dims[1];
}

/*\ INQUIRE NAME OF A GLOBAL ARRAY
 *  Fortran version
\*/
void ga_inquire_name_(g_a, array_name, len)
      Integer *g_a, len;
      char    *array_name;
{
   /* now we need to convert C string into Fortran string */
   c2fstring(GA[*g_a].name, array_name, len);
}


/*\ INQUIRE NAME OF A GLOBAL ARRAY
 *  C version
\*/
void ga_inquire_name(g_a, array_name)
      Integer *g_a;
      char    *array_name;
{ 
   strcpy(array_name, GA[*g_a].name);
}



/*\ RETURN COORDINATES OF A GA PATCH ASSOCIATED WITH CALLING PROCESSOR
\*/
void ga_distribution_(g_a, proc, ilo, ihi, jlo, jhi)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *proc;
{
register int  iproc, jproc, loc;

   ga_check_handleM(g_a, "ga_distribution");
 
   if(*proc > GA[*g_a].nblock1 * GA[*g_a].nblock2 - 1){

/*
         fprintf("fewer blocks %d than procs %d \n", 
                 GA[*g_a].nblock1 * GA[*g_a].nblock2, (int)(*proc));
         fprintf(stderr,"distr:me=%d %d-%d  %d-%d\n",*proc,*ilo, *ihi, *jlo, *jhi);
*/
         *ilo = (Integer)0;    *jlo = (Integer)0; 
         *ihi = (Integer)-1;   *jhi = (Integer)-1; 
   }else{
         jproc =  *proc/GA[*g_a].nblock1 ; 
         iproc = (*proc)%GA[*g_a].nblock1; 


         loc = iproc;
         *ilo = GA[*g_a].mapc[loc]; *ihi = GA[*g_a].mapc[loc+1] -1; 

         /* correction to find the right spot in mapc*/
         loc = jproc + GA[*g_a].nblock1;
         *jlo = GA[*g_a].mapc[loc]; *jhi = GA[*g_a].mapc[loc+1] -1; 

         if( iproc == GA[*g_a].nblock1 -1) *ihi = GA[*g_a].dims[0];
         if( jproc == GA[*g_a].nblock2 -1) *jhi = GA[*g_a].dims[1];
/*
         fprintf(stderr,"%d-> %d-%d  %d-%d\n", *proc, *ilo, *ihi, *jlo, *jhi);
*/
   }
}




/*\ RETURN COORDINATES OF A GA PATCH ASSOCIATED WITH CALLING PROCESSOR 
\*/
void ga_distribution2_(g_a, iproc, ilo, ihi, jlo, jhi)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *iproc;
{
register int  chunkx, chunky, i;
register int  nprocx, nprocy, iprocx, iprocy;
double sqrt();

   ga_check_handleM(g_a, "ga_distribution");
  
   /* 
    * determine  a simple chunking  
    * (needs to be improved to consider page allignment) 
    */
   nprocx= (int)sqrt((double)nproc);
   for(i=nprocx;i>0&& (nproc%i);i--);
   nprocx =i; nprocy=nproc/nprocx;

   iprocx = (*iproc)/nprocy; iprocy = (*iproc)%nprocy;
   fprintf(stderr,"processors %d %d, %d (%d,%d)\n",nprocx,nprocy,
                                           *iproc,iprocx,iprocy); 

   chunkx = (GA[*g_a].dims[0]+nprocx-1)/nprocx;
   chunky = (GA[*g_a].dims[1]+nprocy-1)/nprocy;

   *ilo = (Integer)(iprocx * chunkx +1);
   *jlo = (Integer)(iprocy * chunky +1);

   *ihi = (Integer)(iprocx <nprocx-1 ? *ilo + chunkx -1 : GA[*g_a].dims[0]); 
   *jhi = (Integer)(iprocy <nprocy-1 ? *jlo + chunky -1 : GA[*g_a].dims[1]); 

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

   if (*i <= 0 || *i > GA[*g_a].dims[0]  ||
       *j <= 0 || *j > GA[*g_a].dims[1]){
       *owner = -1;
       return( FALSE);
   }

   iproc = findblock(GA[*g_a].mapc,GA[*g_a].nblock1, GA[*g_a].dims[0], *i);
   jproc = findblock(GA[*g_a].mapc+GA[*g_a].nblock1,GA[*g_a].nblock2,
            GA[*g_a].dims[1], *j);

   *owner = jproc* GA[*g_a].nblock1 + iproc;
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

   if (*ilo <= 0 || *ihi > GA[*g_a].dims[0] ||
       *jlo <= 0 || *jhi > GA[*g_a].dims[1] ||
       *ihi < *ilo ||  *jhi < *jlo){
       fprintf(stderr," me %d-> %d  %d \n",me, *ilo, *ihi);
       fprintf(stderr," me %d-> %d  %d \n",me, *jlo, *jhi);
       ga_error(" ga_locate_region: indices out of range ", *g_a);
   }

   /* find "processor coordinates" for the left top corner */
   iprocLT = findblock(GA[*g_a].mapc,GA[*g_a].nblock1, GA[*g_a].dims[0], *ilo);
   jprocLT = findblock(GA[*g_a].mapc+GA[*g_a].nblock1,GA[*g_a].nblock2,
             GA[*g_a].dims[1], *jlo);

   /* find "processor coordinates" for the right bottom corner */
   iprocRB = findblock(GA[*g_a].mapc,GA[*g_a].nblock1, GA[*g_a].dims[0], *ihi);
   jprocRB = findblock(GA[*g_a].mapc+GA[*g_a].nblock1,GA[*g_a].nblock2,
             GA[*g_a].dims[1], *jhi);

   *np = 0;
   for(i=iprocLT;i<=iprocRB;i++)
       for(j=jprocLT;j<=jprocRB;j++){
           owner = j* GA[*g_a].nblock1 + i;
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
    




/*\ SCATTER OPERATION     
\*/
void ga_dscatter_(g_a, v, i, j, nv)
     Integer *g_a, *nv, *i, *j;
     DoublePrecision *v;
{
register int k;
DoublePrecision *ptr_dst;
 /*
  * Scatter elements of v into the global array
  */

  if (*nv < 1) return;


  ga_check_handleM(g_a, "ga_dscatter");

  if(GA[*g_a].type !=MT_F_DBL)ga_error(" ga_dscatter: type not supported ",*g_a);

  for(k=0; k< *nv; k++){
     if (i[k] <= 0 || i[k] > GA[*g_a].dims[0]  ||
         j[k] <= 0 || j[k] > GA[*g_a].dims[1])
         ga_error("ga_dscatter: invalid i/j",i[k]*100000 + j[k]);
     ptr_dst = ((DoublePrecision *)GA[*g_a].ptr) +((j[k]-1)*GA[*g_a].dims[0] +
               i[k]-1);
     *ptr_dst = v[k];
  }
}



/*\ GATHER OPERATION     
\*/
void ga_dgather_(g_a, v, i, j, nv)
     Integer *g_a, *nv, *i, *j;
     DoublePrecision *v;
{
register int k;
DoublePrecision *ptr_dst;
 /*
  * Gather elements from the global array into v
  */

  if (*nv < 1) return;

  ga_check_handleM(g_a, "ga_dgather");

  if(GA[*g_a].type !=MT_F_DBL)ga_error(" ga_dgather: type not supported ",*g_a);

  for(k=0; k< *nv; k++){
     if (i[k] <= 0 || i[k] > GA[*g_a].dims[0]  ||
         j[k] <= 0 || j[k] > GA[*g_a].dims[1])
         ga_error("ga_dscatter: invalid i/j",i[k]*100000 + j[k]);
     ptr_dst = ((DoublePrecision *)GA[*g_a].ptr) +((j[k]-1)*GA[*g_a].dims[0] +
               i[k]-1);
     v[k] = *ptr_dst ;
  }
}




/*\ READ AND INCREMENT AN ELEMENT OF A GLOBAL ARRAY
\*/
Integer ga_read_inc_(g_a, i, j, inc)
        Integer *g_a, *i, *j, *inc;
{
Integer *ptr, value;

#ifdef GA_TRACE
       trace_stime_();
#endif

  ga_check_handleM(g_a, "ga_read_inc");
  if(GA[*g_a].type !=MT_F_INT)ga_error(" ga_read_inc: must be integer ",*g_a);

  ptr = ((Integer *)GA[*g_a].ptr) +((*j-1)*GA[*g_a].dims[0] + *i -1);

  if(nproc>1)LOCK(ptr);
    value = *ptr;
    (*ptr) += *inc;
  if(nproc>1)UNLOCK(ptr);

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_RDI; 
   trace_genrec_(g_a, i, i, j, j, &op_code);
#endif

  return(value);
}



Integer ga_nodeid_()
{
  return ((Integer)me);
}


Integer ga_nnodes_()
{
  return ((Integer)nproc);
}

void ga_mask_(mask)
  Integer *mask;
{}



void ga_copy_private(g_a, g_b)
     Integer *g_a, *g_b;
{
char    *ptr_src, *ptr_dst ;
register Integer  chunk, nbytes;

   ga_sync_();

   ga_check_handleM(g_a, "ga_copy");
   ga_check_handleM(g_b, "ga_copy");

   if(GA[*g_a].type != GA[*g_b].type ||
     (GA[*g_a].type != MT_F_DBL && GA[*g_a].type != MT_F_INT))
        ga_error("ga_copy: types not identical ", 0L);

   if (GA[*g_a].dims[0]!=GA[*g_b].dims[0] ||
       GA[*g_a].dims[1]!=GA[*g_b].dims[1] )
            ga_error("ga_copy: arrays not conformant", 0L);

   chunk = (GA[*g_a].size -nproc+1)/nproc;

   /* copy "local" block of data */
    nbytes = me<nproc -1 ? chunk: GA[*g_a].size -me*chunk+1;

   ptr_src = GA[*g_a].ptr + me*chunk;
   ptr_dst = GA[*g_b].ptr + me*chunk;
   Copy(ptr_src, ptr_dst, nbytes);

   ga_sync_();
}




void ga_copy_from_patch_(trans, g_a, ilo,ihi,jlo,jhi, g_b)
     Integer *g_a, *ilo,*ihi,*jlo,*jhi, *g_b;
     char *trans;
{
int nbytes, item_size;
register int i, j, jsrc;
char *ptr_src, *ptr_dst;

   ga_sync_();

   ga_check_handleM(g_a, "ga_copy_from_patch");
   ga_check_handleM(g_b, "ga_copy_from_patch");

   if(GA[*g_a].type != GA[*g_b].type ||
     (GA[*g_a].type != MT_F_DBL && GA[*g_a].type != MT_F_INT))
        ga_error("ga_copy_from_patch: wrong types ", 0L);

   /* check if patch indices and g_a dims match */
   if (*ilo <= 0 || *ihi > GA[*g_a].dims[0] ||
       *jlo <= 0 || *jhi > GA[*g_a].dims[1]){
       fprintf(stderr,"me=%d, %d %d %d %d\n", me, *ilo, *ihi, *jlo, *jhi);
       ga_error(" ga_copy_from_patch: g_a indices out of range ", 0L);
   }

   /* check if patch and g_b dimensions are conforming */
   if(*trans == 'n' || *trans == 'N')
     if (*ihi - *ilo +1 != GA[*g_b].dims[0] ||
         *jhi - *jlo +1 != GA[*g_b].dims[1]){
         if(!me)fprintf(stderr,"me=%d, %d %d %d %d\n", me, *ilo,*ihi,*jlo,*jhi);
         ga_error(" ga_copy_from_patch: g_b and patch dims not conforming ",0L);
     }else;
   else
     if (*ihi - *ilo +1 != GA[*g_b].dims[1] ||
         *jhi - *jlo +1 != GA[*g_b].dims[0]){
         if(!me)fprintf(stderr,"me=%d, %d %d %d %d\n", me, *ilo,*ihi,*jlo,*jhi);
         ga_error(" ga_copy_from_patch: g_b and patch dims not conforming ",0L);
     }

   item_size = (int) MAsizeof(GA[*g_a].type);
   nbytes = item_size * (*ihi - *ilo +1);
   
   if(*trans == 'n' || *trans == 'N')
     for(j = me + *jlo-1, jsrc = me; j < *jhi; j += nproc, jsrc += nproc){ 
       ptr_src = GA[*g_a].ptr + item_size*( j*GA[*g_a].dims[0] + *ilo -1);
       ptr_dst = GA[*g_b].ptr + item_size*( jsrc*GA[*g_b].dims[0]);
       Copy(ptr_src, ptr_dst, nbytes);
     }
   else
     for (j = me + *jlo-1, jsrc=me; j < *jhi; j += nproc, jsrc += nproc){
       ptr_src = GA[*g_a].ptr + item_size*( j*GA[*g_a].dims[0] + *ilo -1);
       ptr_dst = GA[*g_b].ptr + item_size*jsrc;

       for (i = 0; i<= *ihi - *ilo; i++){
            Copy(ptr_src, ptr_dst, item_size);
            ptr_src += item_size;
            ptr_dst += item_size * GA[*g_b].dims[0];
       }
   }

   ga_sync_();
}


   

void ga_copy_to_patch_(trans, g_a, ilo,ihi,jlo,jhi, g_b)
     Integer *g_a, *ilo,*ihi,*jlo,*jhi, *g_b;
     char *trans;
{
int nbytes, item_size;
register int i, j, jsrc;
char *ptr_src, *ptr_dst;

   ga_sync_();

   ga_check_handleM(g_a, "ga_copy_to_patch");
   ga_check_handleM(g_b, "ga_copy_to_patch");

   if(GA[*g_a].type != GA[*g_b].type ||
     (GA[*g_a].type != MT_F_DBL && GA[*g_a].type != MT_F_INT))
        ga_error("ga_copy_to_patch: wrong types ", 0L);

   /* check if patch indices and g_a dims match */
   if (*ilo <= 0 || *ihi > GA[*g_a].dims[0] ||
       *jlo <= 0 || *jhi > GA[*g_a].dims[1]){
       fprintf(stderr,"me=%d, %d %d %d %d\n", me, *ilo, *ihi, *jlo, *jhi);
       ga_error(" ga_copy_to_patch: g_a indices out of range ", 0L);
   }

   /* check if patch and g_b dimensions are conforming */
   if(*trans == 'n' || *trans == 'N')
     if (*ihi - *ilo +1 != GA[*g_b].dims[0] ||
         *jhi - *jlo +1 != GA[*g_b].dims[1]){
         if(!me)fprintf(stderr,"me=%d, %d %d %d %d\n", me, *ilo,*ihi,*jlo,*jhi);
         ga_error(" ga_copy_to_patch: g_b and patch dims not conforming ",0L);
     }else;
   else
     if (*ihi - *ilo +1 != GA[*g_b].dims[1] ||
         *jhi - *jlo +1 != GA[*g_b].dims[0]){
         if(!me)fprintf(stderr,"me=%d, %d %d %d %d\n", me, *ilo,*ihi,*jlo,*jhi);
         ga_error(" ga_copy_to_patch: g_b and patch dims not conforming ",0L);
     }

   item_size = (int) MAsizeof(GA[*g_a].type);
   nbytes = item_size * (*ihi - *ilo +1);
   
   if(*trans == 'n' || *trans == 'N')
     for(j = me + *jlo-1, jsrc = me; j < *jhi; j += nproc, jsrc += nproc){ 
       ptr_dst = GA[*g_a].ptr + item_size*( j*GA[*g_a].dims[0] + *ilo -1);
       ptr_src = GA[*g_b].ptr + item_size*( jsrc*GA[*g_b].dims[0]);
       Copy(ptr_src, ptr_dst, nbytes);
     }
   else
     for (j = me + *jlo-1, jsrc=me; j < *jhi; j += nproc, jsrc += nproc){
       ptr_dst = GA[*g_a].ptr + item_size*( j*GA[*g_a].dims[0] + *ilo -1);
       ptr_src = GA[*g_b].ptr + item_size*jsrc;

       for (i = 0; i<= *ihi - *ilo; i++){
            Copy(ptr_src, ptr_dst, item_size);
            ptr_dst += item_size;
            ptr_src += item_size * GA[*g_b].dims[0];
       }
   }

   ga_sync_();
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


Integer MAsizeof(type)    
        Integer type;
{
   return(ma__sizeof_(&type));
}

/*
void Error(string,code)
char * string;
long code;
{
fprintf(stderr, "%s %d\n",string, code);
fflush(stderr);
}
*/

#ifdef SNGL_PROC
long nnodes_()
{
return 1L;
}
long nodeid_()
{
return 0L;
}


void brdcst_(mid,buf, len, originator )
     int mid, *len, *originator;
     char *buf;
{
}

static Integer ncount=0;
Integer nxtval_(mproc)
        Integer *mproc;
{
        if(nproc>1)ga_error("GA - use differrent nxtval",0);
        if(*mproc>=0){ ncount++; return(ncount -1);}
        else {ncount=0; return(0);}
}

#endif


/*************************************************************************/
