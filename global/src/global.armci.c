/* $Id: global.armci.c,v 1.48 2000-11-02 19:17:00 d3h325 Exp $ */
/* 
 * module: global.armci.c
 * author: Jarek Nieplocha
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
 
/*#define PERMUTE_PIDS */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "global.h"
#include "globalp.h"
#include "message.h"
#include "macdecls.h"
#include "global.armci.h"

#define DEBUG 0
#define USE_MALLOC 1
#define INVALID_MA_HANDLE -1 
#define NEAR_INT(x) (x)< 0.0 ? ceil( (x) - 0.5) : floor((x) + 0.5)

#if !defined(CRAY_YMP)
#define BYTE_ADDRESSABLE_MEMORY
#endif

/*uncomment line below to initialize arrays in ga_create/duplicate */
/*#define GA_CREATE_INDEF yes */

/*uncomment line below to verify consistency of MA in every sync */
/*#define CHECK_MA yes */

/* uncomment line below to verify if MA base address is alligned wrt datatype*/ 
#if !(defined(LINUX) || defined(CRAY))
#define CHECK_MA_ALGN 1
#endif

char *fence_array;
static int GA_fence_set=0;

/*\ SYNCHRONIZE ALL THE PROCESSES
\*/
void FATR ga_sync_()
{
extern int GA_fence_set;
#ifdef CHECK_MA
Integer status;
#endif

       ARMCI_AllFence();
       ga_msg_sync_();
       if(GA_fence_set)bzero(fence_array,(int)GAnproc);
       GA_fence_set=0;
#ifdef CHECK_MA
       status = MA_verify_allocator_stuff();
#endif
}


/*\ wait until requests intiated by calling process are completed
\*/
void FATR ga_fence_()
{
    int proc;
    if(GA_fence_set<1)ga_error("ga_fence: fence not initialized",0);
    GA_fence_set--;
    for(proc=0;proc<GAnproc;proc++)if(fence_array[proc])ARMCI_Fence(proc);
    bzero(fence_array,(int)GAnproc);
}

/*\ initialize tracing of request completion
\*/
void FATR ga_init_fence_()
{
    GA_fence_set++;
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


/*\ Register process list
 *  process list can be used to:
 *   1. permute process ids w.r.t. message-passing ids (set PERMUTE_PIDS), or
 *   2. change logical mapping of array blocks to processes
\*/
void ga_register_proclist_(Integer *list, Integer* np)
{
int i;

      GA_PUSH_NAME("ga_register_proclist");
      if( *np <0 || *np >GAnproc) ga_error("invalid number of processors",*np);
      if( *np <GAnproc) ga_error("Invalid number of processors",*np);

      GA_Proc_list = (int*)malloc(GAnproc * sizeof(int)*2);
      GA_inv_Proc_list = GA_Proc_list + *np;
      if(!GA_Proc_list) ga_error("could not allocate proclist",*np);

      for(i=0;i< (int)*np; i++){
          int p  = (int)list[i];
          if(p<0 || p>= GAnproc) ga_error("invalid list entry",p);
          GA_Proc_list[i] = p; 
          GA_inv_Proc_list[p]=i;
      }

      GA_POP_NAME;
}


void GA_Register_proclist(int *list, int np)
{
      int i;
      GA_PUSH_NAME("ga_register_proclist");
      if( np <0 || np >GAnproc) ga_error("invalid number of processors",np);
      if( np <GAnproc) ga_error("Invalid number of processors",np);

      GA_Proc_list = (int*)malloc(GAnproc * sizeof(int)*2);
      GA_inv_Proc_list = GA_Proc_list + np;
      if(!GA_Proc_list) ga_error("could not allocate proclist",np);

      for(i=0; i< np; i++){
          int p  = list[i];
          if(p<0 || p>= GAnproc) ga_error("invalid list entry",p);
          GA_Proc_list[i] = p;
          GA_inv_Proc_list[p]=i;
      }
      GA_POP_NAME;
}



/*\ FINAL CLEANUP of shmem when terminating
\*/
void ga_clean_resources()
{
    ARMCI_Cleanup();
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
void ga_check_handle(Integer *g_a,char * string)
{
  ga_check_handleM(g_a,string);
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
        printf("%d INT_MB=%ld(%lx) DBL_MB=%ld(%lx) DCPL_MB=%ld(%lx)\n",
                (int)GAme, INT_MB,INT_MB, DBL_MB,DBL_MB, DCPL_MB,DCPL_MB);
}



/*\ INITIALIZE GLOBAL ARRAY STRUCTURES
 *
 *  either ga_initialize_ltd or ga_initialize must be the first 
 *         GA routine called (except ga_uses_ma)
\*/
void FATR  ga_initialize_()
{
Integer  i;

    if(GAinitialized) return;

    /* zero in pointers in GA array */
    for(i=0;i<MAX_ARRAYS; i++) {
       GA[i].ptr  = (char**)0;
       GA[i].mapc = (int*)0;
    }
    GAnproc = (Integer)armci_msg_nproc();

#ifdef PERMUTE_PIDS
    ga_sync_();
    ga_hook_();
    if(GA_Proc_list) GAme = (Integer)GA_Proc_list[ga_msg_nodeid_()];
    else
#endif
    GAme = (Integer)armci_msg_me();
    if(GAme<0 || GAme>20000) 
       ga_error("ga_init:message-passing initialization problem: my ID=",GAme);

    MPme= (Integer)armci_msg_me();
    MPnproc = (Integer)armci_msg_nproc();

    if(GA_Proc_list)
      fprintf(stderr,"permutation applied %d now becomes %d\n",(int)MPme,(int)GAme);

    if(GAnproc > MAX_NPROC && MPme==0){
      fprintf(stderr,"Current GA setup is for up to %d processors\n",MAX_NPROC);
      fprintf(stderr,"Please change MAX_NPROC in config.h & recompile\n");
      ga_error("terminating...",0);
    }

    map = (Integer*)malloc((GAnproc*2*MAXDIM +1)*sizeof(Integer));
    if(!map) ga_error("ga_init:malloc failed (map)",0);
    GA_proclist = (Integer*)malloc(GAnproc*sizeof(Integer)); 
    if(!GA_proclist) ga_error("ga_init:malloc failed (proclist)",0);
    fence_array = calloc(GAnproc,1);
    if(!fence_array) ga_error("ga_init:calloc failed",0);

    /* set activity status for all arrays to inactive */
    for(i=0;i<max_global_array;i++)GA[i].actv=0;

    ARMCI_Init(); /* initialize GA run-time library */

    /* assure that GA will not alocate more shared memory than specified */
    if(ARMCI_Uses_shm())
       if(GA_memory_limited) ARMCI_Set_shm_limit(GA_total_memory);

    GAinitialized = 1;

}




/*\ IS MA USED FOR ALLOCATION OF GA MEMORY ?
\*/ 
logical FATR ga_uses_ma_()
{
#ifdef AVOID_MA_STORAGE
   return FALSE;
#else
   if(!GAinitialized) return FALSE;
   
   if(ARMCI_Uses_shm()) return FALSE;
   else return TRUE;
#endif
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
   if(!ga_uses_ma_()) return(GA_total_memory);
   else{
      Integer ma_limit = MA_inquire_avail(MT_F_BYTE);

      if ( GA_memory_limited ) return( MIN(GA_total_memory, ma_limit) );
      else return( ma_limit );
   }
}


/*\ internal malloc that bypasses MA and uses internal buf when possible
\*/
#define MBUFLEN 256
#define MBUF_LEN MBUFLEN+2
static double ga_int_malloc_buf[MBUF_LEN];
static int mbuf_used=0;
#define MBUF_GUARD -1998.1998
void *gai_malloc(int bytes)
{
    void *ptr;
    if(!mbuf_used && bytes <= MBUF_LEN){
       if(DEBUG){
          ga_int_malloc_buf[0]= MBUF_GUARD;
          ga_int_malloc_buf[MBUFLEN]= MBUF_GUARD;
       }
       ptr = ga_int_malloc_buf+1;
       mbuf_used++;
    }else{
        Integer handle, idx, elems = (bytes+sizeof(double)-1)/sizeof(double)+1; 
        if(MA_push_get(MT_DBL, elems, "GA malloc temp", &handle, &idx)){
            MA_get_pointer(handle, &ptr);
            *((Integer*)ptr)= handle;
            ptr = ((double*)ptr)+ 1;  /*needs sizeof(double)>=sizeof(Integer) */
        }else
            ptr=NULL;
    }
    return ptr;
}

void gai_free(void *ptr)
{
    if(ptr == (ga_int_malloc_buf+1)){
        if(DEBUG){
          assert(ga_int_malloc_buf[0]== MBUF_GUARD);
          assert(ga_int_malloc_buf[MBUFLEN]== MBUF_GUARD);
          assert(mbuf_used ==1);
        }
        mbuf_used =0;
    }else{
        Integer handle= *( (Integer*) (-1 + (double*)ptr));
        if(!MA_pop_stack(handle)) ga_error("gai_free:MA_pop_stack failed",0);
    }
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
void FATR  ga_initialize_ltd_(Integer *mem_limit)
{

  GA_total_memory =GA_memory_limit  = *mem_limit; 
  if(*mem_limit >= 0) GA_memory_limited = 1; 
  ga_initialize_();
}


  

#define gam_checktype(_type)\
       if(_type != MT_F_DBL && _type != MT_F_INT &&  _type != MT_F_DCPL)\
         ga_error("type not yet supported ",  _type)

#define gam_checkdim(ndim, dims)\
{\
int _d;\
    if(ndim<1||ndim>MAXDIM) ga_error("unsupported number of dimensions",ndim);\
    for(_d=0; _d<ndim; _d++)\
         if(dims[_d]<1)ga_error("wrong dimension specified",dims[_d]);\
}


/*\ print subscript of ndim dimensional array with two strings before and after
\*/
static void print_subscript(char *pre,int ndim, Integer subscript[], char* post)
{
        int i;

        printf("%s [",pre);
        for(i=0;i<ndim;i++){
                printf("%ld",subscript[i]);
                if(i==ndim-1)printf("] %s",post);
                else printf(",");
        }
}


Integer mapALL[MAX_NPROC+1];

logical nga_create(Integer type,
                   Integer ndim,
                   Integer dims[],
                   char* array_name,
                   Integer chunk[],
                   Integer *g_a)
{
Integer pe[MAXDIM], *pmap[MAXDIM], *map;
Integer d, i;
Integer blk[MAXDIM];
logical status;
/*
extern void ddb(Integer ndims, Integer dims[], Integer npes, Integer blk[], Integer pedims[]);
*/
extern void ddb_h2(Integer ndims, Integer dims[], Integer npes,double thr,Integer bias, Integer blk[], Integer pedims[]);

      GA_PUSH_NAME("nga_create");
      if(!GAinitialized) ga_error("GA not initialized ", 0);
      gam_checktype(type);
      gam_checkdim(ndim, dims);

      if(chunk && chunk[0]!=0) /* for either NULL or chunk[0]=0 compute all */
          for(d=0; d< ndim; d++) blk[d]=MIN(chunk[d],dims[d]);
      else
          for(d=0; d< ndim; d++) blk[d]=-1;

      /* eliminate dimensions =1 from ddb analysis */
      for(d=0; d<ndim; d++)if(dims[d]==1)blk[d]=1;
      
      /* for load balancing overwrite block size if needed */ 
/*      for(d=0; d<ndim; d++)if(blk[d]*GAnproc < dims[d])blk[d]=-1;*/

      if(GAme==0 && DEBUG )for(d=0;d<ndim;d++) fprintf(stderr,"b[%ld]=%ld\n",d,blk[d]);
      ga_sync_();

/*      ddb(ndim, dims, GAnproc, blk, pe);*/
      ddb_h2(ndim, dims, GAnproc, 0.0, (Integer)0, blk, pe);

      for(d=0, map=mapALL; d< ndim; d++){
        Integer nblock;
        int p;
        int pcut; /* # procs that get full blk[] elements; the rest gets less*/

        pmap[d] = map;

        /* RJH ... don't leave some nodes without data if possible
         but respect the users block size */

        if (chunk && chunk[d] > 1) {
          int ddim = (dims[d]-1)/MIN(chunk[d],dims[d]) + 1;
          pcut = (ddim -(blk[d]-1)*pe[d]) ;
        }
        else {
          pcut = (dims[d]-(blk[d]-1)*pe[d]) ;
        }

        for (nblock=i=p=0; (p<pe[d]) && (i<dims[d]); p++, nblock++) {
          int b = blk[d];
          if (p >= pcut)
            b = b-1;
          map[nblock] = i+1;
          if (chunk && chunk[d]>1) b *= MIN(chunk[d],dims[d]);
          i += b;
        }

        pe[d] = MIN(pe[d],nblock);
        map +=  pe[d]; 
      }

      if(GAme==0&& DEBUG){
         print_subscript("pe ",ndim, pe,"\n");
         print_subscript("blocks ",ndim, blk,"\n");
         printf("decomposition map\n");
         for(d=0; d< ndim; d++){
           printf("dim=%ld: ",d); 
           for (i=0;i<pe[d];i++)printf("%ld ",pmap[d][i]);
           printf("\n"); 
         }
         fflush(stdout);
      }

      status = nga_create_irreg(type, ndim, dims, array_name, mapALL, pe, g_a);

      GA_POP_NAME;
      return status;
}


      

logical ga_create(type, dim1, dim2, array_name, chunk1, chunk2, g_a)
     Integer *type, *dim1, *dim2, *chunk1, *chunk2, *g_a;
     char *array_name;
{
Integer ndim=2, dims[2], chunk[2];
logical status;
#ifdef  OLD_DEFAULT_BLK
#define BLK_THR 1
#else
#define BLK_THR 0
#endif

    dims[0]=*dim1;
    dims[1]=*dim2;

    /*block size of 1 is troublesome, old ga treated it as "use default" */
    /* for backward compatibility we use old convention */
    chunk[0] = (*chunk1 ==BLK_THR)? -1: *chunk1;
    chunk[1] = (*chunk2 ==BLK_THR)? -1: *chunk2;

    status = nga_create(*type, ndim,  dims, array_name, chunk, g_a);

    return status;
}


/*\ CREATE A GLOBAL ARRAY
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
logical FATR ga_create_(type, dim1, dim2, array_name, chunk1, chunk2, g_a)
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


/*\ CREATE A GLOBAL ARRAY
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
logical FATR nga_create_(Integer *type, Integer *ndim, Integer *dims,
                   _fcd array_name, Integer *chunk, Integer *g_a)
#else
logical FATR nga_create_(Integer *type, Integer *ndim, Integer *dims,
                   char* array_name, Integer *chunk, Integer *g_a, int slen)
#endif
{
char buf[FNAM];
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

  return (nga_create(*type, *ndim,  dims, buf, chunk, g_a));
}

#if defined(CRAY) || defined(WIN32)
logical FATR nga_create_irreg_(Integer *type, Integer *ndim, Integer *dims,
                 _fcd array_name, Integer map[], Integer block[], Integer *g_a)
#else
logical FATR nga_create_irreg_(Integer *type, Integer *ndim, Integer *dims,
                 char* array_name, Integer map[], Integer block[],
                 Integer *g_a, int slen)
#endif
{
char buf[FNAM];
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

  return (nga_create_irreg(*type, *ndim,  dims, buf, map, block, g_a));
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


#ifdef PERMUTE_PIDS
char* ptr_array[MAX_NPROC];
#endif

/*\ get memory alligned w.r.t. MA base
 *  required on Linux as g77 ignores natural data alignment in common blocks
\*/ 
int gai_get_shmem(char **ptr_arr, int bytes, int type, long *adj)
{
int status=0;
#ifndef _CHECK_MA_ALGN
char *base;
long diff, item_size;  
Integer *adjust;
int i;

    /* need to enforce proper, natural allignment (on size boundary)  */
    switch (type){
      case MT_F_DBL: base =  (char *) DBL_MB; break;
      case MT_F_INT: base =  (char *) INT_MB; break;
      case MT_F_DCPL: base =  (char *) DCPL_MB; break;
      default:        base = (char*)0;
    }

    item_size = GAsizeofM(type);
    bytes += item_size; 

#endif

    *adj = 0;
#ifdef PERMUTE_PIDS
    if(GA_Proc_list){
      bzero(ptr_array,GAnproc*sizeof(char*));
      status = ARMCI_Malloc((void**)ptr_array, bytes);
      for(i=0;i<GAnproc;i++)ptr_arr[i] = ptr_array[GA_inv_Proc_list[i]]; 
    }else
#endif

    status = ARMCI_Malloc((void**)ptr_arr, bytes);
    if(status) return status;

#ifndef _CHECK_MA_ALGN

    /* adjust all addresses if they are not alligned on corresponding nodes*/
    adjust = (Integer*)gai_malloc(GAnproc*sizeof(Integer));

    diff = (ABS( base - (char *) ptr_arr[GAme])) % item_size; 
    for(i=0;i<GAnproc;i++)adjust[i]=0;
    adjust[GAme] = (diff > 0) ? item_size - diff : 0;
    *adj = adjust[GAme];

    ga_igop(GA_TYPE_GSM, adjust, GAnproc, "+");
    
    for(i=0;i<GAnproc;i++){
       ptr_arr[i] = adjust[i] + (char*)ptr_arr[i];
    }
    gai_free(adjust);

#endif
    return status;
}


int gai_getmem(char* name, char **ptr_arr, int bytes, int type, long *id)
{
Integer handle = INVALID_MA_HANDLE, index;
Integer nelem, item_size = GAsizeofM(type);
char *ptr = (char*)0;

#ifdef AVOID_MA_STORAGE
   return gai_get_shmem(ptr_arr, bytes, type, id);
#else
   if(ARMCI_Uses_shm()) return gai_get_shmem(ptr_arr, bytes, type, id);
   else{

     nelem = bytes/item_size + 1;
     if(bytes)
        if(MA_alloc_get(type, nelem, name, &handle, &index))
                MA_get_pointer(handle, &ptr);
     *id   = (long)handle;

     /*
            printf("MA DBL_MB=%ld ptr=%ld index=%d\n",DBL_MB, ptr,index);
            fflush(stdout);
     */

     bzero(ptr_arr,(int)GAnproc*sizeof(char*));
     ptr_arr[GAme] = ptr;
     armci_exchange_address((void**)ptr_arr,(int)GAnproc);
     if(bytes && !ptr) return 1; 
     else return 0;
   }
#endif
}




/*\ RETURN COORDINATES OF A GA PATCH ASSOCIATED WITH PROCESSOR proc
\*/
void FATR nga_distribution_(Integer *g_a, Integer *proc, Integer *lo, Integer *
hi)
{
Integer ga_handle;

   ga_check_handleM(g_a, "nga_distribution");
   ga_handle = (GA_OFFSET + *g_a);
   ga_ownsM(ga_handle, *proc, lo, hi);
}


/*\ CREATE A GLOBAL ARRAY -- IRREGULAR DISTRIBUTION
\*/
logical nga_create_irreg(
        Integer type,     /* MA type */ 
        Integer ndim,     /* number of dimensions */
        Integer dims[],   /* array of dimensions */
        char *array_name, /* array name */
        Integer map[],    /* decomposition map array */ 
        Integer nblock[], /* number of blocks for each dimension in map */
        Integer *g_a)     /* array handle (output) */
{

Integer  hi[MAXDIM];
Integer  mem_size, nelem, mem_size_proc;
Integer  i, ga_handle, status, maplen=0;

      ga_sync_();
      GA_PUSH_NAME("nga_create_irreg");

      if(!GAinitialized) ga_error("GA not initialized ", 0);
      if(!ma_address_init) gai_ma_address_init();

      gam_checktype(type);
      gam_checkdim(ndim, dims);
      for(i=0; i< ndim; i++)
         if(nblock[i]>dims[i]) 
            ga_error("number of blocks must be <= corresponding dimension",i);

      GAstat.numcre ++;

      /*** Get next free global array handle ***/
      ga_handle =-1; i=0;
      do{
          if(!GA[i].actv) ga_handle=i;
          i++;
      }while(i<max_global_array && ga_handle==-1);
      if( ga_handle == -1)
          ga_error(" too many arrays ", (Integer)max_global_array);
      *g_a = (Integer)ga_handle - GA_OFFSET;

      /*** fill in Global Info Record for g_a ***/
      gai_init_struct(ga_handle);
      GA[ga_handle].type = type;
      GA[ga_handle].actv = 1;
      strcpy(GA[ga_handle].name, array_name);
      GA[ga_handle].ndim    = ndim;

      for( i = 0; i< ndim; i++){
         GA[ga_handle].dims[i] = dims[i];
         GA[ga_handle].nblock[i] = nblock[i];
         GA[ga_handle].scale[i] = (double)nblock[i]/(double)dims[i];
         maplen += nblock[i];
      } 
      for(i = 0; i< maplen; i++)GA[ga_handle].mapc[i] = (int)map[i];
      GA[ga_handle].mapc[maplen] = -1;
      GA[ga_handle].elemsize = GAsizeofM(type);

      /*** determine which portion of the array I am supposed to hold ***/
      nga_distribution_(g_a, &GAme, GA[ga_handle].lo, hi);
      for( i = 0, nelem=1; i< ndim; i++){
           GA[ga_handle].chunk[i] = (int)(hi[i]-GA[ga_handle].lo[i]+1);
           nelem *= GA[ga_handle].chunk[i];
      }

      mem_size = mem_size_proc =  nelem * GA[ga_handle].elemsize;
      GA[ga_handle].id = INVALID_MA_HANDLE;
      GA[ga_handle].size = mem_size_proc;

      /* if requested, enforce limits on memory consumption */
      if(GA_memory_limited) GA_total_memory -= mem_size_proc;

      /* check if everybody has enough memory left */
      if(GA_memory_limited){
         status = (GA_total_memory >= 0) ? 1 : 0;
         ga_igop(GA_TYPE_GSM, &status, 1, "*");
      }else status = 1;

/*      fprintf(stderr,"%d, elems=%d size=%d status=%d\n",GAme,nelem,mem_size,status);*/
/*      ga_sync_();*/
      if(status){
          status = !gai_getmem(array_name, GA[ga_handle].ptr,mem_size,
                                 (int)type, &GA[ga_handle].id);
      }else{
          GA[ga_handle].ptr[GAme]=NULL;
      }

      ga_sync_();

      if(status){
         GAstat.curmem += GA[ga_handle].size;
         GAstat.maxmem  = MAX(GAstat.maxmem, GAstat.curmem);
         status = TRUE;
      }else{
         ga_destroy_(g_a);
         status = FALSE;
      }

      GA_POP_NAME;
      return status;
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
Integer  ilo, ihi, jlo, jhi;
Integer  mem_size, nelem, mem_size_proc;
Integer  i, ga_handle, status;

      if(!GAinitialized) ga_error("GA not initialized ", 0);

      ga_sync_();
      if(!ma_address_init) gai_ma_address_init();

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
        fprintf(stderr," array:%d map1:\n", (int)*g_a);
        for (i=0;i<*nblock1;i++)fprintf(stderr," %ld |",map1[i]);
        fprintf(stderr," \n array:%d map2:\n",(int) *g_a);
        for (i=0;i<*nblock2;i++)fprintf(stderr," %ld |",map2[i]);
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
      GA[ga_handle].ndim    = 2;
      GA[ga_handle].dims[0] = (int)*dim1;
      GA[ga_handle].dims[1] = (int)*dim2;
      GA[ga_handle].nblock[0] = (int) *nblock1;
      GA[ga_handle].nblock[1] = (int) *nblock2;
      GA[ga_handle].scale[0] = (double)*nblock1/(double)*dim1;
      GA[ga_handle].scale[1] = (double)*nblock2/(double)*dim2;
      GA[ga_handle].elemsize = GAsizeofM(*type);

      /* Copy distribution maps, map1 & map2, into mapc:
       * . since nblock1*nblock2<=GAnproc,  mapc[GAnproc+1] suffices
       *   to pack everything into it;
       * . the dimension of block i is given as: MAX(mapc[i+1]-mapc[i],dim1/2)
       */
      for(i=0;i< *nblock1; i++) GA[ga_handle].mapc[i] = (int)map1[i];
      for(i=0;i< *nblock2; i++) GA[ga_handle].mapc[i+ *nblock1] = (int)map2[i];
      GA[ga_handle].mapc[*nblock1 + *nblock2] = -1; /* end of block marker */

      if(GAme ==0 && DEBUG){
         fprintf(stderr,"\nmapc %ld elem\n", *nblock1 + *nblock2);
         for(i=0;i<1+*nblock1+ *nblock2;i++)
             fprintf(stderr,"%d,",GA[ga_handle].mapc[i]);
         fprintf(stderr,"\n\n");
      }


      /*** determine which portion of the array I am supposed to hold ***/
      ga_distribution_(g_a, &GAme, &ilo, &ihi, &jlo, &jhi);
      GA[ga_handle].chunk[0] = (int) (ihi-ilo+1);
      GA[ga_handle].chunk[1] = (int) (jhi-jlo+1);
      GA[ga_handle].lo[0] = ilo;
      GA[ga_handle].lo[1] = jlo;
      nelem = (ihi-ilo+1)*(jhi-jlo+1);

      mem_size = mem_size_proc =  nelem * GA[ga_handle].elemsize;
      GA[ga_handle].id = INVALID_MA_HANDLE;

      /* on shmem platforms, we use avg mem_size per processor in cluster */
      GA[ga_handle].size = mem_size_proc;

      /* if requested, enforce limits on memory consumption */
      if(GA_memory_limited) GA_total_memory -= mem_size_proc; 


      /* check if everybody has enough memory left */
      if(GA_memory_limited){
         status = (GA_total_memory >= 0) ? 1 : 0;
         ga_igop(GA_TYPE_GSM, &status, 1, "*");
      }else status = 1;
 
      if(status){
          status = !gai_getmem(array_name, GA[ga_handle].ptr,mem_size,*type,
                              &GA[ga_handle].id);
      }else{
          GA[ga_handle].ptr[GAme]=NULL;
      }

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
logical FATR ga_create_irreg_(type, dim1, dim2, array_name, map1, nblock1,
                         map2, nblock2, g_a)
     Integer *type, *dim1, *dim2, *map1, *map2, *nblock1, *nblock2, *g_a;
     _fcd array_name;
#else
logical FATR ga_create_irreg_(type, dim1, dim2, array_name, map1, nblock1,
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


Integer FATR ga_ndim_(Integer *g_a)
{
      ga_check_handleM(g_a,"ga_ndim");       
      return GA[*g_a +GA_OFFSET].ndim;
}
 


/*\ DUPLICATE A GLOBAL ARRAY
 *  -- new array g_b will have properties of g_a
\*/
logical ga_duplicate(Integer *g_a, Integer *g_b, char* array_name)
     /*
      * array_name    - a character string [input]
      * g_a           - Integer handle for reference array [input]
      * g_b           - Integer handle for new array [output]
      */
{
char     **save_ptr;
Integer  mem_size, mem_size_proc;
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
      mem_size = mem_size_proc = GA[ga_handle].size; 
      GA[ga_handle].id = INVALID_MA_HANDLE;
      /* if requested, enforce limits on memory consumption */
      if(GA_memory_limited) GA_total_memory -= mem_size_proc;

      /* check if everybody has enough memory left */
      if(GA_memory_limited){
         status = (GA_total_memory >= 0) ? 1 : 0;
         ga_igop(GA_TYPE_GSM, &status, 1, "*");
      }else status = 1;

      if(status)
          status = !gai_getmem(array_name, GA[ga_handle].ptr,mem_size,
                               GA[ga_handle].type, &GA[ga_handle].id);
      else{
          GA[ga_handle].ptr[GAme]=NULL;
      }

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
logical FATR ga_duplicate_(g_a, g_b, array_name)
     Integer *g_a, *g_b;
     _fcd array_name;
#else
logical FATR ga_duplicate_(g_a, g_b, array_name, slen)
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
    GA[ga_handle].actv = 0;     
    if(GA[ga_handle].ptr[GAme]==NULL) return TRUE;
 
#ifndef AVOID_MA_STORAGE
    if(ARMCI_Uses_shm()){
#endif
      /* make sure that we free original (before address allignment) pointer */
      ARMCI_Free(GA[ga_handle].ptr[GAme] - GA[ga_handle].id);
#ifndef AVOID_MA_STORAGE
    }else{
      if(GA[ga_handle].id != INVALID_MA_HANDLE) MA_free_heap(GA[ga_handle].id);
    }
#endif

    if(GA_memory_limited) GA_total_memory += GA[ga_handle].size;
    GAstat.curmem -= GA[ga_handle].size;

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
    ga_sync_();

    GA_total_memory = -1; /* restore "unlimited" memory usage status */
    GA_memory_limited = 0;
    free(map);
    free(GA_proclist);

    ARMCI_Finalize();
    GAinitialized = 0;
    ga_sync_();
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
 


#define gaShmemLocation(proc, g_a, _i, _j, ptr_loc, _pld)                      \
{                                                                              \
Integer _ilo, _ihi, _jlo, _jhi, offset, proc_place, g_handle=(g_a)+GA_OFFSET;  \
Integer _lo[2], _hi[2];                                                        \
                                                                               \
      ga_ownsM(g_handle, (proc),_lo,_hi);                                      \
      _ilo = _lo[0]; _ihi=_hi[0];                                              \
      _jlo = _lo[1]; _jhi=_hi[1];                                              \
                                                                               \
      if((_i)<_ilo || (_i)>_ihi || (_j)<_jlo || (_j)>_jhi){                    \
       sprintf(err_string,"%s:p=%ld invalid i/j (%ld,%ld)><(%ld:%ld,%ld:%ld)", \
                 "gaShmemLocation", proc, (_i),(_j), _ilo, _ihi, _jlo, _jhi);  \
          ga_error(err_string, g_a );                                          \
      }                                                                        \
      offset = ((_i) - _ilo) + (_ihi-_ilo+1)*((_j)-_jlo);                      \
                                                                               \
      /* find location of the proc in current cluster pointer array */         \
      proc_place =  proc;                                             \
      *(ptr_loc) = GA[g_handle].ptr[proc_place] +                              \
                   offset*GAsizeofM(GA[g_handle].type);                        \
      *(_pld) = _ihi-_ilo+1;                                                   \
}


#define gaCheckSubscriptM(subscr, lo, hi, ndim)                             \
{                                                                              \
Integer _d;                                                                    \
   for(_d=0; _d<  ndim; _d++)                                                  \
      if( subscr[_d]<  lo[_d] ||  subscr[_d]>  hi[_d]){                  \
        sprintf(err_string,"check subscript failed:%ld not in (%ld:%ld) dim=", \
                  subscr[_d],  lo[_d],  hi[_d]);                            \
          ga_error(err_string, _d);                                            \
      }\
}


#define gam_Location(proc, g_handle,  subscript, ptr_loc, ld)                  \
{                                                                              \
Integer _offset=0, _d, _factor=1, _last=GA[g_handle].ndim-1;                   \
Integer _lo[MAXDIM], _hi[MAXDIM];                                              \
                                                                               \
      ga_ownsM(g_handle, proc, _lo, _hi);                                      \
      gaCheckSubscriptM(subscript, _lo, _hi, GA[g_handle].ndim);               \
      for(_d=0; _d < _last; _d++)            {                                 \
          _offset += (subscript[_d]-_lo[_d]) * _factor;                        \
          ld[_d] = _hi[_d] - _lo[_d]+1;                                        \
          _factor *= ld[_d];                                                   \
      }                                                                        \
      _offset += (subscript[_last]-_lo[_last]) * _factor;                      \
      *(ptr_loc) =  GA[g_handle].ptr[proc]+_offset*GA[g_handle].elemsize;      \
}



#define gam_Loc_ptr(proc, g_handle,  subscript, ptr_loc)                      \
{                                                                             \
Integer _offset=0, _d, _factor=1, _last=GA[g_handle].ndim-1;                  \
Integer _lo[MAXDIM], _hi[MAXDIM];                                             \
                                                                              \
      ga_ownsM(g_handle, proc, _lo, _hi);                                     \
      gaCheckSubscriptM(subscript, _lo, _hi, GA[g_handle].ndim);              \
      for(_d=0; _d < _last; _d++)            {                                \
          _offset += (subscript[_d]-_lo[_d]) * _factor;                       \
          _factor *= _hi[_d] - _lo[_d]+1;                                     \
      }                                                                       \
      _offset += (subscript[_last]-_lo[_last]) * _factor;                     \
      *(ptr_loc) =  GA[g_handle].ptr[proc]+_offset*GA[g_handle].elemsize;     \
}

#define ga_check_regionM(g_a, ilo, ihi, jlo, jhi, string){                     \
   if (*(ilo) <= 0 || *(ihi) > GA[GA_OFFSET + *(g_a)].dims[0] ||               \
       *(jlo) <= 0 || *(jhi) > GA[GA_OFFSET + *(g_a)].dims[1] ||               \
       *(ihi) < *(ilo) ||  *(jhi) < *(jlo)){                                   \
       sprintf(err_string,"%s:request(%ld:%ld,%ld:%ld) out of range (1:%ld,1:%ld)",\
               string, *(ilo), *(ihi), *(jlo), *(jhi),                         \
               GA[GA_OFFSET + *(g_a)].dims[0], GA[GA_OFFSET + *(g_a)].dims[1]);\
       ga_error(err_string, *(g_a));                                           \
   }                                                                           \
}



#define gam_GetRangeFromMap0(p, ndim, plo, phi, proc){\
Integer   _mloc = p* (ndim *2 +1);\
          *plo  = (Integer*)map + _mloc;\
          *phi  = *plo + ndim;\
          *proc = *phi[ndim]; /* proc is immediately after hi */\
}

#define gam_GetRangeFromMap(p, ndim, plo, phi){\
Integer   _mloc = p* ndim *2;\
          *plo  = (Integer*)map + _mloc;\
          *phi  = *plo + ndim;\
}

#define gam_CountElems(ndim, lo, hi, pelems){\
Integer _d;\
     for(_d=0,*pelems=1; _d< ndim;_d++)  *pelems *= hi[_d]-lo[_d]+1;\
}

#define gam_ComputeCount(ndim, lo, hi, count){\
Integer _d;\
          for(_d=0; _d< ndim;_d++) count[_d] = hi[_d]-lo[_d]+1;\
}


#define gam_ComputePatchIndex(ndim, lo, plo, dims, pidx){\
Integer _d, _factor;\
          *pidx = plo[0] -lo[0];\
          for(_d= 0,_factor=1; _d< ndim -1; _d++){\
             _factor *= dims[_d];\
             *pidx += _factor * (plo[_d+1]-lo[_d+1]);\
          }\
}

#define ga_RegionError(ndim, lo, hi, val){\
Integer _d, _l;\
   char *str= "cannot locate region: ";\
   sprintf(err_string, str); \
   _l = strlen(str);\
   for(_d=0; _d< ndim; _d++){ \
        sprintf(err_string+_l, "%ld:%ld ",lo[_d],hi[_d]);\
        _l=strlen(err_string);\
   }\
   ga_error(err_string, val);\
}

/*\fill array with value
\*/
void FATR ga_fill_(Integer *g_a, void* val)
{
int i,elems,handle=GA_OFFSET + *g_a;
char *ptr;

   GA_PUSH_NAME("ga_fill");
   ga_sync_();

   ga_check_handleM(g_a, "ga_fill");
   gam_checktype(GA[handle].type);
   elems = GA[handle].size/GA[handle].elemsize;
   ptr = GA[handle].ptr[GAme];

   switch (GA[handle].type){
   case MT_F_DCPL: 
        for(i=0; i<elems;i++)((DoubleComplex*)ptr)[i]=*(DoubleComplex*)val;
        break;
   case MT_F_DBL:  
        for(i=0; i<elems;i++)((DoublePrecision*)ptr)[i]=*(DoublePrecision*)val;
        break;
   case MT_F_INT:  
        for(i=0; i<elems;i++)((Integer*)ptr)[i]=*(Integer*)val;
        break;
   default:
        ga_error("type not supported",GA[handle].type);
   }
   ga_sync_();
   GA_POP_NAME;
}


#define gam_setstride(ndim, size, ld, ldrem, stride_rem, stride_loc){\
int _i;\
          stride_rem[0]= stride_loc[0] =size;\
          for(_i=0;_i<ndim;_i++){\
                stride_rem[_i] *= ldrem[_i];\
                stride_loc[_i] *= ld[_i];\
                if(_i<ndim-1){\
                     stride_rem[_i+1] = stride_rem[_i]; \
                     stride_loc[_i+1] = stride_loc[_i];\
                }\
          }\
}

/*\ PUT A 2-DIMENSIONAL PATCH OF DATA INTO A GLOBAL ARRAY
\*/
void FATR nga_put_(Integer *g_a, 
                   Integer *lo,
                   Integer *hi,
                   void    *buf,
                   Integer *ld)
{
Integer  p, np, proc, handle=GA_OFFSET + *g_a;
Integer  idx, elems, ndim, size;

      GA_PUSH_NAME("nga_put");

      if(!nga_locate_region_(g_a, lo, hi, map, GA_proclist, &np ))
          ga_RegionError(ga_ndim_(g_a), lo, hi, *g_a);

      size = GA[handle].elemsize;
      ndim = GA[handle].ndim;

      gam_CountElems(ndim, lo, hi, &elems);
      GAbytes.puttot += (double)size*elems;
      GAstat.numput++;

      gaPermuteProcList(np);
      for(idx=0; idx< np; idx++){
          Integer ldrem[MAXDIM];
          int stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
          Integer idx_buf, *plo, *phi;
          char *pbuf, *prem;

          p = (Integer)ProcListPerm[idx];
          gam_GetRangeFromMap(p, ndim, &plo, &phi);
          proc = GA_proclist[p];

          gam_Location(proc,handle, plo, &prem, ldrem); 

          /* find the right spot in the user buffer */
          gam_ComputePatchIndex(ndim,lo, plo, ld, &idx_buf);
          pbuf = size*idx_buf + (char*)buf;        

          gam_ComputeCount(ndim, plo, phi, count); 

          /* scale number of rows by element size */
          count[0] *= size; 
          gam_setstride(ndim, size, ld, ldrem, stride_rem, stride_loc);

          if(GA_fence_set)fence_array[proc]=1;
#ifdef PERMUTE_PIDS
    if(GA_Proc_list){
/*       fprintf(stderr,"permuted %d %d\n",proc,GA_inv_Proc_list[proc]);*/
       proc = GA_inv_Proc_list[proc];
    }
#endif

          if(proc == GAme){
             gam_CountElems(ndim, plo, phi, &elems);
             GAbytes.putloc += (double)size*elems;
          }
          ARMCI_PutS(pbuf, stride_loc, prem, stride_rem, count, ndim -1, proc);

      }

      GA_POP_NAME;
}



void FATR  ga_put_(g_a, ilo, ihi, jlo, jhi, buf, ld)
   Integer  *g_a,  *ilo, *ihi, *jlo, *jhi,  *ld;
   Void  *buf;
{
Integer lo[2], hi[2];

#ifdef GA_TRACE
   trace_stime_();
#endif

   lo[0]=*ilo;
   lo[1]=*jlo;
   hi[0]=*ihi;
   hi[1]=*jhi;
   nga_put_(g_a, lo, hi, buf, ld);

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_PUT; 
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
}




/*\ GET AN N-DIMENSIONAL PATCH OF DATA FROM A GLOBAL ARRAY
\*/
void FATR nga_get_(Integer *g_a,
                   Integer *lo,
                   Integer *hi,
                   void    *buf,
                   Integer *ld)
{
Integer  p, np, proc, handle=GA_OFFSET + *g_a;
Integer  idx, elems, ndim, size ;

      GA_PUSH_NAME("nga_get");

      if(!nga_locate_region_(g_a, lo, hi, map, GA_proclist, &np ))
          ga_RegionError(ga_ndim_(g_a), lo, hi, *g_a);

      size = GA[handle].elemsize;
      ndim = GA[handle].ndim;

      gam_CountElems(ndim, lo, hi, &elems);
      GAbytes.gettot += (double)size*elems;
      GAstat.numget++;

      gaPermuteProcList(np);
      for(idx=0; idx< np; idx++){
          Integer ldrem[MAXDIM];
          int stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
          Integer idx_buf, *plo, *phi;
          char *pbuf, *prem;

          p = (Integer)ProcListPerm[idx];
          gam_GetRangeFromMap(p, ndim, &plo, &phi);
          proc = GA_proclist[p];

          gam_Location(proc,handle, plo, &prem, ldrem);

          /* find the right spot in the user buffer */
          gam_ComputePatchIndex(ndim,lo, plo, ld, &idx_buf);
          pbuf = size*idx_buf + (char*)buf;

          gam_ComputeCount(ndim, plo, phi, count);

          /* scale number of rows by element size */
          count[0] *= size; 

          gam_setstride(ndim, size, ld, ldrem, stride_rem, stride_loc);

#ifdef PERMUTE_PIDS
          if(GA_Proc_list) proc = GA_inv_Proc_list[proc];
#endif
          if(proc == GAme){
             gam_CountElems(ndim, plo, phi, &elems);
             GAbytes.getloc += (double)size*elems;
          }
          ARMCI_GetS(prem, stride_rem, pbuf, stride_loc, count, ndim -1, proc);

      }

      GA_POP_NAME;
}


void FATR  ga_get_(g_a, ilo, ihi, jlo, jhi, buf, ld)
   Integer  *g_a,  *ilo, *ihi, *jlo, *jhi,  *ld;
   Void  *buf;
{
Integer lo[2], hi[2];

#ifdef GA_TRACE
   trace_stime_();
#endif

   lo[0]=*ilo;
   lo[1]=*jlo;
   hi[0]=*ihi;
   hi[1]=*jhi;
   nga_get_(g_a, lo, hi, buf, ld);

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_GET;
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
}



/*\ ACCUMULATE OPERATION FOR A N-DIMENSIONAL PATCH OF GLOBAL ARRAY
 *
 *  g_a += alpha * patch
\*/
void FATR nga_acc_(Integer *g_a,
                   Integer *lo,
                   Integer *hi,
                   void    *buf,
                   Integer *ld,
                   void    *alpha)
{
Integer  p, np, proc, handle=GA_OFFSET + *g_a;
Integer  idx, elems, ndim, size, type;
int optype;

      GA_PUSH_NAME("nga_acc");

      if(!nga_locate_region_(g_a, lo, hi, map, GA_proclist, &np ))
          ga_RegionError(ga_ndim_(g_a), lo, hi, *g_a);

      size = GA[handle].elemsize;
      type = GA[handle].type;
      ndim = GA[handle].ndim;

      if(type==MT_F_DBL) optype= ARMCI_ACC_DBL;
      else if(type==MT_F_DCPL)optype= ARMCI_ACC_DCP;
      else if(size==sizeof(int))optype= ARMCI_ACC_INT;
      else if(size==sizeof(long))optype= ARMCI_ACC_LNG;
      else ga_error("type not supported",type);

      gam_CountElems(ndim, lo, hi, &elems);
      GAbytes.acctot += (double)size*elems;
      GAstat.numacc++;

      gaPermuteProcList(np);
      for(idx=0; idx< np; idx++){
          Integer ldrem[MAXDIM];
          int stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
          Integer idx_buf, *plo, *phi;
          char *pbuf, *prem;

          p = (Integer)ProcListPerm[idx];
          gam_GetRangeFromMap(p, ndim, &plo, &phi);
          proc = GA_proclist[p];

          gam_Location(proc,handle, plo, &prem, ldrem);

          /* find the right spot in the user buffer */
          gam_ComputePatchIndex(ndim,lo, plo, ld, &idx_buf);
          pbuf = size*idx_buf + (char*)buf;

          gam_ComputeCount(ndim, plo, phi, count);

          /* scale number of rows by element size */
          count[0] *= size;
          gam_setstride(ndim, size, ld, ldrem, stride_rem, stride_loc);

          if(GA_fence_set)fence_array[proc]=1;

#ifdef PERMUTE_PIDS
          if(GA_Proc_list) proc = GA_inv_Proc_list[proc];
#endif
          if(proc == GAme){
             gam_CountElems(ndim, plo, phi, &elems);
             GAbytes.accloc += (double)size*elems;
          }

          ARMCI_AccS(optype, alpha, pbuf, stride_loc, prem, stride_rem, count, ndim-1, proc);

      }

      GA_POP_NAME;
}



void FATR  ga_acc_(g_a, ilo, ihi, jlo, jhi, buf, ld, alpha)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *ld;
   void *buf, *alpha;
{
Integer lo[2], hi[2];
#ifdef GA_TRACE
   trace_stime_();
#endif

   lo[0]=*ilo;
   lo[1]=*jlo;
   hi[0]=*ihi;
   hi[1]=*jhi;
   nga_acc_(g_a, lo, hi, buf, ld, alpha);

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_ACC;
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
}


void nga_access_ptr(Integer* g_a, Integer lo[], Integer hi[],
                      void* ptr, Integer ld[])

{
char *lptr;
Integer  handle = GA_OFFSET + *g_a;
Integer  ow,i;

   GA_PUSH_NAME("nga_access_ptr");
   if(!nga_locate_(g_a,lo,&ow))ga_error("locate top failed",0);
   if(ow != GAme) ga_error("cannot access top of the patch",ow);
   if(!nga_locate_(g_a,hi, &ow))ga_error("locate bottom failed",0);
   if(ow != GAme) ga_error("cannot access bottom of the patch",ow);

   for (i=0; i<GA[handle].ndim; i++)
       if(lo[i]>hi[i]) {
           ga_RegionError(GA[handle].ndim, lo, hi, *g_a);
       }

   gam_Location(ow,handle, lo, &lptr, ld);
   *(char**)ptr = lptr; 
   GA_POP_NAME;
}


/*\ PROVIDE ACCESS TO A PATCH OF A GLOBAL ARRAY
\*/
void FATR nga_access_(Integer* g_a, Integer lo[], Integer hi[],
                      Integer* index, Integer ld[])
{
char     *ptr;
Integer  handle = GA_OFFSET + *g_a;
Integer  ow,i;
unsigned long    elemsize;
unsigned long    lref, lptr;

   GA_PUSH_NAME("nga_access");
   if(!nga_locate_(g_a,lo,&ow))ga_error("locate top failed",0);
   if(ow != GAme) ga_error("cannot access top of the patch",ow);
   if(!nga_locate_(g_a,hi, &ow))ga_error("locate bottom failed",0);
   if(ow != GAme) ga_error("cannot access bottom of the patch",ow);

   for (i=0; i<GA[handle].ndim; i++)
       if(lo[i]>hi[i]) {
           ga_RegionError(GA[handle].ndim, lo, hi, *g_a);
       }


   gam_Location(ow,handle, lo, &ptr, ld);

   /*
    * return patch address as the distance elements from the reference address
    *
    * .in Fortran we need only the index to the type array: dbl_mb or int_mb
    *  that are elements of COMMON in the the mafdecls.h include file
    * .in C we need both the index and the pointer
    */

   elemsize = (unsigned long)GA[handle].elemsize;

   /* compute index and check if it is correct */
   switch (GA[handle].type){
     case MT_F_DBL:
        *index = (Integer) ((DoublePrecision*)ptr - DBL_MB);
        lref = (unsigned long)DBL_MB;
        break;

     case MT_F_DCPL:
        *index = (Integer) ((DoubleComplex*)ptr - DCPL_MB);
        lref = (unsigned long)DCPL_MB;
        break;

     case MT_F_INT:
        *index = (Integer) ((Integer*)ptr - INT_MB);
        lref = (unsigned long)INT_MB;
        break;
   }

#ifdef BYTE_ADDRESSABLE_MEMORY
   /* check the allignment */
   lptr = (unsigned long)ptr;
   if( lptr%elemsize != lref%elemsize ){ 
       printf("%d: lptr=%lu(%lu) lref=%lu(%lu)\n",(int)GAme,lptr,lptr%elemsize,
                                                    lref,lref%elemsize);
       ga_error("nga_access: MA addressing problem: base address misallignment",
                 handle);
   }
#endif

   /* adjust index for Fortran addressing */
   (*index) ++ ;
   FLUSH_CACHE;

   GA_POP_NAME;
}



/*\ PROVIDE ACCESS TO A PATCH OF A GLOBAL ARRAY
\*/
void FATR ga_access_(g_a, ilo, ihi, jlo, jhi, index, ld)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *index, *ld;
{
Integer lo[2], hi[2],ndim=ga_ndim_(g_a);

     if(ndim != 2) 
        ga_error("ga_access: 2D API cannot be used for array dimension",ndim);

     lo[0]=*ilo;
     lo[1]=*jlo;
     hi[0]=*ihi;
     hi[1]=*jhi;
     nga_access_(g_a,lo,hi,index,ld);
} 



/*\ RELEASE ACCESS TO A PATCH OF A GLOBAL ARRAY
\*/
void FATR  ga_release_(Integer *g_a, 
                       Integer *ilo, Integer *ihi, Integer *jlo, Integer *jhi)
{}


/*\ RELEASE ACCESS TO A PATCH OF A GLOBAL ARRAY
\*/
void FATR  nga_release_(Integer *g_a, Integer *lo, Integer *hi)
{}


/*\ RELEASE ACCESS & UPDATE A PATCH OF A GLOBAL ARRAY
\*/
void FATR  ga_release_update_(g_a, ilo, ihi, jlo, jhi)
     Integer *g_a, *ilo, *ihi, *jlo, *jhi;
{}


/*\ RELEASE ACCESS & UPDATE A PATCH OF A GLOBAL ARRAY
\*/
void FATR  nga_release_update_(Integer *g_a, Integer *lo, Integer *hi)
{}


/*\ INQUIRE POPERTIES OF A GLOBAL ARRAY
 *  Fortran version
\*/ 
void FATR  ga_inquire_(g_a,  type, dim1, dim2)
      Integer *g_a, *dim1, *dim2, *type;
{
Integer ndim = ga_ndim_(g_a);

   if(ndim != 2)
      ga_error("ga_inquire: 2D API cannot be used for array dimension",ndim);

   *type       = GA[GA_OFFSET + *g_a].type;
   *dim1       = GA[GA_OFFSET + *g_a].dims[0];
   *dim2       = GA[GA_OFFSET + *g_a].dims[1];
}



/*\ INQUIRE POPERTIES OF A GLOBAL ARRAY
 *  Fortran version
\*/
void FATR nga_inquire_(Integer *g_a, Integer *type, Integer *ndim,Integer *dims)
{
Integer handle = GA_OFFSET + *g_a,i;
   ga_check_handleM(g_a, "nga_inquire");
   *type       = GA[handle].type;
   *ndim       = GA[handle].ndim;
   for(i=0;i<*ndim;i++)dims[i]=GA[handle].dims[i];
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
void ga_inquire_name(g_a, array_name)
      Integer *g_a;
      char    **array_name;
{ 
   ga_check_handleM(g_a, "ga_inquire_name");
   *array_name = GA[GA_OFFSET + *g_a].name;
}





/*\ RETURN COORDINATES OF ARRAY BLOCK HELD BY A PROCESSOR
\*/
void FATR nga_proc_topology_(Integer* g_a, Integer* proc, Integer* subscript)
{
Integer d, index, ndim, ga_handle = GA_OFFSET + *g_a;

   ga_check_handleM(g_a, "nga_proc_topology");
   ndim = GA[ga_handle].ndim;

   index = GA_proc_list ? GA_proc_list[*proc]: *proc;

   for(d=0; d<ndim; d++){
       subscript[d] = index% GA[ga_handle].nblock[d];
       index  /= GA[ga_handle].nblock[d];  
   }
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



/*\ LOCATE THE OWNER OF SPECIFIED ELEMENT OF A GLOBAL ARRAY
\*/
logical FATR nga_locate_(Integer *g_a, Integer* subscript, Integer* owner)
{
Integer d, proc, dpos, ndim, ga_handle = GA_OFFSET + *g_a, proc_s[MAXDIM];

   ga_check_handleM(g_a, "nga_locate");
   ndim = GA[ga_handle].ndim;

   for(d=0, *owner=-1; d< ndim; d++) 
       if(subscript[d]< 1 || subscript[d]>GA[ga_handle].dims[d])return FALSE;

   for(d = 0, dpos = 0; d< ndim; d++){
       findblock(GA[ga_handle].mapc + dpos, GA[ga_handle].nblock[d],
                 GA[ga_handle].scale[d], (int)subscript[d], &proc_s[d]);
       dpos += GA[ga_handle].nblock[d];
   }

   ga_ComputeIndexM(&proc, ndim, proc_s, GA[ga_handle].nblock); 

   *owner = GA_proc_list ? GA_proc_list[proc]: proc;

   return TRUE;
}



/*\ LOCATE PROCESSORS/OWNERS OF THE SPECIFIED PATCH OF A GLOBAL ARRAY
\*/
logical FATR nga_locate_region_( Integer *g_a,
                                 Integer *lo,
                                 Integer *hi,
                                 Integer *map,
                                 Integer *proclist,
                                 Integer *np)
{
int  procT[MAXDIM], procB[MAXDIM], proc_subscript[MAXDIM];
Integer  proc, owner, i, ga_handle;
Integer  d, dpos, ndim, elems;

   ga_check_handleM(g_a, "nga_locate_region");

   ga_handle = GA_OFFSET + *g_a;

   for(d = 0; d< GA[ga_handle].ndim; d++)
       if((lo[d]<1 || hi[d]>GA[ga_handle].dims[d]) ||(lo[d]>hi[d]))return FALSE;

   ndim = GA[ga_handle].ndim;

   /* find "processor coordinates" for the top corner */
   for(d = 0, dpos = 0; d< GA[ga_handle].ndim; d++){
       findblock(GA[ga_handle].mapc + dpos, GA[ga_handle].nblock[d], 
                 GA[ga_handle].scale[d], (int)lo[d], &procT[d]);
       dpos += GA[ga_handle].nblock[d];
   }

   /* find "processor coordinates" for the right bottom corner */
   for(d = 0, dpos = 0; d< GA[ga_handle].ndim; d++){
       findblock(GA[ga_handle].mapc + dpos, GA[ga_handle].nblock[d], 
                 GA[ga_handle].scale[d], (int)hi[d], &procB[d]);
       dpos += GA[ga_handle].nblock[d];
   }

   *np = 0;

   ga_InitLoopM(&elems, ndim, proc_subscript, procT,procB,GA[ga_handle].nblock);

   for(i= 0; i< elems; i++){ 
      Integer _lo[MAXDIM], _hi[MAXDIM];
      Integer  offset;

      /* convert i to owner processor id */
      ga_ComputeIndexM(&proc, ndim, proc_subscript, GA[ga_handle].nblock); 
      owner = GA_proc_list ? GA_proc_list[proc]: proc;
      ga_ownsM(ga_handle, owner, _lo, _hi);

      offset = *np *(ndim*2); /* location in mapc to put patch range */

      for(d = 0; d< ndim; d++)
              map[d + offset ] = lo[d] < _lo[d] ? _lo[d] : lo[d];
      for(d = 0; d< ndim; d++)
              map[ndim + d + offset ] = hi[d] > _hi[d] ? _hi[d] : hi[d];

      proclist[i] = owner;
      ga_UpdateSubscriptM(ndim,proc_subscript,procT,procB,GA[ga_handle].nblock);
      (*np)++;
   }
   return(TRUE);
}
    

/*\ returns in nblock array the number of blocks each dimension is divided to
\*/
void GA_Nblock(int g_a, int *nblock)
{
int ga_handle = GA_OFFSET + g_a;
int i, n;

     ga_check_handleM(&g_a, "GA_Nblock");

     n = GA[ga_handle].ndim;

#ifdef USE_FAPI 
     for(i=0; i<n; i++) nblock[i] = GA[ga_handle].nblock[i]);
#else
     for(i=0; i<n; i++) nblock[n-i-1] = GA[ga_handle].nblock[i];
#endif
     
}
     

void FATR ga_nblock_(Integer *g_a, Integer *nblock)
{
Integer ga_handle = GA_OFFSET + *g_a;
int i, n;

     ga_check_handleM(g_a, "ga_nblock");

     n = GA[ga_handle].ndim;

     for(i=0; i<n; i++) nblock[i] = (Integer)GA[ga_handle].nblock[i];
}


void ga_scatter_acc_local(Integer g_a, Void *v,Integer *i,Integer *j,
                          Integer nv, void* alpha, Integer proc) 
{
void **ptr_src, **ptr_dst;
char *ptr_ref;
Integer ldp, item_size, ilo, ihi, jlo, jhi, type;
armci_giov_t desc;
register Integer k, offset;
int rc;

  if (nv < 1) return;

  GA_PUSH_NAME("ga_scatter_local");

  ga_distribution_(&g_a, &proc, &ilo, &ihi, &jlo, &jhi);

  /* get address of the first element owned by proc */
  gaShmemLocation(proc, g_a, ilo, jlo, &ptr_ref, &ldp);

  type = GA[GA_OFFSET + g_a].type;
  item_size = GAsizeofM(type);

  ptr_src = gai_malloc(2*nv*sizeof(void*));
  if(ptr_src==NULL)ga_error("gai_malloc failed",nv);
  else ptr_dst=ptr_src+ nv;

  for(k=0; k< nv; k++){
     if(i[k] < ilo || i[k] > ihi  || j[k] < jlo || j[k] > jhi){
       sprintf(err_string,"proc=%d invalid i/j=(%ld,%ld)>< [%ld:%ld,%ld:%ld]",
               (int)proc, i[k], j[k], ilo, ihi, jlo, jhi); 
       ga_error(err_string,g_a);
     }

     offset  = (j[k] - jlo)* ldp + i[k] - ilo;
     ptr_dst[k] = ptr_ref + item_size * offset;
     ptr_src[k] = ((char*)v) + k*item_size;
  }
  desc.bytes = item_size;
  desc.src_ptr_array = ptr_src;
  desc.dst_ptr_array = ptr_dst;
  desc.ptr_array_len = nv;

  if(GA_fence_set)fence_array[proc]=1;

#ifdef PERMUTE_PIDS
    if(GA_Proc_list) proc = GA_inv_Proc_list[proc];
#endif

  if(alpha != NULL) {
    int optype;
    if(type==MT_F_DBL) optype= ARMCI_ACC_DBL;
    else if(type==MT_F_DCPL)optype= ARMCI_ACC_DCP;
    else if(item_size==sizeof(int))optype= ARMCI_ACC_INT;
    else if(item_size==sizeof(long))optype= ARMCI_ACC_LNG;
    else ga_error("type not supported",type);
    rc= ARMCI_AccV(optype, alpha, &desc, 1, (int)proc);
  }

  if(rc) ga_error("scatter/_acc failed in armci",rc);

  gai_free(ptr_src);

  GA_POP_NAME;
}


/*\ based on subscripts compute pointers
\*/
void gai_sort_proc(Integer* g_a, Integer* sbar, Integer *nv, Integer list[], Integer proc[])
{
Integer k, ndim;
extern void ga_sort_permutation();

   if (*nv < 1) return;

   ga_check_handleM(g_a, "gai_get_pointers");
   ndim = GA[*g_a+GA_OFFSET].ndim;

   for(k=0; k< *nv; k++)if(!nga_locate_(g_a, sbar+k*ndim, proc+k)){
         print_subscript("invalid subscript",ndim, sbar +k*ndim,"\n");
         ga_error("failed -element:",k);
   }
         
   /* Sort the entries by processor */
   ga_sort_permutation(nv, list, proc);
}
 

/*\ permutes input index list using sort routine used in scatter/gather
\*/
void FATR nga_sort_permut_(Integer* g_a, Integer index[], 
                           Integer* subscr_arr, Integer *nv)
{
    /* The new implementation doesn't change the order of the elements
     * They are identical
     */
    /*
Integer pindex, phandle;

  if (*nv < 1) return;

  if(!MA_push_get(MT_F_INT,*nv, "nga_sort_permut--p", &phandle, &pindex))
              ga_error("MA alloc failed ", *g_a);

  gai_sort_proc(g_a, subscr_arr, nv, index, INT_MB+pindex);
  if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);
    */
}


/*\ SCATTER OPERATION elements of v into the global array
\*/
void FATR  ga_scatter_(Integer *g_a, Void *v, Integer *i, Integer *j,
                       Integer *nv)
{
    register Integer k;
    Integer kk;
    Integer pindex, phandle, item_size;
    Integer proc, type=GA[GA_OFFSET + *g_a].type;

    Integer *aproc, naproc; /* active processes and numbers */
    Integer *map;           /* map the active processes to allocated space */
    char *buf1, *buf2;
    
    Integer *count;   /* counters for each process */
    Integer *nelem;   /* number of elements for each process */
    /* source and destination pointers for each process */
    void ***ptr_src, ***ptr_dst; 
    void **ptr_org; /* the entire pointer array */
    armci_giov_t desc;
    Integer *ilo, *ihi, *jlo, *jhi, *ldp;
    char **ptr_ref;
    
    if (*nv < 1) return;
    
    ga_check_handleM(g_a, "ga_scatter");
    GA_PUSH_NAME("ga_scatter");
    GAstat.numsca++;
    
    if(!MA_push_get(MT_F_INT,*nv, "ga_scatter--p", &phandle, &pindex))
        ga_error("MA alloc failed ", *g_a);

    /* allocate temp memory */
    buf1 = gai_malloc(4 * GAnproc * (sizeof(Integer)));
    if(buf1 == NULL) ga_error("gai_malloc failed", 3*GAnproc);
    
    count = (Integer *)buf1;
    nelem = (Integer *)(buf1 + GAnproc * sizeof(Integer));
    aproc = (Integer *)(buf1 + 2 * GAnproc * sizeof(Integer));
    map = (Integer *)(buf1 + 3 * GAnproc * sizeof(Integer));
    
    /* initialize the counters and nelem */
    for(kk=0; kk<GAnproc; kk++) {
        count[kk] = 0; nelem[kk] = 0;
    }
    
    /* find proc that owns the (i,j) element; store it in temp: INT_MB[] */
    for(k=0; k< *nv; k++) {
        if(! ga_locate_(g_a, i+k, j+k, INT_MB+pindex+k)){
            sprintf(err_string,"invalid i/j=(%ld,%ld)", i[k], j[k]);
            ga_error(err_string,*g_a);
        }
        nelem[INT_MB[pindex+k]]++;
    }

    naproc = 0;
    for(k=0; k<GAnproc; k++) if(nelem[k] > 0) {
        aproc[naproc] = k;
        map[k] = naproc;
        naproc ++;
    }
    
    buf2 = gai_malloc(2*naproc*sizeof(void **) + 2*(*nv)*sizeof(void *) +
                      5*naproc*sizeof(Integer) + naproc*sizeof(char*));
    if(buf2 == NULL) ga_error("gai_malloc failed", naproc);
 
    ptr_src = (void ***)buf2;
    ptr_dst = (void ***)(buf2 + naproc*sizeof(void **));
    ptr_org = (void **)(buf2 + 2*naproc*sizeof(void **));
    ilo = (Integer *)(buf2+2*naproc*sizeof(void **)+2*(*nv)*sizeof(void *));
    ihi = ilo + naproc;
    jlo = ihi + naproc;
    jhi = jlo + naproc;
    ldp = jhi + naproc;
    ptr_ref = (char **)(ldp + naproc);

    for(kk=0; kk<naproc; kk++) {
        ga_distribution_(g_a, &aproc[kk],
                         &(ilo[kk]), &(ihi[kk]), &(jlo[kk]), &(jhi[kk]));
        
        /* get address of the first element owned by proc */
        gaShmemLocation(aproc[kk], *g_a, ilo[kk], jlo[kk], &(ptr_ref[kk]),
                        &(ldp[kk]));
    }
    
    /* determine limit for message size --  v,i, & j will travel together */
    item_size = GAsizeofM(type);
    GAbytes.scatot += (double)item_size**nv ;
    GAbytes.scaloc += (double)item_size* nelem[INT_MB[pindex+GAme]];

    ptr_src[0] = ptr_org; ptr_dst[0] = ptr_org + (*nv);
    for(k=1; k<naproc; k++) {
        ptr_src[k] = ptr_src[k-1] + nelem[aproc[k-1]];
        ptr_dst[k] = ptr_dst[k-1] + nelem[aproc[k-1]];
    }
    
    for(k=0; k<(*nv); k++){
        int this_count;
        proc = INT_MB[pindex+k]; this_count = count[proc]; count[proc]++;
        proc = map[proc];
        ptr_src[proc][this_count] = ((char*)v) + k * item_size;
        if(i[k] < ilo[proc] || i[k] > ihi[proc]  ||
           j[k] < jlo[proc] || j[k] > jhi[proc]){
          sprintf(err_string,"proc=%d invalid i/j=(%ld,%ld)><[%ld:%ld,%ld:%ld]",
             (int)proc, i[k], j[k], ilo[proc], ihi[proc], jlo[proc], jhi[proc]);
            ga_error(err_string, *g_a);
        }
        ptr_dst[proc][this_count] = ptr_ref[proc] + item_size *
            ((j[k] - jlo[proc])* ldp[proc] + i[k] - ilo[proc]);
    }
    
    /* source and destination pointers are ready for all processes */
    for(k=0; k<naproc; k++) {
        int rc;

        desc.bytes = item_size;
        desc.src_ptr_array = ptr_src[k];
        desc.dst_ptr_array = ptr_dst[k];
        desc.ptr_array_len = nelem[aproc[k]];
        
        rc = ARMCI_PutV(&desc, 1, aproc[k]);
        if(rc) ga_error("scatter failed in armci",rc);
    }

    gai_free(buf2);
    gai_free(buf1);

    if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);
    GA_POP_NAME;
}
      


/*\ SCATTER OPERATION elements of v into the global array
\*/
void FATR  ga_scatter_acc_(g_a, v, i, j, nv, alpha)
     Integer *g_a, *nv, *i, *j;
     Void *v, *alpha;
{
register Integer k;
Integer pindex, phandle, item_size;
Integer first, nelem, proc, type=GA[GA_OFFSET + *g_a].type;

  if (*nv < 1) return;

  ga_check_handleM(g_a, "ga_scatter_acc");
  GA_PUSH_NAME("ga_scatter_acc");
  GAstat.numsca++;

  if(!MA_push_get(MT_F_INT,*nv, "ga_scatter_acc--p", &phandle, &pindex))
            ga_error("MA alloc failed ", *g_a);

  /* find proc that owns the (i,j) element; store it in temp: INT_MB[] */
  for(k=0; k< *nv; k++) if(! ga_locate_(g_a, i+k, j+k, INT_MB+pindex+k)){
         sprintf(err_string,"invalid i/j=(%ld,%ld)", i[k], j[k]);
         ga_error(err_string,*g_a);
  }

  /* determine limit for message size --  v,i, & j will travel together */
  item_size = GAsizeofM(type);
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
             GAbytes.scaloc += (double)item_size* nelem ;
      }

      ga_scatter_acc_local(*g_a, ((char*)v)+item_size*first, i+first,
                           j+first, nelem, alpha, proc);
      first += nelem;

  }while (first< *nv);

  if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);

  GA_POP_NAME;
}



/*\ permutes input index list using sort routine used in scatter/gather
\*/
void FATR  ga_sort_permut_(g_a, index, i, j, nv)
     Integer *g_a, *nv, *i, *j, *index;
{
    /* The new implementation doesn't change the order of the elements
     * They are identical
     */

#if 0
register Integer k;
Integer pindex, phandle;
extern void ga_sort_permutation();

  if (*nv < 1) return;

  if(!MA_push_get(MT_F_INT,*nv, "ga_sort_permut--p", &phandle, &pindex))
            ga_error("MA alloc failed ", *g_a);

  /* find proc that owns the (i,j) element; store it in temp: INT_MB[] */
  for(k=0; k< *nv; k++) if(! ga_locate_(g_a, i+k, j+k, INT_MB+pindex+k)){
         sprintf(err_string,"invalid i/j=(%ld,%ld)", i[k], j[k]);
         ga_error(err_string,*g_a);
  }

  /* Sort the entries by processor */
  ga_sort_permutation(nv, index, INT_MB+pindex);
  if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);
#endif
}




#define SCATTER -99
#define GATHER -98
#define SCATTER_ACC -97

/*\ GATHER OPERATION elements from the global array into v
\*/
void gai_gatscat(int op, Integer* g_a, void* v, Integer subscript[], 
                 Integer* nv, double *locbytes, double* totbytes, void *alpha)
{
    register Integer k, handle=*g_a+GA_OFFSET;
    Integer  ndim, item_size, type;
    Integer *proc, phandle;

    Integer *aproc, naproc; /* active processes and numbers */
    Integer *map;           /* map the active processes to allocated space */
    char *buf1, *buf2;
    
    Integer *count;        /* counters for each process */
    Integer *nelem;        /* number of elements for each process */
    /* source and destination pointers for each process */
    void ***ptr_src, ***ptr_dst; 
    void **ptr_org; /* the entire pointer array */
    armci_giov_t desc;
    
    GA_PUSH_NAME("gai_gatscat");

    if(!MA_push_stack(MT_F_INT,*nv,"ga_gat-p",&phandle))
        ga_error("MAfailed",*g_a);
    if(!MA_get_pointer(phandle, &proc))
        ga_error("MA pointer failed ", *g_a);

    ndim = GA[handle].ndim;

    type = GA[handle].type;
    item_size = GA[handle].elemsize;
    *totbytes += (double)item_size**nv;

    /* allocate temp memory */
    buf1 = gai_malloc(4 * GAnproc * (sizeof(Integer)));
    if(buf1 == NULL) ga_error("gai_malloc failed", 3*GAnproc);
    
    count = (Integer *)buf1;
    nelem = (Integer *)(buf1 + GAnproc * sizeof(Integer));
    aproc = (Integer *)(buf1 + 2 * GAnproc * sizeof(Integer));
    map = (Integer *)(buf1 + 3 * GAnproc * sizeof(Integer));
    
    /* initialize the counters and nelem */
    for(k=0; k<GAnproc; k++) {count[k] = 0; nelem[k] = 0;}

    /* get the process id that the element should go and count the
     * number of elements for each process
     */
    for(k=0; k<*nv; k++) {
        if(!nga_locate_(g_a, subscript+k*ndim, proc+k)) {
            print_subscript("invalid subscript",ndim, subscript+k*ndim,"\n");
            ga_error("failed -element:",k);
        }
        nelem[proc[k]]++;
    }

    /* find the active processes (with which transfer data) */
    naproc = 0;
    for(k=0; k<GAnproc; k++) if(nelem[k] > 0) {
        aproc[naproc] = k;
        map[k] = naproc;
        naproc ++;
    }

    buf2 = gai_malloc(2*naproc*sizeof(void **) + 2*(*nv)*sizeof(void *));
    if(buf2 == NULL) ga_error("gai_malloc failed", 2*naproc);
    
    ptr_src = (void ***)buf2;
    ptr_dst = (void ***)(buf2 + naproc * sizeof(void **));
    ptr_org = (void **)(buf2 + 2 * naproc * sizeof(void **));
    
    /* set the pointers as
     *    P0            P1                  P0          P1        
     * ptr_src[0]   ptr_src[1] ...       ptr_dst[0]  ptr_dst[1] ...
     *        \          \                    \          \
     * ptr_org |-------------------------------|---------------------------|
     *         |              (*nv)            |            (*nv)          |
     *         | nelem[0] | nelem[1] |...      | nelem[0] | nelem[1] |...
     */
    ptr_src[0] = ptr_org; ptr_dst[0] = ptr_org + (*nv);
    for(k=1; k<naproc; k++) {
        ptr_src[k] = ptr_src[k-1] + nelem[aproc[k-1]];
        ptr_dst[k] = ptr_dst[k-1] + nelem[aproc[k-1]];
    }

    *locbytes += (double)item_size* nelem[GAme];
    
/*
#ifdef PERMUTE_PIDS
    if(GA_Proc_list) p = GA_inv_Proc_list[p];
#endif
*/    
    switch(op) { 
      case GATHER:
        /* go through all the elements
         * for process 0: ptr_src[0][0, 1, ...] = subscript + offset0...
         *                ptr_dst[0][0, 1, ...] = v + offset0...
         * for process 1: ptr_src[1][...] ...
         *                ptr_dst[1][...] ...
         */  
        for(k=0; k<(*nv); k++){
            ptr_dst[map[proc[k]]][count[proc[k]]] = ((char*)v) + k * item_size;
            gam_Loc_ptr(proc[k], handle,  (subscript+k*ndim),
                        ptr_src[map[proc[k]]]+count[proc[k]]);
            count[proc[k]]++;
        }
        
        /* source and destination pointers are ready for all processes */
        for(k=0; k<naproc; k++) {
            int rc;

            desc.bytes = item_size;
            desc.src_ptr_array = ptr_src[k];
            desc.dst_ptr_array = ptr_dst[k];
            desc.ptr_array_len = nelem[aproc[k]];
            
            rc=ARMCI_GetV(&desc, 1, aproc[k]);
            if(rc) ga_error("gather failed in armci",rc);
        }
        break;
      case SCATTER:
        /* go through all the elements
         * for process 0: ptr_src[0][0, 1, ...] = v + offset0...
         *                ptr_dst[0][0, 1, ...] = subscript + offset0...
         * for process 1: ptr_src[1][...] ...
         *                ptr_dst[1][...] ...
         */
        for(k=0; k<(*nv); k++){
            ptr_src[map[proc[k]]][count[proc[k]]] = ((char*)v) + k * item_size;
            gam_Loc_ptr(proc[k], handle,  (subscript+k*ndim),
                        ptr_dst[map[proc[k]]]+count[proc[k]]);
            count[proc[k]]++;
        }

        /* source and destination pointers are ready for all processes */
        for(k=0; k<naproc; k++) {
            int rc;

            desc.bytes = item_size;
            desc.src_ptr_array = ptr_src[k];
            desc.dst_ptr_array = ptr_dst[k];
            desc.ptr_array_len = nelem[aproc[k]];
            
            if(GA_fence_set) fence_array[aproc[k]]=1;
            
            rc=ARMCI_PutV(&desc, 1, aproc[k]);
            if(rc) ga_error("scatter failed in armci",rc);
        }
        break;
      case SCATTER_ACC:
        /* go through all the elements
         * for process 0: ptr_src[0][0, 1, ...] = v + offset0...
         *                ptr_dst[0][0, 1, ...] = subscript + offset0...
         * for process 1: ptr_src[1][...] ...
         *                ptr_dst[1][...] ...
         */
        for(k=0; k<(*nv); k++){
            ptr_src[map[proc[k]]][count[proc[k]]] = ((char*)v) + k * item_size;
            gam_Loc_ptr(proc[k], handle,  (subscript+k*ndim),
                        ptr_dst[map[proc[k]]]+count[proc[k]]);
            count[proc[k]]++;
        }

        /* source and destination pointers are ready for all processes */
        for(k=0; k<naproc; k++) {
            int rc;
            
            desc.bytes = item_size;
            desc.src_ptr_array = ptr_src[k];
            desc.dst_ptr_array = ptr_dst[k];
            desc.ptr_array_len = nelem[aproc[k]];
            
            if(GA_fence_set) fence_array[aproc[k]]=1;
            
            if(alpha != NULL) {
                int optype;
                if(type==MT_F_DBL) optype= ARMCI_ACC_DBL;
                else if(type==MT_F_DCPL)optype= ARMCI_ACC_DCP;
                else if(item_size==sizeof(int))optype= ARMCI_ACC_INT;
                else if(item_size==sizeof(long))optype= ARMCI_ACC_LNG;
                else ga_error("type not supported",type);
                rc= ARMCI_AccV(optype, alpha, &desc, 1, aproc[k]);
            }
            if(rc) ga_error("scatter_acc failed in armci",rc);
        }
        break;        
      default: ga_error("operation not supported",op);
    }

    gai_free(buf2); gai_free(buf1);
    
    if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);
    GA_POP_NAME;
}



/*\ GATHER OPERATION elements from the global array into v
\*/
void FATR nga_gather_(Integer *g_a, void* v, Integer subscript[], Integer *nv)
{

  if (*nv < 1) return;
  ga_check_handleM(g_a, "nga_gather");
  GA_PUSH_NAME("nga_gather");
  GAstat.numgat++;

  gai_gatscat(GATHER,g_a,v,subscript,nv,&GAbytes.gattot,&GAbytes.gatloc, NULL);

  GA_POP_NAME;
}


void FATR nga_scatter_(Integer *g_a, void* v, Integer subscript[], Integer *nv)
{

  if (*nv < 1) return;
  ga_check_handleM(g_a, "nga_scatter");
  GA_PUSH_NAME("nga_scatter");
  GAstat.numsca++;

  gai_gatscat(SCATTER,g_a,v,subscript,nv,&GAbytes.scatot,&GAbytes.scaloc, NULL);

  GA_POP_NAME;
}

void FATR nga_scatter_acc_(Integer *g_a, void* v, Integer subscript[],
                           Integer *nv, void *alpha)
{

  if (*nv < 1) return;
  ga_check_handleM(g_a, "nga_scatter_acc");
  GA_PUSH_NAME("nga_scatter_acc");
  GAstat.numsca++;

  gai_gatscat(SCATTER_ACC, g_a, v, subscript, nv, &GAbytes.scatot,
              &GAbytes.scaloc, alpha);

  GA_POP_NAME;
}

void FATR  ga_gather000_(g_a, v, i, j, nv)
     Integer *g_a, *nv, *i, *j;
     Void *v;
{
int k;
Integer *sbar = (Integer*)malloc(2*sizeof(Integer)* *nv);
     if(!sbar) ga_error("gather:malloc failed",*nv);
     for(k=0;k<*nv;k++){
          sbar[2*k] = i[k];
          sbar[2*k+1] = j[k];
     }
     nga_gather_(g_a,v,sbar,nv);
     free(sbar);
}
  


/*\ SCATTER OPERATION elements of v into the global array
\*/
void FATR  ga_scatter000_(g_a, v, i, j, nv)
     Integer *g_a, *nv, *i, *j;
     Void *v;
{
int k;
Integer *sbar = (Integer*)malloc(2*sizeof(Integer)* *nv);
     if(!sbar) ga_error("scatter:malloc failed",*nv);
     for(k=0;k<*nv;k++){
          sbar[2*k] = i[k];
          sbar[2*k+1] = j[k];
     }
     nga_scatter_(g_a,v,sbar,nv);
     free(sbar);
}


/*\ GATHER OPERATION elements from the global array into v
\*/
void FATR  ga_gather_(Integer *g_a, void *v, Integer *i, Integer *j,
                      Integer *nv)
{
    register Integer k;
    Integer kk;
    Integer pindex, phandle, item_size;
    Integer proc;

    Integer *aproc, naproc; /* active processes and numbers */
    Integer *map;           /* map the active processes to allocated space */
    char *buf1, *buf2;
    
    Integer *count;   /* counters for each process */
    Integer *nelem;   /* number of elements for each process */
    /* source and destination pointers for each process */
    void ***ptr_src, ***ptr_dst; 
    void **ptr_org; /* the entire pointer array */
    armci_giov_t desc;
    Integer *ilo, *ihi, *jlo, *jhi, *ldp;
    char **ptr_ref;
    
    if (*nv < 1) return;

    ga_check_handleM(g_a, "ga_gather");
    GA_PUSH_NAME("ga_gather");
    GAstat.numgat++;

    if(!MA_push_get(MT_F_INT, *nv, "ga_gather--p", &phandle, &pindex))
        ga_error("MA failed ", *g_a);

    /* allocate temp memory */
    buf1 = gai_malloc(4 * GAnproc * (sizeof(Integer)));
    if(buf1 == NULL) ga_error("gai_malloc failed", 3*GAnproc);
    
    count = (Integer *)buf1;
    nelem = (Integer *)(buf1 + GAnproc * sizeof(Integer));
    aproc = (Integer *)(buf1 + 2 * GAnproc * sizeof(Integer));
    map = (Integer *)(buf1 + 3 * GAnproc * sizeof(Integer));
   
    /* initialize the counters and nelem */
    for(kk=0; kk<GAnproc; kk++) {
        count[kk] = 0; nelem[kk] = 0;
    }
    
    /* find proc that owns the (i,j) element; store it in temp: INT_MB[] */
    for(k=0; k< *nv; k++) {
        if(! ga_locate_(g_a, i+k, j+k, INT_MB+pindex+k)){
            sprintf(err_string,"invalid i/j=(%ld,%ld)", i[k], j[k]);
            ga_error(err_string, *g_a);
        }
        nelem[INT_MB[pindex+k]]++;
    }

    naproc = 0;
    for(k=0; k<GAnproc; k++) if(nelem[k] > 0) {
        aproc[naproc] = k;
        map[k] = naproc;
        naproc ++;
    }
    
    buf2 = gai_malloc(2*naproc*sizeof(void **) + 2*(*nv)*sizeof(void *) +
                      5*naproc*sizeof(Integer) + naproc*sizeof(char*));
    if(buf2 == NULL) ga_error("gai_malloc failed", naproc);
 
    ptr_src = (void ***)buf2;
    ptr_dst = (void ***)(buf2 + naproc*sizeof(void **));
    ptr_org = (void **)(buf2 + 2*naproc*sizeof(void **));
    ilo = (Integer *)(buf2+2*naproc*sizeof(void **)+2*(*nv)*sizeof(void *));
    ihi = ilo + naproc;
    jlo = ihi + naproc;
    jhi = jlo + naproc;
    ldp = jhi + naproc;
    ptr_ref = (char **)(ldp + naproc);

    for(kk=0; kk<naproc; kk++) {
        ga_distribution_(g_a, &aproc[kk],
                         &(ilo[kk]), &(ihi[kk]), &(jlo[kk]), &(jhi[kk]));
        
        /* get address of the first element owned by proc */
        gaShmemLocation(aproc[kk], *g_a, ilo[kk], jlo[kk], &(ptr_ref[kk]),
                        &(ldp[kk]));
    }
    
    item_size = GA[GA_OFFSET + *g_a].elemsize;
    GAbytes.gattot += (double)item_size**nv;
    GAbytes.gatloc += (double)item_size * nelem[INT_MB[pindex+GAme]];

    ptr_src[0] = ptr_org; ptr_dst[0] = ptr_org + (*nv);
    for(k=1; k<naproc; k++) {
        ptr_src[k] = ptr_src[k-1] + nelem[aproc[k-1]];
        ptr_dst[k] = ptr_dst[k-1] + nelem[aproc[k-1]];
    }
    
    for(k=0; k<(*nv); k++){
        int this_count;
        proc = INT_MB[pindex+k]; this_count = count[proc]; count[proc]++;
        proc = map[proc]; 
        ptr_dst[proc][this_count] = ((char*)v) + k * item_size;

        if(i[k] < ilo[proc] || i[k] > ihi[proc]  ||
           j[k] < jlo[proc] || j[k] > jhi[proc]){
          sprintf(err_string,"proc=%d invalid i/j=(%ld,%ld)><[%ld:%ld,%ld:%ld]",
                 (int)proc,i[k],j[k],ilo[proc],ihi[proc],jlo[proc], jhi[proc]);
            ga_error(err_string, *g_a);
        }
        ptr_src[proc][this_count] = ptr_ref[proc] + item_size *
            ((j[k] - jlo[proc])* ldp[proc] + i[k] - ilo[proc]);
    }
    
    /* source and destination pointers are ready for all processes */
    for(k=0; k<naproc; k++) {
        int rc;

        desc.bytes = item_size;
        desc.src_ptr_array = ptr_src[k];
        desc.dst_ptr_array = ptr_dst[k];
        desc.ptr_array_len = nelem[aproc[k]];
        
        rc=ARMCI_GetV(&desc, 1, aproc[k]);
        if(rc) ga_error("gather failed in armci",rc);
    }

    gai_free(buf2);
    gai_free(buf1);

    if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);
    GA_POP_NAME;
}
      
           



/*\ READ AND INCREMENT AN ELEMENT OF A GLOBAL ARRAY
\*/
Integer FATR nga_read_inc_(Integer* g_a, Integer* subscript, Integer* inc)
{
Integer *ptr, ldp[MAXDIM], value, proc, handle=GA_OFFSET+*g_a;
int optype;

    ga_check_handleM(g_a, "nga_read_inc");
    GA_PUSH_NAME("ga_read_inc");

    if(GA[handle].type!=MT_F_INT) ga_error("type must be integer",*g_a);

    GAstat.numrdi++;
    GAbytes.rditot += (double)sizeof(Integer);

    /* find out who owns it */
    nga_locate_(g_a, subscript, &proc);

    /* get an address of the g_a(subscript) element */
/*    gaShmemLocation(proc, *g_a, subscript[0],subscript[1],(char**)&ptr,&ldp[0]);*/
    gam_Location(proc, handle,  subscript, (char**)&ptr, ldp);

#   ifdef EXT_INT
      optype = ARMCI_FETCH_AND_ADD_LONG;
#   else
      optype = ARMCI_FETCH_AND_ADD;
#   endif

    if(GAme == proc)GAbytes.rdiloc += (double)sizeof(Integer);

#ifdef PERMUTE_PIDS
    if(GA_Proc_list) proc = GA_inv_Proc_list[proc];
#endif

    ARMCI_Rmw(optype, (int*)&value, (int*)ptr, (int)*inc, (int)proc);


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
logical FATR ga_compare_distr_(Integer *g_a, Integer *g_b)
{
int h_a =*g_a + GA_OFFSET;
int h_b =*g_b + GA_OFFSET;
int i;

   GA_PUSH_NAME("ga_compare_distr");
   ga_check_handleM(g_a, "distribution a");
   ga_check_handleM(g_b, "distribution b");
   
   GA_POP_NAME;

   if(GA[h_a].ndim != GA[h_b].ndim) return FALSE; 

   for(i=0; i <GA[h_a].ndim; i++)
       if(GA[h_a].dims[i] != GA[h_b].dims[i]) return FALSE;

   for(i=0; i <MAPLEN; i++){
      if(GA[h_a].mapc[i] != GA[h_b].mapc[i]) return FALSE;
      if(GA[h_a].mapc[i] == -1) break;
   }
   return TRUE;
}


static int num_mutexes=0;
static int chunk_mutex;

logical FATR ga_create_mutexes_(Integer *num)
{
Integer myshare;

   if (*num <= 0 || *num > MAX_MUTEXES) return(FALSE);
   if(num_mutexes) ga_error("mutexes already created",num_mutexes);

   num_mutexes= (int)*num;

   if(GAnproc == 1) return(TRUE);

   chunk_mutex = (*num + GAnproc-1)/GAnproc;
   if(GAme * chunk_mutex >= *num)myshare =0;
   else myshare=chunk_mutex;

   /* need work here to use permutation */
   if(ARMCI_Create_mutexes(myshare)) return FALSE;
   return TRUE;
}


void FATR ga_lock_(Integer *mutex)
{
int m,p;

   if(GAnproc == 1) return;
   if(num_mutexes< *mutex)ga_error("invalid mutex",*mutex);

   p = num_mutexes/chunk_mutex -1;
   m = num_mutexes%chunk_mutex;

#ifdef PERMUTE_PIDS
    if(GA_Proc_list) p = GA_inv_Proc_list[p];
#endif

   ARMCI_Lock(m,p);
}


void FATR ga_unlock_(Integer *mutex)
{
int m,p;

   if(GAnproc == 1) return;
   if(num_mutexes< *mutex)ga_error("invalid mutex",*mutex);
   
   p = num_mutexes/chunk_mutex -1;
   m = num_mutexes%chunk_mutex;

#ifdef PERMUTE_PIDS
    if(GA_Proc_list) p = GA_inv_Proc_list[p];
#endif

   ARMCI_Unlock(m,p);
}              
   

logical FATR ga_destroy_mutexes_()
{
   if(num_mutexes<1) ga_error("mutexes destroyed",0);
   num_mutexes= 0;
   if(GAnproc == 1) return TRUE;
   if(ARMCI_Destroy_mutexes()) return FALSE;
   return TRUE;
}


/*\ return list of message-passing process ids for GA process ids
\*/
void FATR ga_list_nodeid_(list, num_procs)
     Integer *list, *num_procs;
{
Integer proc;
   for( proc = 0; proc < *num_procs; proc++)

#ifdef PERMUTE_PIDS
       if(GA_Proc_list) list[proc] = GA_inv_Proc_list[proc]; 
       else
#endif
       list[proc]=proc;
}


/*************************************************************************/

logical FATR ga_locate_region_(g_a, ilo, ihi, jlo, jhi, mapl, np )
        Integer *g_a, *ilo, *jlo, *ihi, *jhi, mapl[][5], *np;
{
   logical status;
   Integer lo[2], hi[2], p;
   lo[0]=*ilo; lo[1]=*jlo;
   hi[0]=*ihi; hi[1]=*jhi;

   status = nga_locate_region_(g_a,lo,hi,map, GA_proclist, np);

   /* need to swap elements (ilo,jlo,ihi,jhi) -> (ilo,ihi,jlo,jhi) */
   for(p = 0; p< *np; p++){
     mapl[p][0] = map[4*p];
     mapl[p][1] = map[4*p + 2];
     mapl[p][2] = map[4*p + 1];
     mapl[p][3] = map[4*p + 3];
     mapl[p][4] = GA_proclist[p];
   } 

   return status;
}



void ga_scatter_local(Integer g_a, Void *v,Integer *i,Integer *j,
                      Integer nv, Integer proc) 
{
     ga_error("ga_scatter_local should not be used",0);
}


/*\ LOCATE THE OWNER OF THE (i,j) ELEMENT OF A GLOBAL ARRAY
\*/
logical FATR ga_locate_(g_a, i, j, owner)
        Integer *g_a, *i, *j, *owner;
{
Integer subscript[2];
  
    subscript[0]=*i; subscript[1]=*j;

    return nga_locate_(g_a, subscript, owner);
}


/*\ RETURN COORDINATES OF A 2-D GA PATCH ASSOCIATED WITH PROCESSOR proc
\*/
void FATR  ga_distribution_(g_a, proc, ilo, ihi, jlo, jhi)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *proc;
{
Integer lo[2], hi[2];
Integer ndim = ga_ndim_(g_a);

   if(ndim != 2)
      ga_error("ga_distribution:2D API cannot be used for dimension",ndim);

   nga_distribution_(g_a, proc, lo, hi);
   *ilo = lo[0]; *ihi=hi[0];
   *jlo = lo[1]; *jhi=hi[1]; 
}


/*\ RETURN COORDINATES OF ARRAY BLOCK HELD BY A PROCESSOR
\*/
void FATR ga_proc_topology_(g_a, proc, pr, pc)
   Integer *g_a, *proc, *pr, *pc;
{
Integer subscript[2];
   nga_proc_topology_(g_a, proc,subscript);
   *pr = subscript[0]; 
   *pc = subscript[1]; 
}



Integer ga_read_inc_local(g_a, i, j, inc, proc)
        Integer g_a, i, j, inc, proc;
{
   ga_error("ga_rdi_local should not be used",0);
   return(-1);
}


/*\ READ AND INCREMENT AN ELEMENT OF A GLOBAL ARRAY
\*/
Integer FATR ga_read_inc_(g_a, i, j, inc)
        Integer *g_a, *i, *j, *inc;
{
Integer  value, subscript[2];

#ifdef GA_TRACE
       trace_stime_();
#endif

   subscript[0] =*i;
   subscript[1] =*j;
   value = nga_read_inc_(g_a, subscript, inc);

#  ifdef GA_TRACE
     trace_etime_();
     op_code = GA_OP_RDI;
     trace_genrec_(g_a, i, i, j, j, &op_code);
#  endif

   return(value);
}



/*\ returns true/false depending on validity of the handle
\*/
logical FATR ga_valid_handle_(Integer *g_a)
{
   if(GA_OFFSET+ (*g_a) < 0 || GA_OFFSET+(*g_a) >= max_global_array ||
      ! (GA[GA_OFFSET+(*g_a)].actv) ) return FALSE;
   else return TRUE;
}
