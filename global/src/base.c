/* $Id: base.c,v 1.108 2004-12-08 02:40:50 manoj Exp $ */
/* 
 * module: base.c
 * author: Jarek Nieplocha
 * description: implements GA primitive operations --
 *              create (regular& irregular) and duplicate, destroy
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
#include "globalp.h"
#include "message.h"
#include "base.h"
#include "macdecls.h"
#include "armci.h"


#ifdef GA_USE_VAMPIR
#include "ga_vampir.h"
#endif
#ifdef GA_PROFILE
#include "ga_profile.h"
#endif
/*#define AVOID_MA_STORAGE 1*/ 
#define DEBUG 0
#define USE_MALLOC 1
#define INVALID_MA_HANDLE -1 
#define NEAR_INT(x) (x)< 0.0 ? ceil( (x) - 0.5) : floor((x) + 0.5)

#define MAX_PTR MAX_NPROC
#define MAPLEN  (MIN(GAnproc, MAX_NPROC) +MAXDIM)
#define FLEN        80              /* length of Fortran strings */

/*uncomment line below to verify consistency of MA in every sync */
/*#define CHECK_MA yes */

/*uncomment line below to verify if MA base address is alligned wrt datatype*/
#if !(defined(LINUX) || defined(CRAY) || defined(CYGWIN))
#define CHECK_MA_ALGN 1
#endif

/*uncomment line below to initialize arrays in ga_create/duplicate */
/*#define GA_CREATE_INDEF yes */

/*uncomment line below to introduce padding between shared memory regions 
  of a GA when the region spans in more than 1 process within SMP */
#define GA_ELEM_PADDING yes

global_array_t *_ga_main_data_structure;
global_array_t *GA;
proc_list_t *_proc_list_main_data_structure;
proc_list_t *PGRP_LIST;
static int GAinitialized = 0;
int _ga_sync_begin = 1;
int _ga_sync_end = 1;
int _max_global_array = MAX_ARRAYS;
Integer *GA_proclist;
int* GA_Proc_list = NULL;
int* GA_inv_Proc_list=NULL;
int GA_World_Proc_Group = -1;
int GA_Default_Proc_Group = -1;
int GA_Init_Proc_Group = -2;

/* MA addressing */
DoubleComplex   *DCPL_MB;           /* double precision complex base address */
DoublePrecision *DBL_MB;            /* double precision base address */
Integer         *INT_MB;            /* integer base address */
float           *FLT_MB;            /* float base address */
int** GA_Update_Flags;
int* GA_Update_Signal;

typedef struct {
long id;
long type;
long size;
long dummy;
} getmem_t;

/* set total limit (bytes) for memory usage per processor to "unlimited" */
static Integer GA_total_memory = -1;
static Integer GA_memory_limited = 0;
struct ga_stat_t GAstat;
struct ga_bytes_t GAbytes ={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
long   *GAstat_arr;
static Integer GA_memory_limit=0;
Integer GAme, GAnproc;
static Integer MPme;
Integer mapALL[MAX_NPROC+1];

char *GA_name_stack[NAME_STACK_LEN];  /* stack for storing names of GA ops */
int  GA_stack_size=0;

/* Function prototypes */
extern void gai_init_onesided();
int gai_getmem(char* name, char **ptr_arr, Integer bytes, int type, long *id,
               int grp_id);


/*************************************************************************/

/*\ This macro computes index (place in ordered set) for the element
 *  identified by _subscript in ndim- dimensional array of dimensions _dim[]
 *  assume that first subscript component changes first
\*/
#define ga_ComputeIndexM(_index, _ndim, _subscript, _dims)                     \
{                                                                              \
  Integer  _i, _factor=1;                                                      \
  __CRAYX1_PRAGMA("_CRI novector");                                         \
  for(_i=0,*(_index)=0; _i<_ndim; _i++){                                       \
      *(_index) += _subscript[_i]*_factor;                                     \
      if(_i<_ndim-1)_factor *= _dims[_i];                                      \
  }                                                                            \
}


/*\ updates subscript corresponding to next element in a patch <lo[]:hi[]>
\*/
#define ga_UpdateSubscriptM(_ndim, _subscript, _lo, _hi, _dims)\
{                                                                              \
  Integer  _i;                                                                 \
  __CRAYX1_PRAGMA("_CRI novector");                                         \
  for(_i=0; _i<_ndim; _i++){                                                   \
       if(_subscript[_i] < _hi[_i]) { _subscript[_i]++; break;}                \
       _subscript[_i] = _lo[_i];                                               \
  }                                                                            \
}


/*\ Initialize n-dimensional loop by counting elements and setting subscript=lo
\*/
#define ga_InitLoopM(_elems, _ndim, _subscript, _lo, _hi, _dims)\
{                                                                              \
  Integer  _i;                                                                 \
  *_elems = 1;                                                                 \
  __CRAYX1_PRAGMA("_CRI novector");                                         \
  for(_i=0; _i<_ndim; _i++){                                                   \
       *_elems *= _hi[_i]-_lo[_i] +1;                                          \
       _subscript[_i] = _lo[_i];                                               \
  }                                                                            \
}


Integer GAsizeof(type)    
        Integer type;
{
  switch (type) {
     case C_DBL  : return (sizeof(double));
     case C_INT  : return (sizeof(int));
     case C_DCPL : return (sizeof(DoubleComplex));
     case C_FLOAT : return (sizeof(float));
     case C_LONG : return (sizeof(long));
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

      GA_Proc_list = (int*)malloc((size_t)GAnproc * sizeof(int)*2);
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

      GA_Proc_list = (int*)malloc((size_t)GAnproc * sizeof(int)*2);
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

    if( GA_OFFSET + (*g_a) < 0 || GA_OFFSET + (*g_a) >= _max_global_array){
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


/*\ Initialize MA-like addressing:
 *  get addressees for the base arrays for double, complex and int types
\*/
static int ma_address_init=0;
void gai_ma_address_init()
{
#ifdef CHECK_MA_ALGN
Integer  off_dbl, off_int, off_dcpl, off_flt;
#endif
     ma_address_init=1;
     INT_MB = (Integer*)MA_get_mbase(MT_F_INT);
     DBL_MB = (DoublePrecision*)MA_get_mbase(MT_F_DBL);
     DCPL_MB= (DoubleComplex*)MA_get_mbase(MT_F_DCPL);
     FLT_MB = (float*)MA_get_mbase(MT_F_REAL);  
#   ifdef CHECK_MA_ALGN
        off_dbl = 0 != ((long)DBL_MB)%sizeof(DoublePrecision);
        off_int = 0 != ((long)INT_MB)%sizeof(Integer);
        off_dcpl= 0 != ((long)DCPL_MB)%sizeof(DoublePrecision);
        off_flt = 0 != ((long)FLT_MB)%sizeof(float);  
        if(off_dbl)
           ga_error("GA initialize: MA DBL_MB not alligned", (Integer)DBL_MB);

        if(off_int)
           ga_error("GA initialize: INT_MB not alligned", (Integer)INT_MB);

        if(off_dcpl)
          ga_error("GA initialize: DCPL_MB not alligned", (Integer)DCPL_MB);

        if(off_flt)
           ga_error("GA initialize: FLT_MB not alligned", (Integer)FLT_MB);   

#   endif

    if(DEBUG) printf("%d INT_MB=%p DBL_MB=%p DCPL_MB=%p FLT_MB=%p\n",
                     (int)GAme, INT_MB,DBL_MB, DCPL_MB, FLT_MB);
}



/*\ INITIALIZE GLOBAL ARRAY STRUCTURES
 *
 *  either ga_initialize_ltd or ga_initialize must be the first 
 *         GA routine called (except ga_uses_ma)
\*/
void FATR  ga_initialize_()
{
Integer  i, j,nproc, nnode, zero;
int bytes;

    if(GAinitialized) return;
#ifdef GA_USE_VAMPIR
    vampir_init(NULL,NULL,__FILE__,__LINE__);
    ga_vampir_init(__FILE__,__LINE__);
    vampir_begin(GA_INITIALIZE,__FILE__,__LINE__);
#endif

    GA_Default_Proc_Group = -1;
    /* zero in pointers in GA array */
    _ga_main_data_structure
       = (global_array_t *)malloc(sizeof(global_array_t)*MAX_ARRAYS);
    _proc_list_main_data_structure
       = (proc_list_t *)malloc(sizeof(proc_list_t)*MAX_ARRAYS);
    if(!_ga_main_data_structure)
       ga_error("ga_init:malloc ga failed",0);
    if(!_proc_list_main_data_structure)
       ga_error("ga_init:malloc proc_list failed",0);
    GA = _ga_main_data_structure;
    PGRP_LIST = _proc_list_main_data_structure;
    for(i=0;i<MAX_ARRAYS; i++) {
       GA[i].ptr  = (char**)0;
       GA[i].mapc = (int*)0;
       PGRP_LIST[i].map_proc_list = (int*)0;
       PGRP_LIST[i].inv_map_proc_list = (int*)0;
       PGRP_LIST[i].actv = 0;
    }

    bzero(&GAstat,sizeof(GAstat));

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

    if(GA_Proc_list)
      fprintf(stderr,"permutation applied %d now becomes %d\n",(int)MPme,(int)GAme);

    if(GAnproc > MAX_NPROC && MPme==0){
      fprintf(stderr,"Current GA setup is for up to %d processors\n",MAX_NPROC);
      fprintf(stderr,"Please change MAX_NPROC in config.h & recompile\n");
      ga_error("terminating...",0);
    }

    GA_proclist = (Integer*)malloc((size_t)GAnproc*sizeof(Integer)); 
    if(!GA_proclist) ga_error("ga_init:malloc failed (proclist)",0);
    gai_init_onesided();

    /* set activity status for all arrays to inactive */
    for(i=0;i<_max_global_array;i++)GA[i].actv=0;

    ARMCI_Init(); /* initialize GA run-time library */
    /* Create proc list for mirrored arrays */
    PGRP_LIST[0].map_proc_list = (int*)malloc(GAnproc*sizeof(int)*2);
    PGRP_LIST[0].inv_map_proc_list = PGRP_LIST[0].map_proc_list + GAnproc;
    for (i=0; i<GAnproc; i++) PGRP_LIST[0].map_proc_list[i] = -1;
    for (i=0; i<GAnproc; i++) PGRP_LIST[0].inv_map_proc_list[i] = -1;
    nnode = ga_cluster_nodeid_();
    nproc = ga_cluster_nprocs_((Integer*)&nnode);
    zero = 0;
    j = ga_cluster_procid_((Integer*)&nnode, (Integer*)&zero);
    PGRP_LIST[0].parent = -1;
    PGRP_LIST[0].actv = 1;
    PGRP_LIST[0].map_nproc = nproc;
    PGRP_LIST[0].mirrored = 1;
    for (i=0; i<nproc; i++) {
       PGRP_LIST[0].map_proc_list[i+j] = i;
       PGRP_LIST[0].inv_map_proc_list[i] = i+j;
    }

    /* assure that GA will not alocate more shared memory than specified */
    if(ARMCI_Uses_shm())
       if(GA_memory_limited) ARMCI_Set_shm_limit(GA_total_memory);

    /* Allocate memory for update flags and signal*/
    bytes = 2*MAXDIM*sizeof(int);
    GA_Update_Flags = (int**)malloc(GAnproc*sizeof(void*));
    if (!GA_Update_Flags)
      ga_error("ga_init: Failed to initialize GA_Update_Flags",(int)GAme);
    if (ARMCI_Malloc((void**)GA_Update_Flags, (armci_size_t) bytes))
      ga_error("ga_init:Failed to initialize memory for update flags",GAme);
    if(GA_Update_Flags[GAme]==NULL)ga_error("ga_init:ARMCIMalloc failed",GAme);

    bytes = sizeof(int);
    GA_Update_Signal = ARMCI_Malloc_local((armci_size_t) bytes);

    /* Zero update flags */
    for (i=0; i<2*MAXDIM; i++) GA_Update_Flags[GAme][i] = 0;

    /* set MA error function */
    MA_set_error_callback(ARMCI_Error);

    GAinitialized = 1;

#ifdef GA_PROFILE 
    ga_profile_init();
#endif
#ifdef GA_USE_VAMPIR
    vampir_end(GA_INITIALIZE,__FILE__,__LINE__);
#endif

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
    for(i=0; i<_max_global_array; i++) 
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
#ifdef GA_USE_VAMPIR
  vampir_init(NULL,NULL,__FILE__,__LINE__);
  ga_vampir_init(__FILE__,__LINE__);
  vampir_begin(GA_INITIALIZE_LTD,__FILE__,__LINE__);
#endif
  GA_total_memory =GA_memory_limit  = *mem_limit; 
  if(*mem_limit >= 0) GA_memory_limited = 1; 
  ga_initialize_();
#ifdef GA_USE_VAMPIR
  vampir_end(GA_INITIALIZE_LTD,__FILE__,__LINE__);
#endif
}


  

#define gam_checktype(_type)\
       if(_type != C_DBL  && _type != C_INT &&  \
          _type != C_DCPL && _type != C_FLOAT &&_type != C_LONG)\
         ga_error("ttype not yet supported ",  _type)

#define gam_checkdim(ndim, dims)\
{\
int _d;\
    if(ndim<1||ndim>MAXDIM) ga_error("unsupported number of dimensions",ndim);\
  __CRAYX1_PRAGMA("_CRI novector");                                         \
    for(_d=0; _d<ndim; _d++)\
         if(dims[_d]<1)ga_error("wrong dimension specified",dims[_d]);\
}

/*\ utility function to tell whether or not an array is mirrored
\*/
logical FATR ga_is_mirrored_(Integer *g_a)
{
  Integer ret = FALSE;
  Integer handle = GA_OFFSET + *g_a;
  Integer p_handle = (Integer)GA[handle].p_handle;
  if (p_handle >= 0) {
     if (PGRP_LIST[p_handle].mirrored) ret = TRUE;
  }
  return ret;
}


/*\ UTILITY FUNCTION TO LOCATE THE BOUNDING INDICES OF A CONTIGUOUS CHUNK OF
 *  SHARED MEMORY FOR A MIRRORED ARRAY
\*/
void ngai_get_first_last_indices( Integer *g_a)  /* array handle (input) */
{

  Integer  lo[MAXDIM], hi[MAXDIM];
  Integer  nelems, nnodes, inode, nproc;
  Integer  ifirst, ilast, nfirst, nlast, icnt, np;
  Integer  i, j, itmp, icheck, ndim, map_offset[MAXDIM];
  Integer  index[MAXDIM], subscript[MAXDIM];
  Integer  handle = GA_OFFSET + *g_a;
  Integer  type, size, id, grp_id;
  int Save_default_group;
  char     *fptr, *lptr;

  /* find total number of elements */
  ndim = GA[handle].ndim;
  nelems = 1;
  for (i=0; i<ndim; i++) nelems *= GA[handle].dims[i];

  /* If array is mirrored, evaluate first and last indices */
  if (ga_is_mirrored_(g_a)) {
    /* If default group is not world group, change default group to world group
       temporarily */
    Save_default_group = GA_Default_Proc_Group;
    GA_Default_Proc_Group = -1;
    nnodes = ga_cluster_nnodes_();
    inode = ga_cluster_nodeid_();
    nproc = ga_cluster_nprocs_(&inode);
    grp_id = GA[handle].p_handle;
    ifirst = (int)((double)(inode*nelems)/((double)nnodes));
    if (inode != nnodes-1) {
      ilast = (int)((double)((inode+1)*nelems)/((double)nnodes))-1;
    } else {
      ilast = nelems-1;
    }
    /* ifirst and ilast correspond to offsets in shared memory. Find the
       actual indices of the data elements corresponding to these offsets */
    for (i = 0; i<ndim; i++) {
      map_offset[i] = 0;
      for (j = 0; j<i; j++) {
        map_offset[i] += GA[handle].nblock[j];
      }
    }
    icnt = 0;
    nfirst = -1;
    nlast = -1;
    for (i = 0; i<nproc; i++) {
      /* find block indices corresponding to proc i */
      nga_proc_topology_(g_a, &i, index);
      nelems = 1;
      for (j = 0; j<ndim; j++) {
        if (index[j] < GA[handle].nblock[j]-1) {
          
          itmp = (GA[handle].mapc[map_offset[j]+index[j]+1]
               - GA[handle].mapc[map_offset[j]+index[j]]);
          nelems *= itmp;
        } else {
          itmp = (GA[handle].dims[j]
               - GA[handle].mapc[map_offset[j]+index[j]] + 1);
          nelems *= itmp;
        }
      }
      icnt += nelems;
      if (nfirst == -1 && ifirst < icnt) {
        nfirst = i;
      }
      if (nlast == -1 && ilast < icnt) {
        nlast = i;
      }
    }
    /* Adjust indices corresponding to start and end of block of
       shared memory so that it can be decomposed into large
       rectangular blocks of the global array. Start by
       adusting the lower index */
    icnt = 0;
    for (i = 0; i<nfirst; i++) {
      nga_proc_topology_(g_a, &i, index);
      nelems = 1;
      for (j = 0; j<ndim; j++) {
        if (index[j] < GA[handle].nblock[j]-1) {
          nelems *= (GA[handle].mapc[map_offset[j]+index[j]+1]
                 - GA[handle].mapc[map_offset[j]+index[j]]);
        } else {
          nelems *= (GA[handle].dims[j]
                 - GA[handle].mapc[map_offset[j]+index[j]] + 1);
        }
      }
      icnt += nelems;
    }
    ifirst = ifirst - icnt;
    /* find dimensions of data on block nfirst */
    np = 0;
    for (i=0; i<inode; i++) np += ga_cluster_nprocs_(&i);
    np += nfirst;
    np = PGRP_LIST[grp_id].map_proc_list[np];
    nga_distribution_(g_a, &np, lo, hi);
    for (i=0; i<ndim; i++) {
      subscript[i] = ifirst%(hi[i] - lo[i] + 1);
      ifirst /= (hi[i] - lo[i] + 1);
    }
    /* adjust value of ifirst */
    nga_proc_topology_(g_a, &nfirst, index);
    ifirst = subscript[ndim-1];
    for (i=0; i<ndim-1; i++) {
      subscript[i] = 0;
    }
    /* Finally, evaluate absolute indices of first data point */
    for (i=0; i<ndim; i++) {
      GA[handle].first[i] = GA[handle].mapc[map_offset[i]+index[i]]
                          + subscript[i];
    }
    /* adjust upper bound */
    if (nlast > nfirst) {
      icnt = 0;
      for (i = 0; i<nlast; i++) {
        nga_proc_topology_(g_a, &i, index);
        nelems = 1;
        for (j = 0; j<ndim; j++) {
          if (index[j] < GA[handle].nblock[j]-1) {
            nelems *= (GA[handle].mapc[map_offset[j]+index[j]+1]
                   - GA[handle].mapc[map_offset[j]+index[j]]);
          } else {
            nelems *= (GA[handle].dims[j]
                   - GA[handle].mapc[map_offset[j]+index[j]] + 1);
          }
        }
        icnt += nelems;
      }
    }
    ilast = ilast - icnt;
    /* find dimensions of data on block nfirst */
    np = 0;
    for (i=0; i<inode; i++) np += ga_cluster_nprocs_(&i);
    np += nlast;
    np = PGRP_LIST[grp_id].map_proc_list[np];
    nga_distribution_(g_a, &np, lo, hi);
    for (i=0; i<ndim; i++) {
      subscript[i] = ilast%(hi[i] - lo[i] + 1);
      ilast /= (hi[i] - lo[i] + 1);
    }
    /* adjust value of ilast */
    icheck = 1;
    nga_proc_topology_(g_a, &nlast, index);
    for (i=0; i<ndim-1; i++) {
      if (index[i] < GA[handle].nblock[i]-1) {
        itmp = GA[handle].mapc[map_offset[i]+index[i]+1]
             - GA[handle].mapc[map_offset[i]+index[i]];
      } else {
        itmp = GA[handle].dims[i]
             - GA[handle].mapc[map_offset[i]+index[i]] + 1;
      }
      if (subscript[i] < itmp-1) icheck = 0;
      subscript[i] = itmp-1;
    }
    if (!icheck) {
      subscript[ndim-1]--;
    }
    /* Finally, evaluate absolute indices of last data point */
    for (i=0; i<ndim; i++) {
      GA[handle].last[i] = GA[handle].mapc[map_offset[i]+index[i]]
                          + subscript[i];
    }
    /* find length of shared memory segment owned by this node. Adjust
     * length, if necessary, to account for gaps in memory between
     * processors */
    type = GA[handle].type;
    switch(type) {
      case C_FLOAT: size = sizeof(float); break;
      case C_DBL: size = sizeof(double); break;
      case C_LONG: size = sizeof(long); break;
      case C_INT: size = sizeof(int); break;
      case C_DCPL: size = 2*sizeof(double); break;
      default: ga_error("type not supported",type);
    }
    for (i=0; i<ndim; i++) index[i] = (Integer)GA[handle].last[i];
    i = nga_locate_(g_a, index, &id);
    nga_distribution_(g_a, &id, lo, hi);
    /* check to see if last is the last element on the processor */
    icheck = 1;
    for (i=0; i<ndim; i++) {
      if (GA[handle].last[i] < hi[i]) icheck = 0;
     /* BJP printf("p[%d] last[%d]: %d, hi[%d]: %d\n",GAme,i,GA[handle].last[i],i,hi[i]); */
    }
    /* if last is the last element on a processor that is NOT the last
     * processor on the node then we need to get the first element on the
     * next processor */
    if (inode == nnodes - 1) icheck = 0;
    if (icheck) {
      /* BJP printf("p[%d] Augmenting index, inode: %d nproc: %d\n",GAme,inode,nproc); */
      ilast = GA[handle].last[0]-1;
      for (i=1; i<ndim; i++) {
        ilast = ilast*GA[handle].dims[i-1]+GA[handle].last[i]-1;
      }
      ilast++;
      for (i=ndim-1; i>=1; i--) {
        subscript[i] = ilast%GA[handle].dims[i-1];
        subscript[i]++;
        /* BJP printf("p[%d] subscript[%d]: %d\n",GAme,i,subscript[i]); */
        ilast /= GA[handle].dims[i-1];
      }
      subscript[0] = ilast+1;
      /* BJP printf("p[%d] subscript[%d]: %d\n",GAme,0,subscript[0]); */
      i = nga_locate_(g_a, subscript, &id);
      gam_Loc_ptr(id, handle, subscript, &lptr);
      size = 0;
    } else {
      gam_Loc_ptr(id, handle, GA[handle].last, &lptr);
    }
    for (i=0; i<ndim; i++) index[i] = (Integer)GA[handle].first[i];
    i = nga_locate_(g_a, index, &id);
    gam_Loc_ptr(id, handle, GA[handle].first, &fptr);
    GA[handle].shm_length = lptr - fptr + size;
    GA_Default_Proc_Group = Save_default_group;
  } else {
    for (i=0; i<ndim; i++) {
      GA[handle].first[i] = 0;
      GA[handle].last[i] = -1;
      GA[handle].shm_length = -1;
    }
  }
}

/*\ print subscript of ndim dimensional array with two strings before and after
\*/
void gai_print_subscript(char *pre,int ndim, Integer subscript[], char* post)
{
        int i;

        printf("%s [",pre);
        for(i=0;i<ndim;i++){
                printf("%ld",(long)subscript[i]);
                if(i==ndim-1)printf("] %s",post);
                else printf(",");
        }
}

void gai_init_struct(int handle)
{
     if(!GA[handle].ptr){
        int len = (int)MIN((Integer)GAnproc, MAX_PTR);
        GA[handle].ptr = (char**)malloc(len*sizeof(char**));
     }
     if(!GA[handle].mapc){
        int len = (int)MAPLEN;
        GA[handle].mapc = (int*)malloc(len*sizeof(int*));
        GA[handle].mapc[0] = -1;
     }
     if(!GA[handle].ptr)ga_error("malloc failed: ptr:",0);
     if(!GA[handle].mapc)ga_error("malloc failed: mapc:",0);
     GA[handle].ndim = -1;
}

/*\ SIMPLE FUNCTION TO SET DEFAULT PROCESSOR GROUP
  \*/
void FATR ga_pgroup_set_default_(Integer *grp)
{
    int local_sync_begin,local_sync_end;
 
    local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous sync masking*/
 
    /* force a hang if default group is not being set correctly */
    if (local_sync_begin || local_sync_end) ga_pgroup_sync_(grp);
    GA_Default_Proc_Group = (int)(*grp);
}
 
int FATR ga_pgroup_create_(Integer *list, Integer *count)
{
    Integer pgrp_handle, i, j, nprocs, itmp;
    Integer tmp_list[MAX_NPROC], parent;
    int tmp2_list[MAX_NPROC], tmp_count;
 
    GA_PUSH_NAME("ga_pgroup_create_");
    /*** Get next free process group handle ***/
    pgrp_handle =-1; i=0;
    do{
       if(!PGRP_LIST[i].actv) pgrp_handle=i;
       i++;
    }while(i<_max_global_array && pgrp_handle==-1);
    if( pgrp_handle == -1)
       ga_error(" Too many process groups ", (Integer)_max_global_array);
 
    /* Check list for validity (no duplicates and no out of range entries) */
    nprocs = GAnproc;
    for (i=0; i<*count; i++) {
       if (list[i] <0 || list[i] >= nprocs)
	  ga_error(" invalid element in list ", list[i]);
       for (j=i+1; j<*count; j++) {
	  if (list[i] == list[j])
	     ga_error(" Duplicate elements in list ", list[i]);
       }
    }
 
    /* Allocate memory for arrays containg processor maps and initialize
       values */
  PGRP_LIST[pgrp_handle].map_proc_list
    = (int*)malloc(GAnproc*sizeof(int)*2);
  PGRP_LIST[pgrp_handle].inv_map_proc_list
    = PGRP_LIST[pgrp_handle].map_proc_list + GAnproc;
  for (i=0; i<GAnproc; i++)
     PGRP_LIST[pgrp_handle].map_proc_list[i] = -1;
  for (i=0; i<GAnproc; i++)
     PGRP_LIST[pgrp_handle].inv_map_proc_list[i] = -1;
 
  /* Remap elements in list to absolute processor indices (if necessary)*/
  if (GA_Default_Proc_Group != -1) {
     parent = GA_Default_Proc_Group;
     for (i=0; i<*count; i++) {
	tmp2_list[i] = (int)PGRP_LIST[parent].inv_map_proc_list[list[i]];
     }
  } else {
     for (i=0; i<*count; i++) {
	tmp2_list[i] = (int)list[i];
     }
  }
  /* use a simple sort routine to reorder list into assending order */
  for (j=1; j<*count; j++) {
     itmp = tmp2_list[j];
     i = j-1;
     while(i>=0  && tmp2_list[i] > itmp) {
	tmp2_list[i+1] = tmp2_list[i];
	i--;
     }
     tmp2_list[i+1] = itmp;
  }
 
  tmp_count = (int)(*count);
  /* Create proc list maps */
  for (i=0; i<*count; i++) {
     j = tmp2_list[i];
     PGRP_LIST[pgrp_handle].map_proc_list[j] = i;
     PGRP_LIST[pgrp_handle].inv_map_proc_list[i] = j;
  }
  PGRP_LIST[pgrp_handle].actv = 1;
  PGRP_LIST[pgrp_handle].parent = GA_Default_Proc_Group;
  PGRP_LIST[pgrp_handle].mirrored = 0;
  PGRP_LIST[pgrp_handle].map_nproc = tmp_count;
#ifdef MPI
  ARMCI_Group_create(tmp_count, tmp2_list, &PGRP_LIST[pgrp_handle].group);
#endif
 
 
  GA_POP_NAME;
#ifdef MPI
  return pgrp_handle;
#else
  return ga_pgroup_get_default_();
#endif
}

/*\ SIMPLE FUNCTIONS TO RECOVER STANDARD PROCESSOR LISTS
\*/
Integer FATR ga_pgroup_get_default_()
{
  return GA_Default_Proc_Group;
}

Integer FATR ga_pgroup_get_mirror_()
{
  return 0;
}

Integer FATR ga_pgroup_get_world_()
{
  return -1;
}

#ifdef MPI
ARMCI_Group* ga_get_armci_group_(int grp_id)
{
  return &PGRP_LIST[grp_id].group;
}
#endif

/*\ Return a new global array handle
\*/
Integer ga_create_handle_()
{
  Integer ga_handle, i, g_a;
  /*** Get next free global array handle ***/
  GA_PUSH_NAME("ga_create_handle");
  ga_handle =-1; i=0;
  do{
      if(!GA[i].actv) ga_handle=i;
      i++;
  }while(i<_max_global_array && ga_handle==-1);
  if( ga_handle == -1)
      ga_error(" too many arrays ", (Integer)_max_global_array);
  g_a = (Integer)ga_handle - GA_OFFSET;

  /*** fill in Global Info Record for g_a ***/
  gai_init_struct(ga_handle);
  GA[ga_handle].p_handle = GA_Init_Proc_Group;
  GA[ga_handle].name[0] = '\0';
  GA[ga_handle].mapc[0] = -1;
  GA[ga_handle].irreg = 0;
  GA[ga_handle].ghosts = 0;
  GA[ga_handle].corner_flag = -1;
  GA[ga_handle].cache = NULL;
  GA_POP_NAME;
  return g_a;
}

/*\ Set the dimensions and data type on a new global array
\*/
void ga_set_data_(Integer *g_a, Integer *ndim, Integer *dims, Integer *type)
{
  Integer i;
  Integer ga_handle = *g_a + GA_OFFSET;
  GA_PUSH_NAME("ga_set_data");
  if (GA[ga_handle].actv == 1)
    ga_error("Cannot set data on array that has been allocated",0);
  gam_checkdim(*ndim, dims);
  gam_checktype(ga_type_f2c(*type));

  GA[ga_handle].type = ga_type_f2c((int)(*type));

  for (i=0; i<*ndim; i++) {
    GA[ga_handle].dims[i] = (int)dims[i];
    GA[ga_handle].chunk[i] = 0;
    GA[ga_handle].width[i] = 0;
  }
  GA[ga_handle].ndim = (int)(*ndim);
  GA_POP_NAME;
}

/*\ Set the chunk array on a new global array
\*/
void ga_set_chunk_(Integer *g_a, Integer *chunk)
{
  Integer i;
  Integer ga_handle = *g_a + GA_OFFSET;
  GA_PUSH_NAME("ga_set_chunk");
  if (GA[ga_handle].actv == 1)
    ga_error("Cannot set chunk on array that has been allocated",0);
  if (GA[ga_handle].ndim < 1)
    ga_error("Dimensions must be set before chunk array is specified",0);
  if (chunk) {
    for (i=0; i<GA[ga_handle].ndim; i++) {
      GA[ga_handle].chunk[i] = (int)chunk[i];
    }
  }
  GA_POP_NAME;
}

/*\ Set the array name on a new global array
\*/
void ga_set_array_name(Integer g_a, char *array_name)
{
  Integer ga_handle = g_a + GA_OFFSET;
  GA_PUSH_NAME("ga_set_array_name");
  if (GA[ga_handle].actv == 1)
    ga_error("Cannot set array name on array that has been allocated",0);
  strcpy(GA[ga_handle].name, array_name);
  GA_POP_NAME;
}

/*\ Set the array name on a new global array
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
void FATR ga_set_array_name_(Integer *g_a, _fcd array_name)
#else
void FATR ga_set_array_name_(Integer *g_a, char* array_name, int slen)
#endif
{
  char buf[FNAM];
#if defined(CRAY) || defined(WIN32)
  f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
  f2cstring(array_name ,slen, buf, FNAM);
#endif

  ga_set_array_name(*g_a, buf);
}

/*\ Set the processor configuration on a new global array
\*/
void ga_set_pgroup_(Integer *g_a, Integer *p_handle)
{
  Integer ga_handle = *g_a + GA_OFFSET;
  GA_PUSH_NAME("ga_set_pgroup");
  if (GA[ga_handle].actv == 1)
    ga_error("Cannot set processor configuration on array that has been allocated",0);
  if (*p_handle == GA_World_Proc_Group || PGRP_LIST[*p_handle].actv == 1) {
    GA[ga_handle].p_handle = (int) (*p_handle);
  } else {
    ga_error("Processor group does not exist",0);
  }
  GA_POP_NAME;
}

Integer ga_get_pgroup_(Integer *g_a)
{
    Integer ga_handle = *g_a + GA_OFFSET;
    return (Integer)GA[ga_handle].p_handle;
}
 
Integer ga_get_pgroup_size_(Integer *grp_id)
{
    int p_handle = (int)(*grp_id);
    if (p_handle > 0) {
       return (Integer)PGRP_LIST[p_handle].map_nproc;
    } else {
       return GAnproc;
    }
}

/*\ Add ghost cells to a new global array
\*/
void ga_set_ghosts_(Integer *g_a, Integer *width)
{
  Integer i;
  Integer ga_handle = *g_a + GA_OFFSET;
  GA_PUSH_NAME("ga_set_ghosts");
  if (GA[ga_handle].actv == 1)
    ga_error("Cannot set ghost widths on array that has been allocated",0);
  if (GA[ga_handle].ndim < 1)
    ga_error("Dimensions must be set before array widths are specified",0);
  for (i=0; i<GA[ga_handle].ndim; i++) {
    if ((int)width[i] > GA[ga_handle].dims[i])
      ga_error("Boundary width must be <= corresponding dimension",i);
    if ((int)width[i] < 0)
      ga_error("Boundary width must be >= 0",i);
  }
  for (i=0; i<GA[ga_handle].ndim; i++) {
    GA[ga_handle].width[i] = (int)width[i];
    if (width[i] > 0) GA[ga_handle].ghosts = 1;
  }
  if (GA[ga_handle].actv == 0) {
    if (!ga_set_ghost_info_(g_a))
      ga_error("Could not allocate update information for ghost cells",0);
  }
  GA_POP_NAME;
}

/*\ Set irregular distribution in a new global array
\*/
void ga_set_irreg_distr_(Integer *g_a, Integer *mapc, Integer *nblock)
{
  Integer i, maplen;
  Integer ga_handle = *g_a + GA_OFFSET;
  GA_PUSH_NAME("ga_set_irreg_distr");
  if (GA[ga_handle].actv == 1)
    ga_error("Cannot set irregular data distribution on array that has been allocated",0);
  if (GA[ga_handle].ndim < 1)
    ga_error("Dimensions must be set before irregular distribution is specified",0);
  for (i=0; i<GA[ga_handle].ndim; i++)
    if ((int)nblock[i] > GA[ga_handle].dims[i])
      ga_error("number of blocks must be <= corresponding dimension",i);
  maplen = 0;
  for (i=0; i<GA[ga_handle].ndim; i++) {
    maplen += nblock[i];
    GA[ga_handle].nblock[i] = (int)nblock[i];
  }
  for (i=0; i<maplen; i++) {
    GA[ga_handle].mapc[i] = (int)mapc[i];
  }
  GA[ga_handle].mapc[maplen] = -1;
  GA[ga_handle].irreg = 1;
  GA_POP_NAME;
}

/*\ Overide the irregular data distribution flag on a new global array
\*/
void FATR ga_set_irreg_flag_(Integer *g_a, logical *flag)
{
  Integer ga_handle = *g_a + GA_OFFSET;
  GA_PUSH_NAME("ga_set_irreg");
  GA[ga_handle].irreg = (int)(*flag);
  GA_POP_NAME;
}

/*\ Get dimension on a new global array
\*/
Integer ga_get_dimension_(Integer *g_a)
{
  Integer ga_handle = *g_a + GA_OFFSET;
  return (Integer)GA[ga_handle].ndim;
} 

/*\  Allocate memory and complete setup of global array
\*/
logical ga_allocate_( Integer *g_a)
{

  Integer hi[MAXDIM];
  Integer ga_handle = *g_a + GA_OFFSET;
  Integer d, width[MAXDIM], ndim;
  Integer mem_size, nelem;
  Integer i, status, maplen=0, p_handle;
  Integer dims[MAXDIM], chunk[MAXDIM];
  Integer pe[MAXDIM], *pmap[MAXDIM], *map;
  Integer blk[MAXDIM];
  Integer me_local;
#ifdef GA_USE_VAMPIR
  vampir_begin(GA_ALLOCATE,__FILE__,__LINE__);
#endif

  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous sync masking*/
  if (GA[ga_handle].ndim == -1)
    ga_error("Insufficient data to create global array",0);

  p_handle = (Integer)GA[ga_handle].p_handle;
  if (p_handle == (Integer)GA_Init_Proc_Group) {
    GA[ga_handle].p_handle = GA_Default_Proc_Group;
    p_handle = GA_Default_Proc_Group;
  }
  ga_pgroup_sync_(&p_handle);
  GA_PUSH_NAME("ga_allocate");

  if(!GAinitialized) ga_error("GA not initialized ", 0);
  if(!ma_address_init) gai_ma_address_init();

  ndim = GA[ga_handle].ndim;
  for (i=0; i<ndim; i++) width[i] = GA[ga_handle].width[i];

  /* The data distribution has not been specified by the user. Create
     default distribution */
  if (GA[ga_handle].mapc[0] == -1) {
    extern void ddb_h2(Integer ndims, Integer dims[], Integer npes,
                       double thr, Integer bias, Integer blk[],
                       Integer pedims[]);

    for (d=0; d<ndim; d++) {
      dims[d] = GA[ga_handle].dims[d];
      chunk[d] = GA[ga_handle].chunk[d];
    }
    if(chunk && chunk[0]!=0) /* for either NULL or chunk[0]=0 compute all */
      for(d=0; d< ndim; d++) blk[d]=MIN(chunk[d],dims[d]);
    else
      for(d=0; d< ndim; d++) blk[d]=-1;

    /* eliminate dimensions =1 from ddb analysis */
    for(d=0; d<ndim; d++)if(dims[d]==1)blk[d]=1;
 
    if (GAme==0 && DEBUG )
      for (d=0;d<ndim;d++) fprintf(stderr,"b[%ld]=%ld\n",(long)d,(long)blk[d]);
    ga_pgroup_sync_(&p_handle);

    /* ddb(ndim, dims, GAnproc, blk, pe);*/
    if (p_handle >= 0) {
       ddb_h2(ndim, dims, PGRP_LIST[p_handle].map_nproc, 0.0,
              (Integer)0, blk, pe);
    } else {
       ddb_h2(ndim, dims, GAnproc, 0.0, (Integer)0, blk, pe);
    }

    for(d=0, map=mapALL; d< ndim; d++){
      Integer nblock;
      Integer pcut; /* # procs that get full blk[] elements; the rest gets less*/
      int p;

      pmap[d] = map;

      /* RJH ... don't leave some nodes without data if possible
       but respect the users block size */
      
      if (chunk && chunk[d] > 1) {
        Integer ddim = (dims[d]-1)/MIN(chunk[d],dims[d]) + 1;
        pcut = (ddim -(blk[d]-1)*pe[d]) ;
      }
      else {
        pcut = (dims[d]-(blk[d]-1)*pe[d]) ;
      }

      for (nblock=i=p=0; (p<pe[d]) && (i<dims[d]); p++, nblock++) {
        Integer b = blk[d];
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
      gai_print_subscript("pe ",(int)ndim, pe,"\n");
      gai_print_subscript("blocks ",(int)ndim, blk,"\n");
      printf("decomposition map\n");
      for(d=0; d< ndim; d++){
        printf("dim=%ld: ",(long)d); 
        for (i=0;i<pe[d];i++)printf("%ld ",(long)pmap[d][i]);
        printf("\n"); 
      }
      fflush(stdout);
    }
    maplen = 0;
    for( i = 0; i< ndim; i++){
      GA[ga_handle].nblock[i] = pe[i];
      maplen += pe[i];
    }
    for(i = 0; i< maplen; i++) {
      GA[ga_handle].mapc[i] = (int)mapALL[i];
    }
    GA[ga_handle].mapc[maplen] = -1;
  }

  GAstat.numcre ++;

  GA[ga_handle].actv = 1;
  /* If only one processor is being used and array is mirrored,
   * set proc list to world group */
  if (ga_cluster_nnodes_() == 1 && GA[ga_handle].p_handle == 0) {
    GA[ga_handle].p_handle = ga_pgroup_get_world_();
  }
  /* set corner flag, if it has not already been set and set up message
     passing data */
  if (GA[ga_handle].corner_flag == -1) {
     i = 1;
  } else {
     i = GA[ga_handle].corner_flag;
  }
  ga_set_ghost_corner_flag_(g_a, &i);
 
  for( i = 0; i< ndim; i++){
     GA[ga_handle].scale[i] = (double)GA[ga_handle].nblock[i]
       / (double)GA[ga_handle].dims[i];
  }
  GA[ga_handle].elemsize = GAsizeofM(GA[ga_handle].type);
  /*** determine which portion of the array I am supposed to hold ***/
  if (p_handle >= 0) {
     me_local = (Integer)PGRP_LIST[p_handle].map_proc_list[GAme];
     nga_distribution_(g_a, &me_local, GA[ga_handle].lo, hi);
  } else {
     nga_distribution_(g_a, &GAme, GA[ga_handle].lo, hi);
  }
  for( i = 0, nelem=1; i< ndim; i++){
       GA[ga_handle].chunk[i] = (int)(hi[i]-GA[ga_handle].lo[i]+1);
       nelem *= (int)(hi[i]-GA[ga_handle].lo[i]+1+2*width[i]);
  }
  mem_size = nelem * GA[ga_handle].elemsize;
  GA[ga_handle].id = INVALID_MA_HANDLE;
  GA[ga_handle].size = mem_size;
  /* if requested, enforce limits on memory consumption */
  if(GA_memory_limited) GA_total_memory -= mem_size;
  /* check if everybody has enough memory left */
  if(GA_memory_limited){
     status = (GA_total_memory >= 0) ? 1 : 0;
     if (p_handle > 0) {
        ga_pgroup_igop(p_handle,GA_TYPE_GSM, &status, 1, "*");
     } else {
        ga_igop(GA_TYPE_GSM, &status, 1, "*");
     }
  }else status = 1;

  if (status) {
    status = !gai_getmem(GA[ga_handle].name, GA[ga_handle].ptr,mem_size,
                             GA[ga_handle].type, &GA[ga_handle].id, p_handle);
  } else {
    GA[ga_handle].ptr[GAme]=NULL;
  }

  /* If array is mirrored, evaluate first and last indices */
  ngai_get_first_last_indices(g_a);

  ga_pgroup_sync_(&p_handle);
  if (status) {
    GAstat.curmem += GA[ga_handle].size;
    GAstat.maxmem  = MAX(GAstat.maxmem, GAstat.curmem);
    status = TRUE;
  } else {
    ga_destroy_(g_a);
    status = FALSE;
  }
  GA_POP_NAME;
#ifdef GA_USE_VAMPIR
  vampir_end(GA_ALLOCATE,__FILE__,__LINE__);
#endif
  return status;
}

/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY WITH GHOST CELLS
 *   -- IRREGULAR DISTRIBUTION
 *  This is the master routine. All other creation routines are derived
 *  from this one.
\*/
logical nga_create_ghosts_irreg_config(
        Integer type,     /* MA type */ 
        Integer ndim,     /* number of dimensions */
        Integer dims[],   /* array of dimensions */
        Integer width[],  /* width of boundary cells for each dimension */
        char *array_name, /* array name */
        Integer map[],    /* decomposition map array */ 
        Integer nblock[], /* number of blocks for each dimension in map */
        Integer p_handle, /* processor list handle */
        Integer *g_a)     /* array handle (output) */
{
  logical status;
#ifdef GA_USE_VAMPIR
  vampir_begin(NGA_CREATE_GHOSTS_IRREG_CONFIG,__FILE__,__LINE__);
#endif

  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous sync masking*/
  ga_sync_();
  GA_PUSH_NAME("nga_create_ghosts_irreg_config");

  *g_a = ga_create_handle_();
  ga_set_data_(g_a,&ndim,dims,&type);
  ga_set_ghosts_(g_a,width);
  ga_set_array_name(*g_a,array_name);
  ga_set_irreg_distr_(g_a,map,nblock);
  ga_set_pgroup_(g_a,&p_handle);
  status = ga_allocate_(g_a);

  GA_POP_NAME;
#ifdef GA_USE_VAMPIR
  vampir_end(NGA_CREATE_IRREG_CONFIG,__FILE__,__LINE__);
#endif
  return status;
}

logical nga_create_ghosts_irreg(
        Integer type,     /* MA type */ 
        Integer ndim,     /* number of dimensions */
        Integer dims[],   /* array of dimensions */
        Integer width[],  /* width of boundary cells for each dimension */
        char *array_name, /* array name */
        Integer map[],    /* decomposition map array */ 
        Integer nblock[], /* number of blocks for each dimension in map */
        Integer *g_a)     /* array handle (output) */
{
   Integer p_handle = ga_pgroup_get_default_();
   return nga_create_ghosts_irreg_config(type, ndim, dims, width,
                array_name, map, nblock, p_handle, g_a);
}


/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY
 *  Allow machine to choose location of array boundaries on individual
 *  processors.
\*/
logical nga_create_config(Integer type,
                         Integer ndim,
                         Integer dims[],
                         char* array_name,
                         Integer *chunk,
                         Integer p_handle,
                         Integer *g_a)
{
  logical status;
  GA_PUSH_NAME("nga_create_config");
  *g_a = ga_create_handle_();
  ga_set_data_(g_a,&ndim,dims,&type);
  ga_set_array_name(*g_a,array_name);
  ga_set_chunk_(g_a,chunk);
  ga_set_pgroup_(g_a,&p_handle);
  status = ga_allocate_(g_a);
  GA_POP_NAME;
  return status;
}

logical nga_create(Integer type,
                   Integer ndim,
                   Integer dims[],
                   char* array_name,
                   Integer *chunk,
                   Integer *g_a)
{
  Integer p_handle = ga_pgroup_get_default_();
  return nga_create_config(type, ndim, dims, array_name, chunk, p_handle, g_a);
}

/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY WITH GHOST CELLS
 *  Allow machine to choose location of array boundaries on individual
 *  processors.
\*/
logical nga_create_ghosts_config(Integer type,
                   Integer ndim,
                   Integer dims[],
                   Integer width[],
                   char* array_name,
                   Integer chunk[],
                   Integer p_handle,
                   Integer *g_a)
{
  logical status;
  GA_PUSH_NAME("nga_create_ghosts");
  *g_a = ga_create_handle_();
  ga_set_data_(g_a,&ndim,dims,&type);
  ga_set_ghosts_(g_a,width);
  ga_set_array_name(*g_a,array_name);
  ga_set_chunk_(g_a,chunk);
  ga_set_pgroup_(g_a,&p_handle);
  status = ga_allocate_(g_a);
  GA_POP_NAME;
  return status;
}

logical nga_create_ghosts(Integer type,
                   Integer ndim,
                   Integer dims[],
                   Integer width[],
                   char* array_name,
                   Integer chunk[],
                   Integer *g_a)
{
  Integer p_handle = ga_pgroup_get_default_();
  return nga_create_ghosts_config(type, ndim, dims, width, array_name,
                  chunk, p_handle, g_a);
}

/*\ CREATE A 2-DIMENSIONAL GLOBAL ARRAY
 *  Allow machine to choose location of array boundaries on individual
 *  processors.
\*/
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

/*\ CREATE A GLOBAL ARRAY -- IRREGULAR DISTRIBUTION -- PROCESSOR CONFIGURATION
 *  User can specify location of array boundaries on individual
 *  processors and the processor configuration.
\*/
logical nga_create_irreg_config(
        Integer type,     /* MA type */ 
        Integer ndim,     /* number of dimensions */
        Integer dims[],   /* array of dimensions */
        char *array_name, /* array name */
        Integer map[],    /* decomposition map array */ 
        Integer nblock[], /* number of blocks for each dimension in map */
        Integer p_handle, /* processor list hande */
        Integer *g_a)     /* array handle (output) */
{

Integer  d,width[MAXDIM];
logical status;

      for (d=0; d<ndim; d++) width[d] = 0;
      status = nga_create_ghosts_irreg_config(type, ndim, dims, width,
          array_name, map, nblock, p_handle, g_a);

      return status;
}

/*\ CREATE A GLOBAL ARRAY -- IRREGULAR DISTRIBUTION
 *  User can specify location of array boundaries on individual
 *  processors.
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

Integer  d,width[MAXDIM];
logical status;

      for (d=0; d<ndim; d++) width[d] = 0;
      status = nga_create_ghosts_irreg(type, ndim, dims, width,
          array_name, map, nblock, g_a);

      return status;
}

/*\ CREATE A 2-DIMENSIONAL GLOBAL ARRAY -- IRREGULAR DISTRIBUTION
 *  User can specify location of array boundaries on individual
 *  processors.
\*/
logical ga_create_irreg(type, dim1, dim2, array_name, map1, nblock1, map2,
                        nblock2, g_a)
      Integer *type, *dim1, *dim2, *map1, *nblock1, *map2, *nblock2, *g_a;
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
Integer  ndim, dims[MAXDIM], width[MAXDIM], nblock[MAXDIM], *map;
Integer  i,ctype;
logical status;
 
      ctype = ga_type_f2c((int)(*type));  
      if(ctype != C_DBL  && ctype != C_INT &&  
         ctype != C_DCPL && ctype != C_FLOAT  && ctype != C_LONG)
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
        for (i=0;i<*nblock1;i++)fprintf(stderr," %ld |",(long)map1[i]);
        fprintf(stderr," \n array:%d map2:\n",(int) *g_a);
        for (i=0;i<*nblock2;i++)fprintf(stderr," %ld |",(long)map2[i]);
        fprintf(stderr,"\n\n");
      }
      ndim = 2;
      dims[0] = *dim1;
      dims[1] = *dim2;
      width[0] = 0;
      width[1] = 0;
      nblock[0] = *nblock1;
      nblock[1] = *nblock2;
      map = mapALL;
      for(i=0;i< *nblock1; i++) map[i] = (int)map1[i];
      for(i=0;i< *nblock2; i++) map[i+ *nblock1] = (int)map2[i];
      status = nga_create_ghosts_irreg(*type, ndim, dims, width,
          array_name, mapALL, nblock, g_a);
 
      return status;

}

/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY WITH GHOST CELLS
 * -- IRREGULAR DISTRIBUTION -- PROCESSOR CONFIGURATION
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
logical FATR nga_create_ghosts_irreg_config_(Integer *type,
    Integer *ndim, Integer *dims, Integer width[],
    _fcd array_name, Integer map[], Integer block[],
    Integer *p_handle, Integer *g_a)
#else
logical FATR nga_create_ghosts_irreg_config_(Integer *type,
    Integer *ndim, Integer *dims, Integer width[], char* array_name,
    Integer map[], Integer block[], Integer *p_handle, Integer *g_a,
    int slen)
#endif
{
char buf[FNAM];
Integer st; 
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif
  
      _ga_irreg_flag = 1; /* set this flag=1, to indicate array is irregular */
      st = nga_create_ghosts_irreg_config(*type, *ndim,  dims, width, buf, 
					  map, block, *p_handle, g_a);
      _ga_irreg_flag = 0; /* unset it, after creating array */ 
      return st;
}

/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY WITH GHOST CELLS
 * -- IRREGULAR DISTRIBUTION
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
logical FATR nga_create_ghosts_irreg_(Integer *type, Integer *ndim,
    Integer *dims, Integer width[],
    _fcd array_name, Integer map[], Integer block[], Integer *g_a)
#else
logical FATR nga_create_ghosts_irreg_(Integer *type, Integer *ndim,
    Integer *dims, Integer width[], char* array_name, Integer map[],
    Integer block[], Integer *g_a, int slen)
#endif
{
char buf[FNAM];
Integer st;
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif
      
      _ga_irreg_flag = 1; /* set this flag=1, to indicate array is irregular */
      st = nga_create_ghosts_irreg(*type, *ndim,  dims, width, buf, map,
				   block, g_a);
      _ga_irreg_flag = 0; /* unset it, after creating array */
      return st;
}

/*\ CREATE A 2-DIMENSIONAL GLOBAL ARRAY
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

/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY -- PROCESSOR CONFIGURATION
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
logical FATR nga_create_config_(Integer *type, Integer *ndim,
                    Integer *dims, _fcd array_name, Integer *chunk,
                    Integer *p_handle, Integer *g_a)
#else
logical FATR nga_create_config_(Integer *type, Integer *ndim,
                   Integer *dims, char* array_name, Integer *chunk,
                   Integer *p_handle, Integer *g_a, int slen)
#endif
{
char buf[FNAM];
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

  return (nga_create_config(*type, *ndim,  dims, buf, chunk, *p_handle, g_a));
}

/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY
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

/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY WITH GHOST CELLS -- PROCESSOR
 *  CONFIGURATION
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
logical FATR nga_create_ghosts_config_(Integer *type, Integer *ndim,
                   Integer *dims, Integer *width, _fcd array_name,
                   Integer *chunk, Integer *p_handle, Integer *g_a)
#else
logical FATR nga_create_ghosts_config_(Integer *type, Integer *ndim,
                   Integer *dims, Integer *width, char* array_name,
                   Integer *chunk, Integer *p_handle, Integer *g_a,
                   int slen)
#endif
{
char buf[FNAM];
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

  return (nga_create_ghosts_config(*type, *ndim,  dims, width, buf, chunk,
                   *p_handle, g_a));
}

/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY WITH GHOST CELLS
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
logical FATR nga_create_ghosts_(Integer *type, Integer *ndim, Integer *dims,
                   Integer *width, _fcd array_name, Integer *chunk, Integer *g_a)
#else
logical FATR nga_create_ghosts_(Integer *type, Integer *ndim, Integer *dims,
                   Integer *width, char* array_name, Integer *chunk, Integer *g_a,
                   int slen)
#endif
{
char buf[FNAM];
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

  return (nga_create_ghosts(*type, *ndim,  dims, width, buf, chunk, g_a));
}

/*\ CREATE A 2-DIMENSIONAL GLOBAL ARRAY -- IRREGULAR DISTRIBUTION
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
Integer st;
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif
      _ga_irreg_flag = 1; /* set this flag=1, to indicate array is irregular*/
      st = ga_create_irreg(type, dim1, dim2, buf, map1, nblock1,
			   map2, nblock2, g_a);
      _ga_irreg_flag = 0; /* unset it, after creating array */ 
      return st;
}

/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY -- IRREGULAR DISTRIBUTION
 *  -- PROCESSOR DISTRIBUTION
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
logical FATR nga_create_irreg_config_(Integer *type, Integer *ndim,
                 Integer *dims, _fcd array_name, Integer map[],
                 Integer block[], Integer *p_handle, Integer *g_a)
#else
logical FATR nga_create_irreg_config_(Integer *type, Integer *ndim,
                 Integer *dims, char* array_name, Integer map[],
                 Integer block[], Integer *p_handle, Integer *g_a,
                 int slen)
#endif
{
char buf[FNAM];
Integer st;
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

      _ga_irreg_flag = 1; /* set this flag=1, to indicate array is irregular*/
      st = nga_create_irreg_config(*type, *ndim,  dims, buf, map, block,
				   *p_handle, g_a);
      _ga_irreg_flag = 0; /* unset it, after creating array */ 
      return st;
}

/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY -- IRREGULAR DISTRIBUTION
 *  Fortran version
\*/
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
Integer st;
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif
      
      _ga_irreg_flag = 1; /* set this flag=1, to indicate array is irregular */
      st = nga_create_irreg(*type, *ndim,  dims, buf, map, block, g_a);
      _ga_irreg_flag = 0; /* unset it, after creating array */
      return st;
}

#ifdef PERMUTE_PIDS
char* ptr_array[MAX_NPROC];
#endif

/*\ get memory alligned w.r.t. MA base
 *  required on Linux as g77 ignores natural data alignment in common blocks
\*/ 
int gai_get_shmem(char **ptr_arr, Integer bytes, int type, long *adj,
		  int grp_id)
{
int status=0;
#ifndef _CHECK_MA_ALGN
char *base;
long diff, item_size;  
Integer *adjust;
int i, nproc,grp_me=GAme;

    if (grp_id > 0) {
       nproc  = PGRP_LIST[grp_id].map_nproc;
       grp_me = PGRP_LIST[grp_id].map_proc_list[GAme];
    }
    else
       nproc = GAnproc; 
 
    /* need to enforce proper, natural allignment (on size boundary)  */
    switch (ga_type_c2f(type)){
      case MT_F_DBL:   base =  (char *) DBL_MB; break;
      case MT_F_INT:   base =  (char *) INT_MB; break;
      case MT_F_DCPL:  base =  (char *) DCPL_MB; break;
      case MT_F_REAL:  base =  (char *) FLT_MB; break;  
      default:        base = (char*)0;
    }

    item_size = GAsizeofM(type);
#   ifdef GA_ELEM_PADDING
       bytes += item_size; 
#   endif

#endif

    *adj = 0;
#ifdef PERMUTE_PIDS
    if(GA_Proc_list){
       bzero(ptr_array,nproc*sizeof(char*));
       /* use ARMCI_Malloc_group for groups if proc group is not world group
	  or mirror group */
#  ifdef MPI
       if (grp_id > 0)
	  status = ARMCI_Malloc_group((void**)ptr_array, bytes,
				      &PGRP_LIST[grp_id].group);
       else
#  endif
	  status = ARMCI_Malloc((void**)ptr_array, bytes);
       if(bytes!=0 && ptr_array[grp_me]==NULL) 
	  ga_error("gai_get_shmem: ARMCI Malloc failed", GAme);
       for(i=0;i<nproc;i++)ptr_arr[i] = ptr_array[GA_inv_Proc_list[i]];
    }else
#endif
       
    /* use ARMCI_Malloc_group for groups if proc group is not world group
       or mirror group */
#ifdef MPI
    if (grp_id > 0) {
       status = ARMCI_Malloc_group((void**)ptr_arr, (armci_size_t)bytes,
				   &PGRP_LIST[grp_id].group);
    } else
#endif
      status = ARMCI_Malloc((void**)ptr_arr, (armci_size_t)bytes);

    if(bytes!=0 && ptr_arr[grp_me]==NULL) 
       ga_error("gai_get_shmem: ARMCI Malloc failed", GAme);
    if(status) return status;

#ifndef _CHECK_MA_ALGN

    /* adjust all addresses if they are not alligned on corresponding nodes*/

    /* we need storage for GAnproc*sizeof(Integer) -- _ga_map is bigger */
    adjust = (Integer*)_ga_map;

    diff = (ABS( base - (char *) ptr_arr[grp_me])) % item_size; 
    for(i=0;i<nproc;i++)adjust[i]=0;
    adjust[grp_me] = (diff > 0) ? item_size - diff : 0;
    *adj = adjust[grp_me];

    if (grp_id > 0)
       ga_pgroup_igop(grp_id,GA_TYPE_GSM, adjust, nproc, "+");
    else
       ga_igop(GA_TYPE_GSM, adjust, nproc, "+");
    
    for(i=0;i<nproc;i++){
       ptr_arr[i] = adjust[i] + (char*)ptr_arr[i];
    }

#endif
    return status;
}


int gai_getmem(char* name, char **ptr_arr, Integer bytes, int type, long *id,
	       int grp_id)
{
Integer handle = INVALID_MA_HANDLE, index;
Integer nelem, item_size = GAsizeofM(type);
char *ptr = (char*)0;

#ifdef AVOID_MA_STORAGE
   return gai_get_shmem(ptr_arr, bytes, type, id, grp_id);
#else
   if(ARMCI_Uses_shm()) return gai_get_shmem(ptr_arr, bytes, type, id, grp_id);
   else{
     nelem = bytes/item_size + 1;
     if(bytes)
        if(MA_alloc_get(type, nelem, name, &handle, &index)){
                MA_get_pointer(handle, &ptr);}
     *id   = (long)handle;

     /* 
            printf("bytes=%d ptr=%ld index=%d\n",bytes, ptr,index);
            fflush(stdout);
     */

     bzero((char*)ptr_arr,(int)GAnproc*sizeof(char*));
     ptr_arr[GAme] = ptr;
     armci_exchange_address((void**)ptr_arr,(int)GAnproc);
     if(bytes && !ptr) return 1; 
     else return 0;
   }
#endif
}


/*\ externalized version of gai_getmem to facilitate two-step array creation
\*/
void *GA_Getmem(int type, int nelem, int grp_id)
{
char **ptr_arr=(char**)0;
int  rc,i;
long id;
int bytes;
int extra=sizeof(getmem_t)+GAnproc*sizeof(char*);
char *myptr;
Integer status;
     type = ga_type_f2c(type);	
     bytes = nelem *  GAsizeofM(type);
     if(GA_memory_limited){
         GA_total_memory -= bytes+extra;
         status = (GA_total_memory >= 0) ? 1 : 0;
         ga_igop(GA_TYPE_GSM, &status, 1, "*");
         if(!status)GA_total_memory +=bytes+extra;
     }else status = 1;

     ptr_arr=(char**)_ga_map; /* need memory GAnproc*sizeof(char**) */
     rc= gai_getmem("ga_getmem", ptr_arr,(Integer)bytes+extra, type, &id, grp_id);
     if(rc)ga_error("ga_getmem: failed to allocate memory",bytes+extra);

     myptr = ptr_arr[GAme];  

     /* make sure that remote memory addresses point to user memory */
     for(i=0; i<GAnproc; i++)ptr_arr[i] += extra;

#ifndef AVOID_MA_STORAGE
     if(ARMCI_Uses_shm()) 
#endif
        id += extra; /* id is used to store offset */

     /* stuff the type and id info at the beginning */
     ((getmem_t*)myptr)->id = id;
     ((getmem_t*)myptr)->type = type;
     ((getmem_t*)myptr)->size = bytes+extra;

     /* add ptr info */
     memcpy(myptr+sizeof(getmem_t),ptr_arr,(size_t)GAnproc*sizeof(char**));

     return (void*)(myptr+extra);
}


void GA_Freemem(void *ptr)
{
int extra = sizeof(getmem_t)+GAnproc*sizeof(char*); 
getmem_t *info = (getmem_t *)((char*)ptr - extra);
char **ptr_arr = (char**)(info+1);

#ifndef AVOID_MA_STORAGE
    if(ARMCI_Uses_shm()){
#endif
      /* make sure that we free original (before address alignment) pointer */
      ARMCI_Free(ptr_arr[GAme] - info->id);
#ifndef AVOID_MA_STORAGE
    }else{
      if(info->id != INVALID_MA_HANDLE) MA_free_heap(info->id);
    }
#endif

    if(GA_memory_limited) GA_total_memory += info->size;
}

/*\ RETURN COORDINATES OF A GA PATCH ASSOCIATED WITH PROCESSOR proc
\*/
void FATR nga_distribution_(Integer *g_a, Integer *proc, Integer *lo, Integer *
hi)
{
Integer ga_handle, p_handle, lproc, tproc;

   ga_check_handleM(g_a, "nga_distribution");
   ga_handle = (GA_OFFSET + *g_a);
   /* BJP
   p_handle = GA[ga_handle].p_handle;
   if (GA_Default_Proc_Group != -1) {
      tproc = PGRP_LIST[GA_Default_Proc_Group].inv_map_proc_list[*proc];
   } else {
      tproc = *proc;
   }
   if (p_handle < 0) {
     lproc = tproc;
   } else {
     lproc = PGRP_LIST[p_handle].map_proc_list[tproc];
   }
   */
   /* BJP */
   lproc = *proc;
   ga_ownsM(ga_handle, lproc, lo, hi);

}




/*\ RETURN COORDINATES OF A GA PATCH ASSOCIATED WITH PROCESSOR proc
\*/
void nga_distribution_no_handle_(Integer *ndim, Integer *dims,
        Integer *nblock, Integer *mapc, Integer *proc, Integer *lo,
        Integer * hi)
{
   ga_ownsM_no_handle(*ndim, dims, nblock, mapc, *proc, lo, hi);
}


/*\ Check to see if array has ghost cells.
\*/
logical FATR ga_has_ghosts_(Integer* g_a)
{
      int h_a = (int)*g_a + GA_OFFSET;
      return GA[h_a].ghosts;
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
int local_sync_begin,local_sync_end;
Integer grp_id;

#ifdef GA_USE_VAMPIR
      vampir_begin(GA_DUPLICATE,__FILE__,__LINE__);
#endif


      local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
      _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
      grp_id = ga_get_pgroup_(g_a);
      if(local_sync_begin)ga_pgroup_sync_(&grp_id);

      GAstat.numcre ++; 

      ga_check_handleM(g_a,"ga_duplicate");       

      /* find a free global_array handle for g_b */
      ga_handle =-1; i=0;
      do{
        if(!GA[i].actv) ga_handle=i;
        i++;
      }while(i<_max_global_array && ga_handle==-1);
      if( ga_handle == -1)
          ga_error("ga_duplicate: too many arrays", (Integer)_max_global_array);
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
	 if (grp_id > 0) {
	    int istatus = (int)status;
	    ga_pgroup_igop((int)grp_id,GA_TYPE_GSM, &status, 1, "*");
	    status = (Integer)status;
         } else {
	    ga_igop(GA_TYPE_GSM, &status, 1, "*");
         }
      }else status = 1;

      if(status)
          status = !gai_getmem(array_name, GA[ga_handle].ptr,mem_size,
                               (int)GA[ga_handle].type, &GA[ga_handle].id,
			       (int)grp_id);
      else{
          GA[ga_handle].ptr[GAme]=NULL;
      }

      if(local_sync_end)ga_pgroup_sync_(&grp_id);

#     ifdef GA_CREATE_INDEF
      if(status){
         Integer one = 1; 
         Integer dim1 =GA[ga_handle].dims[1], dim2=GA[ga_handle].dims[2];
         if(GAme==0)fprintf(stderr,"duplicate:initializing GA array%ld\n",*g_b);
         if(GA[ga_handle].type == C_DBL) {
             double bad = (double) DBL_MAX;
             ga_fill_patch_(g_b, &one, &dim1, &one, &dim2,  &bad);
         } else if (GA[ga_handle].type == C_INT) {
             int bad = (int) INT_MAX;
             ga_fill_patch_(g_b, &one, &dim1, &one, &dim2,  &bad);
         } else if (GA[ga_handle].type == C_LONG) {
             long bad = LONG_MAX;
             ga_fill_patch_(g_b, &one, &dim1, &one, &dim2,  &bad);
         } else if (GA[ga_handle].type == C_DCPL) { 
             DoubleComplex bad = {DBL_MAX, DBL_MAX};
             ga_fill_patch_(g_b, &one, &dim1, &one, &dim2,  &bad);
         } else if (GA[ga_handle].type == C_FLOAT) {
             float bad = FLT_MAX;
             ga_fill_patch_(g_b, &one, &dim1, &one, &dim2,  &bad);   
         } else {
             ga_error("ga_duplicate: type not supported ",GA[ga_handle].type);
         }
      }
#     endif
 
#ifdef GA_USE_VAMPIR
      vampir_end(GA_DUPLICATE,__FILE__,__LINE__);
#endif

      if(status){
         GAstat.curmem += GA[ga_handle].size;
         GAstat.maxmem  = MAX(GAstat.maxmem, GAstat.curmem);
         return(TRUE);
      }else{ 
         ga_destroy_(g_b);
         return(FALSE);
      }
}

/*\ DUPLICATE A GLOBAL ARRAY -- memory comes from user
 *  -- new array g_b will have properties of g_a
\*/
int GA_Assemble_duplicate(int g_a, char* array_name, void* ptr)
{
char     **save_ptr;
int      i, ga_handle;
int      *save_mapc;
int extra = sizeof(getmem_t)+GAnproc*sizeof(char*);
getmem_t *info = (getmem_t *)((char*)ptr - extra);
char **ptr_arr = (char**)(info+1);
int g_b;


      ga_sync_();

      GAstat.numcre ++;

      ga_check_handleM(&g_a,"ga_assemble_duplicate");

      /* find a free global_array handle for g_b */
      ga_handle =-1; i=0;
      do{
        if(!GA[i].actv) ga_handle=i;
        i++;
      }while(i<_max_global_array && ga_handle==-1);
      if( ga_handle == -1)
          ga_error("ga_assemble_duplicate: too many arrays ", 
                                           (Integer)_max_global_array);
      g_b = ga_handle - GA_OFFSET;

      gai_init_struct(ga_handle);

      /*** copy content of the data structure ***/
      save_ptr = GA[ga_handle].ptr;
      save_mapc = GA[ga_handle].mapc;
      GA[ga_handle] = GA[GA_OFFSET + g_a];
      strcpy(GA[ga_handle].name, array_name);
      GA[ga_handle].ptr = save_ptr;
      GA[ga_handle].mapc = save_mapc;
      for(i=0;i<MAPLEN; i++)GA[ga_handle].mapc[i] = GA[GA_OFFSET+ g_a].mapc[i];

      /* get ptrs and datatype from user memory */
      gam_checktype(ga_type_f2c(info->type));
      GA[ga_handle].type = ga_type_f2c(info->type);
      GA[ga_handle].size = info->size;
      GA[ga_handle].id = info->id;
      memcpy(GA[ga_handle].ptr,ptr_arr,(size_t)GAnproc*sizeof(char**));

      GAstat.curmem += GA[ga_handle].size;
      GAstat.maxmem  = MAX(GAstat.maxmem, GAstat.curmem);

      ga_sync_();

      return(g_b);
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
Integer ga_handle = GA_OFFSET + *g_a, p_handle;
int local_sync_begin;

#ifdef GA_USE_VAMPIR
    vampir_begin(GA_DESTROY,__FILE__,__LINE__);
#endif

    local_sync_begin = _ga_sync_begin; 
    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
    p_handle = (Integer)GA[ga_handle].p_handle;
    if(local_sync_begin)ga_pgroup_sync_(&p_handle);

    GAstat.numdes ++; /*regardless of array status we count this call */
    /* fails if handle is out of range or array not active */
    if(ga_handle < 0 || ga_handle >= _max_global_array){
#ifdef GA_USE_VAMPIR
       vampir_end(GA_DESTROY,__FILE__,__LINE__);
#endif
       return FALSE;
    }
    if(GA[ga_handle].actv==0){
#ifdef GA_USE_VAMPIR
       vampir_end(GA_DESTROY,__FILE__,__LINE__);
#endif
       return FALSE;
    }
    if (GA[ga_handle].cache)
      free(GA[ga_handle].cache);
    GA[ga_handle].cache = NULL;
    GA[ga_handle].actv = 0;     
    if(GA[ga_handle].ptr[GAme]==NULL){
#ifdef GA_USE_VAMPIR
       vampir_end(GA_DESTROY,__FILE__,__LINE__);
#endif
       return TRUE;
    } 
#ifndef AVOID_MA_STORAGE
    if(ARMCI_Uses_shm()){
#endif
      /* make sure that we free original (before address allignment) pointer */
#ifdef MPI
      if (GA[ga_handle].p_handle > 0){
	 int grp_me = PGRP_LIST[GA[ga_handle].p_handle].map_proc_list[GAme];
	 ARMCI_Free_group(GA[ga_handle].ptr[grp_me] - GA[ga_handle].id,
			  &PGRP_LIST[GA[ga_handle].p_handle].group);
      }
      else
#endif
	 ARMCI_Free(GA[ga_handle].ptr[GAme] - GA[ga_handle].id);
#ifndef AVOID_MA_STORAGE
    }else{
      if(GA[ga_handle].id != INVALID_MA_HANDLE) MA_free_heap(GA[ga_handle].id);
    }
#endif

    if(GA_memory_limited) GA_total_memory += GA[ga_handle].size;
    GAstat.curmem -= GA[ga_handle].size;

#ifdef GA_USE_VAMPIR
    vampir_end(GA_DESTROY,__FILE__,__LINE__);
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

    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
    if(!GAinitialized) return;

#ifdef GA_USE_VAMPIR
    vampir_begin(GA_TERMINATE,__FILE__,__LINE__);
#endif
#ifdef GA_PROFILE 
    ga_profile_terminate();
#endif
    for (i=0;i<_max_global_array;i++){
          handle = i - GA_OFFSET ;
          if(GA[i].actv) ga_destroy_(&handle);
          if(GA[i].ptr) free(GA[i].ptr);
          if(GA[i].mapc) free(GA[i].mapc);
    }
    ga_sync_();

    GA_total_memory = -1; /* restore "unlimited" memory usage status */
    GA_memory_limited = 0;
    free(_ga_map);
    free(GA_proclist);
    ARMCI_Free_local(GA_Update_Signal);

    ARMCI_Finalize();
    GAinitialized = 0;
    ga_sync_();

#ifdef GA_USE_VAMPIR
    vampir_end(GA_TERMINATE,__FILE__,__LINE__);
    vampir_finalize(__FILE__,__LINE__);
#endif
}   

    
/*\ IS ARRAY ACTIVE/INACTIVE
\*/ 
Integer FATR ga_verify_handle_(g_a)
     Integer *g_a;
{
  return (Integer)
    ((*g_a + GA_OFFSET>= 0) && (*g_a + GA_OFFSET< _max_global_array) && 
             GA[GA_OFFSET + (*g_a)].actv);
}
 


/*\fill array with value
\*/
void FATR ga_fill_(Integer *g_a, void* val)
{
int i,elems,handle=GA_OFFSET + (int)*g_a;
char *ptr;
int local_sync_begin,local_sync_end;
Integer grp_id;

#ifdef GA_USE_VAMPIR
   vampir_begin(GA_FILL,__FILE__,__LINE__);
#endif

   GA_PUSH_NAME("ga_fill");

   local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
   _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous sync masking*/
   grp_id = ga_get_pgroup_(g_a);
   if(local_sync_begin)ga_pgroup_sync_(&grp_id);


   ga_check_handleM(g_a, "ga_fill");
   gam_checktype(GA[handle].type);
   elems = (int)GA[handle].size/GA[handle].elemsize;
   
   /* Bruce..Please CHECK if this is correct */
   if (grp_id >= 0){  
      Integer grp_me = PGRP_LIST[GA[handle].p_handle].map_proc_list[GAme];
      ptr = GA[handle].ptr[grp_me];
   }
   else  ptr = GA[handle].ptr[GAme];

   switch (GA[handle].type){
   case C_DCPL: 
        for(i=0; i<elems;i++)((DoubleComplex*)ptr)[i]=*(DoubleComplex*)val;
        break;
   case C_DBL:  
        for(i=0; i<elems;i++)((double*)ptr)[i]=*(double*)val;
        break;
   case C_INT:  
        for(i=0; i<elems;i++)((int*)ptr)[i]=*(int*)val;
        break;
   case C_FLOAT:
        for(i=0; i<elems;i++)((float*)ptr)[i]=*(float*)val;
        break;     
   case C_LONG:
        for(i=0; i<elems;i++)((long*)ptr)[i]=*(long*)val;
        break;
   default:
        ga_error("type not supported",GA[handle].type);
   }

   if(local_sync_end)ga_pgroup_sync_(&grp_id);

   GA_POP_NAME;
 
#ifdef GA_USE_VAMPIR
   vampir_end(GA_FILL,__FILE__,__LINE__);
#endif
}

/*\ INQUIRE POPERTIES OF A GLOBAL ARRAY
 *   Fortran version for internal global array functions
\*/
void FATR ga_inquire_internal_(Integer* g_a, Integer* type, Integer* dim1, Integer* dim2)
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
void FATR  ga_inquire_(Integer* g_a, Integer* type, Integer* dim1,Integer* dim2)
{
Integer ndim = ga_ndim_(g_a);

   if(ndim != 2)
      ga_error("ga_inquire: 2D API cannot be used for array dimension",ndim);

   *type       = (Integer)ga_type_c2f(GA[GA_OFFSET + *g_a].type);
   *dim1       = GA[GA_OFFSET + *g_a].dims[0];
   *dim2       = GA[GA_OFFSET + *g_a].dims[1];
}


/*\ INQUIRE POPERTIES OF A GLOBAL ARRAY
 *  C version
\*/
void ga_inquire(Integer* g_a, Integer* type, Integer* dim1, Integer* dim2)
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
   *type       = (Integer)ga_type_c2f(GA[handle].type);
   *ndim       = GA[handle].ndim;
   for(i=0;i<*ndim;i++)dims[i]=GA[handle].dims[i];
}

/*\ INQUIRE POPERTIES OF A GLOBAL ARRAY
 *  C version
\*/
void nga_inquire(Integer *g_a, Integer *type, Integer *ndim,Integer *dims)
{
Integer handle = GA_OFFSET + *g_a,i;
   ga_check_handleM(g_a, "nga_inquire");
   *type       = GA[handle].type;
   *ndim       = GA[handle].ndim;
   for(i=0;i<*ndim;i++)dims[i]=GA[handle].dims[i];
}

/*\ INQUIRE POPERTIES OF A GLOBAL ARRAY
 *  Fortran version for internal global array routines
\*/
void FATR nga_inquire_internal_(Integer *g_a, Integer *type, Integer *ndim,Integer *dims)
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

   index = GA_Proc_list ? GA_Proc_list[*proc]: *proc;

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
Integer p_handle;

   ga_check_handleM(g_a, "nga_locate");
   ndim = GA[ga_handle].ndim;

   for(d=0, *owner=-1; d< ndim; d++) 
       if(subscript[d]< 1 || subscript[d]>GA[ga_handle].dims[d]) return FALSE;

   for(d = 0, dpos = 0; d< ndim; d++){
       findblock(GA[ga_handle].mapc + dpos, GA[ga_handle].nblock[d],
                 GA[ga_handle].scale[d], (int)subscript[d], &proc_s[d]);
       dpos += GA[ga_handle].nblock[d];
   }

   ga_ComputeIndexM(&proc, ndim, proc_s, GA[ga_handle].nblock); 

/*   printf("p[%d] computed index: %d\n",(int)GAme,(int)proc); */
   /* BJP
   p_handle = GA[ga_handle].p_handle;
   if (p_handle >= 0) {
      proc = PGRP_LIST[p_handle].inv_map_proc_list[proc];
   }
   */
   *owner = GA_Proc_list ? GA_Proc_list[proc]: proc;
   /* BJP
   if (GA_Default_Proc_Group != -1) {
      *owner = PGRP_LIST[GA_Default_Proc_Group].map_proc_list[*owner];
   }
   */
   
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
/*    g_a      [input]  global array handle
      lo       [input]  lower indices of patch in global array
      hi       [input]  upper indices of patch in global array
      map      [input]  list of lower and upper indices for portion of
                        patch that exists on each processor containing a
                        portion of the patch. The map is constructed so
                        that for a D dimensional global array, the first
                        D elements are the lower indices on the first
                        processor in proclist, the next D elements are
                        the upper indices of the first processor in
                        proclist, the next D elements are the lower
                        indices for the second processor in proclist, and
                        so on.
      proclist [output] list of processors containing some portion of the
                        patch
      np       [output] total number of processors containing a portion
                        of the patch
*/
{
int  procT[MAXDIM], procB[MAXDIM], proc_subscript[MAXDIM];
Integer  proc, owner, i, ga_handle;
Integer  d, dpos, ndim, elems, p_handle;

   ga_check_handleM(g_a, "nga_locate_region");

   ga_handle = GA_OFFSET + *g_a;
#ifdef __crayx1
#pragma _CRI novector
#endif
   for(d = 0; d< GA[ga_handle].ndim; d++)
       if((lo[d]<1 || hi[d]>GA[ga_handle].dims[d]) ||(lo[d]>hi[d]))return FALSE;

   ndim = GA[ga_handle].ndim;

   /* find "processor coordinates" for the top left corner and store them
    * in ProcT */
#ifdef __crayx1
#pragma _CRI novector
#endif
   for(d = 0, dpos = 0; d< GA[ga_handle].ndim; d++){
       findblock(GA[ga_handle].mapc + dpos, GA[ga_handle].nblock[d], 
                 GA[ga_handle].scale[d], (int)lo[d], &procT[d]);
       dpos += GA[ga_handle].nblock[d];
   }

   /* find "processor coordinates" for the right bottom corner and store
    * them in procB */
#ifdef __crayx1
#pragma _CRI novector
#endif
   for(d = 0, dpos = 0; d< GA[ga_handle].ndim; d++){
       findblock(GA[ga_handle].mapc + dpos, GA[ga_handle].nblock[d], 
                 GA[ga_handle].scale[d], (int)hi[d], &procB[d]);
       dpos += GA[ga_handle].nblock[d];
   }

   *np = 0;

   /* Find total number of processors containing data and return the
    * result in elems. Also find the lowest "processor coordinates" of the
    * processor block containing data and return these in proc_subscript.
   */
   ga_InitLoopM(&elems, ndim, proc_subscript, procT,procB,GA[ga_handle].nblock);

   p_handle = (Integer)GA[ga_handle].p_handle;
   for(i= 0; i< elems; i++){ 
      Integer _lo[MAXDIM], _hi[MAXDIM];
      Integer  offset;

      /* convert i to owner processor id using the current values in
       proc_subscript */
      ga_ComputeIndexM(&proc, ndim, proc_subscript, GA[ga_handle].nblock); 
      /* get range of global array indices that are owned by owner */
      ga_ownsM(ga_handle, proc, _lo, _hi);

      offset = *np *(ndim*2); /* location in map to put patch range */

#ifdef __crayx1
#pragma _CRI novector
#endif
      for(d = 0; d< ndim; d++)
              map[d + offset ] = lo[d] < _lo[d] ? _lo[d] : lo[d];
#ifdef __crayx1
#pragma _CRI novector
#endif
      for(d = 0; d< ndim; d++)
              map[ndim + d + offset ] = hi[d] > _hi[d] ? _hi[d] : hi[d];

      /* BJP
      if (p_handle < 0) {
        owner = proc;
      } else {
	owner = PGRP_LIST[p_handle].inv_map_proc_list[proc];
      }
      */
      /* BJP */
      owner = proc;
      proclist[i] = owner;
      /* Update to proc_subscript so that it corresponds to the next
       * processor in the block of processors containing the patch */
      ga_UpdateSubscriptM(ndim,proc_subscript,procT,procB,GA[ga_handle].nblock);
      (*np)++;
   }

   /* Remap processor list (if necessary)*/
   /* BJP
   if (GA_Default_Proc_Group != -1) {
      Integer tmp_list[MAX_NPROC];
      for (i=0; i<*np; i++) {
	 tmp_list[i] = proclist[i];
      }
      for (i=0; i<*np; i++) {
	 proclist[i] = PGRP_LIST[GA_Default_Proc_Group].map_proc_list[tmp_list[i]];
      }
   }
   */

   return(TRUE);
}
#ifdef __crayx1
#pragma _CRI inline nga_locate_region_
#endif


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


Integer FATR ga_nodeid_()
{
    if (GA_Default_Proc_Group > 0) {
       return (Integer)PGRP_LIST[GA_Default_Proc_Group].map_proc_list[GAme];
    } else {
       return ((Integer)GAme);
    }
}

Integer FATR ga_pgroup_nodeid_(Integer *grp)
{
    if (*grp >= 0) {
       return (Integer)PGRP_LIST[(int)(*grp)].map_proc_list[GAme];
    } else {
       return GAme;
    }
}


Integer FATR ga_nnodes_()
{
    if (GA_Default_Proc_Group > 0) {
       return (Integer)PGRP_LIST[GA_Default_Proc_Group].map_nproc;
    } else {
       return ((Integer)GAnproc);
    }
}

Integer FATR ga_pgroup_nnodes_(Integer *grp)
{
    return (Integer)PGRP_LIST[(int)(*grp)].map_nproc;
}


/*\ COMPARE DISTRIBUTIONS of two global arrays
\*/
logical FATR ga_compare_distr_(Integer *g_a, Integer *g_b)
{
int h_a =(int)*g_a + GA_OFFSET;
int h_b =(int)*g_b + GA_OFFSET;
int i;

   _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
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
int myshare;

#ifdef GA_USE_VAMPIR
   vampir_begin(GA_CREATE_MUTEXES,__FILE__,__LINE__);
#endif
   _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
   if (*num <= 0 || *num > MAX_MUTEXES) return(FALSE);
   if(num_mutexes) ga_error("mutexes already created",num_mutexes);

   num_mutexes= (int)*num;

   if(GAnproc == 1){
#ifdef GA_USE_VAMPIR
      vampir_end(GA_CREATE_MUTEXES,__FILE__,__LINE__);
#endif
      return(TRUE);
   }
   chunk_mutex = (int)((*num + GAnproc-1)/GAnproc);
   if(GAme * chunk_mutex >= *num)myshare =0;
   else myshare=chunk_mutex;

   /* need work here to use permutation */
   if(ARMCI_Create_mutexes(myshare)){
#ifdef GA_USE_VAMPIR
      vampir_end(GA_CREATE_MUTEXES,__FILE__,__LINE__);
#endif
      return FALSE;
   }
#ifdef GA_USE_VAMPIR
   vampir_end(GA_CREATE_MUTEXES,__FILE__,__LINE__);
#endif
   return TRUE;
}


void FATR ga_lock_(Integer *mutex)
{
int m,p;

   if(GAnproc == 1) return;
   if(num_mutexes< *mutex)ga_error("invalid mutex",*mutex);

#ifdef GA_USE_VAMPIR
   vampir_begin(GA_LOCK,__FILE__,__LINE__);
#endif

   p = num_mutexes/chunk_mutex -1;
   m = num_mutexes%chunk_mutex;

#ifdef PERMUTE_PIDS
    if(GA_Proc_list) p = GA_inv_Proc_list[p];
#endif

   ARMCI_Lock(m,p);
 
#ifdef GA_USE_VAMPIR
   vampir_end(GA_LOCK,__FILE__,__LINE__);
#endif
}


void FATR ga_unlock_(Integer *mutex)
{
int m,p;

   if(GAnproc == 1) return;
   if(num_mutexes< *mutex)ga_error("invalid mutex",*mutex);
   
#ifdef GA_USE_VAMPIR
   vampir_begin(GA_UNLOCK,__FILE__,__LINE__);
#endif

   p = num_mutexes/chunk_mutex -1;
   m = num_mutexes%chunk_mutex;

#ifdef PERMUTE_PIDS
    if(GA_Proc_list) p = GA_inv_Proc_list[p];
#endif

   ARMCI_Unlock(m,p);

#ifdef GA_USE_VAMPIR
   vampir_end(GA_UNLOCK,__FILE__,__LINE__);
#endif
}              
   

logical FATR ga_destroy_mutexes_()
{
   _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
   if(num_mutexes<1) ga_error("mutexes destroyed",0);

#ifdef GA_USE_VAMPIR
   vampir_begin(GA_DESTROY_MUTEXES,__FILE__,__LINE__);
#endif

   num_mutexes= 0;
   if(GAnproc == 1){
#ifdef GA_USE_VAMPIR
      vampir_end(GA_DESTROY_MUTEXES,__FILE__,__LINE__);
#endif
      return TRUE;
   }
   if(ARMCI_Destroy_mutexes()){
#ifdef GA_USE_VAMPIR
      vampir_end(GA_DESTROY_MUTEXES,__FILE__,__LINE__);
#endif
      return FALSE;
   }
#ifdef GA_USE_VAMPIR
   vampir_end(GA_DESTROY_MUTEXES,__FILE__,__LINE__);
#endif
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

   status = nga_locate_region_(g_a,lo,hi,_ga_map, GA_proclist, np);

   /* need to swap elements (ilo,jlo,ihi,jhi) -> (ilo,ihi,jlo,jhi) */
   for(p = 0; p< *np; p++){
     mapl[p][0] = _ga_map[4*p];
     mapl[p][1] = _ga_map[4*p + 2];
     mapl[p][2] = _ga_map[4*p + 1];
     mapl[p][3] = _ga_map[4*p + 3];
     mapl[p][4] = GA_proclist[p];
   } 

   return status;
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


/*\ returns true/false depending on validity of the handle
\*/
logical FATR ga_valid_handle_(Integer *g_a)
{
   if(GA_OFFSET+ (*g_a) < 0 || GA_OFFSET+(*g_a) >= _max_global_array ||
      ! (GA[GA_OFFSET+(*g_a)].actv) ) return FALSE;
   else return TRUE;
}

int gai_getval(int *ptr) { return *ptr;}

/*\ A function that helps user avoid syncs that he thinks are unnecessary
    inside a collective call.
\*/

/*
       Mask flags have to be reset in every collective call. Even if that
       collective call doesnt do any sync at all.
       If masking only the beginning sync is possible, make sure to
       clear even the _sync_end mask to avoid a mask intended for this
       collective_function_call to be carried to next collective_function_call
       or to a collective function called by this function.
       Similarly, make sure to use two copy mask values to local variables
       and reset the global mask variables to avoid carring the mask to a
       collective call inside the current collective call.
*/
void FATR ga_mask_sync_(Integer *begin, Integer *end)
{
  if (*begin) _ga_sync_begin = 1;
  else _ga_sync_begin = 0;

  if (*end) _ga_sync_end = 1;
  else _ga_sync_end = 0;
}

/*\ merge all copies of a mirrored array by adding them together
\*/
void FATR ga_merge_mirrored_(Integer *g_a)
{
  Integer handle = GA_OFFSET + *g_a;
  Integer inode, nprocs, nnodes, zero, zproc, nblocks;
  int *blocks, *map, *dims, *width;
  Integer i, j, index[MAXDIM], itmp, ndim;
  Integer nelem, count, type, atype;
  char *zptr, *bptr, *nptr;
  Integer bytes, total;
  int local_sync_begin, local_sync_end;

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end = 1; /*remove any previous masking */
  if (local_sync_begin) ga_sync_();
  /* don't perform update if node is not mirrored */
  if (!ga_is_mirrored_(g_a)) return;
  GA_PUSH_NAME("ga_merge_mirrored");

  inode = ga_cluster_nodeid_();
  nnodes = ga_cluster_nnodes_(); 
  nprocs = ga_cluster_nprocs_(&inode);
  zero = 0;

  zproc = ga_cluster_procid_(&inode, &zero);
  zptr = GA[handle].ptr[zproc];
  map = GA[handle].mapc;
  blocks = GA[handle].nblock;
  dims = GA[handle].dims;
  width = GA[handle].width;
  type = GA[handle].type;
  ndim = GA[handle].ndim;

  /* Check whether or not all nodes contain the same number
     of processors. */
  if (nnodes*nprocs == ga_nnodes_())  {
    /* check to see if there is any buffer space between the data
       associated with each processor that needs to be zeroed out
       before performing the merge */
    if (zproc == GAme) {
      /* the use of nblocks instead of nprocs is designed to support a peculiar
         coding style in which the dimensions of the block array are all set to
         1 and all the data is restricted to the master processor on the node */
      nblocks = 1;
      for (i=0; i<ndim; i++) {
        nblocks *= blocks[i];
      }
      for (i=0; i<nblocks; i++) {
        /* Find out from mapc data how many elements are supposed to be located
           on this processor. Start by converting processor number to indices */
        itmp = i;
        for (j=0; j<ndim; j++) {
          index[j] = itmp%(Integer)blocks[j];
          itmp = (itmp - index[j])/(Integer)blocks[j];
        }

        nelem = 1;
        count = 0;
        for (j=0; j<ndim; j++) {
          if (index[j] < (Integer)blocks[j]-1) {
            nelem *= (Integer)(map[index[j]+1+count] - map[index[j]+count]
                   + 2*width[j]);
          } else {
            nelem *= (Integer)(dims[j] - map[index[j]+count] + 1 + 2*width[j]);
          }
          count += (Integer)blocks[j];
        }
        /* We now have the total number of elements located on this processor.
           Find out if the location of the end of this data set matches the
           origin of the data on the next processor. If not, then zero data in
           the gap. */
        nelem *= GAsizeof(type);
        bptr = GA[handle].ptr[ga_cluster_procid_(&inode, &i)];
        bptr += nelem;
        if (i<nblocks-1) {
          j = i+1;
          nptr = GA[handle].ptr[ga_cluster_procid_(&inode, &j)];
          if (bptr != nptr) {
            bytes = (long)nptr - (long)bptr;
            /* BJP printf("p[%d] Gap on proc %d is %d\n",GAme,i,bytes); */
            bzero(bptr, bytes);
          }
        }
      }
      /* find total number of bytes containing global array */
      total = (long)bptr - (long)zptr;
      total /= GAsizeof(type);
      /*convert from C data type to ARMCI type */
      switch(type) {
        case C_FLOAT: atype=ARMCI_FLOAT; break;
        case C_DBL: atype=ARMCI_DOUBLE; break;
        case C_LONG: atype=ARMCI_LONG; break;
        case C_INT: atype=ARMCI_INT; break;
        case C_DCPL: atype=ARMCI_DOUBLE; break;
        default: ga_error("type not supported",type);
      }
      /* now that gap data has been zeroed, do a global sum on data */
      armci_msg_gop_scope(SCOPE_MASTERS, zptr, total, "+", atype);
    } 
  } else {
    Integer _ga_tmp;
    Integer lo[MAXDIM], hi[MAXDIM], ld[MAXDIM];
    Integer idims[MAXDIM], iwidth[MAXDIM], ichunk[MAXDIM];
    void *ptr_a;
    void *one;
    double d_one = 1.0;
    int i_one = 1;
    float f_one = 1.0;
    long l_one = 1;
    double c_one[2];
    c_one[0] = 1.0;
    c_one[1] = 0.0;

    /* choose one as scaling factor in accumulate */
    switch (type) {
      case C_FLOAT: one = &f_one; break;
      case C_DBL: one = &d_one; break;
      case C_LONG: one = &l_one; break;
      case C_INT: one = &i_one; break;
      case C_DCPL: one = &c_one; break;
      default: ga_error("type not supported",type);
    }
    
  /* Nodes contain a mixed number of processors. Create a temporary GA to
     complete merge operation. */
    count = 0;
    for (i=0; i<ndim; i++) {
      idims[i] = (Integer)dims[i];
      iwidth[i] = (Integer)width[i];
      ichunk[i] = 0;
    }
    if (!nga_create_ghosts(type, ndim, idims,
        iwidth, "temporary", ichunk, &_ga_tmp)) 
      ga_error("Unable to create work array for merge",GAme);
    ga_zero_(&_ga_tmp);
    /* Find data on this processor and accumulate in temporary global array */
    nga_distribution_(g_a,&GAme,lo,hi);
    nga_access_ptr(g_a, lo, hi, &ptr_a, ld);
    nga_acc_(&_ga_tmp, lo, hi, ptr_a, ld, one);
    /* copy and data back to original global array */
    ga_sync_();
    nga_get_(&_ga_tmp, lo, hi, ptr_a, ld);
    ga_destroy_(&_ga_tmp);
  }
  if (local_sync_end) ga_sync_();
  GA_POP_NAME;
}

/*\ merge all copies of a  patch of a mirrored array into a patch in a
 *  distributed array
\*/
void FATR nga_merge_distr_patch_(Integer *g_a, Integer *alo, Integer *ahi,
                                 Integer *g_b, Integer *blo, Integer *bhi)
/*    Integer *g_a  handle to mirrored array
      Integer *alo  indices of lower corner of mirrored array patch
      Integer *ahi  indices of upper corner of mirrored array patch
      Integer *g_b  handle to distributed array
      Integer *blo  indices of lower corner of distributed array patch
      Integer *bhi  indices of upper corner of distributed array patch
*/
{
  Integer local_sync_begin, local_sync_end;
  Integer a_handle, b_handle, adim, bdim;
  Integer mlo[MAXDIM], mhi[MAXDIM], mld[MAXDIM];
  Integer dlo[MAXDIM], dhi[MAXDIM];
  double d_one;
  Integer type, i_one;
  double z_one[2];
  float f_one;
  long l_one;
  void *src_data_ptr;
  void *one;
  Integer i, idim, intersect, p_handle;

  GA_PUSH_NAME("nga_merge_distr_patch");
  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end = 1; /*remove any previous masking */
  if (local_sync_begin) ga_sync_();
  ga_check_handle(g_a, "nga_merge_distr_patch");
  ga_check_handle(g_b, "nga_merge_distr_patch");

  /* check to make sure that both patches lie within global arrays and
     that patches are the same dimensions */
  a_handle = GA_OFFSET + *g_a;
  b_handle = GA_OFFSET + *g_b;

  if (!ga_is_mirrored_(g_a))
    ga_error("Handle to a non-mirrored array passed",0);

  if (ga_is_mirrored_(g_b) && ga_cluster_nnodes_())
    ga_error("Distributed array is mirrored",0);

  adim = GA[a_handle].ndim;
  bdim = GA[b_handle].ndim;

  p_handle = GA[a_handle].p_handle;

  if (adim != bdim)
    ga_error("Global arrays must have same dimension",0);

  type = GA[a_handle].type;
  if (type != GA[b_handle].type)
    ga_error("Global arrays must be of same type",0);

  for (i=0; i<adim; i++) {
    idim = GA[a_handle].dims[i];
    if (alo[i] < 0 || alo[i] >= idim || ahi[i] < 0 || ahi[i] >= idim ||
        alo[i] > ahi[i])
      ga_error("Invalid patch index on mirrored GA",0);
  }
  for (i=0; i<bdim; i++) {
    idim = GA[b_handle].dims[i];
    if (blo[i] < 0 || blo[i] >= idim || bhi[i] < 0 || bhi[i] >= idim ||
        blo[i] > bhi[i])
      ga_error("Invalid patch index on distributed GA",0);
  }
  for (i=0; i<bdim; i++) {
    idim = GA[b_handle].dims[i];
    if (ahi[i] - alo[i] != bhi[i] - blo[i])
      ga_error("Patch dimensions do not match for index ",i);
  }
  nga_zero_patch_(g_b, blo, bhi);

  /* Find coordinates of mirrored array patch that I own */
  i = PGRP_LIST[p_handle].map_proc_list[GAme];
  nga_distribution_(g_a, &i, mlo, mhi);
  /* Check to see if mirrored array patch intersects my portion of
     mirrored array */
  intersect = 1;
  for (i=0; i<adim; i++) {
    if (mhi[i] < alo[i]) intersect = 0;
    if (mlo[i] > ahi[i]) intersect = 0;
  }
  if (intersect) {
    /* get portion of mirrored array patch that actually resides on this
       processor */
    for (i=0; i<adim; i++) {
      mlo[i] = MAX(alo[i],mlo[i]);
      mhi[i] = MIN(ahi[i],mhi[i]);
    }

    /* get pointer to locally held distribution */
    nga_access_ptr(g_a, mlo, mhi, &src_data_ptr, mld);

    /* find indices in distributed array corresponding to this patch */
    for (i=0; i<adim; i++) {
      dlo[i] = blo[i] + mlo[i]-alo[i];
      dhi[i] = blo[i] + mhi[i]-alo[i];
    }

    /* perform accumulate */
    if (type == C_DBL) {
      d_one = 1.0;
      one = &d_one;
    } else if (type == C_DCPL) {
      z_one[0] = 1.0;
      z_one[1] = 0.0;
      one = &z_one;
    } else if (type == C_FLOAT) {
      f_one = 1.0;
      one = &f_one;
    } else if (type == C_INT) {
      i_one = 1;
      one = &i_one;
    } else if (type == C_LONG) {
      l_one = 1;
      one = &l_one;
    } else {
      ga_error("Type not supported",type);
    }
    nga_acc_(g_b, dlo, dhi, src_data_ptr, mld, one);
  }
  if (local_sync_end) ga_sync_();
  GA_POP_NAME;
}

/*\ get number of distinct patches corresponding to a contiguous shared
 *  memory segment
\*/
Integer FATR ga_num_mirrored_seg_(Integer *g_a)
{
  Integer handle = *g_a + GA_OFFSET;
  Integer i, j, ndim, map_offset[MAXDIM];
  int *first, *last, *nblock;
  Integer lower[MAXDIM], upper[MAXDIM];
  Integer istart = 0, nproc, inode;
  Integer ret = 0, icheck, np;

  if (!ga_is_mirrored_(g_a)) return ret;
  GA_PUSH_NAME("ga_num_mirrored_seg");
  ndim = GA[handle].ndim;
  first = GA[handle].first;
  last = GA[handle].last;
  nblock = GA[handle].nblock;
  for (i=0; i<ndim; i++) {
    map_offset[i] = 0;
    for (j=0; j<i; j++) {
      map_offset[i] += nblock[j];
    }
  }
  inode = ga_cluster_nodeid_();
  nproc = ga_cluster_nprocs_(&inode);
  /* loop over all data blocks on this node to find out how many
   * separate data blocks correspond to this segment of shared
   * memory */
  for (i=0; i<nproc; i++) {
    /* BJP np = nproc*inode + i; */
    nga_distribution_(g_a,&i,lower,upper);
    icheck = 1;
    /* see if processor corresponds to block of array data
     * that contains start of shared memory segment */
    if (!istart) {
      for (j=0; j<ndim; j++) {
        if (!(first[j] >= lower[j] && first[j] <= upper[j])) {
          icheck = 0;
          break;
        }
      }
    }
    if (icheck && !istart) {
      istart = 1;
    }
    icheck = 1;
    for (j=0; j<ndim; j++) {
      if (!(last[j] >= lower[j] && last[j] <= upper[j])) {
        icheck = 0;
        break;
      }
    }
    if (istart) ret++;
    if (istart && icheck) {
      GA_POP_NAME;
      return ret;
    }
  }
  GA_POP_NAME;
  return ret;
}

/*\ Get patch corresponding to one of the blocks of data
 *  identified using ga_num_mirrored_seg_
\*/
void FATR ga_get_mirrored_block_(Integer *g_a,
                               Integer *npatch,
                               Integer *lo,
                               Integer *hi)
{
  Integer handle = *g_a + GA_OFFSET;
  Integer i, j, ndim, map_offset[MAXDIM];
  int *first, *last, *nblock;
  Integer lower[MAXDIM], upper[MAXDIM];
  Integer istart = 0, nproc, inode;
  Integer ret = 0, icheck, np;

  if (!ga_is_mirrored_(g_a)) {
    for (j=0; j<GA[handle].ndim; j++) {
      lo[j] = 0;
      hi[j] = -1;
    }
    return;
  }
  GA_PUSH_NAME("ga_get_mirrored_block");
  ndim = GA[handle].ndim;
  first = GA[handle].first;
  last = GA[handle].last;
  nblock = GA[handle].nblock;
  for (i=0; i<ndim; i++) {
    map_offset[i] = 0;
    for (j=0; j<i; j++) {
      map_offset[i] += nblock[j];
    }
  }
  inode = ga_cluster_nodeid_();
  nproc = ga_cluster_nprocs_(&inode);
  /* loop over all data blocks on this node to find out how many
   * separate data blocks correspond to this segment of shared
   * memory */
  for (i=0; i<nproc; i++) {
    /* BJP np = nproc*inode + i; */
    nga_distribution_(g_a,&i,lower,upper);
    icheck = 1;
    /* see if processor corresponds to block of array data
     * that contains start of shared memory segment */
    if (!istart) {
      for (j=0; j<ndim; j++) {
        if (!(first[j] >= lower[j] && first[j] <= upper[j])) {
          icheck = 0;
          break;
        }
      }
    }
    if (icheck && !istart) {
      istart = 1;
    }
    icheck = 1;
    for (j=0; j<ndim; j++) {
      if (!(last[j] >= lower[j] && last[j] <= upper[j])) {
        icheck = 0;
        break;
      }
    }
    if (istart && ret == *npatch) {
      if (!icheck) {
        if (ret == 0) {
          for (j=0; j<ndim; j++) {
            lo[j] = first[j];
            hi[j] = upper[j];
          }
        } else {
          for (j=0; j<ndim; j++) {
            lo[j] = lower[j];
            hi[j] = upper[j];
          }
        }
      } else {
        if (ret == 0) {
          for (j=0; j<ndim; j++) {
            lo[j] = first[j];
            hi[j] = last[j];
          }
        } else {
          for (j=0; j<ndim; j++) {
            lo[j] = lower[j];
            hi[j] = last[j];
          }
        }
      }
      GA_POP_NAME;
      return;
    }
    if (istart) ret++;
  }
  for (j=0; j<ndim; j++) {
    lo[j] = 0;
    hi[j] = -1;
  }
  GA_POP_NAME;
  return;
}

/*\ do a fast merge of all copies of a mirrored array only passing
 *  around non-zero data
\*/
void FATR ga_fast_merge_mirrored_(Integer *g_a)
{
  Integer handle = GA_OFFSET + *g_a;
  Integer inode, new_inode, nprocs, nnodes, new_nnodes, zero, zproc;
  int *blocks, *map, *dims, *width;
  Integer i, j, index[MAXDIM], itmp, ndim;
  Integer nelem, count, type;
  int slength, rlength, nsize;
  char  *bptr, *nptr, *fptr;
  Integer bytes;
  Integer ilast,inext,id;
  int Save_default_group;
  int local_sync_begin, local_sync_end;

  /* declarations for message exchanges */
  int next_node,next;
  int armci_tag = 88000;
  char *dstn,*srcp;
  int next_nodel=0;
  int dummy=1, LnB, powof2nodes;
  int groupA, groupB, sizeB;
  void armci_util_wait_int(volatile int *,int,int);

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end = 1; /*remove any previous masking */
  if (local_sync_begin) ga_sync_();
  /* don't perform update if node is not mirrored */
  if (!ga_is_mirrored_(g_a)) return;
  
  /* If default group is not world group, change default group to world group
     temporarily */
  Save_default_group = GA_Default_Proc_Group;
  GA_Default_Proc_Group = -1;

  GA_PUSH_NAME("ga_fast_merge_mirrored");

  inode = ga_cluster_nodeid_();
  /* BJP printf("p[%d] inode: %d\n",GAme,inode); */
  nnodes = ga_cluster_nnodes_(); 
  nprocs = ga_cluster_nprocs_(&inode);
  zero = 0;

  powof2nodes=1;
  LnB = floor(log(nnodes)/log(2))+1;
  if(pow(2,LnB-1)<nnodes){powof2nodes=0;}
  /* Partition nodes into groups A and B. Group A contains a power of 2
   * nodes, group B contains the remainder */
  if (powof2nodes) {
    groupA = 1;
    groupB = 0;
    sizeB = 0;
    new_nnodes = nnodes;
  } else {  
    new_nnodes = pow(2,LnB-1);
    sizeB = nnodes-new_nnodes;
    if (inode<2*sizeB) {
      if (inode%2 == 0) {
        groupA = 1;
        groupB = 0;
      } else {
        groupA = 0;
        groupB = 1;
      }
    } else {
      groupA = 1;
      groupB = 0;
    }
  }
  /*if (groupA) printf("p[%d] Group A\n",GAme);
  if (groupB) printf("p[%d] Group B\n",GAme);*/

  zproc = ga_cluster_procid_(&inode, &zero);
  map = GA[handle].mapc;
  blocks = GA[handle].nblock;
  dims = GA[handle].dims;
  width = GA[handle].width;
  type = GA[handle].type;
  ndim = GA[handle].ndim;

  /* Check whether or not all nodes contain the same number
     of processors. */
  if (nnodes*nprocs == ga_nnodes_())  {
    /* check to see if there is any buffer space between the data
       associated with each processor that needs to be zeroed out
       before performing the merge */
    if (zproc == GAme) {
      nsize = 0;
      for (i=0; i<nprocs; i++) {
        /* Find out from mapc data how many elements are supposed to be located
           on this processor. Start by converting processor number to indices */
        itmp = i;
        for (j=0; j<ndim; j++) {
          index[j] = itmp%(Integer)blocks[j];
          itmp = (itmp - index[j])/(Integer)blocks[j];
        }

        nelem = 1;
        count = 0;
        for (j=0; j<ndim; j++) {
          if (index[j] < (Integer)blocks[j]-1) {
            nelem *= (Integer)(map[index[j]+1+count] - map[index[j]+count]
                   + 2*width[j]);
          } else {
            nelem *= (Integer)(dims[j] - map[index[j]+count] + 1 + 2*width[j]);
          }
          count += (Integer)blocks[j];
        }
        /* We now have the total number of elements located on this processor.
           Find out if the location of the end of this data set matches the
           origin of the data on the next processor. If not, then zero data in
           the gap. */
        nelem *= GAsizeof(type);
        nsize += (int)nelem;
        bptr = GA[handle].ptr[ga_cluster_procid_(&inode, &i)];
        bptr += nelem;
        if (i<nprocs-1) {
          j = i+1;
          nptr = GA[handle].ptr[ga_cluster_procid_(&inode, &j)];
          if (bptr != nptr) {
            bytes = (long)nptr - (long)bptr;
            nsize += (int)bytes;
            bzero(bptr, bytes);
          }
        }
      }
      /* The gaps have now been zeroed out. Begin exchange of data */
      /* This algorith is based on the armci_msg_barrier code */
      /* Locate pointers to beginning and end of non-zero data */
      for (i=0;i<ndim;i++) index[i] = (Integer)GA[handle].first[i];
      i = nga_locate_(g_a, index, &id);
      gam_Loc_ptr(id, handle, GA[handle].first, &fptr);
      for (i=0;i<ndim;i++) index[i] = (Integer)GA[handle].last[i];
      slength = GA[handle].shm_length;
      if(nnodes>1){
        if(!powof2nodes && inode < 2*sizeB && groupA) {
          ilast = inode + 1;
          next_nodel = ga_cluster_procid_(&ilast, &zero);
        } else if (groupB) {
          ilast = inode - 1;
          next_nodel = ga_cluster_procid_(&ilast, &zero);
        }
        ilast = ((int)pow(2,(LnB-1)))^inode;
        /*printf("p[%d] Value of next nodel: %d\n",GAme,next_nodel);*/
        /*three step exchange if num of nodes is not pow of 2*/
        /*divide _nodes_ into two sets, first set "pow2" will have a power of 
         *two nodes, the second set "not-pow2" will have the remaining.
         *Each node in the not-pow2 set will have a pair node in the pow2 set.
         *Step-1:each node in pow2 set with a pair in not-pow2 set first recvs 
         *      :a message from its pair in not-pow2. 
         *step-2:All nodes in pow2 do a Rercusive Doubling based Pairwise exng.
         *step-3:Each node in pow2 with a pair in not-pow2 snds msg to its 
         *      :pair node.
         *if num of nodes a pow of 2, only step 2 executed
         */
        if(/*ilast>inode &&*/ groupA){ /*the pow2 set of procs*/
          /* Use actual index of processor you are recieving from in group B
           * and perform first exchange (for non-power of 2) */
          if(!powof2nodes && inode < 2*sizeB){ /*step 1*/
            dstn = (char *)&rlength;
            armci_msg_rcv(armci_tag, dstn,4,NULL,next_nodel);
            if (GAme > next_nodel) {
              dstn = fptr - rlength;
            } else {
              dstn = fptr + slength;
            }
            armci_msg_rcv(armci_tag, dstn,rlength,NULL,next_nodel);
            if (GAme > next_nodel)
              fptr -= rlength;
            slength += rlength;
          }
          /* Assign inode = new_inode */
          if (inode < 2*sizeB) {
            new_inode  = inode/2;
          } else {
            new_inode  = inode - sizeB;
          }
          /*LnB=1;*/ /*BJP*/
          for(i=0;i<LnB-1;i++){ /*step 2*/
            next=((int)pow(2,i))^new_inode;
            if(next>=0 && next<new_nnodes){
              /* Translate back from relative_next_node to actual_next_node */
              if (next < sizeB)
                inext = (Integer)2*next;
              else
                inext = (Integer)(next+sizeB);
              next_node = ga_cluster_procid_(&inext, &zero);
              srcp = (char *)&slength;
              dstn = (char *)&rlength;
              if(next_node > GAme){
                armci_msg_snd(armci_tag, srcp,4,next_node);
                armci_msg_rcv(armci_tag, dstn,4,NULL,next_node);
              }
              else{
                /*would we gain anything by doing a snd,rcv instead of rcv,snd*/
                armci_msg_rcv(armci_tag, dstn,4,NULL,next_node);
                armci_msg_snd(armci_tag, srcp,4,next_node);
              }
              srcp = fptr;
              if (GAme > next_node) {
                dstn = fptr - rlength;
              } else {
                dstn = fptr + slength;
              }
              /* Translate back from relative_next_node to actual_next_node */
              if(next_node > GAme){
                armci_msg_snd(armci_tag, srcp,slength,next_node);
                armci_msg_rcv(armci_tag, dstn,rlength,NULL,next_node);
              }
              else{
                /*would we gain anything by doing a snd,rcv instead of rcv,snd*/
                armci_msg_rcv(armci_tag, dstn,rlength,NULL,next_node);
                armci_msg_snd(armci_tag, srcp,slength,next_node);
              }
              if (GAme > next_node)
                fptr -= rlength;
              slength += rlength;
            }
          }
              /* Use actual index of processor that you already recieved from
               * and that you will be sending to in group B*/
          if(!powof2nodes && inode < 2*sizeB){ /*step 3*/
            srcp = GA[handle].ptr[GAme];
            armci_msg_snd(armci_tag, srcp,nsize,next_nodel);
          }
        }
        else if (groupB) {
          /* Send data from group B to group A and then wait to
           * recieve data from group A to group B */
          if(!powof2nodes){
            /* printf("p[%d] Sending (1) data to %d\n",GAme,next_nodel); */
            srcp = (char *)&slength;
            armci_msg_snd(armci_tag, srcp,4,next_nodel);
            srcp = fptr;
            armci_msg_snd(armci_tag, srcp,slength,next_nodel);
            dstn = GA[handle].ptr[GAme];
            rlength = nsize;
            armci_msg_rcv(armci_tag, dstn,rlength,NULL,next_nodel);
            /*printf("p[%d] Recieved (2) data from %d\n",GAme,next_nodel);*/
          }
        }
      }
      /*printf("p[%d] About to execute armci_msg_gop_scope\n",GAme);*/
      armci_msg_gop_scope(SCOPE_NODE,&dummy,1,"+",ARMCI_INT);
    } else {
      /*printf("p[%d] About to execute armci_msg_gop_scope\n",GAme);*/
      armci_msg_gop_scope(SCOPE_NODE,&dummy,1,"+",ARMCI_INT);
    }
    /*printf("p[%d] Executed armci_msg_gop_scope\n",GAme);*/
  } else {
    ga_merge_mirrored_(g_a);
  }

  GA_Default_Proc_Group = Save_default_group;
  if (local_sync_end) ga_sync_();
  GA_POP_NAME;
}
