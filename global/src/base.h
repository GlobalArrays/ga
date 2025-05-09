/*$Id: base.h,v 1.40.2.4 2007/12/18 18:41:27 d3g293 Exp $ */
#include "armci.h"
#include "gaconfig.h"
#include "typesf2c.h"

#ifdef MSG_COMMS_MPI
#include <mpi.h>
#include "ga-mpi.h"
#endif

extern int _max_global_array;
extern Integer GAme, GAnproc;
extern int GA_Default_Proc_Group;
extern int** GA_Update_Flags;
extern int* GA_Update_Signal;
extern short int _ga_irreg_flag; 
extern Integer GA_Debug_flag;

#define FNAM        31              /* length of array names   */
#define CACHE_SIZE  512             /* size of the cache inside GA DS*/

/* #define USE_GA_MALLOC */
enum data_distribution {REGULAR, BLOCK_CYCLIC, SCALAPACK, TILED, TILED_IRREG};

typedef int ARMCI_Datatype;
typedef struct {
       int mirrored;
       int map_nproc;
       int actv;
       int parent;
       int *map_proc_list;
       int *inv_map_proc_list;
#ifdef MSG_COMMS_MPI
       ARMCI_Group group;
#endif
} proc_list_t;

typedef Integer C_Integer;
typedef armci_size_t C_Long;

typedef struct cache_struct{
  int lo[MAXDIM];
  int hi[MAXDIM];
  void* cache_buf;
  struct cache_struct *next;
} cache_struct_t;

typedef struct {
       short int  ndim;             /* number of dimensions                 */
       short int  irreg;            /* 0-regular; 1-irregular distribution  */
       int  type;                   /* data type in array                   */
       int  actv;                   /* activity status, GA is allocated     */
       int  actv_handle;            /* handle is created                    */
       C_Long   size;               /* size of local data in bytes          */
       int  elemsize;               /* sizeof(datatype)                     */
       int  ghosts;                 /* flag indicating presence of ghosts   */
       long lock;                   /* lock                                 */
       long id;                     /* ID of shmem region / MA handle       */
       C_Integer  dims[MAXDIM];     /* global array dimensions              */
       C_Integer  chunk[MAXDIM];    /* chunking                             */
       int  nblock[MAXDIM];         /* number of blocks per dimension in    */
                                    /* processor grid                       */
       C_Integer  width[MAXDIM];    /* boundary cells per dimension         */
       C_Integer  first[MAXDIM];    /* (Mirrored only) first local element  */
       C_Integer  last[MAXDIM];     /* (Mirrored only) last local element   */
       C_Long  shm_length;          /* (Mirrored only) local shmem length   */
       C_Integer  lo[MAXDIM];       /* top/left corner in local patch       */
       double scale[MAXDIM];        /* nblock/dim (precomputed)             */
       char **ptr;                  /* arrays of pointers to remote data    */
       C_Integer  *mapc;            /* block distribution map               */
       char name[FNAM+1];           /* array name                           */
       int p_handle;                /* pointer to processor list for array  */
       double *cache;               /* store for frequently accessed ptrs   */
       int corner_flag;             /* flag for updating corner ghost cells */
       int distr_type;              /* tag for data distribution type       */
       C_Integer block_dims[MAXDIM];/* array of block dimensions            */
       C_Integer num_blocks[MAXDIM];/* number of blocks in each dimension   */
       C_Integer block_total;       /* total number of blocks in array      */
                                    /* using restricted arrays              */
       C_Integer *rstrctd_list;     /* list of processors with data         */
       C_Integer num_rstrctd;       /* number of processors with data       */
       C_Integer has_data;          /* flag that processor has data         */
       C_Integer rstrctd_id;        /* rank of processor in restricted list */
       C_Integer *rank_rstrctd;     /* ranks of processors with data        */
                                    /* Properties                           */
       int property;                /* property type for GA                 */
       Integer *old_mapc;           /* copy of original map                 */
       int old_nblock[MAXDIM];      /* copy of original nblock array        */
       int old_handle;              /* original group handle                */
       int old_lo[MAXDIM];          /* original lo array                    */
       int old_chunk[MAXDIM];       /* original chunk array                 */
#ifdef ENABLE_CHECKPOINT
       int record_id;               /* record id for writing ga to disk     */
#endif
       /* new */
       int read_cache;              /* flag for read only pointer in cache  */
       cache_struct_t *cache_head;  /* linked list of cached reads          */
       int mem_dev_set;             /* flag for setting memory device       */
       char mem_dev[FNAM+1];        /* memory device type                   */
       int overlay;                 /* GA uses memory from another GA       */

} global_array_t;

enum property_type { NO_PROPERTY,
                     READ_ONLY,
                     READ_CACHE /* new */
};

extern global_array_t *_ga_main_data_structure; 
extern proc_list_t *_proc_list_main_data_structure; 
/*\
 *The following statement had to be moved here because of a problem in the c
 *compiler on SV1. The problem is that when a c file is compiled with a 
 *-htaskprivate option on SV1, all global objects are given task-private status
 *even static variables are supposed to be initialized and given a task-private
 *memory/status. Somehow SV1 fails to do this for global variables that are 
 *initialized during declaration.
 *So to handle that,we cannot initialize global variables to be able to run 
 *on SV1.
\*/
extern global_array_t *GA;
extern proc_list_t *PGRP_LIST;

/*\
 * Copy of the world communicator used by GA so that applications using MPI
 * libraries will not use the same communicator that GA uses internally (at
 * least for the world communictor)
\*/
#ifdef MSG_COMMS_MPI
extern MPI_Comm GA_MPI_World_comm_dup;
#endif

#define ERR_STR_LEN 256               /* length of string for error reporting */

/**************************** MACROS ************************************/


#define ga_check_handleM(g_a, string) \
{\
    if(GA_OFFSET+ (g_a) < 0 || GA_OFFSET+(g_a) >=_max_global_array){   \
      char err_string[ERR_STR_LEN];                                    \
      sprintf(err_string, "%s: INVALID ARRAY HANDLE", string);         \
      pnga_error(err_string, (g_a));                                   \
    }                                                                  \
    if( ! (GA[GA_OFFSET+(g_a)].actv) ){                                \
      char err_string[ERR_STR_LEN];                                    \
      sprintf(err_string, "%s: ARRAY NOT ACTIVE", string);             \
      pnga_error(err_string, (g_a));                                   \
    }                                                                  \
}

/* this macro finds coordinates of the chunk of array owned by processor proc */
#define ga_ownsM_no_handle(ndim, dims, nblock, mapc, proc, lo, hi)             \
{                                                                              \
   Integer _loc, _nb, _d, _index, _dim=ndim,_dimstart=0, _dimpos;              \
   for(_nb=1, _d=0; _d<_dim; _d++)_nb *= (Integer)nblock[_d];                  \
   if((Integer)proc > _nb - 1 || proc<0){                                      \
        for(_d=0; _d<_dim; _d++){                                              \
         lo[_d] = (Integer)0;                                                  \
         hi[_d] = (Integer)-1;}                                                \
   }                                                                           \
   else{                                                                       \
         _index = proc;                                                        \
         for(_d=0; _d<_dim; _d++){                                             \
             _loc = _index% (Integer)nblock[_d];                               \
             _index  /= (Integer)nblock[_d];                                   \
             _dimpos = _loc + _dimstart; /* correction to find place in mapc */\
             _dimstart += (Integer)nblock[_d];                                 \
             lo[_d] = (Integer)mapc[_dimpos];                                  \
             if (_loc==nblock[_d]-1) hi[_d]=dims[_d];                          \
             else hi[_d] = mapc[_dimpos+1]-1;                                  \
         }                                                                     \
   }                                                                           \
}

/* this macro finds the block indices for a given block */
#define gam_find_block_indices(ga_handle,nblock,index) {                       \
  int _itmp, _i;                                                               \
  int _ndim = GA[ga_handle].ndim;                                              \
  _itmp = nblock;                                                              \
  index[0] = _itmp%GA[ga_handle].num_blocks[0];                                \
  for (_i=1; _i<_ndim; _i++) {                                                 \
    _itmp = (_itmp-index[_i-1])/GA[ga_handle].num_blocks[_i-1];                \
    index[_i] = _itmp%GA[ga_handle].num_blocks[_i];                            \
  }                                                                            \
}

/* this macro finds the ScaLAPACK indices for a given processor */
/* gam_find_proc_indices(ga_handle,proc,index) */
#define gam_find_tile_proc_indices(ga_handle,proc,index) {                     \
  Integer _itmp, _i;                                                           \
  Integer _ndim = GA[ga_handle].ndim;                                          \
  _itmp = proc;                                                                \
  index[0] = _itmp%GA[ga_handle].nblock[0];                                    \
  for (_i=1; _i<_ndim; _i++) {                                                 \
    _itmp = (_itmp-index[_i-1])/GA[ga_handle].nblock[_i-1];                    \
    index[_i] = _itmp%GA[ga_handle].nblock[_i];                                \
  }                                                                            \
}
/*
#define gam_find_proc_indices(ga_handle,proc,index) {                          \
  Integer _itmp, _i;                                                           \
  Integer _ndim = GA[ga_handle].ndim;                                          \
  _itmp = proc;                                                                \
  index[_ndim-1] = _itmp%GA[ga_handle].nblock[_ndim-1];                        \
  for (_i=_ndim-2; _i>=0; _i--) {                                              \
    _itmp = (_itmp-index[_i+1])/GA[ga_handle].nblock[_i+1];                    \
    index[_i] = _itmp%GA[ga_handle].nblock[_i];                                \
  }                                                                            \
}
*/
#define gam_find_proc_indices(ga_handle,proc,index) {                          \
  Integer _itmp, _i;                                                           \
  Integer _ndim = GA[ga_handle].ndim;                                          \
  _itmp = proc;                                                                \
  index[0] = _itmp%GA[ga_handle].nblock[0];                        \
  for (_i=1; _i<_ndim; _i++) {                                              \
    _itmp = (_itmp-index[_i-1])/GA[ga_handle].nblock[_i-1];                    \
    index[_i] = _itmp%GA[ga_handle].nblock[_i];                                \
  }                                                                            \
}

/* this macro finds cordinates of the chunk of array owned by processor proc
 * ga_handle: global array handle
 * proc: processor (or block) index
 * lo: lower indices of elements owned by processor (or block)
 * hi: upper indices of elements owned by processor (or block)
 */
#define ga_ownsM(ga_handle, proc, lo, hi)                                      \
{                                                                              \
  if (GA[ga_handle].distr_type == REGULAR) {                                   \
    if (GA[ga_handle].num_rstrctd == 0) {                                      \
      ga_ownsM_no_handle(GA[ga_handle].ndim, GA[ga_handle].dims,               \
                         GA[ga_handle].nblock, GA[ga_handle].mapc,             \
                         proc,lo, hi )                                         \
    } else {                                                                   \
      if (proc < GA[ga_handle].num_rstrctd) {                                  \
        ga_ownsM_no_handle(GA[ga_handle].ndim, GA[ga_handle].dims,             \
                           GA[ga_handle].nblock, GA[ga_handle].mapc,           \
                           proc,lo, hi )                                       \
      } else {                                                                 \
        int _i;                                                                \
        int _ndim = GA[ga_handle].ndim;                                        \
        for (_i=0; _i<_ndim; _i++) {                                           \
          lo[_i] = 0;                                                          \
          hi[_i] = -1;                                                         \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } else if (GA[ga_handle].distr_type == BLOCK_CYCLIC ||                       \
      GA[ga_handle].distr_type == SCALAPACK ||                                 \
      GA[ga_handle].distr_type == TILED) {                                     \
    int _index[MAXDIM];                                                        \
    int _i;                                                                    \
    int _ndim = GA[ga_handle].ndim;                                            \
    gam_find_block_indices(ga_handle,proc,_index);                             \
    for (_i=0; _i<_ndim; _i++) {                                               \
      lo[_i] = _index[_i]*GA[ga_handle].block_dims[_i]+1;                      \
      hi[_i] = (_index[_i]+1)*GA[ga_handle].block_dims[_i];                    \
      if (hi[_i] > GA[ga_handle].dims[_i]) hi[_i]=GA[ga_handle].dims[_i];      \
    }                                                                          \
  } else if (GA[ga_handle].distr_type == TILED_IRREG) {                        \
    int _index[MAXDIM];                                                        \
    int _i;                                                                    \
    int _ndim = GA[ga_handle].ndim;                                            \
    int _offset = 0;                                                           \
    gam_find_block_indices(ga_handle,proc,_index);                             \
    for (_i=0; _i<_ndim; _i++) {                                               \
      lo[_i] = GA[ga_handle].mapc[_offset+_index[_i]];                         \
      if (_index[_i] < GA[ga_handle].num_blocks[_i]-1) {                       \
        hi[_i] = GA[ga_handle].mapc[_offset+_index[_i]+1]-1;                   \
      } else {                                                                 \
        hi[_i] = GA[ga_handle].dims[_i];                                     \
      }                                                                        \
      _offset += GA[ga_handle].num_blocks[_i];                                   \
    }                                                                          \
  }                                                                            \
}

/* this macro finds the block index corresponding to a given set of indices */
#define gam_find_block_from_indices(ga_handle,nblock,index) {                  \
  int _ndim = GA[ga_handle].ndim;                                              \
  int _i;                                                                      \
  nblock = index[_ndim-1];                                                     \
  for (_i=_ndim-2; _i >= 0; _i--) {                                            \
    nblock  = nblock*GA[ga_handle].num_blocks[_i]+index[_i];                   \
  }                                                                            \
}

/* this macro finds the proc that owns a given set block indices
   using the ScaLAPACK data distribution */
/* gam_find_proc_from_sl_indices(ga_handle,proc,index) */
#define gam_find_tile_proc_from_indices(ga_handle,proc,index) {                \
  int _ndim = GA[ga_handle].ndim;                                              \
  int _i;                                                                      \
  Integer _index2[MAXDIM];                                                     \
  for (_i=0; _i<_ndim; _i++) {                                                 \
    _index2[_i] = index[_i]%GA[ga_handle].nblock[_i];                          \
  }                                                                            \
  proc = _index2[_ndim-1];                                                     \
  for (_i=_ndim-2; _i >= 0; _i--) {                                            \
    proc = proc*GA[ga_handle].nblock[_i]+_index2[_i];                          \
  }                                                                            \
}
#define gam_find_proc_from_sl_indices(ga_handle,proc,index) {                  \
  int _ndim = GA[ga_handle].ndim;                                              \
  int _i;                                                                      \
  Integer _index2[MAXDIM];                                                     \
  for (_i=0; _i<_ndim; _i++) {                                                 \
    _index2[_i] = index[_i]%GA[ga_handle].nblock[_i];                          \
  }                                                                            \
  proc = _index2[_ndim-1];                                                           \
  for (_i=_ndim-2; _i >= 0; _i--) {                                               \
    proc = proc*GA[ga_handle].nblock[_i]+_index2[_i];                          \
  }                                                                            \
}

/* this macro computes the strides on both the remote and local
   processors that map out the data. ld and ldrem are the physical dimensions
   of the memory on both the local and remote processors. */
/* NEEDS C_INT64 CONVERSION */
#define gam_setstride(ndim, size, ld, ldrem, stride_rem, stride_loc){\
  int _i;                                                            \
  stride_rem[0]= stride_loc[0] = (int)size;                          \
  for(_i=0;_i<ndim-1;_i++){                                          \
    stride_rem[_i] *= (int)ldrem[_i];                                \
    stride_loc[_i] *= (int)ld[_i];                                   \
      stride_rem[_i+1] = stride_rem[_i];                             \
      stride_loc[_i+1] = stride_loc[_i];                             \
  }                                                                  \
}

/* Count total number of elmenents in array based on values of ndim,
      lo, and hi */
#define gam_CountElems(ndim, lo, hi, pelems){                        \
  int _d;                                                            \
  for(_d=0,*pelems=1; _d< ndim;_d++)  *pelems *= hi[_d]-lo[_d]+1;    \
}

/* NEEDS C_INT64 CONVERSION */
#define gam_ComputeCount(ndim, lo, hi, count){                       \
  int _d;                                                            \
  for(_d=0; _d< ndim;_d++) count[_d] = (int)(hi[_d]-lo[_d])+1;       \
}

#define ga_RegionError(ndim, lo, hi, val){                           \
  int _d, _l;                                                        \
  const char *str= "cannot locate region: ";                         \
  char err_string[ERR_STR_LEN];                                      \
  sprintf(err_string, "%s", str);                                    \
  _d=0;                                                              \
  _l = strlen(str);                                                  \
  sprintf(err_string+_l, "%s", GA[val+GA_OFFSET].name);              \
  _l = strlen(err_string);                                           \
  sprintf(err_string+_l, " [%ld:%ld ",(long)lo[_d],(long)hi[_d]);    \
  _l=strlen(err_string);                                             \
  for(_d=1; _d< ndim; _d++){                                         \
    sprintf(err_string+_l, ",%ld:%ld ",(long)lo[_d],(long)hi[_d]);   \
    _l=strlen(err_string);                                           \
  }                                                                  \
  sprintf(err_string+_l, "%s", "]");                                 \
  _l=strlen(err_string);                                             \
  pnga_error(err_string, val);                                       \
}

/*\ Just return pointer (ptr_loc) to location in memory of element with
 *  subscripts (subscript).
\*/
#define gam_Loc_ptr(proc, g_handle,  subscript, ptr_loc)                      \
{                                                                             \
Integer _offset=0, _d, _w, _factor=1, _last=GA[g_handle].ndim-1;              \
Integer _lo[MAXDIM], _hi[MAXDIM], _p_handle, _iproc;                          \
                                                                              \
      ga_ownsM(g_handle, proc, _lo, _hi);                                     \
      _p_handle = GA[g_handle].p_handle;                                      \
      _iproc = proc;                                                          \
      gaCheckSubscriptM(subscript, _lo, _hi, GA[g_handle].ndim);              \
      for(_d=0; _d < _last; _d++)            {                                \
          _w = (Integer)GA[g_handle].width[_d];                               \
          _offset += (subscript[_d]-_lo[_d]+_w) * _factor;                    \
          _factor *= _hi[_d] - _lo[_d]+1+2*_w;                                \
      }                                                                       \
      _offset += (subscript[_last]-_lo[_last]                                 \
               + (Integer)GA[g_handle].width[_last])                          \
               * _factor;                                                     \
      if (_p_handle == 0) {                                                   \
        _iproc = PGRP_LIST[_p_handle].inv_map_proc_list[_iproc];              \
      }                                                                       \
      if (GA[g_handle].num_rstrctd > 0)                                       \
        _iproc = GA[g_handle].rstrctd_list[_iproc];                           \
      *(ptr_loc) =  GA[g_handle].ptr[_iproc]+_offset*GA[g_handle].elemsize;   \
}

#define ga_check_regionM(g_a, ilo, ihi, jlo, jhi, string){                     \
   if (*(ilo) <= 0 || *(ihi) > GA[GA_OFFSET + *(g_a)].dims[0] ||               \
       *(jlo) <= 0 || *(jhi) > GA[GA_OFFSET + *(g_a)].dims[1] ||               \
       *(ihi) < *(ilo) ||  *(jhi) < *(jlo)){                                   \
       char err_string[ERR_STR_LEN];                                           \
       sprintf(err_string,"%s:req(%ld:%ld,%ld:%ld) out of range (1:%ld,1:%ld)",\
               string, (long)*(ilo), (long)*(ihi), (long)*(jlo), (long)*(jhi), \
               (long)GA[GA_OFFSET + *(g_a)].dims[0],                           \
               (long)GA[GA_OFFSET + *(g_a)].dims[1]);                          \
       pnga_error(err_string, *(g_a));                                         \
   }                                                                           \
}

#define gaCheckSubscriptM(subscr, lo, hi, ndim)                                \
{                                                                              \
Integer _d;                                                                    \
   for(_d=0; _d<  ndim; _d++)                                                  \
      if( subscr[_d]<  lo[_d] ||  subscr[_d]>  hi[_d]){                        \
        char err_string[ERR_STR_LEN];                                          \
        sprintf(err_string,"check subscript failed:%ld not in (%ld:%ld) dim=%d", \
                  (long)subscr[_d],  (long)lo[_d],  (long)hi[_d], (int)_d);    \
          pnga_error(err_string, _d);                                          \
      }\
}

extern void pna_access_block_grid_ptr(Integer g_a, Integer *index, void *ptr,
    Integer ld);
