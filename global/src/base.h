/*$Id: base.h,v 1.32 2004-10-20 21:25:58 vinod Exp $ */
extern int _max_global_array;
extern Integer *_ga_map;
extern Integer GAme, GAnproc;
extern Integer *GA_proclist;
extern int GA_Default_Proc_Group;
extern int* GA_Proc_list;
extern int* GA_inv_Proc_list;
extern int** GA_Update_Flags;
extern int* GA_Update_Signal;
extern short int _ga_irreg_flag; 

#define FNAM        31              /* length of array names   */
#define CACHE_SIZE  512             /* size of the cache inside GA DS*/

#ifdef __crayx1
#define __CRAYX1_PRAGMA _Pragma
#else
#define __CRAYX1_PRAGMA(_pragf)
#endif

typedef int ARMCI_Datatype;
#include "armci.h"
typedef struct {
       int mirrored;
       int map_nproc;
       int actv;
       int parent;
       int *map_proc_list;
       int *inv_map_proc_list;
#ifdef MPI
       ARMCI_Group group;
#endif
} proc_list_t;

typedef struct {
       short int  ndim;         /* number of dimensions                 */
       short int  irreg;        /* 0-regular; 1-irregular distribution  */
       int  type;               /* data type in array                   */
       int  actv;               /* activity status                      */
       int  size;               /* size of local data in bytes          */
       int  elemsize;           /* sizeof(datatype)                     */
       int  ghosts;             /* flag indicating presence of ghosts   */
       long lock;               /* lock                                 */
       long id;                 /* ID of shmem region / MA handle       */
       int  dims[MAXDIM];       /* global array dimensions              */
       int  chunk[MAXDIM];      /* chunking                             */
       int  nblock[MAXDIM];     /* number of blocks per dimension       */
       int  width[MAXDIM];      /* boundary cells per dimension         */
       int  first[MAXDIM];      /* (Mirrored only) first local element  */
       int  last[MAXDIM];       /* (Mirrored only) last local element   */
       int  shm_length;         /* (Mirrored only) local shmem length   */
       Integer lo[MAXDIM];      /* top/left corner in local patch       */
       double scale[MAXDIM];    /* nblock/dim (precomputed)             */
       char **ptr;              /* arrays of pointers to remote data    */
       int  *mapc;              /* block distribution map               */
       char name[FNAM+1];       /* array name                           */
       int p_handle;            /* pointer to processor list for array  */
       double *cache;           /* store for frequently accessed ptrs   */
       int corner_flag;         /* flag for updating corner ghost cells */
} global_array_t;

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


#define ERR_STR_LEN 256               /* length of string for error reporting */
static char err_string[ ERR_STR_LEN]; /* string for extended error reporting */

/**************************** MACROS ************************************/


#define ga_check_handleM(g_a, string) \
{\
    if(GA_OFFSET+ (*g_a) < 0 || GA_OFFSET+(*g_a) >=_max_global_array){ \
      sprintf(err_string, "%s: INVALID ARRAY HANDLE", string);         \
      ga_error(err_string, (*g_a));                                    \
    }\
    if( ! (GA[GA_OFFSET+(*g_a)].actv) ){                               \
      sprintf(err_string, "%s: ARRAY NOT ACTIVE", string);             \
      ga_error(err_string, (*g_a));                                    \
    }                                                                  \
}

/* this macro finds cordinates of the chunk of array owned by processor proc */
#define ga_ownsM_no_handle(ndim, dims, nblock, mapc, proc, lo, hi)             \
{                                                                              \
   Integer _loc, _nb, _d, _index, _dim=ndim,_dimstart=0, _dimpos;              \
   for(_nb=1, _d=0; _d<_dim; _d++)_nb *= nblock[_d];                           \
   if(proc > _nb - 1 || proc<0){                                               \
      __CRAYX1_PRAGMA("_CRI novector");                                               \
           for(_d=0; _d<_dim; _d++){                                           \
         lo[_d] = (Integer)0;                                                  \
         hi[_d] = (Integer)-1;}                                                \
   }                                                                           \
   else{                                                                       \
         _index = proc;                                                        \
         if(GA_inv_Proc_list) _index = GA_inv_Proc_list[proc];                 \
      __CRAYX1_PRAGMA("_CRI novector");                                               \
         for(_d=0; _d<_dim; _d++){                                             \
             _loc = _index% nblock[_d];                                        \
             _index  /= nblock[_d];                                            \
             _dimpos = _loc + _dimstart; /* correction to find place in mapc */\
             _dimstart += nblock[_d];                                          \
             lo[_d] = mapc[_dimpos];                                           \
             if(_loc==nblock[_d]-1)hi[_d]=dims[_d];                            \
             else hi[_d] = mapc[_dimpos+1]-1;                                  \
         }                                                                     \
   }                                                                           \
}

/* this macro finds cordinates of the chunk of array owned by processor proc */
#define ga_ownsM(ga_handle, proc, lo, hi)                                      \
  ga_ownsM_no_handle(GA[ga_handle].ndim, GA[ga_handle].dims,                   \
                     GA[ga_handle].nblock, GA[ga_handle].mapc,                 \
                     proc,lo, hi )

/* this macro computes the strides on both the remote and local
   processors that map out the data. ld and ldrem are the physical dimensions
   of the memory on both the local and remote processors. */
#define gam_setstride(ndim, size, ld, ldrem, stride_rem, stride_loc){\
  int _i;                                                            \
  stride_rem[0]= stride_loc[0] = (int)size;                          \
  __CRAYX1_PRAGMA("_CRI novector");                                         \
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
  __CRAYX1_PRAGMA("_CRI novector");                                         \
  for(_d=0,*pelems=1; _d< ndim;_d++)  *pelems *= hi[_d]-lo[_d]+1;    \
}

#define gam_ComputeCount(ndim, lo, hi, count){                       \
  int _d;                                                            \
  __CRAYX1_PRAGMA("_CRI novector");                                         \
  for(_d=0; _d< ndim;_d++) count[_d] = (int)(hi[_d]-lo[_d])+1;       \
}

#define ga_RegionError(ndim, lo, hi, val){                           \
  int _d, _l;                                                        \
  char *str= "cannot locate region: ";                               \
  sprintf(err_string, str);                                          \
  _d=0;                                                              \
  _l = strlen(str);                                                  \
  sprintf(err_string+_l, "[%ld:%ld ",(long)lo[_d],(long)hi[_d]);     \
  _l=strlen(err_string);                                             \
  __CRAYX1_PRAGMA("_CRI novector");                                         \
  for(_d=0; _d< ndim-1; _d++){                                       \
    sprintf(err_string+_l, ",%ld:%ld ",(long)lo[_d],(long)hi[_d]);   \
    _l=strlen(err_string);                                           \
  }                                                                  \
  sprintf(err_string+_l, "]");                                       \
  _l=strlen(err_string);                                             \
  ga_error(err_string, val);                                         \
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
      if (_p_handle != 0) {                                                   \
        _iproc = proc;                                                        \
      } else {                                                                \
        _iproc = PGRP_LIST[_p_handle].inv_map_proc_list[proc];                \
      }                                                                       \
      gaCheckSubscriptM(subscript, _lo, _hi, GA[g_handle].ndim);              \
  __CRAYX1_PRAGMA("_CRI novector");                                         \
      for(_d=0; _d < _last; _d++)            {                                \
          _w = GA[g_handle].width[_d];                                        \
          _offset += (subscript[_d]-_lo[_d]+_w) * _factor;                    \
          _factor *= _hi[_d] - _lo[_d]+1+2*_w;                                \
      }                                                                       \
      _offset += (subscript[_last]-_lo[_last]+GA[g_handle].width[_last])      \
               * _factor;                                                     \
      *(ptr_loc) =  GA[g_handle].ptr[_iproc]+_offset*GA[g_handle].elemsize;   \
}

#define ga_check_regionM(g_a, ilo, ihi, jlo, jhi, string){                     \
   if (*(ilo) <= 0 || *(ihi) > GA[GA_OFFSET + *(g_a)].dims[0] ||               \
       *(jlo) <= 0 || *(jhi) > GA[GA_OFFSET + *(g_a)].dims[1] ||               \
       *(ihi) < *(ilo) ||  *(jhi) < *(jlo)){                                   \
       sprintf(err_string,"%s:req(%ld:%ld,%ld:%ld) out of range (1:%ld,1:%ld)",\
               string, (long)*(ilo), (long)*(ihi), (long)*(jlo), (long)*(jhi), \
               (long)GA[GA_OFFSET + *(g_a)].dims[0],                           \
               (long)GA[GA_OFFSET + *(g_a)].dims[1]);                          \
       ga_error(err_string, *(g_a));                                           \
   }                                                                           \
}

#define gaCheckSubscriptM(subscr, lo, hi, ndim)                                \
{                                                                              \
Integer _d;                                                                    \
  __CRAYX1_PRAGMA("_CRI novector");                                         \
   for(_d=0; _d<  ndim; _d++)                                                  \
      if( subscr[_d]<  lo[_d] ||  subscr[_d]>  hi[_d]){                        \
        sprintf(err_string,"check subscript failed:%ld not in (%ld:%ld) dim=", \
                  (long)subscr[_d],  (long)lo[_d],  (long)hi[_d]);             \
          ga_error(err_string, _d);                                            \
      }\
}
