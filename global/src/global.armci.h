/*Wed Jan 25 10:25:49 PST 1995*/
#include "config.h"
  
#if !defined(__STDC__) || !defined(__cplusplus) && !defined(LINUX)
#  define volatile
#endif

#define MAX_REG     128             /* max number of shmem regions per array */
#define RESERVED    2*sizeof(long)  /* used for shmem buffer management */  
#define FNAM        31              /* length of Fortran names   */
#define FLEN        80              /* length of Fortran strings */
#define BUF_SIZE    4096            /* size of shmem buffer */ 
#define ERR_STR_LEN 256             /* length of string for error reporting */

#ifdef  CRAY_T3D
#       define ALLIGN_SIZE      32
#else
#       define ALLIGN_SIZE      128
#endif

#define MAX_PTR MAX_NPROC
#define MAPLEN  (MIN(GAnproc, MAX_NPROC) +MAXDIM)


typedef struct {
       int  ndim;               /* number of dimensions                 */
       int  dims[MAXDIM];       /* global array dimensions              */
       int  chunk[MAXDIM];      /* chunking                             */
       int  nblock[MAXDIM];     /* number of blocks per dimension       */
       double scale[MAXDIM];    /* nblock/dim (precomputed)             */
       char **ptr;              /* arrays of pointers to remote data    */
       int  *mapc;              /* block distribution map               */
       Integer type;            /* type of array                        */
       int  actv;               /* activity status                      */
       Integer lo[MAXDIM];      /* top/left corner in local patch       */
       Integer size;            /* size of local data in bytes          */
       int elemsize;            /* sizeof(datatype)                     */
       long lock;               /* lock                                 */
       long id;			/* ID of shmem region / MA handle       */
       char name[FNAM+1];       /* array name                           */
} global_array_t;


static global_array_t GA[MAX_ARRAYS]; 
static int max_global_array = MAX_ARRAYS;
Integer *map;       /* used in get/put/acc */
Integer *proclist;
extern Integer in_handler;                   /* set in interrupt handler*/


char err_string[ ERR_STR_LEN];        /* string for extended error reporting */
char *GA_name_stack[NAME_STACK_LEN];  /* stack for storing names of GA ops */ 
int  GA_stack_size=0;

/**************************** MACROS ************************************/

#define allign__(n, SIZE) \
        (((n)%SIZE) ? (n)+SIZE - (n)%SIZE: (n))

#define allign_size(n) allign__((long)(n), ALLIGN_SIZE)
#define allign_page(n) allign__((long)(n), PAGE_SIZE)

#define ga_check_handleM(g_a, string) \
{\
    if(GA_OFFSET+ (*g_a) < 0 || GA_OFFSET+(*g_a) >= max_global_array){ \
      sprintf(err_string, "%s: INVALID ARRAY HANDLE", string);         \
      ga_error(err_string, (*g_a));                                    \
    }\
    if( ! (GA[GA_OFFSET+(*g_a)].actv) ){                               \
      sprintf(err_string, "%s: ARRAY NOT ACTIVE", string);             \
      ga_error(err_string, (*g_a));                                    \
    }                                                                  \
}
      
/* this macro finds cordinates of the chunk of array owned by processor proc */
#define ga_ownsM(ga_handle, proc, lo, hi)                                      \
{                                                                              \
   Integer _loc, _nb, _d, _index, _dim=GA[ga_handle].ndim,_dimstart=0, _dimpos;\
   for(_nb=1, _d=0; _d<_dim; _d++)_nb *= GA[ga_handle].nblock[_d];             \
   if(proc > _nb - 1 || proc<0)for(_d=0; _d<_dim; _d++){                       \
         lo[_d] = (Integer)0;                                                  \
         hi[_d] = (Integer)-1;                                                 \
   }else{                                                                      \
         for(_d=0, _index = proc; _d<_dim; _d++){                              \
             _loc = _index% GA[ga_handle].nblock[_d];                          \
             _index  /= GA[ga_handle].nblock[_d];                              \
             _dimpos = _loc + _dimstart; /* correction to find place in mapc */\
             _dimstart += GA[ga_handle].nblock[_d];                            \
             lo[_d] = GA[ga_handle].mapc[_dimpos];                             \
             if(_loc==GA[ga_handle].nblock[_d]-1)hi[_d]=GA[ga_handle].dims[_d];\
             else hi[_d] = GA[ga_handle].mapc[_dimpos+1]-1;                    \
         }                                                                     \
   }                                                                           \
}


/*\ This macro computes index (place in ordered set) for the element
 *  identified by _subscript in ndim- dimensional array of dimensions _dim[]
 *  assume that first subscript component changes first
\*/
#define ga_ComputeIndexM(_index, _ndim, _subscript, _dims)                     \
{                                                                              \
  Integer  _i, _factor=1;                                                      \
  for(_i=0,*(_index)=0; _i<_ndim; _i++){                                       \
      *(_index) += _subscript[_i]*_factor;                                     \
      if(_i<_ndim-1)_factor *= _dims[_i];  \
  }                                                                            \
}


/*\ updates subscript corresponding to next element in a patch <lo[]:hi[]>
\*/
#define ga_UpdateSubscriptM(_ndim, _subscript, _lo, _hi, _dims)\
{                                                                              \
  Integer  _i;                                                                 \
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
  for(_i=0; _i<_ndim; _i++){                                                   \
       *_elems *= _hi[_i]-_lo[_i] +1;                                          \
       _subscript[_i] = _lo[_i];                                               \
  }                                                                            \
}


/*****************************************************************************/


#include "armci.h"


/* MA addressing */
DoubleComplex   *DCPL_MB;           /* double precision complex base address */
DoublePrecision *DBL_MB;            /* double precision base address */
Integer         *INT_MB;            /* integer base address */


/* cache numbers of GA/message-passing processes and ids */
static Integer GAme, GAnproc, GAmaster;
static Integer MPme, MPnproc;

static int GAinitialized = 0;
int ProcListPerm[MAX_NPROC];            /* permuted list of processes */
Integer local_buf_req=0;
Integer *NumRecReq = &local_buf_req;/* # received requests by data server */
                                    /* overwritten by shmem buf ptr if needed */
struct ga_stat_t GAstat = {0,0,0,0,0,0,0,0,0,0,0};
struct ga_bytes_t GAbytes ={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
long   *GAstat_arr;  

    
#ifdef CRAY_T3D
#      include <fortran.h>
#endif

/* set total limit (bytes) for memory usage per processor to "unlimited" */ 
static Integer GA_total_memory = -1;
static Integer GA_memory_limited = 0;


#if defined(__STDC__) || defined(__cplusplus)
# define ARGS_(s) s
#else
# define ARGS_(s) ()
#endif

extern logical gaDirectAccess ARGS_((Integer, int  ));
extern void ma_ga_get_ptr_ ARGS_((char **, char *));
extern Integer ma_ga_diff_ ARGS_((char *, char *));
extern void ma_ga_base_address_ ARGS_((Void*, Void**));
extern void ga_sort_scat ARGS_((Integer*,Void*,Integer*,Integer*,Integer*, Integer));
extern void ga_sort_gath_ ARGS_((Integer*, Integer*, Integer*, Integer*));


#undef ARGS_

#ifdef GA_TRACE
  static Integer     op_code;
#endif

#define FLUSH_CACHE 
