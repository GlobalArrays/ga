/*Wed Jan 25 10:25:49 PST 1995*/
  
#if !defined(__STDC__) || !defined(__cplusplus) && !defined(LINUX)
#  define volatile
#endif

#define MAX_REG     128             /* max number of shmem regions per array */
#define RESERVED    2*sizeof(long)  /* used for shmem buffer management */  
#define FNAM        31              /* length of Fortran names   */
#define FLEN        80              /* length of Fortran strings */
#define BUF_SIZE    4096            /* size of shmem buffer */ 
#define ERR_STR_LEN 200             /* length of string for error reporting */

#ifdef  CRAY_T3D
#       define ALLIGN_SIZE      32
#else
#       define ALLIGN_SIZE      128
#endif

#define MAX_PTR MAX_NPROC
#define   MAPLEN  (MIN(GAnproc, MAX_NPROC) +2)


typedef struct {
       int  dims[2];            /* global dimensions [i,j]              */
       int  chunk[2];           /* chunking                             */
       int  nblock[2];          /* number of chunks (blocks)            */
       double scale[2];         /* nblock/dim (precomputed)             */
       char **ptr;              /* arrays of pointers to remote data    */
       int  *mapc;              /* block distribution map               */
       Integer type;            /* type of array                        */
       int  actv;               /* activity status                      */
       Integer ilo;             /* coordinates of local patch           */
       Integer jlo;
       Integer size;            /* size of local data in bytes          */
       long lock;               /* lock                                 */
       long id;			/* ID of shmem region / MA handle       */
       char name[FNAM+1];       /* array name                           */
} global_array_t;


static global_array_t GA[MAX_ARRAYS]; 
static int max_global_array = MAX_ARRAYS;
Integer map[MAX_NPROC][5];               /* used in get/put/acc */
extern Integer in_handler;               /* set in interrupt handler*/


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
      

#define ga_ownsM(ga_handle, proc, ilo,ihi, jlo, jhi)                           \
{                                                                              \
   Integer loc, iproc, jproc;                                                  \
   if(proc > GA[ga_handle].nblock[0] * GA[ga_handle].nblock[1] - 1 || proc<0){ \
         ilo = (Integer)0;    jlo = (Integer)0;                                \
         ihi = (Integer)-1;   jhi = (Integer)-1;                               \
   }else{                                                                      \
         jproc = proc/GA[ga_handle].nblock[0];                                 \
         iproc = proc%GA[ga_handle].nblock[0];                                 \
         loc = iproc;                                                          \
         ilo = GA[ga_handle].mapc[loc];  ihi = GA[ga_handle].mapc[loc+1] -1;   \
         /* correction to find the right spot in mapc*/                        \
         loc = jproc + GA[ga_handle].nblock[0];                                \
         jlo = GA[ga_handle].mapc[loc];   jhi = GA[ga_handle].mapc[loc+1] -1;  \
         if( iproc == GA[ga_handle].nblock[0] -1)  ihi = GA[ga_handle].dims[0];\
         if( jproc == GA[ga_handle].nblock[1] -1)  jhi = GA[ga_handle].dims[1];\
   }                                                                           \
}


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
