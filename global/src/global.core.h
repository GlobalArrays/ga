/*Wed Jan 25 10:25:49 PST 1995*/
  
#ifdef SUN
#  define volatile   
#endif

#ifdef KSR
#  define PRIVATE  __private 
#else
#  define PRIVATE  
#endif

#define MAX_ARRAYS  32             /* max number of global arrays */

#ifndef MAX_NPROC                  /* default max number of processors  */
#   ifdef PARAGON
#     define MAX_NPROC    1024
#   elif defined(DELTA)
#     define MAX_NPROC     512 
#   elif defined(SP1)
#     define MAX_NPROC     400 
#   elif defined(CRAY_T3D)
#     define MAX_NPROC     256 
#   elif defined(KSR)
#     define MAX_NPROC      80 
#   else
#     define MAX_NPROC      64     /* default for everything else */ 
#   endif 
#endif

#define MAX_REG     128             /* max number of shmem regions per array */
#define RESERVED    2*sizeof(long)  /* used for shmem buffer management */  
#define FNAM        35              /* length of Fortran names   */
#define FLEN        80              /* length of Fortran strings */
#define GA_OFFSET   1000            /* offset for handle numbering */
#define BUF_SIZE    4096 

#ifdef  SHMEM
#  define MAX_PTR MAX_NPROC
#else
#  define MAX_PTR 1
#endif


typedef struct {
       int  dims[2];            /* global dimensions [i,j]              */
       int  chunk[2];           /* chunking                             */
       int  nblock[2];          /* number of chunks (blocks)            */
       double scale[2];         /* nblock/dim (precomputed)             */
       char *ptr[MAX_PTR];      /* pointers to local/remote data        */
       int  mapc[MAX_NPROC+2];  /* block distribution map               */
       Integer type;            /* type of array                        */
       int  actv;               /* activity status                      */
       Integer ilo;             /* coordinates of local patch           */
       Integer jlo;
       Integer size;            /* size of local data in bytes          */
       long lock;               /* lock                                 */
       long id;			/* ID of shmem region / MA handle       */
       char name[FNAM+1];       /* array name                           */
} global_array;


PRIVATE static global_array GA[MAX_ARRAYS]; 
PRIVATE static int max_global_array = MAX_ARRAYS;
PRIVATE Integer map[MAX_NPROC][5];               /* used in get/put/acc */



#ifdef  CRAY_T3D
#define ALLIGN_SIZE      32 
#else
#define ALLIGN_SIZE      128
#endif

#define ERR_STR_LEN 200
char err_string[ ERR_STR_LEN];         /* string for extended error reporting */

char *GA_name_stack[NAME_STACK_LEN];   /* stack for storing names of GA ops */ 
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

#define GAsizeofM(type)  ( (type)==MT_F_DBL? sizeof(DoublePrecision): \
                           (type)==MT_F_INT? sizeof(Integer): 0)


/**************** Shared Memory and Mutual Exclusion Co.  **************/
#ifdef SYSV
       /* SHARED MEMORY */
       PRIVATE static  volatile int    barrier_size;
       PRIVATE static  volatile int    *barrier, *barrier1;
       PRIVATE static  long            shmSIZE, shmID;
       PRIVATE static  DoublePrecision *shmBUF;
#      ifdef KSR
           /* lock entire block owned by proc
            * Note that this to work in data server mode we need use
            * (proc - cluster_master) instead of proc
            */
#          define LOCK(g_a, proc, x)    _gspwt(GA[GA_OFFSET + g_a].ptr[(proc)])
#          define UNLOCK(g_a, proc, x)    _rsp(GA[GA_OFFSET + g_a].ptr[(proc)])
#          define UNALIGNED(x)    (((unsigned long) (x)) % sizeof(long))
           typedef __align128 unsigned char subpage[128];
#      else
           /* define LOCK OPERATIONS using SYSV semaphores */
#          include "semaphores.h"
#          define NUM_SEM  SEMMSL
#          define ARR_SEM  (NUM_SEM -1) /* num of sems for locking arrays */
#          define MUTEX    ARR_SEM      /* semid  for synchronization */
#          define LOCK(g_a,proc, x)\
                  P(((proc)-cluster_master)%ARR_SEM)
#          define UNLOCK(g_a,proc, x)\
                   V(((proc)-cluster_master)%ARR_SEM)
#      endif
#else
#      ifdef CRAY_T3D
#          include <limits.h>
#          include <mpp/shmem.h>
#          define INVALID (long)(_INT_MIN_64 +1)
#          define LOCK_AND_GET(x, y, proc) \
              while( ((x) = shmem_swap((long*)(y),INVALID,(proc)))== INVALID)
#          define UNLOCK_AND_PUT(x, y, proc) shmem_swap((long*)(y),(x),(proc))

#          define LOCK(g_a, proc, x) \
              while( shmem_swap(&GA[GA_OFFSET +g_a].lock,INVALID,(proc))\
                     == INVALID)

#          define UNLOCK(g_a, proc, x)\
                 shmem_swap(&GA[GA_OFFSET +g_a].lock, 1, (proc))
#      elif defined(NX) || defined(SP1)
#            include "interrupt.h"
             extern Integer in_handler;
             long oldmask;
#            define LOCK(g_a, proc, x) \
                    { if(  in_handler == 0) ga_mask(1L, &oldmask) }
#            ifdef PARAGON
#              define UNLOCK(g_a,proc, x) \
                    { if( in_handler == 0) ga_mask(0L, &oldmask) }
#            else
#              define UNLOCK(g_a,proc, x) \
                    { if( in_handler == 0) ga_mask(oldmask, &oldmask) }
#            endif
#      endif
#endif

#include "mem.ops.h"
/************************************************************************/

/* cache coherency ? */
#ifndef CRAY_T3D
#       define  FLUSH_CACHE
#       define  FLUSH_CACHE_LINE(x) 
#else
#       define FLUSH_CACHE shmem_udcflush()
#       define FLUSH_CACHE_LINE(x)   shmem_udcflush_line((long*)(x))
#endif


#ifdef KSR
        int    KSRbarrier_mem_req();
        void   KSRbarrier(), KSRbarrier_init(int, int, int, char*);
#endif


/* MA addressing */
DoublePrecision *DBL_MB;            /* double precision base address */
Integer         *INT_MB;            /* integer base address */


PRIVATE static int GAinitialized = 0;
PRIVATE static Integer GAme, GAnproc, GAmaster;
PRIVATE static Integer MPme, MPnproc;
int ProcListPerm[MAX_NPROC];            /* permuted list of processes */
#if defined(DATA_SERVER)
    Integer *NumRecReq;                 /* # received requests by data server */
#else
    Integer local_buf_req;
    Integer *NumRecReq = &local_buf_req;/* # received requests by data server */
#endif
    
#if !(defined(SGI)|| defined(AIX))
#   ifndef CRAY_T3D
       int  fprintf();
#   else
#      include <fortran.h>
#   endif
#endif

/* set total limit (bytes) for memory usage per processor to "unlimited" */ 
static Integer GA_total_memory = -1;
static Integer GA_memory_limited = 0;



#if defined(__STDC__) || defined(__cplusplus)
# define ARGS_(s) s
#else
# define ARGS_(s) ()
#endif

extern char* strcpy      ARGS_((char*, char*));
extern void srand        ARGS_((Integer));
extern void srandom      ARGS_((Integer));
extern char *malloc      ARGS_((int));
extern logical gaDirectAccess ARGS_((Integer ));
extern void ma__get_ptr_ ARGS_((char **, char *));
extern Integer ma__diff_ ARGS_((char *, char *));
extern void ma__base_address_ ARGS_((Void*, Void**));

extern char *Create_Shared_Region ARGS_((long *idlist, long *size, long *offset)
);
extern char *Attach_Shared_Region ARGS_((long *idlist, long size, long *offset))
;
extern void Free_Shmem_Ptr ARGS_((Integer id, Integer size, char *addr));
extern long Detach_Shared_Region ARGS_((long id, long size, char *addr));
extern long Delete_Shared_Region ARGS_((long id));
extern long Delete_All_Regions ARGS_(( void));

extern Void* memcpy ARGS_((Void*, Void*, Integer));
extern double sqrt ARGS_((double));
extern Integer MA_push_get ARGS_((Integer, Integer, char*, Integer*, Integer*));
extern Integer MA_pop_stack ARGS_((Integer));
extern void ga_sort_scat_dbl_ ARGS_((Integer*,DoublePrecision*,Integer*,                                             Integer*,Integer*));
extern void ga_sort_scat_int_ ARGS_((Integer*,Integer*,Integer*,Integer*,                                            Integer*));
extern void ga_sort_gath_ ARGS_((Integer*, Integer*, Integer*, Integer*));

extern void free ARGS_((Void*));

#undef ARGS_


#ifdef GA_TRACE
  static Integer     op_code;
#endif
