/*Thu Sep 29 09:40:30 PDT 1994*/
#include "global.c.h"

#ifdef SUN
#define volatile   
#endif

#ifdef KSR
#define PRIVATE  __private 
#else
#define PRIVATE  
#endif

#define MAX_ARRAYS  32
#define BUF_SIZE    4096 
#define MAX_REG     128             /* max number of shmem regions per array */
#define RESERVED    2*sizeof(long)  /* used for shmem buffer management */  
#define FNAM        30              /* length of Fortran names   */
#define FLEN        80              /* length of Fortran strings */
#define max_nproc   256		    /* max number of processors  */
#define GA_OFFSET   1000            /* offset for handle numbering */


typedef struct {
       Integer type;            /* type of array                        */
       int  actv;               /* activity status                      */
       char *ptr[max_nproc];    /* pointers to local/remote data        */
       long ilo;                /* coordinates of local patch           */
       long ihi;
       long jlo;
       long jhi;
       long size;               /* size of local data in bytes          */
       int  dims[2];            /* global dimensions [i,j]              */
       int  chunk[2];           /* chunking                             */
       int  nblock[2];          /* number of chunks (blocks)            */
       int  mapc[max_nproc+1];  /* block distribution map               */
       long lock;
       long id;			/* ID of shmem region / MA handle       */
       char name[FNAM+1];       /* array name                           */
} global_array;


PRIVATE static global_array GA[MAX_ARRAYS]; 
PRIVATE static int max_global_array = MAX_ARRAYS;
PRIVATE Integer map[max_nproc][5];               /* used in get/put/acc */



#ifdef  CRAY_T3D
#define ALLIGN_SIZE      32 
#else
#define ALLIGN_SIZE      128
#endif



/********************** MACROS ******************************/
#define allign_size(n) \
        (((n)%ALLIGN_SIZE) ? (n)+ALLIGN_SIZE - (n)%ALLIGN_SIZE: (n))

#define ga_check_handleM(g_a, string) \
{\
    if(GA_OFFSET+ (*g_a) < 0 || GA_OFFSET+(*g_a) >= max_global_array){\
      fprintf(stderr, " ga_check_handle: %s ", string); \
      ga_error(" invalid global array handle ", (*g_a));\
    }\
    if( ! (GA[GA_OFFSET+(*g_a)].actv) ){                \
      fprintf(stderr, " ga_check_handle: %s ", string); \
      ga_error(" global array is not active ", (*g_a)); \
    }                                                   \
}
      
#define MIN(a,b) ((a)>(b)? (b) : (a))
#define MAX(a,b) ((a)>(b)? (a) : (b))




/*********** Shared Memory, Mutual Exclusion & Memory Copy Co.  *********/
#ifdef SYSV
       /* SHARED MEMORY */
       PRIVATE static  volatile int    barrier_size;
       PRIVATE static  volatile int    *barrier, *barrier1;
       PRIVATE static  long            shmSIZE, shmID;
       PRIVATE static  DoublePrecision *shmBUF;
       char    *Create_Shared_Region();
       long    Detach_Shared_Region();
       long    Delete_Shared_Region();
       char    *Attach_Shared_Region();
#      ifdef KSR
#          include "global.KSR.h"
#          define SUBPAGE 128              /* subpage size on KSR */
#          define PAGE_SIZE  128
#      else
#          define PAGE_SIZE  4096
           /* define LOCK OPERATIONS using SYSV semaphores */
#          include "semaphores.h"
#          define NUM_SEM  SEMMSL
#          define ARR_SEM  (NUM_SEM -1) /* num of sems for locking arrays */
#          define MUTEX    ARR_SEM    /* semid  for synchronization */
#          define LOCK(g_a,proc, x)\
                  P(((proc)-cluster_master)%ARR_SEM)
#          define UNLOCK(g_a,proc, x)\
                   V(((proc)-cluster_master)%ARR_SEM)
#      endif
#else
#        ifdef CRAY_T3D
#            include <limits.h>
#            include <mpp/shmem.h>
#            define INVALID (long)(_INT_MIN_64 +1)
#            define LOCK_AND_GET(x, y, proc) \
                while( ((x) = shmem_swap((long*)(y),INVALID,(proc)))== INVALID)
#            define UNLOCK_AND_PUT(x, y, proc) shmem_swap((long*)(y),(x),(proc))

#            define LOCK(g_a, proc, x) \
                while( shmem_swap(&GA[GA_OFFSET +g_a].lock,INVALID,(proc))\
                       == INVALID)

#            define UNLOCK(g_a, proc, x)\
                   shmem_swap(&GA[GA_OFFSET +g_a].lock, 1, (proc))
#        elif defined(NX) || defined(SP1)
#            include "interrupt.h"
             extern Integer in_handler;
             long oldmask;
#            define LOCK(g_a, proc, x) \
                    { if(  in_handler == 0) ga_mask(1L, &oldmask) }
#            define UNLOCK(g_a,proc, x) \
                    { if( in_handler == 0) ga_mask(oldmask, &oldmask) }
#        endif
#endif

/* MEMORY COPY */
#ifdef SUN
       char* memcpy();
#else
       void* memcpy();
#endif
#ifdef CRAY_T3D
       /*#          define Copy(src,dst,n)          memcpy((dst),(src),(n))*/
       /*            memcpy on this sytem is slow */
#      define Copy(src,dst,n)          copyto((src), (dst),(n))
#      define CopyTo(src,dst,n,proc)   \
                  shmem_put((long*) (dst), (long*)(src), (n), (proc)) 
#      define CopyFrom(src,dst,n,proc) \
                  shmem_get((long*) (dst), (long*)(src), (n), (proc)) 
#elif !defined(KSR)
#      define Copy(src,dst,n)      memcpy((dst),(src),(n))
#      define CopyTo(src,dst,n)    memcpy((dst),(src),(n))
#      define CopyFrom(src,dst,n)  memcpy((dst),(src),(n))
#endif
/************************************************************************/

/* cache coherency ? */
#ifndef CRAY_T3D
#       define  FLUSH_CACHE
#       define  FLUSH_CACHE_LINE(x) 
#else
#       define FLUSH_CACHE shmem_udcflush()
#       define FLUSH_CACHE_LINE(x)   shmem_udcflush_line((long*)(x))
#endif


/* MA addressing */
#include <macommon.h>
DoublePrecision *DBL_MB;            /* double precision base address */
Integer         *INT_MB;            /* integer base address */
void 	ma__base_address_();
Integer ma__sizeof_();

Integer GAsizeof();

PRIVATE static int GAinitialized = 0;
PRIVATE static Integer GAme, GAnproc, GAmaster;
PRIVATE static Integer MPme, MPnproc;
int ProcListPerm[max_nproc];        /* permuted list of processes */
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

char *malloc();
