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


typedef struct {
       char *ptr;		/* pointer to storage */ 
       Integer type;		/* type of array      */ 
       int  actv;		/* activity status    */ 
       long id;			/* ID of shmem region */
       long size;		/* array size in bytes*/       
       int  dims[2];		/* dimensions [i,j]   */ 
       int  chunk1;		/* chunking           */
       int  chunk2;		/* chunking           */
       int  nblock1;
       int  nblock2;
       int  mapc[max_nproc+1];  /* block distribution map */
       char name[FNAM+1];	/* array name (possibly Fortran string) */ 
} global_array;


PRIVATE static global_array GA[MAX_ARRAYS];
PRIVATE static int max_global_array = MAX_ARRAYS;

PRIVATE static  volatile int    barrier_size;
PRIVATE static  volatile int    *barrier, *barrier1;
PRIVATE static  long            shmSIZE, shmID;
PRIVATE static  DoublePrecision *shmBUF;

PRIVATE static int GAinitialized = 0;
PRIVATE static Integer me, nproc;
DoublePrecision *DBL_MB;
Integer *INT_MB;


/********************** MACROS ******************************/

#define ga_check_handleM(g_a, string) \
{\
    if( (*g_a) < 0 || (*g_a) >= max_global_array){      \
      fprintf(stderr, " ga_check_handle: %s ", string); \
      fflush(stderr);					\
      ga_error(" invalid global array handle ", (*g_a));\
    }\
    if( ! (GA[(*g_a)].actv) ){\
      fprintf(stderr, " ga_check_handle: %s ", string); \
      fflush(stderr);					\
      ga_error(" global array is not active ", (*g_a)); \
    }                                                   \
}
      
      
#define MIN(a,b) ((a)>(b)? (b) : (a))
#define MAX(a,b) ((a)>(b)? (a) : (b))

/*********** KSR or not to KSR *********/
#ifdef KSR
#include "global.KSR.h"
#define SUBPAGE 128              /* subpage size on KSR */
#define PAGE_SIZE  128
/**/
#else
#define PAGE_SIZE  4096
#include "semaphores.h"
/* define lock operations using SYSV semaphores */
#define MUTEX 0
#define LOCK(x)   P(MUTEX) 
#define UNLOCK(x) V(MUTEX)
/**/
#ifdef SUN
char* memcpy();
#else
void memcpy();
#endif
#define Copy(src,dst,n)      memcpy((dst),(src),(n))
#define CopyTo(src,dst,n)    memcpy((dst),(src),(n))
#define CopyFrom(src,dst,n)  memcpy((dst),(src),(n))
#endif
/*************************************/

#include <macommon.h>

char 	*Create_Shared_Region();
long 	Detach_Shared_Region();
long 	Delete_Shared_Region();
char 	*Attach_Shared_Region();
void 	ma__base_address_();
Integer ma__sizeof_();
Integer MAsizeof();


#define     GA_TYPE_REQ 32760 - 1
#define     GA_TYPE_GET 32760 - 2
#define     GA_TYPE_SYN 32760 - 3
#define     GA_TYPE_PUT 32760 - 4
#define     GA_TYPE_ACC 32760 - 5
#define     GA_TYPE_GSM 32760 - 6
#define     GA_TYPE_ACK 32760 - 7
#define     GA_TYPE_ADD 32760 - 8
#define     GA_TYPE_DCV 32760 - 9
#define     GA_TYPE_DCI 32760 - 10
#define     GA_TYPE_DCJ 32760 - 11
#define     GA_TYPE_DSC 32760 - 12
#define     GA_TYPE_RDI 32760 - 13

