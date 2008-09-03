#ifndef  _GLOBALP_H_
#define _GLOBALP_H_

#include "config.h"
#include "global.h"

#ifdef __crayx1
#undef CRAY
#endif

#ifdef FALSE
#undef FALSE
#endif
#ifdef TRUE
#undef TRUE
#endif
#ifdef CRAY
#include <fortran.h>
#endif
#ifdef CRAY_YMP
#define FALSE _btol(0)
#define TRUE  _btol(1)
#else
#define FALSE (logical) 0
#define TRUE  (logical) 1
#endif

#if defined(WIN32)
#   include "winutil.h"
#endif
#include "macdecls.h"

#if (defined(CRAY) && !defined(__crayx1)) || defined(NEC) 
#  define NO_REAL_32  
#endif

#define GA_OFFSET   1000           /* offset for handle numbering */

#ifndef MAX_NPROC                  /* default max number of processors  */
#   ifdef PARAGON
#     define MAX_NPROC    1024
#   elif defined(DELTA)
#     define MAX_NPROC     512
#   elif defined(SP1) || defined(SP)
#     define MAX_NPROC     512
#   elif defined(LAPI)
#     define MAX_NPROC     512
#   elif defined(CRAY_T3D)
#     define MAX_NPROC     256
#   elif defined(KSR)
#     define MAX_NPROC      80
#   elif defined(LINUX64)
#     define MAX_NPROC    2048
#   elif defined(BGML)
#     define MAX_NPROC    2048
#   elif defined(BGP)
#     define MAX_NPROC     8192
#   else
#     define MAX_NPROC    2048     /* default for everything else */
#   endif
#endif


/* types/tags of messages used internally by GA */
#define     GA_TYPE_SYN   GA_MSG_OFFSET + 1
#define     GA_TYPE_GSM   GA_MSG_OFFSET + 5
#define     GA_TYPE_GOP   GA_MSG_OFFSET + 15
#define     GA_TYPE_BRD   GA_MSG_OFFSET + 16

/* GA operation ids */
#define     GA_OP_GET 1          /* Get                         */
#define     GA_OP_END 2          /* Terminate                   */
#define     GA_OP_CRE 3          /* Create                      */
#define     GA_OP_PUT 4          /* Put                         */
#define     GA_OP_ACC 5          /* Accumulate                  */
#define     GA_OP_DES 6          /* Destroy                     */
#define     GA_OP_DUP 7          /* Duplicate                   */
#define     GA_OP_ZER 8          /* Zero                        */
#define     GA_OP_DDT 9          /* dot product                 */
#define     GA_OP_SCT 10         /* scatter                     */
#define     GA_OP_GAT 11         /* gather                      */
#define     GA_OP_RDI 15         /* Integer read and increment  */
#define     GA_OP_ACK 16         /* acknowledgment              */
#define     GA_OP_LCK 17         /* acquire lock                */
#define     GA_OP_UNL 18         /* release lock                */


#ifdef GA_TRACE
  static Integer     op_code;
#endif


#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define ABS(a)   (((a) >= 0) ? (a) : (-(a)))

#define GAsizeofM(type)  ( (type)==C_DBL? sizeof(double): \
                           (type)==C_INT? sizeof(int): \
                           (type)==C_DCPL? sizeof(DoubleComplex): \
                           (type)==C_SCPL? sizeof(SingleComplex): \
                           (type)==C_LONG? sizeof(long): \
                           (type)==C_LONGLONG? sizeof(long long): \
                           (type)==C_FLOAT? sizeof(float):0)

#define NAME_STACK_LEN 10
#define PAGE_SIZE  4096

struct ga_stat_t {
         long   numcre; 
         long   numdes;
         long   numget;
         long   numput;
         long   numacc;
         long   numsca;
         long   numgat;
         long   numrdi;
         long   numser;
         long   curmem; 
         long   maxmem; 
         long   numget_procs;
         long   numput_procs;
         long   numacc_procs;
         long   numsca_procs;
         long   numgat_procs;
};

struct ga_bytes_t{ 
         double acctot;
         double accloc;
         double gettot;
         double getloc;
         double puttot;
         double putloc;
         double rditot;
         double rdiloc;
         double gattot;
         double gatloc;
         double scatot;
         double scaloc;
};

#define STAT_AR_SZ sizeof(ga_stat_t)/sizeof(long)

extern long *GAstat_arr;  
extern struct ga_stat_t GAstat;
extern struct ga_bytes_t GAbytes;
extern char *GA_name_stack[NAME_STACK_LEN];    /* stack for names of GA ops */ 
extern int GA_stack_size;
extern int _ga_sync_begin;
extern int _ga_sync_end;
extern int *_ga_argc;
extern char ***_ga_argv;


#define  GA_PUSH_NAME(name) (GA_name_stack[GA_stack_size++] = (name)) 
#define  GA_POP_NAME        (GA_stack_size--)


extern void f2cstring(char*, Integer, char*, Integer);
extern void c2fstring( char*, char*, Integer);
extern void ga_clean_resources( void);

/* periodic operations */
#define PERIODIC_GET 1
#define PERIODIC_PUT 2
#define PERIODIC_ACC 3

extern void ngai_periodic_(Integer *g_a, Integer *lo, Integer *hi, void *buf,
                           Integer *ld, void *alpha, Integer op_code);

#define FLUSH_CACHE
#ifdef  CRAY_T3D
#       define ALLIGN_SIZE      32
#else
#       define ALLIGN_SIZE      128
#endif

#define allign__(n, SIZE) (((n)%SIZE) ? (n)+SIZE - (n)%SIZE: (n))
#define allign_size(n) allign__((long)(n), ALLIGN_SIZE)
#define allign_page(n) allign__((long)(n), PAGE_SIZE)

extern void gai_print_subscript(char *pre,int ndim, Integer subscript[], char* post);
extern void ngai_dest_indices(Integer ndims, Integer *los, Integer *blos, Integer *dimss,
               Integer ndimd, Integer *lod, Integer *blod, Integer *dimsd);

extern logical ngai_patch_intersect(Integer *lo, Integer *hi,
                        Integer *lop, Integer *hip, Integer ndim);

extern logical ngai_comp_patch(Integer andim, Integer *alo, Integer *ahi,
                          Integer bndim, Integer *blo, Integer *bhi);
extern logical ngai_test_shape(Integer *alo, Integer *ahi, Integer *blo,
                          Integer *bhi, Integer andim, Integer bndim);

extern void xb_sgemm (char *transa, char *transb, int *M, int *N, int *K,
		      float *alpha, const float *a, int *p_lda,const float *b,
		      int *p_ldb, float *beta, float *c, int *p_ldc);

extern void xb_dgemm (char *transa, char *transb, int *M, int *N, int *K,
		      double *alpha,const double *a,int *p_lda,const double *b,
		      int *p_ldb, double *beta, double *c, int *p_ldc);

extern void xb_zgemm (char * transa, char *transb, int *M, int *N, int *K,
		      const void *alpha,const void *a,int *p_lda,const void *b,
		      int *p_ldb, const void *beta, void *c, int *p_ldc);

/* GA Memory allocation routines and variables */
extern short int ga_usesMA;

extern void* ga_malloc(Integer nelem, int type, char *name);

extern void ga_free(void *ptr);

extern Integer ga_memory_avail(Integer datatype);

extern void ga_init_nbhandle(Integer *nbhandle);
extern int nga_wait_internal(Integer *nbhandle);
#endif
