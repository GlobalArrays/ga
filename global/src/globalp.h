#ifndef  _GLOBALP_H_
#define _GLOBALP_H_

#include "config.h"

#ifdef FALSE
#undef FALSE
#endif
#ifdef TRUE
#undef TRUE
#endif
#ifdef CRAY_YMP
#include <fortran.h>
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

#define GA_OFFSET   1000           /* offset for handle numbering */

#ifndef MAX_NPROC                  /* default max number of processors  */
#   ifdef PARAGON
#     define MAX_NPROC    1024
#   elif defined(DELTA)
#     define MAX_NPROC     512
#   elif defined(SP1) || defined(SP)
#     define MAX_NPROC     400
#   elif defined(LAPI)
#     define MAX_NPROC     512
#   elif defined(CRAY_T3D)
#     define MAX_NPROC     256
#   elif defined(KSR)
#     define MAX_NPROC      80
#   else
#     define MAX_NPROC     128     /* default for everything else */
#   endif
#endif

#ifdef SYSV
#  define RESERVED_LOCKS  1        /* reserved for barrier */
#endif


/* types/tags of messages used internally by GA */
#define     GA_TYPE_REQ   GA_MSG_OFFSET + 1
#define     GA_TYPE_GET   GA_MSG_OFFSET + 2
#define     GA_TYPE_PUT   GA_MSG_OFFSET + 3
#define     GA_TYPE_ACC   GA_MSG_OFFSET + 4
#define     GA_TYPE_GSM   GA_MSG_OFFSET + 5
#define     GA_TYPE_ACK   GA_MSG_OFFSET + 6
#define     GA_TYPE_ADD   GA_MSG_OFFSET + 7
#define     GA_TYPE_DCV   GA_MSG_OFFSET + 8
#define     GA_TYPE_DCI   GA_MSG_OFFSET + 9
#define     GA_TYPE_DCJ   GA_MSG_OFFSET + 10
#define     GA_TYPE_SCT   GA_MSG_OFFSET + 11
#define     GA_TYPE_RDI   GA_MSG_OFFSET + 12
#define     GA_TYPE_GAT   GA_MSG_OFFSET + 13
#define     GA_TYPE_SYN   GA_MSG_OFFSET + 14
#define     GA_TYPE_GOP   GA_MSG_OFFSET + 15
#define     GA_TYPE_BRD   GA_MSG_OFFSET + 16
#define     GA_TYPE_LCK   GA_MSG_OFFSET + 17
#define     GA_TYPE_UNL   GA_MSG_OFFSET + 18
#define     GA_TYPE_MAS   GA_MSG_OFFSET + 20

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

#define GAsizeofM(type)  ( (type)==MT_F_DBL? sizeof(DoublePrecision): \
                           (type)==MT_F_INT? sizeof(Integer): \
                           (type)==MT_F_DCPL? sizeof(DoubleComplex): \
                           (type)==MT_F_REAL? sizeof(float):0)

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

#define  GA_PUSH_NAME(name) (GA_name_stack[GA_stack_size++] = (name)) 
#define  GA_POP_NAME        (GA_stack_size--)


extern void f2cstring(char*, Integer, char*, Integer);
extern void c2fstring( char*, char*, Integer);
extern void ga_clean_resources( void);


#ifndef ARMCI
extern void ga_put_local(Integer g_a, Integer ilo, Integer ihi, Integer jlo, 
                         Integer jhi, void* buf, Integer offset, Integer ld, 
                         Integer proc);
extern void ga_get_local(Integer g_a, Integer ilo, Integer ihi, Integer jlo, 
                         Integer jhi, void* buf, Integer offset, Integer ld, 
                         Integer proc);
extern Integer ga_read_inc_local(Integer g_a, Integer i, Integer j, Integer inc,
                                 Integer proc);
extern void ga_check_req_balance();
#endif

#ifdef ARMCI
/* periodic operations */
#define PERIODIC_GET 1
#define PERIODIC_PUT 2
#define PERIODIC_ACC 3

extern void ngai_periodic_(Integer *g_a, Integer *lo, Integer *hi, void *buf,
                           Integer *ld, void *alpha, Integer op_code);
#else

extern void ga_put_local(Integer g_a, Integer ilo, Integer ihi, Integer jlo,
                         Integer jhi, void* buf, Integer offset, Integer ld,
                         Integer proc);
extern void ga_get_local(Integer g_a, Integer ilo, Integer ihi, Integer jlo,
                         Integer jhi, void* buf, Integer offset, Integer ld,
                         Integer proc);
extern Integer ga_read_inc_local(Integer g_a, Integer i, Integer j, Integer inc,
                                 Integer proc);
extern void ga_check_req_balance();
#endif


#endif
