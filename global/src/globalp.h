#ifndef _GLOBALP_H_
#define _GLOBALP_H_

#include <stdio.h>

#include "gaconfig.h"

#ifdef __crayx1
#undef CRAY
#endif

#ifdef FALSE
#undef FALSE
#endif
#ifdef TRUE
#undef TRUE
#endif
#ifdef CRAY_YMP
#define FALSE _btol(0)
#define TRUE  _btol(1)
#else
#define FALSE (logical) 0
#define TRUE  (logical) 1
#endif

#if HAVE_WINDOWS_H
#   include <windows.h>
#   define sleep(x) Sleep(1000*(x))
#endif
#include "macdecls.h"

#define GA_OFFSET   1000           /* offset for handle numbering */

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


#ifdef ENABLE_TRACE
  static Integer     op_code;
#endif


#define GA_MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define GA_MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define GA_ABS(a)   (((a) >= 0) ? (a) : (-(a)))

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

/* periodic operations */
#define PERIODIC_GET 1
#define PERIODIC_PUT 2
#define PERIODIC_ACC 3

#define FLUSH_CACHE
#ifdef  CRAY_T3D
#       define ALLIGN_SIZE      32
#else
#       define ALLIGN_SIZE      128
#endif

#define allign__(n, SIZE) (((n)%SIZE) ? (n)+SIZE - (n)%SIZE: (n))
#define allign_size(n) allign__((long)(n), ALLIGN_SIZE)
#define allign_page(n) allign__((long)(n), PAGE_SIZE)

extern void    ga_free(void *ptr);
extern void*   ga_malloc(Integer nelem, int type, char *name);
extern void    gai_print_subscript(char *pre,int ndim, Integer subscript[], char* post);
extern Integer GAsizeof(Integer type);
extern void    ga_sort_gath(Integer *pn, Integer *i, Integer *j, Integer *base);
extern void    ga_sort_permutation(Integer *pn, Integer *index, Integer *base);
extern void    ga_sort_scat(Integer *pn, void *v, Integer *i, Integer *j, Integer *base, Integer type);
extern void    gai_hsort(Integer *list, int num);
extern void    ga_init_nbhandle(Integer *nbhandle);
extern int     nga_test_internal(Integer *nbhandle);
extern int     nga_wait_internal(Integer *nbhandle);
extern int     ga_icheckpoint_init(Integer *gas, int num);
extern int     ga_icheckpoint(Integer *gas, int num);
extern int     ga_irecover(int rid);
extern int     ga_icheckpoint_finalize(int g_a);

/* the following are in the process of moving to papi.h */
#if 0

extern void    ga_matmul(char *transa, char *transb, void *alpha, void *beta, Integer *g_a, Integer *ailo, Integer *aihi, Integer *ajlo, Integer *ajhi, Integer *g_b, Integer *bilo, Integer *bihi, Integer *bjlo, Integer *bjhi, Integer *g_c, Integer *cilo, Integer *cihi, Integer *cjlo, Integer *cjhi);
extern void    ga_matmul_mirrored(char *transa, char *transb, void *alpha, void *beta, Integer *g_a, Integer *ailo, Integer *aihi, Integer *ajlo, Integer *ajhi, Integer *g_b, Integer *bilo, Integer *bihi, Integer *bjlo, Integer *bjhi, Integer *g_c, Integer *cilo, Integer *cihi, Integer *cjlo, Integer *cjhi);

extern void  ga_clean_resources( void);
extern void  ga_msg_brdcst(Integer type, void *buffer, Integer len, Integer root);
extern void  ga_print_file(FILE *, Integer *);
extern short ga_usesMA;

extern void  gai_gop(Integer type, void *x, Integer n, char *op);

extern void  gac_igop(int *x, Integer n, char *op);
extern void  gac_lgop(long *x, Integer n, char *op);
extern void  gac_llgop(long long *x, Integer n, char *op);
extern void  gac_fgop(float *x, Integer n, char *op);
extern void  gac_dgop(double *x, Integer n, char *op);
extern void  gac_cgop(SingleComplex *x, Integer n, char *op);
extern void  gac_zgop(DoubleComplex *x, Integer n, char *op);

extern void  gai_igop(Integer type, Integer  *x, Integer n, char *op);
extern void  gai_sgop(Integer type, Real *x, Integer n, char *op);
extern void  gai_dgop(Integer type, double  *x, Integer n, char *op);
extern void  gai_cgop(Integer type, SingleComplex *x, Integer n, char *op);
extern void  gai_zgop(Integer type, DoubleComplex *x, Integer n, char *op);

extern void  gai_pgroup_gop(Integer p_grp, Integer type, void *x, Integer n, char *op);

extern void  gai_pgroup_igop(Integer p_grp, Integer type, Integer  *x, Integer n, char *op);
extern void  gai_pgroup_sgop(Integer p_grp, Integer type, Real *x, Integer n, char *op);
extern void  gai_pgroup_dgop(Integer p_grp, Integer type, double  *x, Integer n, char *op);
extern void  gai_pgroup_cgop(Integer p_grp, Integer type, SingleComplex *x, Integer n, char *op);
extern void  gai_pgroup_zgop(Integer p_grp, Integer type, DoubleComplex *x, Integer n, char *op);

extern void  gai_check_handle(Integer *, char *);
extern void  gai_copy_patch(char *trans, Integer *g_a, Integer *ailo, Integer *aihi, Integer *ajlo, Integer *ajhi, Integer *g_b, Integer *bilo, Integer *bihi, Integer *bjlo, Integer *bjhi);
extern logical   gai_create(Integer *type, Integer *dim1, Integer *dim2, char *array_name, Integer *chunk1, Integer *chunk2, Integer *g_a);
extern logical   gai_create_irreg(Integer *type, Integer *dim1, Integer *dim2, char *array_name, Integer *map1, Integer *nblock1, Integer *map2, Integer *nblock2, Integer *g_a);
extern void  gai_dot(int Type, Integer *g_a, Integer *g_b, void *value);
extern logical   gai_duplicate(Integer *g_a, Integer *g_b, char* array_name);
extern void  gai_inquire(Integer* g_a, Integer* type, Integer* dim1, Integer* dim2);
extern void  gai_lu_solve_seq(char *trans, Integer *g_a, Integer *g_b);
extern void  gai_matmul_patch(char *transa, char *transb, void *alpha, void *beta, Integer *g_a,Integer *ailo,Integer *aihi,Integer *ajlo,Integer *ajhi, Integer *g_b,Integer *bilo,Integer *bihi,Integer *bjlo,Integer *bjhi, Integer *g_c,Integer *cilo,Integer *cihi,Integer *cjlo,Integer *cjhi);
extern Integer   gai_memory_avail(Integer datatype);
extern void  gai_print_distribution(int fstyle, Integer g_a);
extern void  gai_set_array_name(Integer g_a, char *array_name);

extern void  nga_acc_common(Integer *g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha, Integer *nbhandle);
extern void  nga_access_block_grid_ptr(Integer* g_a, Integer *index, void* ptr, Integer *ld);
extern void  nga_access_block_ptr(Integer* g_a, Integer *idx, void* ptr, Integer *ld);
extern void  nga_access_block_segment_ptr(Integer* g_a, Integer *proc, void* ptr, Integer *len);
extern void  nga_access_ghost_ptr(Integer* g_a, Integer dims[], void* ptr, Integer ld[]);
extern void  nga_access_ghost_element_ptr(Integer* g_a, void *ptr, Integer subscript[], Integer ld[]);
extern void  nga_access_ptr(Integer* g_a, Integer lo[], Integer hi[], void* ptr, Integer ld[]);
extern void  nga_get_common(Integer *g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle);
extern void  nga_put_common(Integer *g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle);

extern logical   ngai_comp_patch(Integer andim, Integer *alo, Integer *ahi, Integer bndim, Integer *blo, Integer *bhi);
extern void  ngai_copy_patch(char *trans, Integer *g_a, Integer *alo, Integer *ahi, Integer *g_b, Integer *blo, Integer *bhi);
extern void  ngai_dest_indices(Integer ndims, Integer *los, Integer *blos, Integer *dimss, Integer ndimd, Integer *lod, Integer *blod, Integer *dimsd);
extern void  ngai_dot_patch(Integer *g_a, char *t_a, Integer *alo, Integer *ahi, Integer *g_b, char *t_b, Integer *blo, Integer *bhi, void *retval);
extern void  ngai_inquire(Integer *g_a, Integer *type, Integer *ndim, Integer *dims);
extern void  ngai_matmul_patch(char *transa, char *transb, void *alpha, void *beta, Integer *g_a, Integer alo[], Integer ahi[], Integer *g_b, Integer blo[], Integer bhi[], Integer *g_c, Integer clo[], Integer chi[]);
extern logical   ngai_patch_intersect(Integer *lo, Integer *hi, Integer *lop, Integer *hip, Integer ndim);
extern void  ngai_periodic_(Integer *g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha, Integer op_code);
extern logical   ngai_test_shape(Integer *alo, Integer *ahi, Integer *blo, Integer *bhi, Integer andim, Integer bndim);

extern void  xb_dgemm (char *transa, char *transb, int *M, int *N, int *K, double *alpha,const double *a,int *p_lda,const double *b, int *p_ldb, double *beta, double *c, int *p_ldc);
extern void  xb_sgemm (char *transa, char *transb, int *M, int *N, int *K, float *alpha, const float *a, int *p_lda,const float *b, int *p_ldb, float *beta, float *c, int *p_ldc);
extern void  xb_zgemm (char * transa, char *transb, int *M, int *N, int *K, const void *alpha,const void *a,int *p_lda,const void *b, int *p_ldb, const void *beta, void *c, int *p_ldc);

extern double   gai_ddot_patch(Integer *g_a, char *t_a, Integer *ailo, Integer *aihi, Integer *ajlo, Integer *ajhi, Integer *g_b, char *t_b, Integer *bilo, Integer *bihi, Integer *bjlo, Integer *bjhi);
extern Integer   gai_idot_patch(Integer *g_a, char *t_a, Integer *ailo, Integer *aihi, Integer *ajlo, Integer *ajhi, Integer *g_b, char *t_b, Integer *bilo, Integer *bihi, Integer *bjlo, Integer *bjhi);
extern Real  gai_sdot_patch(Integer *g_a, char *t_a, Integer *ailo, Integer *aihi, Integer *ajlo, Integer *ajhi, Integer *g_b, char *t_b, Integer *bilo, Integer *bihi, Integer *bjlo, Integer *bjhi);

extern double   ngai_ddot_patch(Integer *g_a, char *t_a, Integer *alo, Integer *ahi, Integer *g_b, char *t_b, Integer *blo, Integer *bhi);
extern Integer   ngai_idot_patch(Integer *g_a, char *t_a, Integer *alo, Integer *ahi, Integer *g_b, char *t_b, Integer *blo, Integer *bhi);
extern Real  ngai_sdot_patch(Integer *g_a, char *t_a, Integer *alo, Integer *ahi, Integer *g_b, char *t_b, Integer *blo, Integer *bhi);

#endif

#endif /* _GLOBALP_H_ */
