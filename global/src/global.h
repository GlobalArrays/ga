/* file global.h */

#ifndef GLOBAL_H
#define GLOBAL_H

#include <stdio.h>
#include "typesf2c.h"


/* Maximum number of array dimensions supported by GA
 * NOTE: Must be changed in tandem with the Fortran definition 
 *  in global.fh! 
 */
#define GA_MAX_DIM 7

#include "c.names.h"
#if defined(CRAY) || defined(WIN32)
#  include "cray.names.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__STDC__) || defined(__cplusplus) || defined(WIN32)
# define ARGS_(s) s
#else
# define ARGS_(s) ()
#endif

extern void    FATR ga_set_memory_limit_ ARGS_((Integer *mem_limit));
extern logical FATR ga_valid_handle_ ARGS_((Integer *g_a));
extern Integer FATR ga_nnodes_   ARGS_(( void));
extern Integer FATR ga_nodeid_   ARGS_(( void));
extern Integer FATR ga_inquire_memory_  ARGS_(( void));
extern Integer FATR ga_memory_avail_ ARGS_(( void));
extern Integer FATR ga_read_inc_ ARGS_((Integer*, Integer*, Integer*, Integer* ));
extern Integer FATR ga_verify_handle_ ARGS_((Integer* ));
extern logical ga_create ARGS_((Integer*, Integer*, Integer*, char*, Integer*,                                  Integer*, Integer*));
extern logical ga_create_irreg ARGS_((Integer*, Integer*, Integer*, char*,                                            Integer*, Integer*, Integer*, Integer*,                                         Integer* ));
extern logical FATR ga_create_mutexes_ ARGS_((Integer*));
extern logical FATR ga_destroy_  ARGS_((Integer* ));
extern logical FATR ga_destroy_mutexes_  ARGS_((void ));
extern logical ga_duplicate ARGS_((Integer*, Integer*, char* ));
extern logical FATR ga_compare_distr_ ARGS_((Integer*, Integer* ));
extern logical FATR ga_locate_   ARGS_((Integer*, Integer*, Integer*, Integer* ));
extern void FATR ga_lock_        ARGS_((Integer* ));
extern void FATR ga_unlock_      ARGS_((Integer* ));
extern void ga_check_handle ARGS_((Integer*, char*));
extern logical FATR ga_locate_region_ ARGS_((Integer*, Integer*, Integer*, Integer*,                                         Integer*, Integer map[][5], Integer* ));
extern void  FATR ga_acc_   ARGS_((Integer*, Integer*, Integer*, Integer*, Integer*,                               void*, Integer*, void* ));
extern void FATR ga_access_ ARGS_((Integer*, Integer*, Integer*, Integer*, Integer*,                               Integer*, Integer* ));
extern void FATR ga_brdcst_ ARGS_((Integer*, Void*, Integer*, Integer* ));
extern void FATR ga_gather_ ARGS_((Integer*, Void*, Integer*, Integer*, Integer* ));
extern void ga_dgop ARGS_((Integer, DoublePrecision*, Integer, char* ));
extern void FATR ga_distribution_ ARGS_((Integer*, Integer*, Integer*, Integer*,                                         Integer*, Integer* ));
extern void FATR ga_scatter_ ARGS_((Integer*, Void*, Integer*, Integer*, Integer*));
extern void ga_error    ARGS_((char*, Integer));
extern void FATR ga_init_fence_   ARGS_(( void));
extern void FATR ga_fence_   ARGS_(( void));
extern void FATR ga_get_     ARGS_((Integer*, Integer*, Integer*, Integer*, Integer*,                               Void*, Integer* ));
extern void ga_igop     ARGS_((Integer, Integer*, Integer, char* ));
extern void FATR ga_initialize_ ARGS_(( void));
extern void FATR ga_initialize_ltd_ ARGS_(( Integer* ));
extern void FATR ga_inquire_ ARGS_((Integer*, Integer*, Integer*, Integer* ));
extern void ga_inquire_name ARGS_((Integer*, char** ));
extern void FATR ga_list_data_servers_ ARGS_((Integer* ));
extern void FATR ga_list_nodeid_ ARGS_((Integer*, Integer* ));
extern void FATR ga_num_data_servers_ ARGS_((Integer* ));
extern void FATR ga_put_  ARGS_((Integer*, Integer*, Integer*, Integer*, Integer*,                               Void*, Integer* ));
extern void FATR ga_release_ ARGS_((Integer*, Integer*, Integer*,Integer*,Integer*));
extern void FATR ga_release_update_ ARGS_((Integer*, Integer*, Integer*, Integer*,                                         Integer* ));
extern void FATR ga_sync_ ARGS_(( void));
extern void FATR ga_terminate_ ARGS_(( void));
extern logical FATR ga_uses_ma_ ARGS_(( void));
extern logical FATR ga_memory_limited_ ARGS_(( void));


extern void ga_copy_patch ARGS_((char *, Integer *, Integer *, Integer *,                                        Integer *, Integer *, Integer *, Integer *,                                     Integer *, Integer *, Integer *));
extern DoublePrecision ga_ddot_patch ARGS_((Integer *, char*, Integer *,                                                   Integer *, Integer *, Integer *,                                                Integer *, char*, Integer *,
                                           Integer *, Integer *, Integer *));
extern DoubleComplex ga_zdot_patch ARGS_((Integer *, char*, Integer *,                                                   Integer *, Integer *, Integer *,                                                Integer *, char*, Integer *,
                                           Integer *, Integer *, Integer *));
extern void FATR ga_fill_patch_  ARGS_((Integer *, Integer *, Integer *, Integer *,                                     Integer *, Void *));
extern void FATR ga_scale_patch_  ARGS_((Integer *, Integer *, Integer *, Integer *,                                     Integer *, DoublePrecision *));
extern void FATR ga_add_patch_   ARGS_((DoublePrecision *, Integer *,                                                   Integer *, Integer *, Integer *, Integer *,                                     DoublePrecision *, Integer *,                                                   Integer *, Integer *, Integer *, Integer *,                                     Integer *, Integer *, Integer *, Integer *,                                     Integer *  ));
extern void ga_matmul_patch  ARGS_((char *, char *,                                                                 DoublePrecision *, DoublePrecision *,                                           Integer *, Integer *, Integer *, Integer *,                                     Integer *, Integer *, Integer *,                                                Integer *, Integer *, Integer *, Integer *, 
                                    Integer *, Integer *, Integer *, Integer*));

extern void FATR ga_copy_   ARGS_((Integer *, Integer *));
extern void      ga_print_file ARGS_((FILE *, Integer *));
extern void FATR ga_print_  ARGS_((Integer *));
extern void FATR ga_print_stats_();
extern void FATR ga_zero_   ARGS_((Integer *));
extern void FATR ga_fill_   ARGS_((Integer *, void *));
extern void FATR ga_scale_  ARGS_((Integer *, void *));
extern void FATR ga_add_   ARGS_((Void *, Integer *, Void *,                                Integer *, Integer *));
extern Integer FATR ga_idot_ ARGS_((Integer *, Integer *));
extern DoublePrecision FATR ga_ddot_ ARGS_((Integer *, Integer *));
extern DoubleComplex ga_zdot ARGS_((Integer *, Integer *));
extern void FATR ga_print_patch_ ARGS_((Integer *, Integer *, Integer *, Integer *,                                     Integer *, Integer *));

extern void FATR ga_summarize_     ARGS_((logical*));
extern void FATR ga_symmetrize_   ARGS_((Integer *)); 
extern void FATR ga_transpose_    ARGS_((Integer *, Integer *));
extern void FATR ga_diag_seq_     ARGS_((Integer *, Integer *, Integer *,                                                DoublePrecision *));
extern void FATR ga_diag_reuse_   ARGS_((Integer*, Integer *, Integer *, Integer *,
                                  DoublePrecision *));
extern void FATR ga_diag_std_     ARGS_((Integer *, Integer *, DoublePrecision *));
extern void FATR ga_diag_std_seq_ ARGS_((Integer *, Integer *, DoublePrecision *));
extern void FATR ga_lu_solve_      ARGS_((char *, Integer *, Integer *));
extern void FATR ga_lu_solve_alt_  ARGS_((Integer *, Integer *, Integer *));
extern void ga_lu_solve_seq_  ARGS_((char *, Integer *, Integer *));

extern Integer FATR ga_llt_solve_ ARGS_((Integer *, Integer *));
extern Integer FATR ga_solve_ ARGS_((Integer *, Integer *));
extern Integer FATR ga_spd_invert_ ARGS_((Integer *));

extern void ga_dgemm ARGS_((char *, char *, Integer *, Integer *, Integer *,                                DoublePrecision *, Integer *, Integer *,                                        DoublePrecision *, Integer *));
extern void FATR ga_diag_ ARGS_((Integer *, Integer *, Integer *,DoublePrecision *));
extern void FATR ga_proc_topology_ ARGS_((Integer *g_a, Integer *proc,  Integer *pr,\
                                     Integer *pc));

extern void FATR ga_sort_permut_ ARGS_((Integer* g_a, Integer* index, Integer* i, Integer* j, Integer* nv));
#undef ARGS_

extern void FATR ga_print_distribution_(Integer *g_a);
extern Integer FATR ga_ndim_(Integer *g_a);

extern logical nga_create_irreg(
        Integer type,    /* MA type */
        Integer ndim,    /* number of dimensions */
        Integer dims[],   /* array of dimensions */
        char *array_name, /* array name */
        Integer map[],    /* decomposition map array */
        Integer nblock[], /* number of blocks for each dimension in map */
        Integer *g_a);    /* array handle (output) */

extern void FATR  nga_release_(Integer *g_a, Integer *lo, Integer *hi);
extern void FATR  nga_release_update_(Integer *g_a, Integer *lo, Integer *hi);
extern void FATR  nga_inquire_(Integer *g_a, Integer *type, Integer *ndim,Integer *dims); 

extern logical FATR nga_locate_(Integer *g_a,
                                Integer* subscr,
                                Integer* owner);

extern logical FATR nga_locate_region_( Integer *g_a,
                                        Integer *lo,
                                        Integer *hi,
                                        Integer *map,
                                        Integer *proclist,
                                        Integer *np);

extern void nga_access_ptr(Integer* g_a, Integer lo[], Integer hi[],
                           void* ptr, Integer ld[]);

extern void FATR nga_access_(Integer* g_a, Integer lo[], Integer hi[],
                             Integer* index, Integer ld[]);
extern void FATR nga_distribution_(Integer *g_a, Integer *proc, 
                                   Integer *lo, Integer *hi);
extern void FATR nga_put_(Integer *g_a, Integer *lo, Integer *hi, 
                          void *buf, Integer *ld);
extern void FATR nga_get_(Integer *g_a, Integer *lo, Integer *hi, 
                          void *buf, Integer *ld);
extern void FATR nga_acc_(Integer *g_a, Integer *lo, Integer *hi,
                          void *buf, Integer *ld, void *alpha);
extern void FATR nga_scatter_(Integer *g_a, void* v, Integer subscr[], 
                              Integer *nv);
extern void FATR nga_scatter_acc_(Integer *g_a, void* v, Integer subscr[], 
                              Integer *nv, void *alpha);
extern void FATR nga_gather_(Integer *g_a, void* v, Integer subscr[],
                             Integer *nv);
extern Integer FATR nga_read_inc_(Integer* g_a,Integer* subscr,Integer* inc);
extern void FATR nga_periodic_get_(Integer *g_a, Integer *lo, Integer *hi,
                                   void *buf, Integer *ld);
extern void FATR nga_periodic_put_(Integer *g_a, Integer *lo, Integer *hi,
                                   void *buf, Integer *ld);
extern void FATR nga_periodic_acc_(Integer *g_a, Integer *lo, Integer *hi,
                                   void *buf, Integer *ld, void *alpha);
extern void FATR nga_proc_topology_(Integer* g_a, Integer* proc, Integer* subscr);

extern void nga_copy_patch(char *trans,
                           Integer *g_a, Integer *alo, Integer *ahi,
                           Integer *g_b, Integer *blo, Integer *bhi);
extern Integer nga_idot_patch(Integer *g_a, char *t_a, Integer *alo,
          Integer *ahi, Integer *g_b, char *t_b, Integer *blo, Integer *bhi);
extern void FATR nga_print_patch_(Integer *g_a, Integer *lo, Integer *hi, Integer *pretty);

extern DoublePrecision nga_ddot_patch(Integer *g_a, char *t_a, 
          Integer *alo, Integer *ahi, Integer *g_b, char *t_b, Integer *blo,
          Integer *bhi);

extern DoubleComplex nga_zdot_patch(Integer *g_a, char *t_a,
          Integer *alo, Integer *ahi, Integer *g_b, char *t_b, Integer *blo,
          Integer *bhi);

extern void FATR nga_zero_patch_(Integer *g_a, Integer *lo, Integer *hi);

extern void FATR nga_fill_patch_(Integer *g_a, Integer *lo, Integer *hi,
                                 void *val);

extern void FATR nga_scale_patch_(Integer *g_a, Integer *lo, Integer *hi,
                                  void *alpha);

extern void FATR nga_add_patch_(DoublePrecision *alpha, Integer *g_a,
                    Integer *alo, Integer *ahi, DoublePrecision *beta,
                    Integer *g_b, Integer *blo, Integer *bhi, Integer *g_c,
                    Integer *clo, Integer *chi);

#ifdef __cplusplus
}
#endif

extern DoubleComplex   *DCPL_MB;
extern DoublePrecision *DBL_MB;
extern Integer         *INT_MB;

#endif 
