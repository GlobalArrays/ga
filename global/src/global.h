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

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__STDC__) || defined(__cplusplus) || defined(WIN32)
# define ARGS_(s) s
#else
# define ARGS_(s) ()
#endif
  
#define C_DBL MT_C_DBL
#define C_INT MT_C_INT
#define C_FLOAT MT_C_FLOAT
#define C_DCPL MT_C_DCPL
#define C_LONG MT_C_LONGINT
#define C_SCPL MT_C_SCPL
  
extern void *GA_Getmem(int type, int nelem, int grp_id);
extern void GA_Freemem(void* ptr);
extern int GA_Assemble_duplicate(int g_a, char *name, void *ptr);
extern void    FATR ga_set_memory_limit_ ARGS_((Integer *mem_limit));
extern logical FATR ga_valid_handle_ ARGS_((Integer *g_a));
extern void    FATR ga_mask_sync_ ARGS_((Integer *begin, Integer *end));
extern void    FATR ga_ghost_barrier_ ARGS_((void));
extern logical FATR ga_has_ghosts_ ARGS_((Integer *g_a));
extern Integer FATR ga_nnodes_   ARGS_(( void));
extern Integer FATR ga_nodeid_   ARGS_(( void));
extern Integer FATR ga_pgroup_nnodes_   ARGS_((Integer *grp_id));
extern Integer FATR ga_pgroup_nodeid_   ARGS_((Integer *grp_id));
extern Integer FATR ga_get_pgroup_ ARGS_((Integer *g_a));
extern Integer FATR ga_inquire_memory_  ARGS_(( void));
extern Integer FATR ga_memory_avail_ ARGS_(( void));
extern Integer FATR ga_read_inc_ ARGS_((Integer*, Integer*, Integer*,
      Integer* ));
extern Integer FATR ga_verify_handle_ ARGS_((Integer* ));
extern logical ga_create ARGS_((Integer*, Integer*, Integer*, char*, Integer*,
      Integer*, Integer*));
extern logical ga_create_irreg ARGS_((Integer*, Integer*, Integer*, char*,
      Integer*, Integer*, Integer*, Integer*,
      Integer* ));
extern logical FATR ga_create_mutexes_ ARGS_((Integer*));
extern logical FATR ga_destroy_  ARGS_((Integer* ));
extern logical FATR ga_destroy_mutexes_  ARGS_((void ));
extern logical ga_duplicate ARGS_((Integer*, Integer*, char* ));
extern logical FATR ga_compare_distr_ ARGS_((Integer*, Integer* ));
extern logical FATR ga_locate_  ARGS_((Integer*, Integer*, Integer*,
      Integer* ));
extern void FATR ga_lock_       ARGS_((Integer* ));
extern void FATR ga_unlock_     ARGS_((Integer* ));
extern void ga_check_handle ARGS_((Integer*, char*));
extern logical FATR ga_locate_region_ ARGS_((Integer*, Integer*, Integer*,
      Integer*, Integer*, Integer map[][5], Integer* ));
extern void  FATR ga_acc_   ARGS_((Integer*, Integer*, Integer*, Integer*,
      Integer*, void*, Integer*, void* ));
extern void  FATR ga_nbacc_   ARGS_((Integer*, Integer*, Integer*, Integer*,
      Integer*, void*, Integer*, void* ,Integer *));
extern void FATR ga_access_ ARGS_((Integer*, Integer*, Integer*, Integer*,
      Integer*, Integer*, Integer* ));
extern void FATR ga_brdcst_ ARGS_((Integer*, Void*, Integer*, Integer* ));
extern void FATR ga_pgroup_brdcst_ ARGS_((Integer*, Integer*, Void*, Integer*, Integer* ));
extern void FATR ga_gather_ ARGS_((Integer*, Void*, Integer*, Integer*,
      Integer* ));
extern void FATR ga_distribution_ ARGS_((Integer*, Integer*, Integer*,
      Integer*, Integer*, Integer* ));
extern void FATR ga_scatter_ ARGS_((Integer*, Void*, Integer*, Integer*,
      Integer*));
extern void ga_error    ARGS_((char*, Integer));
extern void FATR ga_init_fence_   ARGS_(( void));
extern void FATR ga_fence_   ARGS_(( void));
extern void FATR ga_get_     ARGS_((Integer*, Integer*, Integer*, Integer*,
      Integer*, Void*, Integer* ));
extern void FATR ga_nbget_   ARGS_((Integer*, Integer*, Integer*, Integer*,
      Integer*, Void*, Integer*, Integer* ));
extern void FATR ga_nbwait_  ARGS_((Integer*));
extern void ga_dgop ARGS_((Integer, DoublePrecision*, Integer, char* ));
extern void ga_pgroup_dgop ARGS_((Integer, Integer, DoublePrecision*, Integer, char* ));
extern void ga_fgop     ARGS_((Integer, float*, Integer, char* ));
extern void ga_pgroup_fgop     ARGS_((Integer, Integer, float*, Integer, char* ));
extern void ga_igop     ARGS_((Integer, Integer*, Integer, char* ));
extern void ga_pgroup_igop     ARGS_((Integer, Integer, Integer*, Integer, char* ));
extern void ga_lgop    ARGS_((Integer, long*,Integer, char* ));
extern void ga_pgroup_lgop    ARGS_((Integer, Integer, long*,Integer, char* ));
extern void FATR ga_initialize_ ARGS_(( void));
extern void FATR ga_initialize_ltd_ ARGS_(( Integer* ));
extern void FATR ga_inquire_ ARGS_((Integer*, Integer*, Integer*, Integer* ));
extern void FATR ga_inquire_internal_ ARGS_((Integer*, Integer*, Integer*, Integer* ));
extern void ga_inquire_name ARGS_((Integer*, char** ));
extern void FATR ga_list_data_servers_ ARGS_((Integer* ));
extern void FATR ga_list_nodeid_ ARGS_((Integer*, Integer* ));
extern void FATR ga_num_data_servers_ ARGS_((Integer* ));
extern void FATR ga_put_  ARGS_((Integer*, Integer*, Integer*, Integer*,
      Integer*, Void*, Integer* ));
extern void FATR ga_nbput_ ARGS_((Integer*, Integer*, Integer*, Integer*,
      Integer*, Void*, Integer*, Integer* ));
extern void FATR ga_release_ ARGS_((Integer*, Integer*, Integer*, Integer*,
      Integer*));
extern void FATR ga_release_update_ ARGS_((Integer*, Integer*, Integer*,
      Integer*, Integer* ));
extern void FATR ga_sync_ ARGS_(( void));
extern void FATR ga_pgroup_sync_ ARGS_((Integer*));
extern void FATR ga_terminate_ ARGS_(( void));
extern logical FATR ga_uses_ma_ ARGS_(( void));
extern logical FATR ga_memory_limited_ ARGS_(( void));
extern Integer FATR ga_cluster_nnodes_ ARGS_(( void));
extern Integer FATR ga_cluster_nprocs_ ARGS_((Integer*));
extern Integer FATR ga_cluster_proc_nodeid_ ARGS_((Integer*));
extern Integer FATR ga_cluster_nodeid_ ARGS_(( void));
extern Integer FATR ga_cluster_procid_ ARGS_((Integer*, Integer*));


extern void ga_copy_patch ARGS_((char *, Integer *, Integer *, Integer *,
      Integer *, Integer *, Integer *, Integer *,
      Integer *, Integer *, Integer *));
extern DoublePrecision ga_ddot_patch ARGS_((Integer *, char*, Integer *,
      Integer *, Integer *, Integer *,Integer *, char*, Integer *,
      Integer *, Integer *, Integer *));
extern float ga_sdot_patch ARGS_((Integer *, char*, Integer *,
      Integer *, Integer *, Integer *, Integer *, char*, Integer *,
      Integer *, Integer *, Integer *));
extern DoubleComplex ga_zdot_patch ARGS_((Integer *, char*, Integer *, 
      Integer *, Integer *, Integer *, Integer *, char*, Integer *,
      Integer *, Integer *, Integer *));
extern void FATR ga_fill_patch_  ARGS_((Integer *, Integer *, Integer *,
      Integer *, Integer *, Void *));
extern void FATR ga_scale_patch_  ARGS_((Integer *, Integer *, Integer *,
      Integer *, Integer *, DoublePrecision *));
extern void FATR ga_add_patch_   ARGS_((DoublePrecision *, Integer *,
      Integer *, Integer *, Integer *, Integer *,
      DoublePrecision *, Integer *, Integer *, Integer *, Integer *, Integer *,
      Integer *, Integer *, Integer *, Integer *, Integer *  ));
extern void ga_matmul_patch  ARGS_((char *, char *, void *,
      void *, Integer *, Integer *, Integer *, Integer *, Integer *,
      Integer *, Integer *, Integer *, Integer *, Integer *, Integer *,
      Integer *, Integer *, Integer *, Integer*));

extern void nga_matmul_patch ARGS_((char *transa, char *transb, 
			     Void *alpha, Void *beta, 
			     Integer *g_a, Integer alo[], Integer ahi[], 
			     Integer *g_b, Integer blo[], Integer bhi[], 
			     Integer *g_c, Integer clo[], Integer chi[]));
  
extern void FATR ga_copy_   ARGS_((Integer *, Integer *));
extern void      ga_print_file ARGS_((FILE *, Integer *));
extern void FATR ga_print_  ARGS_((Integer *));
extern void FATR ga_print_stats_();
extern void FATR ga_zero_   ARGS_((Integer *));
extern void FATR ga_fill_   ARGS_((Integer *, void *));
extern void FATR ga_scale_  ARGS_((Integer *, void *));
extern void FATR ga_add_   ARGS_((Void *, Integer *, Void *, Integer *,
      Integer *));
extern Integer FATR ga_pgroup_get_default_();
extern Integer FATR ga_pgroup_get_mirror_();
extern Integer FATR ga_pgroup_get_world_();
extern Integer FATR ga_idot_ ARGS_((Integer *, Integer *));
extern float FATR ga_fdot_ ARGS_((Integer *, Integer *));            
extern DoublePrecision FATR ga_ddot_ ARGS_((Integer *, Integer *));
extern DoubleComplex ga_zdot ARGS_((Integer *, Integer *));
extern void FATR ga_print_patch_ ARGS_((Integer *, Integer *, Integer *,
      Integer *, Integer *, Integer *));

extern void FATR ga_summarize_     ARGS_((logical*));
extern void FATR ga_symmetrize_   ARGS_((Integer *)); 
extern void FATR ga_transpose_    ARGS_((Integer *, Integer *));
extern void FATR ga_diag_seq_     ARGS_((Integer *, Integer *, Integer *,
      DoublePrecision *));
extern void FATR ga_diag_reuse_   ARGS_((Integer*, Integer *, Integer *,
      Integer *, DoublePrecision *));
extern void FATR ga_diag_std_     ARGS_((Integer *, Integer *,
      DoublePrecision *));
extern void FATR ga_diag_std_seq_ ARGS_((Integer *, Integer *,
      DoublePrecision *));
extern void FATR ga_lu_solve_      ARGS_((char *, Integer *, Integer *));
extern void FATR ga_lu_solve_alt_  ARGS_((Integer *, Integer *, Integer *));
extern void ga_lu_solve_seq  ARGS_((char *, Integer *, Integer *));

extern Integer FATR ga_llt_solve_ ARGS_((Integer *, Integer *));
extern Integer FATR ga_solve_ ARGS_((Integer *, Integer *));
extern Integer FATR ga_spd_invert_ ARGS_((Integer *));

extern void ga_dgemm ARGS_((char *, char *, Integer *, Integer *, Integer *,
      DoublePrecision *, Integer *, Integer *, DoublePrecision *, Integer *));
extern void ga_zgemm ARGS_((char *, char *, Integer *, Integer *, Integer *,
      DoubleComplex *, Integer *, Integer *, DoubleComplex *, Integer *));
extern void ga_sgemm ARGS_((char *, char *, Integer *, Integer *, Integer *,
      float *, Integer *, Integer *, float *, Integer *));

extern void FATR ga_diag_ ARGS_((Integer *, Integer *, Integer *,
      DoublePrecision *));
extern void FATR ga_proc_topology_ ARGS_((Integer *g_a, Integer *proc,
      Integer *pr, Integer *pc));

extern void FATR ga_sort_permut_ ARGS_((Integer* g_a, Integer* index,
      Integer* i, Integer* j, Integer* nv));
#undef ARGS_

extern void FATR ga_print_distribution_(Integer *g_a);
extern Integer FATR ga_ndim_(Integer *g_a);

extern logical nga_create_ghosts_irreg(
        Integer type,    /* MA type */
        Integer ndim,    /* number of dimensions */
        Integer dims[],   /* array of dimensions */
        Integer width[],  /* array of ghost cell widths */
        char *array_name, /* array name */
        Integer map[],    /* decomposition map array */
        Integer nblock[], /* number of blocks for each dimension in map */
        Integer *g_a);    /* array handle (output) */

extern logical nga_create_ghosts_irreg_config(
        Integer type,    /* MA type */
        Integer ndim,    /* number of dimensions */
        Integer dims[],   /* array of dimensions */
        Integer width[],  /* array of ghost cell widths */
        char *array_name, /* array name */
        Integer map[],    /* decomposition map array */
        Integer nblock[], /* number of blocks for each dimension in map */
        Integer p_handle, /* processor list handle*/
        Integer *g_a);    /* array handle (output) */

extern logical nga_create_ghosts(Integer type,
                   Integer ndim,
                   Integer dims[],
                   Integer width[],
                   char* array_name,
                   Integer chunk[],
                   Integer *g_a);

extern logical nga_create_ghosts_config(Integer type,
                   Integer ndim,
                   Integer dims[],
                   Integer width[],
                   char* array_name,
                   Integer chunk[],
                   Integer p_handle,
                   Integer *g_a);

extern logical nga_create(Integer type,
                   Integer ndim,
                   Integer dims[],
                   char* array_name,
                   Integer chunk[],
                   Integer *g_a);

extern logical nga_create_config(Integer type,
                   Integer ndim,
                   Integer dims[],
                   char* array_name,
                   Integer chunk[],
                   Integer p_handle,
                   Integer *g_a);

extern logical nga_create_irreg(
        Integer type,    /* MA type */
        Integer ndim,    /* number of dimensions */
        Integer dims[],   /* array of dimensions */
        char *array_name, /* array name */
        Integer map[],    /* decomposition map array */
        Integer nblock[], /* number of blocks for each dimension in map */
        Integer *g_a);    /* array handle (output) */

extern logical nga_create_irreg_config(
        Integer type,    /* MA type */
        Integer ndim,    /* number of dimensions */
        Integer dims[],   /* array of dimensions */
        char *array_name, /* array name */
        Integer map[],    /* decomposition map array */
        Integer nblock[], /* number of blocks for each dimension in map */
        Integer p_handle, /* processor list handle*/
        Integer *g_a);    /* array handle (output) */

extern Integer ga_create_handle();
extern void ga_set_data(Integer *g_a, Integer *ndim, Integer dims[], Integer *type);
extern void ga_set_chunk(Integer *g_a, Integer chunk[]);
extern void ga_set_array_name(Integer g_a, char *array_name);
extern void ga_set_ghosts(Integer *g_a, Integer width[]);
extern void ga_set_irreg_distr(Integer *g_a, Integer map[], Integer nblock[]);
extern void ga_set_irreg_flag(Integer *g_a, logical flag);
extern void ga_set_ghost_corner_flag(Integer *g_a, logical flag);
extern Integer nga_get_dimension(Integer *g_a);
extern logical ga_allocate(Integer *g_a);
extern Integer ga_pgroup_create(Integer list[], Integer *count);
extern Integer ga_pgroup_destroy(Integer *grp);
extern Integer FATR ga_pgroup_split_( Integer *grp, Integer *num_group);
extern Integer FATR ga_pgroup_split_irreg_(Integer *grp, Integer *mycolor, Integer *key);
   
extern void ga_update_ghosts(Integer *g_a);
extern void ga_update1_ghosts(Integer *g_a);
extern logical ga_update2_ghosts(Integer *g_a);
extern logical ga_update3_ghosts(Integer *g_a);
extern logical ga_update4_ghosts(Integer *g_a);
extern logical ga_update5_ghosts(Integer *g_a);
extern logical ga_update6_ghosts(Integer *g_a);
extern logical ga_update7_ghosts(Integer *g_a);
extern logical ga_set_update4_info(Integer *g_a);
extern logical ga_set_update5_info(Integer *g_a);
extern logical nga_update_ghost_dir(Integer *g_a, Integer *idim, 
                                    Integer *idir, logical *flag);
extern void FATR ga_merge_mirrored_(Integer *g_a);
extern void FATR ga_fast_merge_mirrored_(Integer *g_a);
extern logical FATR ga_is_mirrored_(Integer *g_a);
extern Integer FATR ga_num_mirrored_seg_(Integer *g_a);
extern void nga_merge_distr_patch(Integer *g_a, Integer *alo, Integer *ahi,
                                  Integer *g_b, Integer *blo, Integer *bhi);
extern void FATR ga_get_mirrored_block_(Integer *g_a, Integer *npatch,
                                        Integer *lo,  Integer *hi);
extern void FATR nga_merge_distr_patch_(Integer *g_a, Integer *alo, Integer *ahi,
                                        Integer *g_b, Integer *blo, Integer *bhi);
extern void  FATR nga_nbget_ghost_dir_(Integer *g_a, Integer *mask, Integer *nbhandle);
extern void FATR nga_get_common(Integer *g_a, Integer *lo, Integer *hi,
                                void *buf, Integer *ld, Integer *nbhandle);
extern void FATR nga_acc_common(Integer *g_a, Integer *lo, Integer *hi,
                 void *buf, Integer *ld, void *alpha, Integer *nbhandle);


extern void FATR  nga_release_(Integer *g_a, Integer *lo, Integer *hi);
extern void FATR  nga_release_update_(Integer *g_a, Integer *lo, Integer *hi);
extern void FATR  nga_inquire_(Integer *g_a, Integer *type, Integer *ndim,
    Integer *dims); 
extern void FATR  nga_inquire_internal_(Integer *g_a, Integer *type, Integer *ndim,
    Integer *dims);

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
extern void nga_access_ghost_ptr(Integer* g_a, Integer dims[],
                           void* ptr, Integer ld[]);
extern void nga_access_ghost_element(Integer* g_a, void* ptr, Integer subscript[],
                              Integer ld[]);

extern void FATR nga_access_(Integer* g_a, Integer lo[], Integer hi[],
                             Integer* index, Integer ld[]);
extern void FATR nga_access_ghosts_(Integer* g_a, Integer dims[],
                             Integer* index, Integer ld[]);
extern void FATR nga_distribution_(Integer *g_a, Integer *proc, 
                                   Integer *lo, Integer *hi);
extern void FATR nga_put_(Integer *g_a, Integer *lo, Integer *hi, 
                          void *buf, Integer *ld);
extern void FATR nga_nbput_(Integer *g_a, Integer *lo, Integer *hi, 
                            void *buf, Integer *ld, Integer *nbhdl);
extern void FATR nga_strided_put_(Integer *g_a, Integer *lo, Integer *hi, 
                                  Integer *skip, void *buf, Integer *ld);
extern void FATR nga_get_(Integer *g_a, Integer *lo, Integer *hi, 
                          void *buf, Integer *ld);
extern void FATR nga_nbget_(Integer *g_a, Integer *lo, Integer *hi, 
                          void *buf, Integer *ld, Integer *nbhdl);
extern void FATR nga_nbwait_(Integer *nbhdl);
extern void FATR nga_acc_(Integer *g_a, Integer *lo, Integer *hi,
                          void *buf, Integer *ld, void *alpha);
extern void FATR nga_nbacc_(Integer *g_a, Integer *lo, Integer *hi,
                          void *buf, Integer *ld, void *alpha,Integer *nbhdl);
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
extern void FATR nga_proc_topology_(Integer* g_a, Integer* proc,
    Integer* subscr);

extern void nga_copy_patch(char *trans,
                           Integer *g_a, Integer *alo, Integer *ahi,
                           Integer *g_b, Integer *blo, Integer *bhi);
extern Integer nga_idot_patch(Integer *g_a, char *t_a, Integer *alo,
          Integer *ahi, Integer *g_b, char *t_b, Integer *blo, Integer *bhi);
extern void FATR nga_print_patch_(Integer *g_a, Integer *lo, Integer *hi,
    Integer *pretty);

extern DoublePrecision nga_ddot_patch(Integer *g_a, char *t_a, 
          Integer *alo, Integer *ahi, Integer *g_b, char *t_b, Integer *blo,
          Integer *bhi);

extern float nga_fdot_patch(Integer *g_a, char *t_a,
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
extern int ga_type_c2f(int type);
extern int ga_type_f2c(int type);
extern void ga_type_gop(int type, void *x, int n, char* op);
extern void FATR ga_pack_(Integer* g_a, Integer* g_b, Integer* g_sbit,
                     Integer* lo, Integer* hi, Integer* icount);
extern void FATR ga_unpack_(Integer* g_a, Integer* g_b, Integer* g_sbit,
                       Integer* lo, Integer* hi, Integer* icount);
extern void FATR ga_scan_copy_(Integer* g_a, Integer* g_b, Integer* g_sbit,
                          Integer* lo, Integer* hi);
extern void FATR ga_scan_add_(Integer* g_a, Integer* g_b, Integer* g_sbit,
                          Integer* lo, Integer* hi);
extern void nga_select_elem_(Integer *g_a, char* op, void* val,
                             Integer *subscript);


/* added by Limin */
extern void FATR ga_add_constant_(Integer *g_a, void *);
extern void FATR ga_abs_value_(Integer *);
extern void FATR ga_recip_(Integer *g_a);
extern void FATR ga_abs_value_patch_ (Integer *, Integer *, Integer *);
extern void FATR ga_add_constant_patch_(Integer *, Integer *, Integer *, 
					 void *);

extern void FATR ga_recip_patch_(Integer *g_a, Integer *lo, Integer *hi);

extern void FATR ga_elem_multiply_(Integer *g_a, Integer *g_b, Integer *g_c);
extern void FATR ga_elem_divide_(Integer *g_a, Integer *g_b, Integer *g_c);
extern void FATR ga_elem_maximum_(Integer *g_a, Integer *g_b, Integer *g_c);
extern void FATR ga_elem_minimum_(Integer *g_a, Integer *g_b, Integer *g_c);
extern void FATR ga_elem_multiply_patch_(Integer *g_a,Integer *alo,Integer *ahi, 
				    Integer *g_b,Integer *blo,Integer *bhi,
				    Integer *g_c,Integer *clo,Integer *chi);
extern void FATR ga_elem_divide_patch_(Integer *g_a,Integer *alo,Integer *ahi,
				  Integer *g_b,Integer *blo,Integer *bhi,
				  Integer *g_c,Integer *clo,Integer *chi);
extern void FATR ga_elem_maximum_patch_(Integer *g_a,Integer *alo,Integer *ahi,
				   Integer *g_b,Integer *blo,Integer *bhi,
				   Integer *g_c,Integer *clo,Integer *chi);
extern void FATR ga_elem_minimum_patch_(Integer *g_a,Integer *alo,Integer *ahi,
				   Integer *g_b,Integer *blo,Integer *bhi,
				   Integer *g_c,Integer *clo,Integer *chi);
extern void FATR ga_step_max_patch_(Integer *g_a, Integer *alo, Integer *ahi, 
			       Integer *g_b,  Integer *blo, Integer *bhi, 
			       void *step);
extern void FATR ga_step_max_(Integer *g_a, Integer *g_b, void *step);
extern void FATR ga_step_bound_info_patch_(Integer *g_xx, Integer *xxlo,Integer *xxhi,
				Integer *g_vv, Integer *vvlo, Integer *vvhi, 
				Integer *g_xxll, Integer *xxlllo, 
				Integer *xxllhi, Integer *g_xxuu, 
				Integer *xxuulo, Integer *xxuuhi, 
				void *boundmin, void *wolfemin,
				void *boundmax);
extern void FATR ga_step_bound_info_(Integer *g_xx, Integer *g_vv, Integer *g_xxll, 
			  Integer *g_xxuu, void *boundmin,
			  void *wolfemin, void *boundmax);

extern void FATR ga_shift_diagonal_(Integer *g_a, void *c);
extern void FATR ga_set_diagonal_(Integer *g_a, Integer *g_v);
extern void FATR ga_zero_diagonal_(Integer *g_a);
extern void FATR ga_add_diagonal_(Integer *g_a, Integer *g_v);
extern void FATR ga_get_diag_(Integer *g_a, Integer *g_v);
extern void FATR ga_scale_rows_(Integer *g_a, Integer *g_v);
extern void FATR ga_scale_cols_(Integer *g_a, Integer *g_v);
extern void FATR ga_norm1_(Integer *g_a, double *nm);
extern void FATR ga_norm_infinity_(Integer *g_a, double *nm);
extern void FATR ga_median_(Integer *g_a, Integer *g_b, Integer *g_c, Integer *g_m);
extern void FATR ga_median_patch_(Integer *g_a, Integer *alo, Integer *ahi, 
			     Integer *g_b, Integer *blo, Integer *bhi, 
			     Integer *g_c, Integer *clo, Integer *chi, 
			     Integer *g_m, Integer *mlo, Integer *mhi);

extern int GA_Cluster_nnodes();         
extern int GA_Cluster_nodeid();
extern int GA_Cluster_nprocs(int x);
extern int GA_Cluster_procid(int x, int y);
extern int GA_Cluster_proc_nodeid(int proc);

#ifdef __cplusplus
}
#endif

extern DoubleComplex   *DCPL_MB;
extern DoublePrecision *DBL_MB;
extern Integer             *INT_MB;
extern float           *FLT_MB;
#endif 
