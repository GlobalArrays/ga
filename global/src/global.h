/* file global.h */

#include "types.f2c.h"


#ifdef FALSE
#undef FALSE
#endif
#ifdef TRUE 
#undef TRUE
#endif
#define FALSE (logical) 0
#define TRUE  (logical) 1

#ifdef CRAY_T3D
#include "cray.names.h"
#endif


#if !defined(NX) && defined(__STDC__) || defined(__cplusplus)
# define ARGS_(s) s
#else
# define ARGS_(s) ()
#endif

extern Integer ga_nnodes_   ARGS_(( void));
extern Integer ga_nodeid_   ARGS_(( void));
extern Integer ga_read_inc_ ARGS_((Integer*, Integer*, Integer*, Integer* ));
extern Integer ga_verify_handle_ ARGS_((Integer* ));
extern logical ga_create ARGS_((Integer*, Integer*, Integer*, char*, Integer*,                                  Integer*, Integer*));
extern logical ga_create_irreg ARGS_((Integer*, Integer*, Integer*, char*,                                            Integer*, Integer*, Integer*, Integer*,                                         Integer* ));
extern logical ga_destroy_  ARGS_((Integer* ));
extern logical ga_duplicate ARGS_((Integer*, Integer*, char* ));
extern logical ga_locate_   ARGS_((Integer*, Integer*, Integer*, Integer* ));
extern void ga_check_handle ARGS_((Integer*, char*));
extern logical ga_locate_region_ ARGS_((Integer*, Integer*, Integer*, Integer*,                                         Integer*, Integer map[][5], Integer* ));
extern void  ga_acc_   ARGS_((Integer*, Integer*, Integer*, Integer*, Integer*,                               DoublePrecision*, Integer*, DoublePrecision* ));
extern void ga_access_ ARGS_((Integer*, Integer*, Integer*, Integer*, Integer*,                               Integer*, Integer* ));
extern void ga_brdcst_ ARGS_((Integer*, Void*, Integer*, Integer* ));
extern void ga_clean_mem ARGS_(( void));
extern void ga_gather_ ARGS_((Integer*, Void*, Integer*, Integer*, Integer* ));
extern void ga_dgop ARGS_((Integer, DoublePrecision*, Integer, char* ));
extern void ga_distribution_ ARGS_((Integer*, Integer*, Integer*, Integer*,                                         Integer*, Integer* ));
extern void ga_scatter_ ARGS_((Integer*, Void*, Integer*, Integer*, Integer*));
extern void ga_error ARGS_((char*, Integer));
extern void ga_get_  ARGS_((Integer*, Integer*, Integer*, Integer*, Integer*,                               Void*, Integer* ));
extern void ga_igop  ARGS_((Integer, Integer*, Integer, char* ));
extern void ga_initialize_ ARGS_(( void));
extern void ga_inquire_ ARGS_((Integer*, Integer*, Integer*, Integer* ));
extern void ga_inquire_name ARGS_((Integer*, char* ));
extern void ga_list_data_servers_ ARGS_((Integer* ));
extern void ga_list_nodeid_ ARGS_((Integer*, Integer* ));
extern void ga_num_data_servers_ ARGS_((Integer* ));
extern void ga_put_  ARGS_((Integer*, Integer*, Integer*, Integer*, Integer*,                               Void*, Integer* ));
extern void ga_release_ ARGS_((Integer*, Integer*, Integer*,Integer*,Integer*));
extern void ga_release_update_ ARGS_((Integer*, Integer*, Integer*, Integer*,                                         Integer* ));
extern void ga_sync_ ARGS_(( void));
extern void ga_terminate_ ARGS_(( void));


extern void ga_copy_patch ARGS_((char *, Integer *, Integer *, Integer *,                                        Integer *, Integer *, Integer *, Integer *,                                     Integer *, Integer *, Integer *));
extern DoublePrecision ga_ddot_patch ARGS_((Integer *, char*, Integer *,                                                   Integer *, Integer *, Integer *,                                                Integer *, char*, Integer *,
                                           Integer *, Integer *, Integer *));
extern void ga_ifill_patch_  ARGS_((Integer *, Integer *, Integer *, Integer *,                                     Integer *, Integer *));
extern void ga_dfill_patch_  ARGS_((Integer *, Integer *, Integer *, Integer *,                                     Integer *, DoublePrecision *));
extern void ga_dscal_patch_  ARGS_((Integer *, Integer *, Integer *, Integer *,                                     Integer *, DoublePrecision *));
extern void ga_dadd_patch_   ARGS_((DoublePrecision *, Integer *,                                                   Integer *, Integer *, Integer *, Integer *,                                     DoublePrecision *, Integer *,                                                   Integer *, Integer *, Integer *, Integer *,                                     Integer *, Integer *, Integer *, Integer *,                                     Integer *  ));
extern void ga_matmul_patch  ARGS_((char *, char *,                                                                 DoublePrecision *, DoublePrecision *,                                           Integer *, Integer *, Integer *, Integer *,                                     Integer *, Integer *, Integer *,                                                Integer *, Integer *, Integer *, Integer *, 
                                    Integer *, Integer *, Integer *, Integer*));

extern void ga_copy_   ARGS_((Integer *, Integer *));
extern void ga_print_  ARGS_((Integer *));
extern void ga_zero_   ARGS_((Integer *));
extern void ga_dscal_  ARGS_((Integer *, DoublePrecision *));
extern void ga_dadd_   ARGS_((DoublePrecision *, Integer *, DoublePrecision *,                                Integer *, Integer *));
extern DoublePrecision ga_ddot_ ARGS_((Integer *, Integer *));
extern void ga_print_patch_ ARGS_((Integer *, Integer *, Integer *, Integer *,                                     Integer *, Integer *));

extern void ga_summarize     ARGS_((logical*));
extern void ga_symmetrize_   ARGS_((Integer *)); 
extern void ga_transpose_    ARGS_((Integer *, Integer *));
extern void ga_diag_seq_     ARGS_((Integer *, Integer *, Integer *,                                                DoublePrecision *));
extern void ga_diag_reuse_   ARGS_((Integer*, Integer *, Integer *, Integer *,
                                  DoublePrecision *));
extern void ga_diag_std_     ARGS_((Integer *, Integer *, DoublePrecision *));
extern void ga_diag_std_seq_ ARGS_((Integer *, Integer *, DoublePrecision *));
extern void ga_lu_solve      ARGS_((char *, Integer *, Integer *));
extern void ga_lu_solve_seq  ARGS_((char *, Integer *, Integer *));

extern void ga_dgemm ARGS_((char *, char *, Integer *, Integer *, Integer *,                                DoublePrecision *, Integer *, Integer *,                                        DoublePrecision *, Integer *));
extern void ga_diag_ ARGS_((Integer *, Integer *, Integer *,DoublePrecision *));
#undef ARGS_


extern DoublePrecision *DBL_MB;
extern Integer         *INT_MB;

