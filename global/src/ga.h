#ifndef _GA_H_
#define _GA_H_
#include <stdio.h>
#ifdef WIN32
#include <windows.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/types.h>
#include "global.h"


extern void GA_Abs_value(int g_a); 
extern void GA_Add_constant(int g_a, void* alpha);
extern void GA_Recip(int g_a);
extern void GA_Elem_multiply(int g_a, int g_b, int g_c);
extern void GA_Elem_divide(int g_a, int g_b, int g_c);
extern void GA_Elem_maximum(int g_a, int g_b, int g_c);
extern void GA_Elem_minimum(int g_a, int g_b, int g_c);
extern void GA_Abs_value_patch(int g_a, int *lo, int *hi);
extern void GA_Add_constant_patch(int g,int *lo,int *hi,void *alpha);
extern void GA_Recip_patch(int g_a,int *lo, int *hi);
extern void GA_Step_max(int g_a, int g_b, void *step);
extern void GA_Step_bound_info(int g_xx, int g_vv, int g_xxll, int g_xxuu, void *boundmin, void *wolfemin, void *boundmax);
extern void GA_Step_max_patch(int g_a, int *alo, int *ahi, int g_b, int *blo, int *bhi, void *step);
extern void GA_Step_bound_info_patch(int g_xx, int *xxlo, int *xxhi, int g_vv, int *vvlo, int *vvhi, int g_xxll, int *xxlllo, int *xxllhi, int g_xxuu, int *xxuulo, int *xxuuhi, void *boundmin, void *wolfemin, void *boundmax);
extern void GA_Elem_multiply_patch(int g_a,int *alo,int *ahi,
				     	int g_b,int *blo,int *bhi,int g_c,int *clo,int *chi);
extern void GA_Elem_divide_patch(int g_a,int *alo,int *ahi,
				     	int g_b,int *blo,int *bhi,int g_c,int *clo,int *chi);
extern void GA_Elem_maximum_patch(int g_a,int *alo,int *ahi,
				     	int g_b,int *blo,int *bhi,int g_c,int *clo,int *chi);
extern void GA_Elem_minimum_patch(int g_a,int *alo,int *ahi,
				     	int g_b,int *blo,int *bhi,int g_c,int *clo,int *chi);


/*Added by Limin for matrix operations*/
extern void GA_Shift_diagonal(int g_a, void *c);
extern void GA_Set_diagonal(int g_a, int g_v);
extern void GA_Zero_diagonal(int g_a);
extern void GA_Add_diagonal(int g_a, int g_v);
extern void GA_Get_diag(int g_a, int g_v);
extern void GA_Scale_rows(int g_a, int g_v);
extern void GA_Scale_cols(int g_a, int g_v);
extern void GA_Norm1(int g_a, double *nm);
extern void GA_Norm_infinity(int g_a, double *nm);
extern void GA_Median(int g_a, int g_b, int g_c, int g_m);
extern void GA_Median_patch(int g_a, int *alo, int *ahi, int g_b, int *blo, int *bhi,
                            int g_c, int *clo, int *chi, int g_m, int *mlo, int *mhi);

extern void GA_Initialize(void);
extern void GA_Initialize_ltd(size_t limit);
extern int NGA_Create(int type,int ndim,int dims[], char *name, int chunk[]);
extern int NGA_Create_irreg(int type,int ndim,int dims[],char *name,
                            int map[], int block[]);
extern int NGA_Create_ghosts(int type,int ndim,int dims[], int width[], char *name, int chunk[]);
extern int NGA_Create_ghosts_irreg(int type,int ndim,int dims[], int width[],
                                   char *name, int map[], int nblock[]);
extern int NGA_Create_config(int type,int ndim,int dims[], char *name,
                             int chunk[], int p_handle);
extern int NGA_Create_irreg_config(int type,int ndim,int dims[],char *name,
                                   int map[], int block[], int p_handle);
extern int NGA_Create_ghosts_config(int type,int ndim,int dims[], int width[],
                                    char *name, int chunk[], int p_handle);
extern int NGA_Create_ghosts_irreg_config(int type,int ndim,int dims[], int width[],
                                          char *name, int map[], int nblock[], int p_handle);
extern int GA_Create_handle();
extern void GA_Set_data(int g_a, int ndim, int dims[], int type);
extern void GA_Set_chunk(int g_a, int chunk[]);
extern void GA_Set_array_name(int g_a, char *name);
extern void GA_Set_pgroup(int g_a, int p_handle);
extern int GA_Get_pgroup(int g_a);
extern void GA_Set_ghosts(int g_a, int width[]);
extern void GA_Set_irreg_distr(int g_a, int map[], int block[]);
extern void GA_Set_irreg_flag(int g_a, int flag);
extern void GA_Set_ghost_corner_flag(int g_a, int flag);
extern int GA_Get_dimension(int g_a);
extern int GA_Allocate(int g_a);
extern int GA_Pgroup_create(int *list, int count);
extern int GA_Pgroup_destroy(int grp);
extern int GA_Pgroup_split(int grp_id, int num_group);
extern int GA_Pgroup_split_irreg(int grp_id, int color, int key);
extern void GA_Update_ghosts(int g_a);
extern void GA_Merge_mirrored(int g_a);
extern void GA_Fast_merge_mirrored(int g_a);
extern void NGA_Merge_distr_patch(int g_a, int alo[], int ahi[],
                                  int g_b, int blo[], int bhi[]);
extern int GA_Is_mirrored(int g_a);
extern int GA_Num_mirrored_seg(int g_a);
extern void GA_Get_mirrored_block(int g_a, int nblock, int lo[], int hi[]);
extern int NGA_Update_ghost_dir(int g_a, int dimension, int idir, int flag);
extern int GA_Has_ghosts(int g_a);
extern void NGA_Access_ghosts(int g_a, int dims[], void *ptr, int ld[]);
extern void NGA_Access_ghost_element(int g_a,  void *ptr, int subscript[], int ld[]);
extern void GA_Mask_sync(int first, int last);
extern int GA_Duplicate(int g_a, char* array_name);
extern void GA_Destroy(int g_a);
extern void GA_Terminate(void);
extern void GA_Zero(int g_a); 
extern void GA_Fill(int g_a, void *value);
extern int GA_Pgroup_get_default();
extern void GA_Pgroup_set_default(int p_handle);
extern int GA_Pgroup_get_mirror();
extern int GA_Pgroup_get_world();
extern int GA_Idot(int g_a, int g_b);
extern long GA_Ldot(int g_a, int g_b);
extern double GA_Ddot(int g_a, int g_b); 
extern DoubleComplex GA_Zdot(int g_a, int g_b); 
extern float GA_Fdot(int g_a, int g_b);
extern void GA_Scale(int g_a, void *value); 
extern void GA_Add(void *alpha, int g_a, void* beta, int g_b, int g_c); 
extern void GA_Copy(int g_a, int g_b); 
extern void NGA_Get(int g_a, int lo[], int hi[], void* buf, int ld[]); 
extern void NGA_Strided_get(int g_a, int lo[], int hi[], int skip[],
                            void* buf, int ld[]); 
extern void NGA_Put(int g_a, int lo[], int hi[], void* buf, int ld[]); 
extern void NGA_Strided_put(int g_a, int lo[], int hi[], int skip[],
                            void* buf, int ld[]); 
extern void NGA_Acc(int g_a, int lo[], int hi[],void* buf,int ld[],void* alpha);
extern void NGA_Strided_acc(int g_a, int lo[], int hi[], int skip[],
                            void* buf, int ld[], void *alpha); 
extern void NGA_Periodic_get(int g_a, int lo[], int hi[], void* buf, int ld[]); 
extern void NGA_Periodic_put(int g_a, int lo[], int hi[], void* buf, int ld[]); 
extern void NGA_Periodic_acc(int g_a, int lo[], int hi[],void* buf,int ld[],void* alpha);
extern long NGA_Read_inc(int g_a, int subscript[], long inc);
extern void NGA_Distribution(int g_a, int iproc, int lo[], int hi[]); 
extern void NGA_Distribution_no_handle(int ndim, const int dims[], const int nblock[], const int mapc[], int iproc, int lo[], int hi[]);
extern int GA_Compare_distr(int g_a, int g_b); 
extern void GA_Print_distribution(int g_a); 
extern void NGA_Access(int g_a, int lo[], int hi[], void *ptr, int ld[]);
extern void NGA_Release(int g_a, int lo[], int hi[]);
extern void NGA_Release_update(int g_a, int lo[], int hi[]);
extern void NGA_Scatter(int g_a, void *v, int* subsArray[], int n);
extern void NGA_Gather(int g_a, void *v, int* subsArray[], int n);
extern void GA_Error(char *message, int code);
extern int NGA_Locate(int g_a, int subscript[]);
extern int NGA_Locate_num_blocks(int g_a, int lo[], int hi[]);
extern int NGA_Locate_region(int g_a,int lo[],int hi[],int map[],int procs[]);
extern void NGA_Inquire(int g_a, int *type, int *ndim, int dims[]);
extern size_t GA_Inquire_memory(void);
extern char* GA_Inquire_name(int g_a);
extern size_t GA_Memory_avail(void);
extern int GA_Uses_fapi(void);
extern int GA_Uses_ma(void);
extern int GA_Memory_limited(void);
extern void GA_Set_memory_limit(size_t limit);
extern int NGA_Create(int type,int ndim,int dims[], char *name, int chunk[]);
extern void GA_Proc_topology(int g_a, int proc, int *prow, int *pcol);
extern void NGA_Proc_topology(int g_a, int proc, int coord[]);
extern void GA_Print_patch(int g_a,int ilo,int ihi,int jlo,int jhi,int pretty);
extern void GA_Print_stats(void);
extern void GA_Check_handle(int g_a, char *string);
extern int GA_Create_mutexes(int number);
extern int GA_Destroy_mutexes();
extern void GA_Lock(int mutex);
extern void GA_Unlock(int mutex);
extern int GA_Nodeid();
extern int GA_Nnodes();
extern int GA_Pgroup_nodeid(int grp_id);
extern int GA_Pgroup_nnodes(int grp_id);
extern void GA_Dgemm(char ta, char tb, int m, int n, int k, 
                     double alpha, int g_a, int g_b, double beta, int g_c );
extern void GA_Zgemm(char ta, char tb, int m, int n, int k, 
                     DoubleComplex alpha, int g_a, int g_b, 
		     DoubleComplex beta, int g_c );
extern void GA_Sgemm(char ta, char tb, int m, int n, int k, 
                     float alpha, int g_a, int g_b, float beta, int g_c );
extern void GA_Copy_patch(char ta,int g_a, int ailo, int aihi,int ajlo,int ajhi,
                             int g_b, int bilo, int bihi, int bjlo,int bjhi);
extern void GA_Brdcst(void *buf, int lenbuf, int root);
extern void GA_Pgroup_brdcst(int grp, void *buf, int lenbuf, int root);
extern void GA_Pgroup_sync(int grp_id);
extern void GA_Gop(int type, void *x, int n, char *op);
extern void GA_Dgop(double x[], int n, char *op);
extern void GA_Pgroup_dgop(int grp, double x[], int n, char *op);
extern void GA_Lgop(long x[], int n, char *op);
extern void GA_Pgroup_lgop(int grp, long x[], int n, char *op);
extern void GA_Igop(Integer x[], int n, char *op);
extern void GA_Pgroup_igop(int grp, Integer x[], int n, char *op);

extern void NGA_Copy_patch(char trans, int g_a, int alo[], int ahi[],
                           int g_b, int blo[], int bhi[]);
extern int NGA_Idot_patch(int g_a, char t_a, int alo[], int ahi[],
                          int g_b, char t_b, int blo[], int bhi[]);
extern double NGA_Ddot_patch(int g_a, char t_a, int alo[], int ahi[],
                             int g_b, char t_b, int blo[], int bhi[]);
extern DoubleComplex NGA_Zdot_patch(int g_a, char t_a, int alo[], int ahi[],
                                    int g_b, char t_b, int blo[], int bhi[]);
extern float NGA_Fdot_patch(int g_a, char t_a, int alo[], int ahi[],
                                    int g_b, char t_b, int blo[], int bhi[]);
extern void NGA_Zero_patch(int g_a, int lo[], int hi[]);
    
extern void NGA_Fill_patch(int g_a, int lo[], int hi[], void *val);
extern void NGA_Scale_patch(int g_a, int lo[], int hi[], void *alpha);
extern void NGA_Add_patch(void * alpha, int g_a, int alo[], int ahi[],
                          void * beta,  int g_b, int blo[], int bhi[],
                          int g_c, int clo[], int chi[]);

extern void NGA_Print_patch(int g_a, int lo[], int hi[], int pretty);
extern void GA_Print(int g_a);
extern void GA_Print_file(FILE *file, int g_a);
extern void GA_Diag(int g_a, int g_s, int g_v, void *eval);
extern void GA_Diag_std(int g_a, int g_v, void *eval);
extern void GA_Diag_seq(int g_a, int g_s, int g_v, void *eval);
extern void GA_Diag_std_seq(int g_a, int g_v, void *eval);
extern void GA_Diag_reuse(int reuse, int g_a, int g_s, int g_v, void *eval);
extern void GA_Lu_solve(char tran, int g_a, int g_b);
extern int GA_Llt_solve(int g_a, int g_b);
extern int GA_Solve(int g_a, int g_b);
extern int GA_Spd_invert(int g_a);
extern void GA_Summarize(int verbose);
extern void GA_Symmetrize(int g_a);
extern void NGA_Select_elem(int g_a, char* op, void* val, int *index);
extern void GA_Transpose(int g_a, int g_b);
extern int  GA_Ndim(int g_a);

extern int  GA_Valid_handle(int g_a);
extern void GA_Nblock(int g_a, int *nblock);
extern void GA_Matmul_patch(char transa, char transb, void* alpha, void *beta,
                            int g_a, int ailo, int aihi, int ajlo, int ajhi,
                            int g_b, int bilo, int bihi, int bjlo, int bjhi,
                            int g_c, int cilo, int cihi, int cjlo, int cjhi);
extern void NGA_Matmul_patch(char transa, char transb, void* alpha, void *beta,
			     int g_a, int alo[], int ahi[], 
			     int g_b, int blo[], int bhi[], 
			     int g_c, int clo[], int chi[]) ;
#define GA_Initialize ga_initialize_
#define GA_Terminate ga_terminate_
#define GA_Sync ga_sync_
#define GA_Error(str,code) ga_error((str),(int)(code))
#define GA_Inquire_memory (size_t)ga_inquire_memory_
#define GA_Memory_avail (size_t)ga_memory_avail_
#define GA_Uses_ma (int)ga_uses_ma_
#define GA_Print_stats ga_print_stats_
#define GA_Init_fence  ga_init_fence_
#define GA_Fence  ga_fence_
#define GA_Nodeid (int)ga_nodeid_
#define GA_Nnodes (int)ga_nnodes_
#define GA_Pgroup_nodeid (int)ga_pgroup_nodeid_
#define GA_Pgroup_nnodes (int)ga_pgroup_nnodes_
#define ga_nbhdl_t Integer

extern int GA_Cluster_nnodes();
extern int GA_Cluster_nodeid();
extern int GA_Cluster_nprocs(int x);
extern int GA_Cluster_procid(int x, int y);
extern void GA_Register_stack_memory(void * (*ext_alloc)(), 
				     void (*ext_free)());

/* Non-blocking APIs */
extern void NGA_NbGet(int g_a, int lo[], int hi[], void* buf, int ld[],
		      ga_nbhdl_t* nbhandle);
extern void NGA_NbPut(int g_a, int lo[], int hi[], void* buf, int ld[],
		      ga_nbhdl_t* nbhandle);
extern void NGA_NbAcc(int g_a,int lo[], int hi[],void* buf,int ld[],void* alpha,
		      ga_nbhdl_t* nbhandle);
extern int NGA_NbWait(ga_nbhdl_t* nbhandle);
extern void NGA_NbGet_ghost_dir(int g_a, int mask[], ga_nbhdl_t* handle);


#ifdef __cplusplus
}
#endif

#endif
