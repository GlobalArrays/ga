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

extern void GA_Initialize(void);
extern void GA_Initialize_ltd(size_t limit);
extern int NGA_Create(int type,int ndim,int dims[], char *name, int chunk[]);
extern int NGA_Create_irreg(int type,int ndim,int dims[],char *name,
                            int map[], int block[]);
extern int GA_Duplicate(int g_a, char* array_name);
extern void GA_Destroy(int g_a);
extern void GA_Terminate(void);
extern void GA_Zero(int g_a); 
extern void GA_Fill(int g_a, void *value);
extern Integer GA_Idot(int g_a, int g_b);
extern double GA_Ddot(int g_a, int g_b); 
extern DoubleComplex GA_Zdot(int g_a, int g_b); 
extern void GA_Scale(int g_a, void *value); 
extern void GA_Add(void *alpha, int g_a, void* beta, int g_b, int g_c); 
extern void GA_Copy(int g_a, int g_b); 
extern void NGA_Get(int g_a, int lo[], int hi[], void* buf, int ld[]); 
extern void NGA_Put(int g_a, int lo[], int hi[], void* buf, int ld[]); 
extern void NGA_Acc(int g_a, int lo[], int hi[],void* buf,int ld[],void* alpha);
extern void NGA_Periodic_get(int g_a, int lo[], int hi[], void* buf, int ld[]); 
extern void NGA_Periodic_put(int g_a, int lo[], int hi[], void* buf, int ld[]); 
extern void NGA_Periodic_acc(int g_a, int lo[], int hi[],void* buf,int ld[],void* alpha);
extern long NGA_Read_inc(int g_a, int subscript[], long inc);
extern void NGA_Distribution(int g_a, int iproc, int lo[], int hi[]); 
extern int GA_Compare_distr(int g_a, int g_b); 
extern void GA_Print_distribution(int g_a); 
extern void NGA_Access(int g_a, int lo[], int hi[], void *ptr, int ld[]);
extern void NGA_Release(int g_a, int lo[], int hi[]);
extern void NGA_Release_update(int g_a, int lo[], int hi[]);
extern void NGA_Scatter(int g_a, void *v, int* subsArray[], int n);
extern void NGA_Gather(int g_a, void *v, int* subsArray[], int n);
extern void GA_Error(char *message, int code);
extern int NGA_Locate(int g_a, int subscript[]);
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
extern void GA_Init_fence(void);
extern void GA_Fence(void);
extern int GA_Create_mutexes(int number);
extern int GA_Destroy_mutexes();
extern void GA_Lock(int mutex);
extern void GA_Unlock(int mutex);
extern int GA_Nodeid();
extern int GA_Nnodes();
extern void GA_Dgemm(char ta, char tb, int m, int n, int k, 
                     double alpha, int g_a, int g_b, double beta, int g_c );
extern void GA_Copy_patch(char ta,int g_a, int ailo, int aihi,int ajlo,int ajhi,
                             int g_b, int bilo, int bihi, int bjlo,int bjhi);
extern void GA_Brdcst(void *buf, int lenbuf, int root);
extern void GA_Dgop(double x[], int n, char *op);
extern void GA_Igop(Integer x[], int n, char *op);

extern void NGA_Copy_patch(char trans, int g_a, int alo[], int ahi[],
                           int g_b, int blo[], int bhi[]);
extern int NGA_Idot_patch(int g_a, char t_a, int alo[], int ahi[],
                          int g_b, char t_b, int blo[], int bhi[]);
extern double NGA_Ddot_patch(int g_a, char t_a, int alo[], int ahi[],
                             int g_b, char t_b, int blo[], int bhi[]);
extern DoubleComplex NGA_Zdot_patch(int g_a, char t_a, int alo[], int ahi[],
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
extern void GA_Transpose(int g_a, int g_b);

extern int  GA_Valid_handle(int g_a);

#define GA_Initialize ga_initialize_
#define GA_Terminate ga_terminate_
#define GA_Sync ga_sync_
#define GA_Error(str,code) ga_error((str),(Integer)(code))
#define GA_Inquire_memory (size_t)ga_inquire_memory_
#define GA_Memory_avail (size_t)ga_memory_avail_
#define GA_Uses_ma (int)ga_uses_ma_
#define GA_Print_stats ga_print_stats_
#define GA_Init_fence  ga_init_fence_
#define GA_Nodeid (int)ga_nodeid_
#define GA_Nnodes (int)ga_nnodes_

#ifdef __cplusplus
}
#endif

#endif
