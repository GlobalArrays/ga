#ifndef _MESSAGE_H_
#define _MESSAGE_H_

#include "armci.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#define ARMCI_INT       -99
#define ARMCI_LONG      -101
#define ARMCI_LONG_LONG -102
#define ARMCI_FLOAT     -306
#define ARMCI_DOUBLE    -307

#define SCOPE_ALL     333
#define SCOPE_NODE    337
#define SCOPE_MASTERS 339

#define armci_msg_sel(x,n,op,type,contribute)\
        armci_msg_sel_scope(SCOPE_ALL,(x),(n),(op),(type),(contribute)) 
#if 0
#define armci_msg_bcast(buffer, len, root)\
        armci_msg_bcast_scope(SCOPE_ALL, (buffer), (len), (root))
#else
extern void armci_msg_bcast(void *buffer, int len, int root);
#endif

extern void armci_msg_sel_scope(int scope, void *x, int n, char* op, 
                                int type, int contribute);
extern void armci_msg_bcast_scope(int scope, void* buffer, int len, int root);
extern void armci_msg_brdcst(void* buffer, int len, int root);
extern void armci_msg_snd(int tag, void* buffer, int len, int to);
extern void armci_msg_rcv(int tag, void* buffer, int buflen, int *msglen, int from);
extern int  armci_msg_rcvany(int tag, void* buffer, int buflen, int *msglen);
extern void armci_msg_reduce(void *x, int n, char *op, int type);
extern void armci_msg_reduce_scope(int scope, void *x, int n, char *op, int type);

extern void armci_msg_gop_scope(int scope, void *x, int n, char* op, int type);
extern void armci_msg_igop(int *x, int n, char* op);
extern void armci_msg_lgop(long *x, int n, char* op);
extern void armci_msg_llgop(long long *x, int n, char* op);
extern void armci_msg_fgop(float *x, int n, char* op);
extern void armci_msg_dgop(double *x, int n, char* op);
extern void armci_exchange_address(void *ptr_ar[], int n);
extern void armci_msg_barrier();
extern void armci_msg_bintree(int scope, int* Root, int *Up, int *Left, int *Right);

extern int  armci_msg_me();
extern int  armci_msg_nproc();
extern void armci_msg_abort(int code);
extern void armci_msg_init(int *argc, char ***argv);
extern void armci_msg_init_comm(MPI_Comm comm);
extern void armci_msg_finalize();
extern double armci_timer();

extern void armci_msg_clus_brdcst(void *buf, int len);
extern void armci_msg_clus_igop(int *x, int n, char* op); 
extern void armci_msg_clus_fgop(float *x, int n, char* op); 
extern void armci_msg_clus_lgop(long *x, int n, char* op); 
extern void armci_msg_clus_llgop(long long *x, int n, char* op); 
extern void armci_msg_clus_dgop(double *x, int n, char* op); 

extern void armci_msg_group_gop_scope(int scope, void *x, int n, char* op, int type, ARMCI_Group *group);
extern void armci_msg_group_igop(int *x, int n, char* op,ARMCI_Group *group);
extern void armci_msg_group_lgop(long *x, int n, char* op,ARMCI_Group *group);
extern void armci_msg_group_llgop(long long *x, int n, char* op,ARMCI_Group *group);
extern void armci_msg_group_fgop(float *x, int n, char* op,ARMCI_Group *group);
extern void armci_msg_group_dgop(double *x, int n,char* op,ARMCI_Group *group);
extern void armci_exchange_address_grp(void *ptr_arr[], int n, ARMCI_Group *group);
extern void armci_msg_group_barrier(ARMCI_Group *group);
extern void armci_msg_group_bcast_scope(int scope, void *buf, int len, int root, ARMCI_Group *group);
extern void armci_grp_clus_brdcst(void *buf, int len, int grp_master, int grp_clus_nproc,ARMCI_Group *mastergroup);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

  
#endif
