#ifndef _MESSAGE_H_
#define _MESSAGE_H_

#define BUF_SIZE 256
#define ARMCI_TAG 30000

extern void armci_msg_brdcst(void* buffer, int len, int root);
extern void armci_msg_snd(int tag, void* buffer, int len, int to);
extern void armci_msg_rcv(int tag, void* buffer, int buflen, int *msglen, int from);
extern void armci_msg_clus_brdcst(void *buf, int len);
extern void armci_msg_clus_igop(long *x, int n, char* op, int logint); 
extern void armci_msg_igop(long *x, int n, char* op, int logint);
extern void armci_exchange_address(void *ptr_ar[], int n);
extern void armci_msg_barrier();
extern void armci_msg_barrier();
extern int  armci_msg_me();
extern int  armci_msg_nproc();
extern void armci_msg_abort(int code);

#define MPI 

#ifdef MPI
#include <mpi.h>
#endif

#endif
