#ifndef _MESSAGE_H_
#define _MESSAGE_H_

#define BUF_SIZE  1024
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

#if defined(PVM)
#   include <pvm3.h>
#elif defined(TCG)
#   include <sndrcv.h>
#else
#   ifndef MPI
#      define MPI 
#   endif
#   include <mpi.h>
#endif

#endif
