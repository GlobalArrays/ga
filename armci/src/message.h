#ifndef _MESSAGE_H_
#define _MESSAGE_H_

extern void armci_msg_brdcst(void* buffer, int len, int root);
extern void armci_msg_snd(int tag, void* buffer, int len, int to);
extern void armci_msg_rcv(int tag, void* buffer, int buflen, int *msglen, int from);
extern int  armci_msg_rcvany(int tag, void* buffer, int buflen, int *msglen);

extern void armci_msg_igop(int *x, int n, char* op);
extern void armci_msg_lgop(long *x, int n, char* op);
extern void armci_msg_dgop(double *x, int n, char* op);
extern void armci_exchange_address(void *ptr_ar[], int n);
extern void armci_msg_barrier();
extern int  armci_msg_me();
extern int  armci_msg_nproc();
extern void armci_msg_abort(int code);

extern void armci_msg_clus_brdcst(void *buf, int len);
extern void armci_msg_clus_igop(int *x, int n, char* op); 
extern void armci_msg_clus_lgop(long *x, int n, char* op); 
extern void armci_msg_clus_dgop(double *x, int n, char* op); 

#endif
