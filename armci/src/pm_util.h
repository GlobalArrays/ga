#ifndef _PM_UTIL_H_
#define _PM_UTIL_H_

void pm_init(int *argc, char *(*argv[]));
void pm_finalize();
double pm_time(void);
void pm_alltoall(void *src, int sbytes, void *dst, int rbytes);
void pm_barrier(void);
int PM_UNDEFINED();
int PM_ERR_GROUP();
int PM_SUCCESS();

int pm_rank();
int pm_nproc();
void pm_abort(int code);
void pm_bcast(void *buffer, int len, int root);
void pm_send(void *buffer, int len, int to, int tag);
int pm_recv(void *buffer, int buflen, int *from, int tag);

#endif
