#ifndef __GPCDEF
#define __GPCDEF
#define ARMCI_GPC_HLEN 1024
#define ARMCI_GPC_DLEN 1024*1024
extern int ARMCI_Gpc_register( void (*func) ());
extern void ARMCI_Gpc_release(int handle); 
extern void * ARMCI_Gpc_translate(void *ptr, int proc, int from); 
extern void ARMCI_Gpc_lock(int proc); 
extern void ARMCI_Gpc_unlock(int proc); 
extern void ARMCI_Gpc_trylock(int proc); 
extern int ARMCI_Gpc_exec(int h,int p, void *hdr, int hlen, void *data,int dlen,
                                  void *rhdr, int rhlen, void *rdata, int rdlen,
                                                              armci_hdl_t* nbh);
#endif
