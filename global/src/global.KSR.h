/*$Id: global.KSR.h,v 1.4 1995-02-02 23:13:23 d3g681 Exp $*/
/* lock entire block owned by proc 
 * Note that this to work in data server mode we need use 
 * (proc - cluster_master) instead of proc
 */
#define LOCK(g_a, proc, x)      _gspwt(GA[GA_OFFSET + g_a].ptr[(proc)])
#define UNLOCK(g_a, proc, x)       _rsp(GA[GA_OFFSET + g_a].ptr[(proc)])

#define UNALIGNED(x)    (((unsigned long) (x)) % sizeof(long))
typedef __align128 unsigned char subpage[128];

int    KSRbarrier_mem_req();
void   KSRbarrier(), KSRbarrier_init(int, int, int, char*);

#define Copy(src,dst,n)      memcpy((char*)(dst),(char*)(src),(n))
void   CopyTo(char*, char*, Integer);
void   CopyFrom(char*, char*, Integer);
