#ifndef _SHMEM_H_
#define _SHMEM_H_
extern void  Set_Shmem_Limit(unsigned long shmemlimit);
extern void  Delete_All_Regions();
extern char* Create_Shared_Region(long idlist[], long size, long *offset);
extern char* Attach_Shared_Region(long idlist[], long size, long offset);
extern void Free_Shmem_Ptr(long id, long size, char* addr);

#if defined(WIN32)
#define SHMIDLEN 3
#elif defined(SUN)
#define SHMIDLEN 128
#else
#define SHMIDLEN 64
#endif

#endif
