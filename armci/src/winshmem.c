/* $Id: winshmem.c,v 1.3 1999-07-28 00:48:06 d3h325 Exp $ */
/* WIN32 & Posix SysV-like shared memory allocation and management
 * 
 *
 * Interface:
 * ~~~~~~~~~
 *  char *Create_Shared_Region(long *idlist, long *size, long *offset)
 *       . to be called by just one process. 
 *       . calls shmalloc, a modified by Robert Harrison version of malloc-like
 *         memory allocator from K&R. shmalloc in turn calls allocate() that
 *         does shmget() and shmat(). 
 *       . idlist might be just a pointer to integer or a true array in the
 *         MULTIPLE_REGIONS versions (calling routine has to take cere of it) 
 *  char *Attach_Shared_Region(long *idlist, long *size, long *offset)
 *       . called by any other process to attach to existing shmem region or
 *         if attached just calculate the address based on the offset
 *       . has to be called after shmem region was created
 *  void  Free_Shmem_Ptr(long id, long size, char* addr)
 *       . called ONLY by the process that created shmem region (id) to return
 *         pointer to shmalloc (shmem is not destroyed)
 *  long  Delete_All_Regions()
 *       . destroys all shared memory regions
 *       . can be called by any process assuming that all processes attached
 *         to alllocated shmem regions 
 *       . needs to by called by cleanup procedure(s)
 */



#include <stdio.h>

#ifdef WIN32
#  include <windows.h>
#  include <process.h>
#  define  GETPID _getpid
#else
#  ifndef _POSIX_C_SOURCE
#    define  _POSIX_C_SOURCE 199309L
#  endif
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/stat.h>
#  include <sys/mman.h>
   typedef int HANDLE; 
   typedef void* LPVOID; 
#  define  GETPID getpid
#endif

#include <assert.h>
#include "shmalloc.h"
#include "shmem.h"

#define DEBUG 0
#define SHM_UNIT (1024)
#define MAX_REGIONS 16

/* default unit for shared memory allocation in KB! */
#ifdef WIN32
#  define _SHMMAX  32678      
#else
#  define _SHMMAX  131072      
#endif

#define SET_MAPNAME(id) sprintf(map_fname,"/tmp/ARMCIshmem.%d.%d",parent_pid,(id))

/*********************** global data structures ****************************/

/* Terminology
 *   region - actual piece of shared memory allocated from OS
 *   block  - a part of allocated shmem that is given to the requesting process
 */


/* array holds handles and addreses for each shmem region*/ 
static struct shm_region_list{
   char     *addr;
   HANDLE   id;
   int      size;
}region_list[MAX_REGIONS];

static char map_fname[64];
static  int alloc_regions=0;   /* counter to identify mapping handle */
static  int last_allocated=0; /* counter trailing alloc_regions by 0/1 */

/* Min and Max amount of aggregate memory that can be allocated */
static  unsigned long MinShmem = _SHMMAX;  
static  unsigned long MaxShmem = MAX_REGIONS*_SHMMAX;
static  int parent_pid=-1;  /* process id of process 0 "parent" */

extern void armci_die(char*,int);

/* not done here yet */
void armci_shmem_init(){};

/*\ application can reset the upper limit for memory allocation
\*/
void armci_set_shmem_limit(unsigned long shmemlimit) /* comes in bytes */ 
{
     unsigned long kbytes;
     kbytes = (shmemlimit + SHM_UNIT -1)/SHM_UNIT;
     if(MaxShmem > kbytes) MaxShmem = kbytes;
     if(MinShmem > kbytes) MinShmem = kbytes;
}


void Delete_All_Regions()
{
int reg;
long code=0;

  for(reg = 0; reg < alloc_regions; reg++){
    if(region_list[reg].addr != (char*)0){
#       if defined(WIN32)
          UnmapViewOfFile(region_list[reg].addr);
          CloseHandle(region_list[reg].id);
#       else
          munmap(region_list[reg].addr, region_list[reg].size);
#       endif
        region_list[reg].addr = (char*)0;
    }
  }
}


/*\ only process that created shared region returns the pointer to shmalloc 
\*/
void Free_Shmem_Ptr(long id, long size, char* addr)
{  
  shfree(addr);
}


char *armci_get_core_from_map_file(int id, int exists, int size)
{
    HANDLE  h_shm_map;
    LPVOID  ptr;
    SET_MAPNAME(id);
    region_list[alloc_regions].addr = (char*)0;

#if defined(WIN32)
    h_shm_map = CreateFileMapping(INVALID_HANDLE_VALUE,
                NULL, PAGE_READWRITE, 0, size, map_fname);
    if(h_shm_map == NULL) return NULL;

    if(exists){
       /* get an error code when mapping should exist */
       if (GetLastError() != ERROR_ALREADY_EXISTS){
          CloseHandle(h_shm_map);
          fprintf(stderr,"map handle does not exist (attach)\n");
          return NULL;
       }else { /* OK */ }
    } else {
        /* problem if mapping it should not be there */
        if (GetLastError() == ERROR_ALREADY_EXISTS){
          CloseHandle(h_shm_map);
          fprintf(stderr,"map handle already exists (create)\n");
          return NULL;
        }
    }
     /* now map file into process address space */
    ptr = MapViewOfFile(h_shm_map, 
                        FILE_MAP_WRITE | FILE_MAP_READ, 0, 0, 0);
    if((char*)ptr == NULL){
       CloseHandle(h_shm_map);
       h_shm_map = INVALID_HANDLE_VALUE;
    }
    
#else

    if(exists){
       h_shm_map = shm_open(map_fname, O_RDWR, S_IRWXU);
       if(h_shm_map == -1) return NULL;
    }else{
       (void*)shm_unlink(map_fname); /* sanity cleanup */
       h_shm_map = shm_open(map_fname, O_CREAT|O_RDWR, S_IRWXU);
       if(h_shm_map) perror("shm_open");
       if(h_shm_map == -1) return NULL;
       if(ftruncate(h_shm_map,size) < 0) return NULL;
    }

fprintf(stderr,"h_shm_map =%d\n",h_shm_map );
    ptr = mmap(0, size, PROT_READ|PROT_WRITE, MAP_SHARED, h_shm_map, 0L);
fprintf(stderr,"ptr =%x\n",ptr);

    close(h_shm_map);
    h_shm_map = -1;

#endif

    /*     save file handle in the array to close it in the future */
    region_list[alloc_regions].id   = h_shm_map;
    region_list[alloc_regions].addr = (char*)ptr;
    region_list[alloc_regions].size = size;

    return((char*)ptr);
}


/*\ function called by shared memory allocator (shmalloc)
\*/
char *allocate(long size)
{
    char *ptr;

    if(alloc_regions>= MAX_REGIONS)
        armci_die("max alloc regions exceeded", alloc_regions);

    ptr = armci_get_core_from_map_file(alloc_regions, 0, (int)size);
    if(ptr !=NULL) alloc_regions++;

    return ptr;
}


char* Create_Shared_Region(long idlist[], long size, long *offset)
{
    char *temp;

    /*initialization */
    if(!alloc_regions){
          int reg;
          for(reg=0;reg<MAX_REGIONS;reg++){
            region_list[reg].addr=(char*)0;
            region_list[reg].id=0;
            parent_pid = GETPID();
          }
          shmalloc_request((unsigned)MinShmem, (unsigned)MaxShmem);
     }

     temp = shmalloc((unsigned)size);
     if(temp == (char*)0 )
           armci_die("Create_Shared_Region: shmalloc failed ",0L);
    
     /* find if shmalloc allocated a new shmem region */
     if(last_allocated == alloc_regions){
         *offset = (long) (temp - region_list[last_allocated-1].addr);
     } else if(last_allocated == alloc_regions -1){
         *offset = (long) (temp - region_list[last_allocated].addr);
         last_allocated++;
     }else{
         armci_die(" Create_Shared_Region:inconsitency in counters",
             last_allocated - alloc_regions);
     }

     idlist[0] = alloc_regions;
     idlist[1] = parent_pid;
     return (temp);
}



char *Attach_Shared_Region(long id[], long size, long offset)
{
    char *temp;

    /*initialization */
    if(!alloc_regions){
          int reg;
          for(reg=0;reg<MAX_REGIONS;reg++){
            region_list[reg].addr=(char*)0;
            region_list[reg].id=0;
            parent_pid= id[1];
          }
     }

     /* find out if a new shmem region was allocated */
     if(alloc_regions == id[0] -1){
         if(DEBUG)printf("alloc_regions=%d size=%d\n",alloc_regions,size);
         temp = armci_get_core_from_map_file(alloc_regions,1,size);
         assert(temp);
         if(temp != NULL)alloc_regions++;
         else return NULL;
     }

     if( alloc_regions == id[0]){
         temp = region_list[alloc_regions-1].addr + offset; 
         assert(temp);
     }else armci_die("Attach_Shared_Region:iconsistency in counters",
         alloc_regions - id[0]);

     assert(temp);
     return(temp);
}
