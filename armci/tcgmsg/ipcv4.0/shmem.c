#if HAVE_CONFIG_H
#   include "config.h"
#endif

/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/shmem.c,v 1.13 2000-10-13 20:55:40 d3h325 Exp $ */

/*
  This stuff attempts to provide a simple interface to temporary shared
  memory regions, loosely modelled after that of Alliant Concentrix 5.0 


  Note that the input arguments switch between integers and pointers
  to integers depending on if they are modified on return.


  Create a shared region of at least size bytes, returning the actual size,
  the id associated with the region. The return value is a pointer to the
  the region. Any error is a hard fail.

  (char *) CreateSharedRegion((long *) id, (long *) size)


  Detach a process from a shared memory region. 0 is returned on success,
  -1 for failure. id, size, and addr must match exactly those items returned
  from CreateSharedRegion

  long DetachSharedRegion((long) id, (long) size, (char *) addr)


  Delete a shared region from the system. This has to be done on the SUN
  to remove it from the system. Returns 0 on success, -1 on error.

  long DeleteSharedRegion((long) id)


  Delete all the shared regions associated with this process.

  long DeleteSharedAll()


  Attach to a shared memory region of known id and size. Returns the
  address of the mapped memory. Size must exactly match the size returned
  from CreateSharedRegion (which in turn is the requested size rounded
  up to a multiple of 4096). Any error is a hard fail. 

  (char *) AttachSharedRegion((long) id, (long) size))

*/

extern void Error();

   /* Bizarre sequent has sysv semaphores but proprietary shmem */
   /* Encore has sysv shmem but is limited to total of 16384bytes! */
#if defined(SYSV)

#include <stdio.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

char *CreateSharedRegion(id, size)
     long *size, *id;
{
  char *temp;

  /* Create the region */

  if ( (*id = shmget(IPC_PRIVATE, (int) *size, 
                     (int) (IPC_CREAT | 00600))) < 0 )
    Error("CreateSharedRegion: failed to create shared region", (long) *id);

  /* Attach to the region */

  if ( (long) (temp = shmat((int) *id, (char *) NULL, 0)) == -1L)
    Error("CreateSharedRegion: failed to attach to shared region", (long) 0);

  return temp;
}

/*ARGSUSED*/
long DetachSharedRegion( id, size, addr)
     long id, size;
     char *addr;
{
  return shmdt(addr);
}

long DeleteSharedRegion(id)
     long id;
{
  return shmctl((int) id, IPC_RMID, (struct shmid_ds *) NULL);
}

/*ARGSUSED*/
char *AttachSharedRegion(id, size)
     long id, size;
{
  char *temp;

  if ( (long) (temp = shmat((int) id, (char *) NULL, 0)) == -1L)
    Error("AttachSharedRegion: failed to attach to shared region", (long) 0);

  return temp;
}

#endif
