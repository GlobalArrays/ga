#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_SYS_IPC_H
#   include <sys/ipc.h>
#endif
#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_SYS_TYPES_H
#   include <sys/types.h>
#endif
#if HAVE_SYS_SHM_H
#   include <sys/shm.h>
#endif

#include "typesf2c.h"
#include "sndrcv.h"

char *CreateSharedRegion(Integer *id, Integer *size)
{
    char *temp;

    /* Create the region */

    if ( (*id = shmget(IPC_PRIVATE, (int) *size, 
                    (int) (IPC_CREAT | 00600))) < 0 ) {
        Error("CreateSharedRegion: failed to create shared region",
                (Integer) *id);
    }

    /* Attach to the region */

    if ( (Integer) (temp = shmat((int) *id, (char *) NULL, 0)) == -1L) {
        Error("CreateSharedRegion: failed to attach to shared region",
                (Integer) 0);
    }

    return temp;
}


Integer DetachSharedRegion(Integer id, Integer size, char *addr)
{
    return shmdt(addr);
}


Integer DeleteSharedRegion(Integer id)
{
    return shmctl((int) id, IPC_RMID, (struct shmid_ds *) NULL);
}


char *AttachSharedRegion(Integer id, Integer size)
{
    char *temp;

    if ( (Integer) (temp = shmat((int) id, (char *) NULL, 0)) == -1L) {
        Error("AttachSharedRegion: failed to attach to shared region",
                (Integer) 0);
    }

    return temp;
}
