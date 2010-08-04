#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_ERRNO_H
#   include <errno.h>
#endif
#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_SETJMP_H
#   include <setjmp.h>
#endif
#if HAVE_SIGNAL_H
#   include <signal.h>
#endif

extern void exit(int status);

#include "sndrcvP.h"
#include "sndrcv.h"
#include "signals.h"
#include "sockets.h"

#if HAVE_SYS_IPC_H
#   include "sema.h"
#   include "shmem.h"
#endif

extern jmp_buf SR_jmp_buf;   /* Jumped to on soft error */ 
extern int SR_caught_sigint;


void Error(const char *string, Integer integer)
{
    (void) signal(SIGCHLD, SIG_DFL); /* Death of children to be expected */
    (void) signal(SIGINT, SIG_IGN);

    (void) fflush(stdout);
    if (SR_caught_sigint) {
        (void) fprintf(stderr,"%3ld: interrupt(%d)\n",NODEID_(), SR_caught_sigint);
        (void) fflush(stderr);
    }
    else {
        (void) fprintf(stdout,"%3ld: %s %ld (%#lx).\n", NODEID_(), string,
                       integer,integer);
        (void) fflush(stdout);
        (void) fprintf(stderr,"%3ld: %s %ld (%#lx).\n", NODEID_(), string,
                       integer,integer);
        if (errno != 0)
            perror("system error message");
        if (DEBUG_)
            PrintProcInfo();
    }
    (void) fflush(stdout);
    (void) fflush(stderr);

    /* Shut down the sockets and remove shared memory and semaphores to
       propagate an error condition to anyone that is trying to communicate
       with me */

    ZapChildren();  /* send interrupt to children which should trap it
                       and call Error in the handler */

#if HAVE_SYS_IPC_H
    (void) SemSetDestroyAll();
    (void) DeleteSharedRegion(SR_proc_info[NODEID_()].shmem_id);
#endif
    ShutdownAll(); /* Close sockets for machines with static kernel */

    /*  abort(); */

    if (SR_exit_on_error) {
        exit(1);
    }
    else {
        SR_error = 1;
        (void) longjmp(SR_jmp_buf, 1); /* For NXTVAL server */
    }
}

/**
 * Interface from fortran to c error routine.
 */
void PARERR_(Integer *code)
{
    Error("User detected error in FORTRAN", *code);
}
