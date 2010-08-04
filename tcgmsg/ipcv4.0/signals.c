#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_SIGNAL_H
#   include <signal.h>
#endif
#if HAVE_SYS_WAIT_H
#   include <sys/wait.h>
#endif
#if HAVE_SYS_TYPES_H
#   include <sys/types.h>
#endif

#include "sndrcvP.h"

extern void Error();
int SR_caught_sigint = 0;

#ifndef SIG_ERR
#   define SIG_ERR (RETSIGTYPE (*)())-1
#endif


RETSIGTYPE SigintHandler(int sig)
{
    SR_caught_sigint = 1;
    Error("SigintHandler: signal was caught",(Integer) sig);
}


/**
 * Trap the signal SIGINT so that we can propagate error
 * conditions and also tidy up shared system resources in a
 * manner not possible just by killing everyone
 */
void TrapSigint()
{
    if ( signal(SIGINT, SigintHandler) == SIG_ERR) {
        Error("TrapSigint: error from signal setting SIGINT",(Integer) SIGINT);
    }
}


/**
 * kill -SIGINT all of my beloved children
 */
void ZapChildren()
{
    while (SR_numchild--) {
        (void) kill((int) SR_pids[SR_numchild], SIGINT);
    }
}


RETSIGTYPE SigchldHandler(int sig)
{
    int status;

    (void) wait(&status);
    SR_caught_sigint = 1;
    Error("Child process terminated prematurely, status=",(Integer) status);
}


/**
 * Trap SIGCHLD so that can tell if children die unexpectedly.
 */
void TrapSigchld()
{
    if ( signal(SIGCHLD, SigchldHandler) == SIG_ERR) {
        Error("TrapSigchld: error from signal setting SIGCHLD",
                (Integer) SIGCHLD);
    }
}


RETSIGTYPE SigsegvHandler(int sig)
{
    SR_caught_sigint = 1;
    Error("SigsegvHandler: signal was caught",(Integer) sig);
}


/**
 * parallel needs to trap the signal SIGSEGV under Solaris 
 * that is generated when interrupted in NxtVal  
 */
void TrapSigsegv()
{
    if ( signal(SIGSEGV, SigsegvHandler) == SIG_ERR) {
        Error("TrapSigsegv: error from signal setting SIGSEGV",
                (Integer) SIGSEGV);
    }
}


RETSIGTYPE SigtermHandler(int sig)
{
    SR_caught_sigint = 1;
    Error("SigtermHandler: signal was caught",(Integer) sig);
}


/**
 * parallel needs to trap the SIGTERM for batch jobs
 */
void TrapSigterm()
{
    if ( signal(SIGTERM, SigtermHandler) == SIG_ERR) {
        Error("TrapSigterm: error from signal setting SIGTERM",
                (Integer) SIGTERM);
    }
}
