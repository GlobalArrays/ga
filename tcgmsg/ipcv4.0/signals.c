/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/signals.c,v 1.4 1995-02-24 02:17:45 d3h325 Exp $ */

#include <signal.h>
#include "sndrcvP.h"
#if defined(SUN) || defined(ALLIANT) || defined(ENCORE) || defined(SEQUENT) || \
    defined(AIX) || defined(NEXT)
#include <sys/wait.h>
#endif

extern void Error();
int SR_caught_sigint = 0;

/*ARGSUSED*/
#if (defined(ENCORE) || defined(SEQUENT) || defined(ARDENT))
int SigintHandler(sig, code, scp, addr)
#else
void SigintHandler(sig, code, scp, addr)
#endif
     int sig, code;
     struct sigcontext *scp;
     char *addr;
{
  SR_caught_sigint = 1;
  Error("SigintHandler: signal was caught",(long) code);
}

void TrapSigint()
/*
  Trap the signal SIGINT so that we can propagate error
  conditions and also tidy up shared system resources in a
  manner not possible just by killing everyone
*/
{
#if defined(ENCORE) || defined(SEQUENT) || defined(ARDENT)
  if ( signal(SIGINT, SigintHandler) == (int (*)()) -1)
    Error("TrapSigint: error from signal setting SIGINT",(long) SIGINT);
#else
  if ( signal(SIGINT, SigintHandler) == (void (*)()) -1)
    Error("TrapSigint: error from signal setting SIGINT",(long) SIGINT);
#endif
}

void ZapChildren()
/*
  kill -SIGINT all of my beloved children
*/
{
  while (SR_nchild--)
    (void) kill((int) SR_pids[SR_nchild], SIGINT);
}

/*ARGSUSED*/
#if (defined(ENCORE) || defined(SEQUENT) || defined(ARDENT))
int SigchldHandler(sig, code, scp, addr)
#else
void SigchldHandler(sig, code, scp, addr)
#endif
     int sig, code;
     struct sigcontext *scp;
     char *addr;
{
  int status, pid;
  
#if defined(ALLIANT) || defined(ENCORE) || defined(SEQUENT) || defined(NEXT)
  union wait ustatus;
#endif

#if defined(ALLIANT) || defined(ENCORE) || defined(SEQUENT) || defined(NEXT)
  pid = wait(&ustatus);
  status = ustatus.w_status;
#else
  pid = wait(&status);
#endif
  SR_caught_sigint = 1;
  Error("Child process terminated prematurely, status=",(long) status);
}

void TrapSigchld()
/*
  Trap SIGCHLD so that can tell if children die unexpectedly.
*/
{
#if defined(ENCORE) || defined(SEQUENT) || defined(ARDENT)
  if ( signal(SIGCHLD, SigchldHandler) == (int (*)()) -1)
    Error("TrapSigchld: error from signal setting SIGCHLD",
		  (long) SIGCHLD);
#else
  if ( signal(SIGCHLD, SigchldHandler) == (void (*)()) -1)
    Error("TrapSigchld: error from signal setting SIGCHLD",
		  (long) SIGCHLD);
#endif
}
