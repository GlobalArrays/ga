/* $Id: signals.c,v 1.3 2000-11-14 20:43:56 d3h325 Exp $ */

#include "tcgmsgP.h"
#ifdef LINUX
#include "sys/wait.h"
#endif

#if defined(SUN) || defined(ALLIANT) || defined(ENCORE) || \
    defined(SEQUENT) || defined(AIX) || defined(NEXT)
#include <sys/wait.h>
#endif

#if (defined(ENCORE) || defined(SEQUENT) || defined(ARDENT))
#   define SigType  int
#else
#   define SigType  void
#endif

#ifndef SIG_ERR
#       define SIG_ERR         (SigType (*)(int))-1
#endif

SigType SigintHandler(int sig)
{
  TCGMSG_caught_sigint = 1L;
  Error("SigintHandler: signal was caught",0L);
}

void TrapSigint()
/*
  Trap the signal SIGINT so that we can propagate error
  conditions and also tidy up shared system resources in a
  manner not possible just by killing everyone
*/
{
  if ( signal(SIGINT, SigintHandler) == SIG_ERR)
    Error("TrapSigint: error from signal setting SIGINT",(long) SIGINT);
}

void ZapChildren()
/*
  kill -SIGINT all of my beloved children
*/
{
  long node;

  for (node=0; node<TCGMSG_nnodes; node++)
    if (node != TCGMSG_nodeid)
      (void) kill((int) TCGMSG_proc_info[node].pid, SIGINT);
}

/*ARGSUSED*/
SigType SigchldHandler(int sig)
{
  int status;
  
  (void) wait(&status);
  TCGMSG_caught_sigint = 1;
  Error("Child process terminated prematurely, status=",(long) status);
}

void TrapSigchld()
/*
  Trap SIGCHLD so that can tell if children die unexpectedly.
*/
{
  if ( signal(SIGCHLD, SigchldHandler) == SIG_ERR)
    Error("TrapSigchld: error from signal setting SIGCHLD", (long) SIGCHLD);
}
