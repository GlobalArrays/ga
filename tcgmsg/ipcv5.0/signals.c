/* $$ */

#include "tcgmsgP.h"
#ifdef LINUX
#include "sys/wait.h"
#endif

#if defined(SUN) || defined(ALLIANT) || defined(ENCORE) || \
    defined(SEQUENT) || defined(AIX) || defined(NEXT)
#include <sys/wait.h>
#endif

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
  TCGMSG_caught_sigint = 1L;
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
  if ( signal(SIGINT, SigintHandler) == (int (*)()) -1L)
    Error("TrapSigint: error from signal setting SIGINT",(long) SIGINT);
#else
  if ( signal(SIGINT, SigintHandler) == (void (*)()) -1L)
    Error("TrapSigint: error from signal setting SIGINT",(long) SIGINT);
#endif
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
#if (defined(ENCORE) || defined(SEQUENT) || defined(ARDENT))
int SigchldHandler(sig, code, scp, addr)
#else
void SigchldHandler(sig, code, scp, addr)
#endif
     int sig, code;
     struct sigcontext *scp;
     char *addr;
{
  int status;
  
#if defined(ALLIANT) || defined(ENCORE) || defined(SEQUENT) || defined(NEXT)
  union wait ustatus;
#endif

#if defined(ALLIANT) || defined(ENCORE) || defined(SEQUENT) || defined(NEXT)
  (void) wait(&ustatus);
  status = ustatus.w_status;
#else
  (void) wait(&status);
#endif
  TCGMSG_caught_sigint = 1;
  Error("Child process terminated prematurely, status=",(long) status);
}

void TrapSigchld()
/*
  Trap SIGCHLD so that can tell if children die unexpectedly.
*/
{
#if defined(ENCORE) || defined(SEQUENT) || defined(ARDENT)
  if ( signal(SIGCHLD, SigchldHandler) == (int (*)()) -1L)
    Error("TrapSigchld: error from signal setting SIGCHLD",
		  (long) SIGCHLD);
#else
  if ( signal(SIGCHLD, SigchldHandler) == (void (*)()) -1L)
    Error("TrapSigchld: error from signal setting SIGCHLD",
		  (long) SIGCHLD);
#endif
}
