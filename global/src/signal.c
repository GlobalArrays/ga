 /******************************************************\
 * Signal handler functions for the following signals:  *
 *        SIGINT, SIGCHLD, SIGBUS, SIGFPE, SIGILL,      *
 *        SIGSEGV, SIGSYS, SIGTRAP, SIGHUP, SIGTERM     *
 * Used to call ga_error that frees up IPC resources    *
 \******************************************************/


#include <signal.h>

#define  ERROR ga_error

#if (defined(ENCORE) || defined(SEQUENT) || defined(ARDENT))
#   define SigType  int
#else
#   define SigType  void
#endif


#ifndef SIG_ERR
#   define SIG_ERR         (SigType (*)())-1
#endif

#if defined(SUN) || defined(ALLIANT) || defined(ENCORE) || defined(SEQUENT) || \
    defined(AIX) || defined(NEXT)
#include <sys/wait.h>
#endif

extern void ERROR();

extern int SR_caught_sigint;

SigType (*SigChldOrig)(), (*SigIntOrig)(), (*SigHupOrig)();


/*********************** SIGINT *************************************/
#if defined(SUN) && !defined(SOLARIS)
SigType SigIntHandler(sig, code, scp, addr)
     int code;
     struct sigcontext *scp;
     char *addr;
#else
SigType SigIntHandler(sig)
#endif
     int sig;
{
  SR_caught_sigint = 1;
  ERROR("SigIntHandler: interrupt signal was caught",(long) sig);
}

void TrapSigInt()
/*
  Trap the signal SIGINT so that we can propagate error
  conditions and also tidy up shared system resources in a
  manner not possible just by killing everyone
*/
{
  if ( (SigIntOrig = signal(SIGINT, SigIntHandler)) == SIG_ERR)
    ERROR("TrapSigInt: error from signal setting SIGINT",0);
}

void RestoreSigInt()
/*
 Restore the original signal handler
*/
{
  if ( signal(SIGINT, SigIntOrig) == SIG_ERR)
    ERROR("RestoreSigInt: error from restoring signal SIGINT",0);
}



/*********************** SIGCHLD *************************************/
#if defined(SUN) && !defined(SOLARIS)
SigType SigChldHandler(sig, code, scp, addr)
     int code;
     struct sigcontext *scp;
     char *addr;
#else
SigType SigChldHandler(sig)
#endif
     int sig;
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
  ERROR("Child process terminated prematurely, status=",(long) status);
}

void TrapSigChld()
/*
  Trap SIGCHLD so that can tell if children die unexpectedly.
*/
{
  if ( (SigChldOrig = signal(SIGCHLD, SigChldHandler)) == SIG_ERR)
    ERROR("TrapSigChld: error from signal setting SIGCHLD",0);
}


void RestoreSigChld(d)
/*
 Restore the original signal handler
*/
{
  if ( signal(SIGCHLD, SigChldOrig) == SIG_ERR)
    ERROR("RestoreSigChld: error from restoring signal SIGChld",0);
}




/*********************** SIGBUS *************************************/
#if defined(SUN) && !defined(SOLARIS)
SigType SigBusHandler(sig, code, scp, addr)
     int code;
     struct sigcontext *scp;
     char *addr;
#else
SigType SigBusHandler(sig)
#endif
     int sig;
{
  SR_caught_sigint = 1;
  ERROR("Bus error, status=",(long) sig);
}

void TrapSigBus()
/*
  Trap SIGBUS 
*/
{
  if ( signal(SIGBUS, SigBusHandler) == SIG_ERR)
    ERROR("TrapSigBus: error from signal setting SIGBUS", 0);
}




/*********************** SIGFPE *************************************/
#if defined(SUN) && !defined(SOLARIS)
SigType SigFpeHandler(sig, code, scp, addr)
     int code;
     struct sigcontext *scp;
     char *addr;
#else
SigType SigFpeHandler(sig)
#endif
     int sig;
{
  SR_caught_sigint = 1;
  ERROR("Floating Point Exception error, status=",(long) sig);
}

void TrapSigFpe()
/*
  Trap SIGFPE
*/
{
  if ( signal(SIGFPE, SigFpeHandler) == SIG_ERR)
    ERROR("TrapSigFpe: error from signal setting SIGFPE", 0);
}




/*********************** SIGILL *************************************/
#if defined(SUN) && !defined(SOLARIS)
SigType SigIllHandler(sig, code, scp, addr)
     int code;
     struct sigcontext *scp;
     char *addr;
#else
SigType SigIllHandler(sig)
#endif
     int sig;
{
  SR_caught_sigint = 1;
  ERROR("Illegal Instruction error, status=",(long) sig);
}

void TrapSigIll()
/*
  Trap SIGILL
*/
{
  if ( signal(SIGILL, SigIllHandler) == SIG_ERR)
    ERROR("TrapSigIll: error from signal setting SIGILL", 0);
}




/*********************** SIGSEGV *************************************/
#if defined(SUN) && !defined(SOLARIS)
SigType SigSegvHandler(sig, code, scp, addr)
     int code;
     struct sigcontext *scp;
     char *addr;
#else
SigType SigSegvHandler(sig)
#endif
     int sig;
{
  SR_caught_sigint = 1;
  ERROR("Segmentation Violation error, status=",(long) sig);
}

void TrapSigSegv()
/*
  Trap SIGSEGV
*/
{
  if ( signal(SIGSEGV, SigSegvHandler) == SIG_ERR)
    ERROR("TrapSigSegv: error from signal setting SIGSEGV", 0);
}




/*********************** SIGSYS *************************************/
#if defined(SUN) && !defined(SOLARIS)
SigType SigSysHandler(sig, code, scp, addr)
     int code;
     struct sigcontext *scp;
     char *addr;
#else
SigType SigSysHandler(sig)
#endif
     int sig;
{
  SR_caught_sigint = 1;
  ERROR("Bad Argument To System Call error, status=",(long) sig);
}

void TrapSigSys()
/*
  Trap SIGSYS
*/
{
#ifndef LINUX
  if ( signal(SIGSYS, SigSysHandler) == SIG_ERR)
    ERROR("TrapSigSys: error from signal setting SIGSYS", 0);
#endif
}



/*********************** SIGTRAP *************************************/
#if defined(SUN) && !defined(SOLARIS)
SigType SigTrapHandler(sig, code, scp, addr)
     int code;
     struct sigcontext *scp;
     char *addr;
#else
SigType SigTrapHandler(sig)
#endif
     int sig;
{
  SR_caught_sigint = 1;
  ERROR("Trace Trap error, status=",(long) sig);
}

void TrapSigTrap()
/*
  Trap SIGTRAP
*/
{
  if ( signal(SIGTRAP, SigTrapHandler) == SIG_ERR)
    ERROR("TrapSigTrap: error from signal setting SIGTRAP", 0);
}



/*********************** SIGHUP *************************************/
#if defined(SUN) && !defined(SOLARIS)
SigType SigHupHandler(sig, code, scp, addr)
     int code;
     struct sigcontext *scp;
     char *addr;
#else
SigType SigHupHandler(sig)
#endif
     int sig;
{
  SR_caught_sigint = 1;
  ERROR("Hangup error, status=",(long) sig);
}

void TrapSigHup()
/*
  Trap SIGHUP
*/
{
  if ( (SigHupOrig = signal(SIGHUP, SigHupHandler)) == SIG_ERR)
    ERROR("TrapSigHup: error from signal setting SIGHUP", 0);
}


void RestoreSigHup()
/*
 Restore the original signal handler
*/
{
  if ( signal(SIGHUP, SigHupOrig) == SIG_ERR)
    ERROR("RestoreSigHUP: error from restoring signal SIGHUP",0);
}



/*********************** SIGTERM *************************************/
#if defined(SUN) && !defined(SOLARIS)
SigType SigTermHandler(sig, code, scp, addr)
     int code;
     struct sigcontext *scp;
     char *addr;
#else
SigType SigTermHandler(sig)
#endif
     int sig;
{
  SR_caught_sigint = 1;
  ERROR("Terminate signal was sent, status=",(long) sig);
}

void TrapSigTerm()
/*
  Trap SIGTERM
*/
{
  if ( signal(SIGTERM, SigTermHandler) == SIG_ERR)
    ERROR("TrapSigTerm: error from signal setting SIGTERM", 0);
}


/*********************** SIGIOT *************************************/
#if defined(SUN) && !defined(SOLARIS)
SigType SigIotHandler(sig, code, scp, addr)
     int code;
     struct sigcontext *scp;
     char *addr;
#else
SigType SigIotHandler(sig)
#endif
     int sig;
{
  SR_caught_sigint = 1;
  ERROR("IOT signal was sent, status=",(long) sig);
}

void TrapSigIot()
/*
  Trap SIGIOT
*/
{
      if ( signal(SIGIOT, SigIotHandler) == SIG_ERR)
          ERROR("TrapSigIot: error from signal setting SIGIOT", 0);
}
