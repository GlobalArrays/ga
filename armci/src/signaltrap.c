 /******************************************************\
 * Signal handler functions for the following signals:  *
 *        SIGINT, SIGCHLD, SIGBUS, SIGFPE, SIGILL,      *
 *        SIGSEGV, SIGSYS, SIGTRAP, SIGHUP, SIGTERM     *
 * Used to call armci_error that frees up IPC resources *
 \******************************************************/


#include <signal.h>

#define  Error armci_die 

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

extern void Error();

int SR_caught_sigint=0;

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
  Error("SigIntHandler: interrupt signal was caught",(int) sig);
}

void TrapSigInt()
/*
  Trap the signal SIGINT so that we can propagate error
  conditions and also tidy up shared system resources in a
  manner not possible just by killing everyone
*/
{
  if ( (SigIntOrig = signal(SIGINT, SigIntHandler)) == SIG_ERR)
    Error("TrapSigInt: error from signal setting SIGINT",0);
}

void RestoreSigInt()
/*
 Restore the original signal handler
*/
{
  if ( signal(SIGINT, SigIntOrig) == SIG_ERR)
    Error("RestoreSigInt: error from restoring signal SIGINT",0);
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
  SR_caught_sigint = 2;
  Error("Child process terminated prematurely, status=",(int) status);
}

void TrapSigChld()
/*
  Trap SIGCHLD so that can tell if children die unexpectedly.
*/
{
  if ( (SigChldOrig = signal(SIGCHLD, SigChldHandler)) == SIG_ERR)
    Error("TrapSigChld: error from signal setting SIGCHLD",0);
}


void RestoreSigChld()
/*
 Restore the original signal handler
*/
{
  if ( signal(SIGCHLD, SigChldOrig) == SIG_ERR)
    Error("RestoreSigChld: error from restoring signal SIGChld",0);
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
  SR_caught_sigint = 3;
  Error("Bus error, status=",(int) sig);
}

void TrapSigBus()
/*
  Trap SIGBUS 
*/
{
  if ( signal(SIGBUS, SigBusHandler) == SIG_ERR)
    Error("TrapSigBus: error from signal setting SIGBUS", 0);
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
  SR_caught_sigint = 4;
  Error("Floating Point Exception error, status=",(int) sig);
}

void TrapSigFpe()
/*
  Trap SIGFPE
*/
{
  if ( signal(SIGFPE, SigFpeHandler) == SIG_ERR)
    Error("TrapSigFpe: error from signal setting SIGFPE", 0);
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
  SR_caught_sigint = 5;
  Error("Illegal Instruction error, status=",(int) sig);
}

void TrapSigIll()
/*
  Trap SIGILL
*/
{
  if ( signal(SIGILL, SigIllHandler) == SIG_ERR)
    Error("TrapSigIll: error from signal setting SIGILL", 0);
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
  SR_caught_sigint = 6;
  Error("Segmentation Violation error, status=",(int) sig);
}

void TrapSigSegv()
/*
  Trap SIGSEGV
*/
{
  if ( signal(SIGSEGV, SigSegvHandler) == SIG_ERR)
    Error("TrapSigSegv: error from signal setting SIGSEGV", 0);
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
  SR_caught_sigint = 7;
  Error("Bad Argument To System Call error, status=",(int) sig);
}

void TrapSigSys()
/*
  Trap SIGSYS
*/
{
#ifndef LINUX
  if ( signal(SIGSYS, SigSysHandler) == SIG_ERR)
    Error("TrapSigSys: error from signal setting SIGSYS", 0);
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
  SR_caught_sigint = 8;
  Error("Trace Trap error, status=",(int) sig);
}

void TrapSigTrap()
/*
  Trap SIGTRAP
*/
{
  if ( signal(SIGTRAP, SigTrapHandler) == SIG_ERR)
    Error("TrapSigTrap: error from signal setting SIGTRAP", 0);
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
  SR_caught_sigint = 9;
  Error("Hangup error, status=",(int) sig);
}

void TrapSigHup()
/*
  Trap SIGHUP
*/
{
  if ( (SigHupOrig = signal(SIGHUP, SigHupHandler)) == SIG_ERR)
    Error("TrapSigHup: error from signal setting SIGHUP", 0);
}


void RestoreSigHup()
/*
 Restore the original signal handler
*/
{
  if ( signal(SIGHUP, SigHupOrig) == SIG_ERR)
    Error("RestoreSigHUP: error from restoring signal SIGHUP",0);
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
  SR_caught_sigint = 10;
  Error("Terminate signal was sent, status=",(int) sig);
}

void TrapSigTerm()
/*
  Trap SIGTERM
*/
{
  if ( signal(SIGTERM, SigTermHandler) == SIG_ERR)
    Error("TrapSigTerm: error from signal setting SIGTERM", 0);
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
  SR_caught_sigint = 11;
  Error("IOT signal was sent, status=",(int) sig);
}

void TrapSigIot()
/*
  Trap SIGIOT
*/
{
      if ( signal(SIGIOT, SigIotHandler) == SIG_ERR)
          Error("TrapSigIot: error from signal setting SIGIOT", 0);
}



/*********************** SIGCONT *************************************/
#if defined(SUN) && !defined(SOLARIS)
SigType SigContHandler(sig, code, scp, addr)
     int code;
     struct sigcontext *scp;
     char *addr;
#else
SigType SigContHandler(sig)
#endif
     int sig;
{
/*  SR_caught_sigint = 12;*/
/*  Error("Trace Cont error, status=",(int) sig);*/
}

void TrapSigCont()
/*
  Trap SIGCONT
*/
{
  if ( signal(SIGCONT, SigContHandler) == SIG_ERR)
    Error("TrapSigCont: error from signal setting SIGCONT", 0);
}

/*********************** SIGXCPU *************************************/
#if defined(SUN) && !defined(SOLARIS)
SigType SigXcpuHandler(sig, code, scp, addr)
     int code;
     struct sigcontext *scp;
     char *addr;
#else
SigType SigXcpuHandler(sig)
#endif
     int sig;
{
  SR_caught_sigint = 13;
  Error("Terminate signal was sent, status=",(int) sig);
}

void TrapSigXcpu()
/*
  Trap SIGXCPU
*/
{
  if ( signal(SIGXCPU, SigXcpuHandler) == SIG_ERR)
    Error("TrapSigXcpu: error from signal setting SIGXCPU", 0);
}

/******************* external API *********************************/

void ARMCI_ChildrenTrapSignals()
{

     TrapSigBus();
     TrapSigFpe();
     TrapSigIll();
     TrapSigSegv();
     TrapSigSys();
     TrapSigTrap();
     TrapSigTerm();

#ifdef SGI
     TrapSigIot();
     TrapSigXcpu();
#endif

}


void ARMCI_ParentTrapSignals()
{
     TrapSigChld();
     TrapSigInt();
     TrapSigHup();
#ifdef SGI
     TrapSigXcpu();
#endif
}



void ARMCI_ParentRestoreSignals()
{
     RestoreSigChld();
     RestoreSigInt();
     RestoreSigHup();
}
