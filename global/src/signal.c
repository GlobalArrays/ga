 /******************************************************\
 * Signal handler functions for the following signals:  *
 *        SIGINT, SIGCHLD, SIGBUS, SIGFPE, SIGILL,      *
 *        SIGSEGV, SIGSYS, SIGTRAP, SIGHUP, SIGTERM     *
 * Used to call ga_error that frees up IPC resources    *
 \******************************************************/


#include <signal.h>

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

extern void ga_error();

extern int SR_caught_sigint;



/*********************** SIGINT *************************************/
SigType SigIntHandler(sig, code, scp, addr)
     int sig, code;
     struct sigcontext *scp;
     char *addr;
{
  SR_caught_sigint = 1;
  ga_error("SigIntHandler: interrupt signal was caught",(long) code);
}

void TrapSigInt()
/*
  Trap the signal SIGINT so that we can propagate error
  conditions and also tidy up shared system resources in a
  manner not possible just by killing everyone
*/
{
  if ( signal(SIGINT, SigIntHandler) == SIG_ERR)
    ga_error("TrapSigInt: error from signal setting SIGINT",(long) SIGINT);
}



/*********************** SIGCHLD *************************************/
SigType SigChldHandler(sig, code, scp, addr)
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
  ga_error("Child process terminated prematurely, status=",(long) status);
}

void TrapSigChld()
/*
  Trap SIGCHLD so that can tell if children die unexpectedly.
*/
{
  if ( signal(SIGCHLD, SigChldHandler) == SIG_ERR)
    ga_error("TrapSigChld: error from signal setting SIGCHLD",
		  (long) SIGCHLD);
}




/*********************** SIGBUS *************************************/
SigType SigBusHandler(sig, code, scp, addr)
     int sig, code;
     struct sigcontext *scp;
     char *addr;
{
  SR_caught_sigint = 1;
  ga_error("Bus error, status=",(long) code);
}

void TrapSigBus()
/*
  Trap SIGBUS 
*/
{
  if ( signal(SIGBUS, SigBusHandler) == SIG_ERR)
    ga_error("TrapSigBus: error from signal setting SIGBUS",
                  (long) SIGBUS);
}




/*********************** SIGFPE *************************************/
SigType SigFpeHandler(sig, code, scp, addr)
     int sig, code;
     struct sigcontext *scp;
     char *addr;
{
  SR_caught_sigint = 1;
  ga_error("Floating Point Exception error, status=",(long) code);
}

void TrapSigFpe()
/*
  Trap SIGFPE
*/
{
  if ( signal(SIGFPE, SigFpeHandler) == SIG_ERR)
    ga_error("TrapSigFpe: error from signal setting SIGFPE",
                  (long) SIGFPE);
}




/*********************** SIGILL *************************************/
SigType SigIllHandler(sig, code, scp, addr)
     int sig, code;
     struct sigcontext *scp;
     char *addr;
{
  SR_caught_sigint = 1;
  ga_error("Illegal Instruction error, status=",(long) code);
}

void TrapSigIll()
/*
  Trap SIGILL
*/
{
  if ( signal(SIGILL, SigIllHandler) == SIG_ERR)
    ga_error("TrapSigIll: error from signal setting SIGILL",
                  (long) SIGILL);
}




/*********************** SIGSEGV *************************************/
SigType SigSegvHandler(sig, code, scp, addr)
     int sig, code;
     struct sigcontext *scp;
     char *addr;
{
  SR_caught_sigint = 1;
  ga_error("Segmentation Violation error, status=",(long) code);
}

void TrapSigSegv()
/*
  Trap SIGSEGV
*/
{
  if ( signal(SIGSEGV, SigSegvHandler) == SIG_ERR)
    ga_error("TrapSigSegv: error from signal setting SIGSEGV",
                  (long) SIGSEGV);
}




/*********************** SIGSYS *************************************/
SigType SigSysHandler(sig, code, scp, addr)
     int sig, code;
     struct sigcontext *scp;
     char *addr;
{
  SR_caught_sigint = 1;
  ga_error("Bad Argument To System Call error, status=",(long) code);
}

void TrapSigSys()
/*
  Trap SIGSYS
*/
{
  if ( signal(SIGSYS, SigSysHandler) == SIG_ERR)
    ga_error("TrapSigSys: error from signal setting SIGSYS",
                  (long) SIGSYS);
}



/*********************** SIGTRAP *************************************/
SigType SigTrapHandler(sig, code, scp, addr)
     int sig, code;
     struct sigcontext *scp;
     char *addr;
{
  SR_caught_sigint = 1;
  ga_error("Trace Trap error, status=",(long) code);
}

void TrapSigTrap()
/*
  Trap SIGTRAP
*/
{
  if ( signal(SIGTRAP, SigTrapHandler) == SIG_ERR)
    ga_error("TrapSigTrap: error from signal setting SIGTRAP",
                  (long) SIGTRAP);
}



/*********************** SIGHUP *************************************/
SigType SigHupHandler(sig, code, scp, addr)
     int sig, code;
     struct sigcontext *scp;
     char *addr;
{
  SR_caught_sigint = 1;
  ga_error("Hangup error, status=",(long) code);
}

void TrapSigHup()
/*
  Trap SIGHUP
*/
{
  if ( signal(SIGHUP, SigHupHandler) == SIG_ERR)
    ga_error("TrapSigHup: error from signal setting SIGHUP",
                  (long) SIGHUP);
}



/*********************** SIGTERM *************************************/
SigType SigTermHandler(sig, code, scp, addr)
     int sig, code;
     struct sigcontext *scp;
     char *addr;
{
  SR_caught_sigint = 1;
  ga_error("Terminate signal was sent, status=",(long) code);
}

void TrapSigTerm()
/*
  Trap SIGTERM
*/
{
  if ( signal(SIGTERM, SigTermHandler) == SIG_ERR)
    ga_error("TrapSigTerm: error from signal setting SIGTERM",
                  (long) SIGTERM);
}


/*********************** SIGIOT *************************************/
SigType SigIotHandler(sig, code, scp, addr)
     int sig, code;
     struct sigcontext *scp;
     char *addr;
{
  SR_caught_sigint = 1;
  ga_error("IOT signal was sent, status=",(long) code);
}

void TrapSigIot()
/*
  Trap SIGIOT
*/
{
      if ( signal(SIGIOT, SigIotHandler) == SIG_ERR)
          ga_error("TrapSigIot: error from signal setting SIGIOT",
                    (long) SIGIOT);
}
