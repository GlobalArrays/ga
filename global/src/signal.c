 /******************************************************\
 * Signal handler functions for the following signals:  *
 *        SIGINT, SIGCHLD, SIGBUS, SIGFPE, SIGILL,      *
 *        SIGSEGV, SIGSYS, SIGTRAP, SIGHUP, SIGTERM     *
 * Used to call ga_error that frees up IPC resources    *
 \******************************************************/


#include <signal.h>

#if defined(SUN) || defined(ALLIANT) || defined(ENCORE) || defined(SEQUENT) || \
    defined(IBM) || defined(NEXT)
#include <sys/wait.h>
#endif

extern void ga_error();

#ifdef TCGMSG
extern int SR_caught_sigint;
#else
int SR_caught_sigint;
#endif



/*********************** SIGINT *************************************/
#if (defined(ENCORE) || defined(SEQUENT) || defined(ARDENT))
int SigIntHandler(sig, code, scp, addr)
#else
void SigIntHandler(sig, code, scp, addr)
#endif
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
#if defined(ENCORE) || defined(SEQUENT) || defined(ARDENT)
  if ( signal(SIGINT, SigIntHandler) == (int (*)()) -1)
    ga_error("TrapSigInt: error from signal setting SIGINT",(long) SIGINT);
#else
  if ( signal(SIGINT, SigIntHandler) == (void (*)()) -1)
    ga_error("TrapSigInt: error from signal setting SIGINT",(long) SIGINT);
#endif
}



/*********************** SIGCHLD *************************************/
#if (defined(ENCORE) || defined(SEQUENT) || defined(ARDENT))
int SigChldHandler(sig, code, scp, addr)
#else
void SigChldHandler(sig, code, scp, addr)
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
  ga_error("Child process terminated prematurely, status=",(long) status);
}

void TrapSigChld()
/*
  Trap SIGCHLD so that can tell if children die unexpectedly.
*/
{
#if defined(ENCORE) || defined(SEQUENT) || defined(ARDENT)
  if ( signal(SIGCHLD, SigChldHandler) == (int (*)()) -1)
    ga_error("TrapSigChld: error from signal setting SIGCHLD",
		  (long) SIGCHLD);
#else
  if ( signal(SIGCHLD, SigChldHandler) == (void (*)()) -1)
    ga_error("TrapSigChld: error from signal setting SIGCHLD",
		  (long) SIGCHLD);
#endif
}




/*********************** SIGBUS *************************************/
#if (defined(ENCORE) || defined(SEQUENT) || defined(ARDENT))
int SigBusHandler(sig, code, scp, addr)
#else
void SigBusHandler(sig, code, scp, addr)
#endif
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
#if defined(ENCORE) || defined(SEQUENT) || defined(ARDENT)
  if ( signal(SIGBUS, SigBusHandler) == (int (*)()) -1)
    ga_error("TrapSigBus: error from signal setting SIGBUS",
                  (long) SIGBUS);
#else
  if ( signal(SIGBUS, SigBusHandler) == (void (*)()) -1)
    ga_error("TrapSigBus: error from signal setting SIGBUS",
                  (long) SIGBUS);
#endif
}




/*********************** SIGFPE *************************************/
#if (defined(ENCORE) || defined(SEQUENT) || defined(ARDENT))
int SigFpeHandler(sig, code, scp, addr)
#else
void SigFpeHandler(sig, code, scp, addr)
#endif
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
#if defined(ENCORE) || defined(SEQUENT) || defined(ARDENT)
  if ( signal(SIGFPE, SigFpeHandler) == (int (*)()) -1)
    ga_error("TrapSigFpe: error from signal setting SIGFPE",
                  (long) SIGFPE);
#else
  if ( signal(SIGFPE, SigFpeHandler) == (void (*)()) -1)
    ga_error("TrapSigFpe: error from signal setting SIGFPE",
                  (long) SIGFPE);
#endif
}




/*********************** SIGILL *************************************/
#if (defined(ENCORE) || defined(SEQUENT) || defined(ARDENT))
int SigIllHandler(sig, code, scp, addr)
#else
void SigIllHandler(sig, code, scp, addr)
#endif
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
#if defined(ENCORE) || defined(SEQUENT) || defined(ARDENT)
  if ( signal(SIGILL, SigIllHandler) == (int (*)()) -1)
    ga_error("TrapSigIll: error from signal setting SIGILL",
                  (long) SIGILL);
#else
  if ( signal(SIGILL, SigIllHandler) == (void (*)()) -1)
    ga_error("TrapSigIll: error from signal setting SIGILL",
                  (long) SIGILL);
#endif
}




/*********************** SIGSEGV *************************************/
#if (defined(ENCORE) || defined(SEQUENT) || defined(ARDENT))
int SigSegvHandler(sig, code, scp, addr)
#else
void SigSegvHandler(sig, code, scp, addr)
#endif
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
#if defined(ENCORE) || defined(SEQUENT) || defined(ARDENT)
  if ( signal(SIGSEGV, SigSegvHandler) == (int (*)()) -1)
    ga_error("TrapSigSegv: error from signal setting SIGSEGV",
                  (long) SIGSEGV);
#else
  if ( signal(SIGSEGV, SigSegvHandler) == (void (*)()) -1)
    ga_error("TrapSigSegv: error from signal setting SIGSEGV",
                  (long) SIGSEGV);
#endif
}




/*********************** SIGSYS *************************************/
#if (defined(ENCORE) || defined(SEQUENT) || defined(ARDENT))
int SigSysHandler(sig, code, scp, addr)
#else
void SigSysHandler(sig, code, scp, addr)
#endif
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
#if defined(ENCORE) || defined(SEQUENT) || defined(ARDENT)
  if ( signal(SIGSYS, SigSysHandler) == (int (*)()) -1)
    ga_error("TrapSigSys: error from signal setting SIGSYS",
                  (long) SIGSYS);
#else
  if ( signal(SIGSYS, SigSysHandler) == (void (*)()) -1)
    ga_error("TrapSigSys: error from signal setting SIGSYS",
                  (long) SIGSYS);
#endif
}



/*********************** SIGTRAP *************************************/
#if (defined(ENCORE) || defined(SEQUENT) || defined(ARDENT))
int SigTrapHandler(sig, code, scp, addr)
#else
void SigTrapHandler(sig, code, scp, addr)
#endif
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
#if defined(ENCORE) || defined(SEQUENT) || defined(ARDENT)
  if ( signal(SIGTRAP, SigTrapHandler) == (int (*)()) -1)
    ga_error("TrapSigTrap: error from signal setting SIGTRAP",
                  (long) SIGTRAP);
#else
  if ( signal(SIGTRAP, SigTrapHandler) == (void (*)()) -1)
    ga_error("TrapSigTrap: error from signal setting SIGTRAP",
                  (long) SIGTRAP);
#endif
}



/*********************** SIGHUP *************************************/
#if (defined(ENCORE) || defined(SEQUENT) || defined(ARDENT))
int SigHupHandler(sig, code, scp, addr)
#else
void SigHupHandler(sig, code, scp, addr)
#endif
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
#if defined(ENCORE) || defined(SEQUENT) || defined(ARDENT)
  if ( signal(SIGHUP, SigHupHandler) == (int (*)()) -1)
    ga_error("TrapSigHup: error from signal setting SIGHUP",
                  (long) SIGHUP);
#else
  if ( signal(SIGHUP, SigHupHandler) == (void (*)()) -1)
    ga_error("TrapSigHup: error from signal setting SIGHUP",
                  (long) SIGHUP);
#endif
}



/*********************** SIGTERM *************************************/
#if (defined(ENCORE) || defined(SEQUENT) || defined(ARDENT))
int SigTermHandler(sig, code, scp, addr)
#else
void SigTermHandler(sig, code, scp, addr)
#endif
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
#if defined(ENCORE) || defined(SEQUENT) || defined(ARDENT)
  if ( signal(SIGTERM, SigTermHandler) == (int (*)()) -1)
    ga_error("TrapSigTerm: error from signal setting SIGTERM",
                  (long) SIGTERM);
#else
  if ( signal(SIGTERM, SigTermHandler) == (void (*)()) -1)
    ga_error("TrapSigTerm: error from signal setting SIGTERM",
                  (long) SIGTERM);
#endif
}

