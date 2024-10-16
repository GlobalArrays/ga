#if HAVE_CONFIG_H
#   include "config.h"
#endif

/* $Id: signaltrap.c,v 1.28 2005-05-13 19:06:40 vinod Exp $ */
 /******************************************************\
 * Signal handler functions for the following signals:  *
 *        SIGINT, SIGCHLD, SIGBUS, SIGFPE, SIGILL,      *
 *        SIGSEGV, SIGSYS, SIGTRAP, SIGHUP, SIGTERM     *
 * Used to call armci_error that frees up IPC resources *
 \******************************************************/


#if HAVE_SIGNAL_H
#   include <signal.h>
#endif
#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_SYS_TYPES_H
#   include <sys/types.h>
#endif
#if HAVE_SYS_WAIT_H
#   include <sys/wait.h>
#endif
#if HAVE_UNISTD_H
#   include <unistd.h>
#endif
#if HAVE_ERRNO_H
#   include <errno.h>
#endif
#include "armci.h"
#include "armcip.h"

#define PAUSE_ON_ERROR__

#define  Error armci_die 
#if !defined(armci_die)
extern void Error();
#endif
#   define SigType  void

#ifndef SIG_ERR
#   define SIG_ERR         (SigType (*)())-1
#endif

extern int armci_me;

int AR_caught_sigint=0;
int AR_caught_sigterm=0;
int AR_caught_sigchld=0;
int AR_caught_sigsegv=0;
int AR_caught_sig=0;

SigType (*SigChldOrig)(), (*SigIntOrig)(), (*SigHupOrig)(), (*SigTermOrig)();
SigType (*SigSegvOrig)();


/*********************** SIGINT *************************************/
SigType SigIntHandler(sig)
     int sig;
{
  AR_caught_sigint = 1;
  AR_caught_sig= sig;
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
  if(AR_caught_sigint) SigIntOrig(SIGINT);
  if ( signal(SIGINT, SigIntOrig) == SIG_ERR)
    Error("RestoreSigInt: error from restoring signal SIGINT",0);
}


/*********************** SIGABORT *************************************/
SigType SigAbortHandler(sig)
     int sig;
{
  AR_caught_sig= sig;
  Error("SigIntHandler: abort signal was caught: cleaning up",(int) sig);
}

void TrapSigAbort()
/*
  Trap the signal SIGINT so that we can propagate error
  conditions and also tidy up shared system resources in a
  manner not possible just by killing everyone
*/
{
  if (  signal(SIGINT, SigAbortHandler) == SIG_ERR)
    Error("TrapSigAbort: error from signal setting SIGABORT",0);
}



/*********************** SIGCHLD *************************************/
SigType SigChldHandler(sig)
     int sig;
{
  int status;

#if defined(LINUX)
  pid_t ret;
  /* Trap signal as soon as possible to avoid race */
  if ( (SigChldOrig = signal(SIGCHLD, SigChldHandler)) == SIG_ERR)
    Error("SigChldHandler: error from signal setting SIGCHLD",0);
#endif

# if defined(LINUX)
  ret = waitpid(0, &status, WNOHANG);
  if((ret == 0) || ((ret == -1) && (errno == ECHILD))) { return; }
# else
  (void)wait(&status);
# endif

      AR_caught_sigchld=1;
      AR_caught_sig= sig;
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
{
    if(AR_caught_sigchld) 
        SigChldOrig(SIGCHLD);
    if (signal(SIGCHLD, SigChldOrig) == SIG_ERR)
        Error("RestoreSigChld: error from restoring signal SIGChld",0);
}


void RestoreSigChldDfl()
{
(void) signal(SIGCHLD, SIG_DFL);
}


/*********************** SIGBUS *************************************/
SigType SigBusHandler(sig)
     int sig;
{
  AR_caught_sig= sig;
#ifdef PAUSE_ON_ERROR
  fprintf(stderr,"%d(%d): Bus Error ... pausing\n",
          armci_me, getpid() );pause();
#endif
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
SigType SigFpeHandler(sig)
     int sig;
{
  AR_caught_sig= sig;
#ifdef PAUSE_ON_ERROR
  fprintf(stderr,"%d(%s:%d): Sig FPE ... pausing\n",
          armci_me, armci_clus_info[armci_clus_me].hostname,
	  getpid() );pause(); 
#endif
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
SigType SigIllHandler(sig)
     int sig;
{
  AR_caught_sig= sig;
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
SigType SigSegvHandler(sig)
     int sig;
{
  AR_caught_sig= sig;
  AR_caught_sigsegv=1;
#ifdef PAUSE_ON_ERROR
  fprintf(stderr,"%d(%s:%d): Segmentation Violation ... pausing\n",
          armci_me, armci_clus_info[armci_clus_me].hostname,
	  getpid() );pause(); 
#endif

  Error("Segmentation Violation error, status=",(int) sig);
}
#ifdef ENABLE_CHECKPOINT
static void * signal_arr[100];
SigType SigSegvActionSa(int sig,siginfo_t *sinfo, void *ptr)
{
  int (*func)();      
  AR_caught_sig= sig;
  AR_caught_sigsegv=1;
  func = signal_arr[sig];
  /*printf("\n%d:in sigaction %p, %d\n",armci_me,sinfo->si_addr,sinfo->si_errno);fflush(stdout);*/

  if(func(sinfo->si_addr,sinfo->si_errno,sinfo->si_fd))
     Error("Segmentation Violation error, status=",(int) SIGSEGV);
}

void TrapSigSegvSigaction()
{
  struct sigaction sa;
    sa.sa_sigaction = (void *)SigSegvActionSa;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGSEGV, &sa, NULL);
}
#endif

void TrapSigSegv()
/*
  Trap SIGSEGV
*/
{
  if ( (SigSegvOrig=signal(SIGSEGV, SigSegvHandler)) == SIG_ERR)
    Error("TrapSigSegv: error from signal setting SIGSEGV", 0);
}


void RestoreSigSegv()
/*
 Restore the original signal handler
*/
{
/*
  if(AR_caught_sigsegv) SigSegvOrig(SIGSEGV);
*/
#ifdef ENABLE_CHECKPOINT__
  struct sigaction sa;
  sa.sa_handler = (void *)SigSegvOrig;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;
  sigaction(SIGSEGV, &sa, NULL);
  sigaction(SIGSEGV,&sa,NULL);
#else
  if ( signal(SIGSEGV,SigSegvOrig) == SIG_ERR)
    Error("RestoreSigSegv: error from restoring signal SIGSEGV",0);
#endif
}


/*********************** SIGSYS *************************************/
SigType SigSysHandler(sig)
     int sig;
{
  AR_caught_sig= sig;
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
SigType SigTrapHandler(sig)
     int sig;
{
  AR_caught_sig= sig;
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
SigType SigHupHandler(sig)
     int sig;
{
  AR_caught_sig= sig;
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
  if(AR_caught_sig== SIGHUP) SigHupOrig(SIGHUP);
  if ( signal(SIGHUP, SigHupOrig) == SIG_ERR)
    Error("RestoreSigHUP: error from restoring signal SIGHUP",0);
}



/*********************** SIGTERM *************************************/
SigType SigTermHandler(sig)
     int sig;
{
  AR_caught_sigterm = 1;
  AR_caught_sig= sig;
  Error("Terminate signal was sent, status=",(int) sig);
}

void TrapSigTerm()
/*
  Trap SIGTERM
*/
{
  if ( (SigTermOrig = signal(SIGTERM, SigTermHandler)) == SIG_ERR)
    Error("TrapSigTerm: error from signal setting SIGTERM", 0);
}

void RestoreSigTerm()
/*
 Restore the original signal handler
*/
{
  if(AR_caught_sigterm && (SigTermOrig != SIG_DFL) ) SigTermOrig(SIGTERM);
  if ( signal(SIGTERM, SigTermOrig) == SIG_ERR)
    Error("RestoreSigTerm: error from restoring signal SIGTerm",0);
}


/*********************** SIGIOT *************************************/
#ifdef SIGIOT
SigType SigIotHandler(sig)
     int sig;
{
  AR_caught_sig= sig;
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
#endif



/*********************** SIGCONT *************************************/
SigType SigContHandler(sig)
     int sig;
{
/*  Error("Trace Cont error, status=",(int) sig);*/
  AR_caught_sig= sig;
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
SigType SigXcpuHandler(sig)
     int sig;
{
  AR_caught_sig= sig;
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
#ifdef ENABLE_CHECKPOINT
     TrapSigSegvSigaction();
#else
     TrapSigSegv(); 
#endif
     TrapSigSys();
     TrapSigTrap();
     TrapSigAbort();
     TrapSigTerm();
     TrapSigInt();
}


void ARMCI_ParentTrapSignals()
{
     TrapSigChld();
     TrapSigHup();
}


void ARMCI_RestoreSignals()
{
     RestoreSigTerm();
     RestoreSigInt();
     RestoreSigSegv();
}


void ARMCI_ParentRestoreSignals()
{
     RestoreSigChld();
     ARMCI_RestoreSignals();
     RestoreSigHup();
}

#ifdef ENABLE_CHECKPOINT
/*user can register a function with 3 parameters, 1st offending address
 * 2nd err number and third file descriptor*/
void ARMCI_Register_Signal_Handler(int sig, void  (*func)())
{
    signal_arr[sig]=func;
}
#endif
