/*
 * This code is used to test shared memory limits within
 * a separately forked child process.
 * This has to be done in a child process to make sure that
 * swap space allocated in test is not counted against ARMCI processes.
 * Some systems do not release swap after shmem ids are deleted
 * until the process exits.
 * JN/07.07.99
 */
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <signal.h>

#define DEBUG_ 0

void (*Sig_Chld_Orig)();
static int status=0;
static int caught_sigchld=0;

#if defined(SUN) && !defined(SOLARIS)
static void SigChldHandler(sig, code, scp, addr)
     int code;
     struct sigcontext *scp;
     char *addr;
#else
static void SigChldHandler(sig)
#endif
     int sig;
{
#ifdef DISABLED
  int pid;
  pid = wait(&status);
  caught_sigchld=1;
#endif
}

static void TrapSigChld()
{
  if ( (Sig_Chld_Orig = signal(SIGCHLD, SigChldHandler)) == SIG_ERR)
    armci_die("TrapSigChld: error from signal setting SIGCHLD",0);
}


static void RestoreSigChld()
{
  if ( signal(SIGCHLD, Sig_Chld_Orig) == SIG_ERR)
    armci_die("Restore_SigChld: error from restoring signal SIGChld",0);
}


int armci_child_shmem_init()
{
    pid_t pid;
    int x;
    
    TrapSigChld();
    if ( (pid = fork() ) < 0)
        armci_die("armci shmem_test fork failed", (int)pid);
    else if(pid == 0){

       x= armci_shmem_test();
       exit(x);
    }else{

       pid_t rc;
       
       /* we might already got status from wait in SIGCHLD handler */
       if(!caught_sigchld){
again:   rc = wait (&status);
         /* can get SIGCHLD while waiting */
/*         if(rc!=pid) perror("ARMCI: wait for child process Shm failed:");*/
         if(rc == -1 && errno == EINTR) goto again;
       }
       if (!WIFEXITED(status)) armci_die("ARMCI: child did not return rc",0);
       x = WEXITSTATUS(status);
    }
    RestoreSigChldDfl();
    return x;
}
