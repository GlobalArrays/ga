/* $Id: shmlimit.c,v 1.8 2000-04-17 22:31:41 d3h325 Exp $ */
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

/* We got 2 ways to return shmmax value to parent by: exit status or pipe 
 * the pipe mechanism is preferable since it allows to pass arbitrary values 
 * from child to parent. The exit mechanism allows values from 0 to 255 (MB)
 * only!
 */
#define USE_PIPE 
#define DEBUG_ 0

#if defined(DECOSF) || defined(SOLARIS64)
#define PIPE_AFTER_FORK_BUG
#endif

void (*Sig_Chld_Orig)();
static int status=0;
static int caught_sigchld=0;
extern int armci_me;

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
#ifdef PIPE_AFTER_FORK_BUG
    int i;
#endif

#ifdef USE_PIPE
    int y;
    int fd[2];

    if(pipe(fd)==-1) armci_die("armci shmem_test pipe failed",0);
#endif

    TrapSigChld();

    if ( (pid = fork() ) < 0)

        armci_die("armci shmem_test fork failed", (int)pid);

    else if(pid == 0){

#ifdef USE_PIPE
       x= armci_shmem_test();

#ifdef PIPE_AFTER_FORK_BUG
       /* due to a bug in OSF1 V4.0/1229/alpha first item written gets hosed*/
       for(i=0;i<2;i++)
#endif
           if(write(fd[1],&x,sizeof(int)) <sizeof(int))
                         armci_die("armci shmem_test: write failed",0);
#endif
       _exit(x);

    }else{

       pid_t rc;

#ifdef USE_PIPE
       int val;
#ifdef PIPE_AFTER_FORK_BUG
       /* due to a bug in OSF1 V4.0/1229/alpha first item read is garbage */
       for(i=0;i<2;i++)
#endif
          if((val=read(fd[0],&y,sizeof(int)))<sizeof(int))
                         armci_die("armci shmem_test: read failed",val);
#endif

       /* we might already got status from wait in SIGCHLD handler */
       if(!caught_sigchld){
again:   rc = wait (&status);
         if(rc == -1 && errno == EINTR) goto again;
       }
       if (!WIFEXITED(status)) armci_die("ARMCI: child did not return rc",0);
       x = WEXITSTATUS(status);
    }
   
    /* restore previous signal handler */
    RestoreSigChld();

#ifdef USE_PIPE
    close(fd[0]);
    close(fd[1]);
    if(DEBUG_)
       printf("%d:in parent: x=%d y=%d\n",armci_me,x,y);fflush(stdout);sleep(1);
    return y;
#else
    return x;
#endif

}
