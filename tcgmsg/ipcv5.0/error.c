/* $$ */

#include "tcgmsgP.h"

#include <errno.h>

extern void perror(const char *);
extern void exit(int);
extern void ZapChildren(void);

void Error(const char *string, long integer)
{
  (void) signal(SIGINT, SIG_IGN);
  (void) signal(SIGCHLD, SIG_DFL); /* Death of children to be expected */

  (void) fflush(stdout);
  if (TCGMSG_caught_sigint) {
    (void) fprintf(stderr,"%2ld: interrupt\n",NODEID_());
  }
  else {
    (void) fprintf(stderr,"%2ld: %s %ld (%#lx).\n", NODEID_(), string,
		   integer,integer);
    if (errno != 0)
      perror("system error message");
  }
  (void) fflush(stderr);

  /* Shut down the sockets and remove shared memory and semaphores to
     propagate an error condition to anyone that is trying to communicate
     with me */

  ZapChildren();  /* send interrupt to children which should trap it
		     and call Error in the handler */

#ifdef SHMEM
  DeleteSharedRegion(TCGMSG_shmem_id);
#endif

  abort();
}

void PARERR_(code)
   long *code;
/*
  Interface from fortran to c error routine
*/
{
  Error("User detected error in FORTRAN", *code);
}
