/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv5.0/usleep.c,v 1.4 1997-11-07 23:44:20 d3h325 Exp $ */

#ifdef AIX
#include <stdio.h>
#include <sys/select.h>
#endif
#include <sys/types.h>
#include <sys/time.h>

#include "tcgmsgP.h"

#ifndef AIX
extern int select(int, fd_set *, fd_set *, fd_set *, struct timeval *);
#endif

#ifdef STUPIDUSLEEP
void USleep(us)
     long us;
{
  int s = us/1000000;
  if (s == 0)
	s = 1;
  (void) sleep(s);
}
#else
void USleep(long us)
/*
  Sleep for the specified no. of micro-seconds ... uses the timeout
  on select ... it seems to be accurate to about a few centiseconds
  on a sun.  I don't know how much system resources it eats.
*/
{
  int width=0;
  struct timeval timelimit;

/*  printf("%2ld: sleeping for %ldus\n", TCGMSG_nodeid, us);
  fflush(stdout);*/

  timelimit.tv_sec = (int) (us/1000000);
  timelimit.tv_usec = (int) (us - timelimit.tv_sec*1000000);

  (void) select(width, (fd_set *) 0, (fd_set *) 0, (fd_set *) 0,
		&timelimit);
}
#endif

