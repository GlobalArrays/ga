/*$Id: llog.c,v 1.2 1995-02-02 23:25:13 d3g681 Exp $*/
/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/llog.c,v 1.2 1995-02-02 23:25:13 d3g681 Exp $ */

#include <stdio.h>

#include <sys/types.h>
#include <sys/time.h>

#include "sndrcv.h"

#if defined(SUN)
extern char *sprintf();
#endif
#ifndef SGI
extern time_t time();
#endif

extern void Error();

void LLOG_()
/*
  close and open stdin and stdout to append to a local logfile
  with the name log.<process#> in the current directory
*/
{
  char name[12];
  time_t t;

  (void) sprintf(name, "log.%03ld",NODEID_());

  (void) fflush(stdout);
  (void) fflush(stderr);

  if (freopen(name, "a", stdout) == (FILE *) NULL)
    Error("LLOG_: error re-opening stdout", (long) -1);

  if (freopen(name, "a", stderr) == (FILE *) NULL)
    Error("LLOG_: error re-opening stderr", (long) -1);

  (void) time(&t);
  (void) printf("\n\nLog file opened : %s\n\n",ctime(&t));
  (void) fflush(stdout);
}
