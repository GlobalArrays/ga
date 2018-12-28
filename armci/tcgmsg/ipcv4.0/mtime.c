#if HAVE_CONFIG_H
#   include "config.h"
#endif

/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/mtime.c,v 1.4 1995-02-24 02:17:28 d3h325 Exp $ */

#include <stdio.h>
#include "sndrcv.h"

long MTIME_()
/*
  return wall clock time in centiseconds
*/
{
  return (long) (TCGTIME_()*100.0);
}

#include <sys/types.h>
#include <sys/time.h>

static unsigned firstsec=0;     /* Reference for timer */
static unsigned firstusec=0;    /* Reference for timer */

void MtimeReset()               /* Sets timer reference */
{
  struct timeval tp;
  struct timezone tzp;
  
  (void) gettimeofday(&tp,&tzp);

  firstsec = tp.tv_sec;
  firstusec = tp.tv_usec;
}

double TCGTIME_()
/*
  Return wall clock time in seconds as accurately as possible
*/
{
  static int firstcall=1;
  double low, high;

  struct timeval tp;
  struct timezone tzp;

  if (firstcall) {
    MtimeReset();
    firstcall = 0;
  }

  (void) gettimeofday(&tp,&tzp);

  low = (double) (tp.tv_usec>>1) - (double) (firstusec>>1);
  high = (double) (tp.tv_sec - firstsec);

  return high + 1.0e-6*(low+low);
}
