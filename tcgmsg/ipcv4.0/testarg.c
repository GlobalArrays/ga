/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/testarg.c,v 1.1.1.1 1994-03-29 06:44:51 d3g681 Exp $ */

/*
  This checks the functioning of the include file farg.h
*/

#include "farg.h"

void parg()
{
  int i;

  for (i=0; i<ARGC_; i++)
    (void) printf("argv(%d)=%s\n", i, ARGV_[i]);
}
