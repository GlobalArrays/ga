/*$Id: testarg.c,v 1.2 1995-02-02 23:26:00 d3g681 Exp $*/
/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/testarg.c,v 1.2 1995-02-02 23:26:00 d3g681 Exp $ */

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
