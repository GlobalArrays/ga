/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/setdbg.c,v 1.1.1.1 1994-03-29 06:44:49 d3g681 Exp $ */

#include "sndrcv.h"
#include "sndrcvP.h"

void SETDBG_(value)
    long *value;
/*
  set global debug flag for this process to value
*/
{
  SR_debug = *value;
}

