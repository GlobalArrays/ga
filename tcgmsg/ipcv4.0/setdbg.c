/*$Id: setdbg.c,v 1.2 1995-02-02 23:25:37 d3g681 Exp $*/
/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/setdbg.c,v 1.2 1995-02-02 23:25:37 d3g681 Exp $ */

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

