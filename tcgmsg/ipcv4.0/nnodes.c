/*$Id: nnodes.c,v 1.2 1995-02-02 23:25:23 d3g681 Exp $*/
/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/nnodes.c,v 1.2 1995-02-02 23:25:23 d3g681 Exp $ */

#include "sndrcv.h"
#include "sndrcvP.h"

long NNODES_()
/*
  return total no. of processes
*/
{
  return SR_n_proc;
}

