/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/nnodes.c,v 1.3 1995-02-24 02:14:06 d3h325 Exp $ */

#include "sndrcv.h"
#include "sndrcvP.h"

long NNODES_()
/*
  return total no. of processes
*/
{
  return SR_n_proc;
}

