/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/nnodes.c,v 1.1.1.1 1994-03-29 06:44:48 d3g681 Exp $ */

#include "sndrcv.h"
#include "sndrcvP.h"

long NNODES_()
/*
  return total no. of processes
*/
{
  return SR_n_proc;
}

