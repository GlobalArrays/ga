/*$Id: nodeid.c,v 1.2 1995-02-02 23:25:24 d3g681 Exp $*/
/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/nodeid.c,v 1.2 1995-02-02 23:25:24 d3g681 Exp $ */

#include "sndrcv.h"
#include "sndrcvP.h"

long NODEID_()
/*
  return logical node no. of current process
*/
{
  return SR_proc_id;
}
