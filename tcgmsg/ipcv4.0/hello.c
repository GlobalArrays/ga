/*$Id: hello.c,v 1.2 1995-02-02 23:25:07 d3g681 Exp $*/
/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/hello.c,v 1.2 1995-02-02 23:25:07 d3g681 Exp $ */

#include "sndrcv.h"

int main(argc, argv)
     int argc;
     char **argv;
/*
  Traditional first parallel program
*/
{
  PBEGIN_(argc, argv);

  (void) printf("Hello from node %ld\n",NODEID_());

  PEND_();

  return 0;
}
