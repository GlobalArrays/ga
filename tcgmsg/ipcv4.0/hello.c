/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/hello.c,v 1.4 1995-02-24 02:17:20 d3h325 Exp $ */

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
