/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/hello.c,v 1.1.1.1 1994-03-29 06:44:47 d3g681 Exp $ */

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
