/* $$ */

#include "tcgmsg.h"
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
