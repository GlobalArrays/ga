#include "srftoc.h"
#include "tcgmsgP.h"


/*\ Define value of debug flag
\*/
void SETDBG_(onoff)
     long *onoff;
{
  DEBUG_ = *onoff;
}


/*\ Print out statistics for communications ... not yet implemented
\*/
void STATS_()
{
  (void) fprintf(stderr,"STATS_ not yet supported\n");
  (void) fflush(stderr);
}


