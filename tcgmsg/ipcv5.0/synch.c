#include "tcgmsgP.h"

void SYNCH_(Integer *ptype)
{
  Integer junk = 0, n = 1;
#ifdef LAPI
  extern int _armci_called_from_barrier;
  if(!_armci_called_from_barrier)
    armci_msg_barrier();
  else
#endif
  IGOP_(ptype, &junk, &n, "+");

}
