#include "tcgmsg.h"

extern int sleep(int);

void SYNCH_(Integer *ptype)
{
  sleep(2);
}
