#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "typesf2c.h"
#include "srftoc.h"

extern Integer random();
extern int srandom();

DoublePrecision DRAND48_()
{
    return ( (DoublePrecision) random() ) * 4.6566128752458e-10;
}

void SRAND48_(unsigned *seed)
{
    (void) srandom(*seed);
}
