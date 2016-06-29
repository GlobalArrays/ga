#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif

#include "typesf2c.h"

static DoublePrecision gai_drand_(Integer *flag)
{
    if (*flag)
        srandom((unsigned) *flag);

    return ((DoublePrecision) random()) * 4.6566128752458e-10;
}

#define drand_ F77_FUNC(drand,DRAND)
DoublePrecision drand_(Integer *flag)
{
    return (gai_drand_(flag));
}
