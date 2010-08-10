#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "papi.h"
#include "typesf2c.h"

void wnga_nbput(Integer *g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle)
{
    pnga_nbput(g_a, lo, hi, buf, ld, nbhandle);
}

void wnga_put(Integer *g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
    pnga_put(g_a, lo, hi, buf, ld);
}

