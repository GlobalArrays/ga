/**
 * @file fapi.c
 *
 * Implements the Fortran interface.
 * These calls forward to the (possibly) weak symbols of the internal
 * implementations.
 */
#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "c.names.h"

#if PROFILING_DEFINES
#   include "wapidefs.h"
#endif
#include "wapi.h"


/**
 * (Non-blocking) put a 2-dimensional patch of data into a global array.
 */
void FATR ga_nbput_(Integer *g_a, Integer *ilo, Integer *ihi, Integer *jlo, Integer *jhi, void *buf, Integer *ld, Integer *nbhandle)
{
    Integer lo[2], hi[2];
    lo[0]=*ilo;
    lo[1]=*jlo;
    hi[0]=*ihi;
    hi[1]=*jhi;
    wnga_nbput(g_a, lo, hi, buf, ld, nbhandle);
}


/**
 * Put a 2-dimensional patch of data into a global array.
 */
void FATR ga_put_(Integer *g_a, Integer *ilo, Integer *ihi, Integer *jlo, Integer *jhi, void *buf, Integer *ld)
{
    Integer lo[2], hi[2];
    lo[0]=*ilo;
    lo[1]=*jlo;
    hi[0]=*ihi;
    hi[1]=*jhi;
    wnga_put(g_a, lo, hi, buf, ld);
}


/**
 * (Non-blocking) put an n-dimensional patch of data into a global array.
 */
void FATR nga_nbput_(Integer *g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle)
{
    wnga_nbput(g_a, lo, hi, buf, ld, nbhandle);
}


/**
 * Put an n-dimensional patch of data into a global array.
 */
void FATR nga_put_(Integer *g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
    wnga_put(g_a, lo, hi, buf, ld);
}
