#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "sndrcv.h"

/**
 * Return the minimum no. of integers which will hold n DoublePrecisions.
 *
 * These routines use C's knowledge of the sizes of data types
 * to generate a portable mechanism for FORTRAN to translate
 * between bytes, integers and DoublePrecisions. Note that we assume that
 * FORTRAN integers are the same size as C Integers.
*/

Integer MDTOI_(Integer *n)
{
    if (*n < 0) {
        Error("MDTOI_: negative argument",*n);
    }
    return (Integer) ( (MDTOB_(n) + sizeof(Integer) - 1) / sizeof(Integer) );
}
