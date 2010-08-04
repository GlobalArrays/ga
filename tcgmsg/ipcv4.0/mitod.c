#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "sndrcv.h"

/**
 * Return the minimum no. of DoublePrecisions in which we can store n Integers.
 *
 * These routines use C's knowledge of the sizes of data types
 * to generate a portable mechanism for FORTRAN to translate
 * between bytes, integers and DoublePrecisions. Note that we assume that
 * FORTRAN integers are the same size as C Integers.
 */

Integer MITOD_(Integer *n)
{
    if (*n < 0) {
        Error("MITOD_: negative argument",*n);
    }
    return (Integer) ( (MITOB_(n) + sizeof(DoublePrecision) - 1) / sizeof(DoublePrecision) );
}
