#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "sndrcv.h"

/**
 * Return the no. of bytes that n DoublePrecisions occupy
 *
 * These routines use C's knowledge of the sizes of data types
 * to generate a portable mechanism for FORTRAN to translate
 * between bytes, integers and DoublePrecisions. Note that we assume that
 * FORTRAN integers are the same size as C Integers.
 */
Integer MDTOB_(Integer *n)
{
    if (*n < 0) {
        Error("MDTOB_: negative argument",*n);
    }

    return (Integer) (*n * sizeof(DoublePrecision));
}
