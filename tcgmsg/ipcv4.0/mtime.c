#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_SYS_TYPES_H
#   include <sys/types.h>
#endif
#if HAVE_SYS_TIME_H
#   include <sys/time.h>
#endif

#include "sndrcv.h"

/**
 * Return wall clock time in centiseconds.
 */
Integer MTIME_()
{
    return (Integer) (TCGTIME_()*100.0);
}

static unsigned firstsec=0;  /**< Reference for timer */
static unsigned firstusec=0; /**< Reference for timer */

/**
 * Sets timer reference.
 */
void MtimeReset()
{
    struct timeval tp;
    struct timezone tzp;

    (void) gettimeofday(&tp,&tzp);

    firstsec = tp.tv_sec;
    firstusec = tp.tv_usec;
}

/**
 * Return wall clock time in seconds as accurately as possible.
 */
DoublePrecision TCGTIME_()
{
    static int firstcall=1;
    DoublePrecision low, high;

    struct timeval tp;
    struct timezone tzp;

    if (firstcall) {
        MtimeReset();
        firstcall = 0;
    }

    (void) gettimeofday(&tp,&tzp);

    low = (DoublePrecision) (tp.tv_usec>>1) - (DoublePrecision) (firstusec>>1);
    high = (DoublePrecision) (tp.tv_sec - firstsec);

    return high + 1.0e-6*(low+low);
}
