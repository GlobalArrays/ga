#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_MEMORY_H
#   include <memory.h>
#endif
#if HAVE_STRING_H
#   include <string.h>
#endif

#define UNALIGNED(a) (((unsigned long) (a)) % sizeof(int))

/**
 * Move n bytes from b to a.
 */
void SRmover(char *a, char *b, long n)
{
    if (UNALIGNED(a) || UNALIGNED(b)) {
        (void) memcpy(a, b, (int) n);      /* abdicate responsibility */
    } else {
        /* Data is integer aligned ... move first n/sizeof(int) bytes
         * as integers and the remainder as bytes */

        int ni = n/sizeof(int);
        int *ai = (int *) a;
        int *bi = (int *) b;
        int i;

        for (i=0; i<ni; i++) {
            ai[i] = bi[i];
        }

        /* Handle the remainder */

        a += ni*sizeof(int);
        b += ni*sizeof(int);
        n -= ni*sizeof(int);

        for (i=0; i<n; i++) {
            a[i] = b[i];
        }
    }
}
