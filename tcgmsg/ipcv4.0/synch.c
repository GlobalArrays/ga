#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "typesf2c.h"
#include "sndrcv.h"

#ifdef USE_VAMPIR
#   include "tcgmsg_vampir.h"
#endif

#ifdef OLDSYNC
/**
 * Synchronize by forcing all process to exchange a zero length message
 * of given type with process 0.
*/
void SYNCH_(Integer *type)
{
    Integer me = NODEID_();
    Integer nproc = NNODES_();
    char *buf = "";
    Integer zero = 0;
    Integer sync = 1;
    Integer from, lenmes, i;

#ifdef USE_VAMPIR
    vampir_begin(TCGMSG_SYNCH,__FILE__,__LINE__);
#endif

    /* First everyone sends null message to zero */

    if (me == 0) {
        for (i=1; i<nproc; i++) {
            RCV_(type, buf, &zero, &lenmes, &i, &from, &sync);
        }
    } else {
        SND_(type, buf, &zero, &zero, &sync);
    }

    /* Zero broadcasts message null message to everyone */

    BRDCST_(type, buf, &zero, &zero);
#ifdef USE_VAMPIR
    vampir_end(TCGMSG_SYNCH,__FILE__,__LINE__);
#endif
}

#else /* OLDSYNC */

/**
 * Synchronize by doing a global sum of a single integer variable
 * ... as Integer type is unique there should be no problems.
 */
void SYNCH_(Integer *type)
{
    Integer junk = 0, n = 1;
#ifdef USE_VAMPIR
    vampir_begin(TCGMSG_SYNCH,__FILE__,__LINE__);
#endif
    IGOP_(type, &junk, &n, "+", 1);
#ifdef USE_VAMPIR
    vampir_end(TCGMSG_SYNCH,__FILE__,__LINE__);
#endif
}

#endif /* OLDSYNC */
