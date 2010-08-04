#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDIO_H
#   include <stdio.h>
#endif

#include "typesf2c.h"
#include "sndrcvP.h"
#include "sndrcv.h"
#include "srftoc.h"
#ifdef USE_VAMPIR
#   include "tcgmsg_vampir.h"
#endif

/**
 * broadcast buffer to all other processes from process originator
 * ... all processes call this routine specifying the same
 * orginating process.
 *
 * Optimized for communicating clusters of processes ... broadcast
 * amoung cluster masters, and then amoung slaves in a cluster.
 */
void BRDCST_(Integer *type, void *buf, Integer *lenbuf, Integer *originator)
{
    Integer me = NODEID_();
    Integer master = SR_clus_info[SR_clus_id].masterid;
    Integer nslave = SR_clus_info[SR_clus_id].nslave;
    Integer slaveid = me - master;
    Integer synch = 1;
    Integer lenmes, from, up, left, right;

#ifdef USE_VAMPIR
    vampir_begin(TCGMSG_BRDCST,__FILE__,__LINE__);
#endif

    /* Process zero is at the top of the broadcast tree */

    if ((me == *originator) && (me != 0)) {
        Integer zero = 0;
        SND_(type, buf, lenbuf, &zero, &synch);
    }
    else if ((*originator != 0) && (me == 0)) {
        RCV_(type, buf, lenbuf, &lenmes, originator, &from, &synch);
    }

    if ((*originator != 0) && (SR_n_proc == 2)) return;    /* Special case */

    /* Broadcast amoung cluster masters */

    if (me == master) {
        up    = (SR_clus_id-1)/2;
        left  = 2*SR_clus_id + 1;
        right = 2*SR_clus_id + 2;
        up = SR_clus_info[up].masterid;
        left = (left < SR_n_clus) ? SR_clus_info[left].masterid : -1;
        right = (right < SR_n_clus) ? SR_clus_info[right].masterid : -1;

        if (me != 0)
            RCV_(type, buf, lenbuf, &lenmes, &up, &from, &synch);
        if (left > 0)
            SND_(type, buf, lenbuf, &left, &synch);
        if (right > 0)
            SND_(type, buf, lenbuf, &right, &synch);
    }

    /* Broadcast amoung local slaves */

    up    = master + (slaveid-1)/2;
    left  = master + 2*slaveid + 1;
    right = master + 2*slaveid + 2;

    if (me != master)
        RCV_(type, buf, lenbuf, &lenmes, &up, &from, &synch);
    if (left < (master+nslave))
        SND_(type, buf, lenbuf, &left, &synch);
    if (right < (master+nslave))
        SND_(type, buf, lenbuf, &right, &synch);

#ifdef USE_VAMPIR
    vampir_end(TCGMSG_BRDCST,__FILE__,__LINE__);
#endif
}  
