#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STRINGS_H
#   include <strings.h>
#endif
#if HAVE_STRING_H
#   include <string.h>
#endif
#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif

extern void free(void *ptr);

#include "sndrcv.h"
#include "msgtypesc.h"

#define TCG_MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define TCG_MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define TCG_ABS(a) (((a) >= 0) ? (a) : (-(a)))

#include "sndrcvP.h"

#define GOP_BUF_SIZE 81920

#ifdef USE_VAMPIR
#include "tcgmsg_vampir.h"
#endif

static void idoop(Integer n, char *op, Integer *x, Integer *work)
{
    if (strncmp(op,"+",1) == 0) {
        while(n--) {
            *x++ += *work++;
        }
    } else if (strncmp(op,"*",1) == 0) {
        while(n--) {
            *x++ *= *work++;
        }
    } else if (strncmp(op,"max",3) == 0) {
        while(n--) {
            *x = TCG_MAX(*x, *work);
            x++; work++;
        }
    } else if (strncmp(op,"min",3) == 0) {
        while(n--) {
            *x = TCG_MIN(*x, *work);
            x++; work++;
        }
    } else if (strncmp(op,"absmax",6) == 0) {
        while(n--) {
            register Integer x1 = TCG_ABS(*x), x2 = TCG_ABS(*work);
            *x = TCG_MAX(x1, x2);
            x++; work++;
        }
    } else if (strncmp(op,"absmin",6) == 0) {
        while(n--) {
            register Integer x1 = TCG_ABS(*x), x2 = TCG_ABS(*work);
            *x = TCG_MIN(x1, x2);
            x++; work++;
        }
    } else if (strncmp(op,"or",2) == 0)  {
        while(n--) {
            *x |= *work;
            x++; work++;
        }
    } else {
        Error("idoop: unknown operation requested", (Integer) n);
    }
}

static void ddoop(
        Integer n,
        char *op,
        DoublePrecision *x,
        DoublePrecision *work)
{
    if (strncmp(op,"+",1) == 0) {
        while(n--) {
            *x++ += *work++;
        }
    } else if (strncmp(op,"*",1) == 0) {
        while(n--) {
            *x++ *= *work++;
        }
    } else if (strncmp(op,"max",3) == 0) {
        while(n--) {
            *x = TCG_MAX(*x, *work);
            x++; work++;
        }
    } else if (strncmp(op,"min",3) == 0) {
        while(n--) {
            *x = TCG_MIN(*x, *work);
            x++; work++;
        }
    } else if (strncmp(op,"absmax",6) == 0) {
        while(n--) {
            register DoublePrecision x1 = TCG_ABS(*x), x2 = TCG_ABS(*work);
            *x = TCG_MAX(x1, x2);
            x++; work++;
        }
    } else if (strncmp(op,"absmin",6) == 0) {
        while(n--) {
            register DoublePrecision x1 = TCG_ABS(*x), x2 = TCG_ABS(*work);
            *x = TCG_MIN(x1, x2);
            x++; work++;
        }
    } else {
        Error("ddoop: unknown operation requested", (Integer) n);
    }
}

/**
 * Global summation optimized for networks of clusters of processes.
 */
void DGOP_(
        Integer *ptype,
        DoublePrecision *x,
        Integer *pn,
        char *op,
        Integer oplen)
{
    Integer me = NODEID_();
    Integer master = SR_clus_info[SR_clus_id].masterid;
    Integer nslave = SR_clus_info[SR_clus_id].nslave;
    Integer slaveid = me - master;
    Integer synch = 1;
    Integer type = (*ptype & MSGDBL) ? *ptype : *ptype + MSGDBL;
    Integer nleft = *pn;
    Integer buflen = TCG_MIN(nleft,GOP_BUF_SIZE); /* Try to get even sized buffers */
    Integer nbuf   = (nleft-1) / buflen + 1;
    Integer zero = 0;
    DoublePrecision *tmp = x;
    DoublePrecision *work;
    Integer nb, ndo, lenmes, from, up, left, right;

#ifdef USE_VAMPIR
    vampir_begin(TCGMSG_DGOP,__FILE__,__LINE__);
#endif

    buflen = (nleft-1) / nbuf + 1;
    if (!(work = (DoublePrecision *) malloc((unsigned) (buflen*sizeof(DoublePrecision))))) {
        Error("DGOP: failed to malloc workspace", nleft);
    }

    /* This loop for pipelining and to avoid caller
       having to provide workspace */

    while (nleft) {
        ndo = TCG_MIN(nleft, buflen);
        nb  = ndo * sizeof(DoublePrecision);

        /* Do summation amoung slaves in a cluster */

        up    = master + (slaveid-1)/2;
        left  = master + 2*slaveid + 1;
        right = master + 2*slaveid + 2;

        if (left < (master+nslave)) {
            RCV_(&type, (char *) work, &nb, &lenmes, &left, &from, &synch);
            ddoop(ndo, op, x, work);
        }
        if (right < (master+nslave)) {
            RCV_(&type, (char *) work, &nb, &lenmes, &right, &from, &synch);
            ddoop(ndo, op, x, work);
        }
        if (me != master) {
            SND_(&type, (char *) x, &nb, &up, &synch);
        }

        /* Do summation amoung masters */

        if (me == master) {
            up    = (SR_clus_id-1)/2;
            left  = 2*SR_clus_id + 1;
            right = 2*SR_clus_id + 2;
            up = SR_clus_info[up].masterid;
            left = (left < SR_n_clus) ? SR_clus_info[left].masterid : -1;
            right = (right < SR_n_clus) ? SR_clus_info[right].masterid : -1;

            if (left > 0) {
                RCV_(&type, (char *) work, &nb, &lenmes, &left, &from, &synch);
                ddoop(ndo, op, x, work);
            }
            if (right > 0) {
                RCV_(&type, (char *) work, &nb, &lenmes, &right, &from, &synch);
                ddoop(ndo, op, x, work);
            }
            if (me != 0)
                SND_(&type, (char *) x, &nb, &up, &synch);
        }
        nleft -= ndo;
        x     += ndo;
        type  += 13;   /* Temporary hack for hippi switch */
    }
    free((char *) work);

    /* Zero has the results ... broadcast them back */
    nb = *pn * sizeof(DoublePrecision);
    BRDCST_(&type, (char *) tmp, &nb, &zero);

#ifdef USE_VAMPIR
    vampir_end(TCGMSG_DGOP,__FILE__,__LINE__);
#endif
}

/**
 * Global summation optimized for networks of clusters of processes.
 */
void IGOP_(Integer *ptype, Integer *x, Integer *pn, char *op, Integer oplen)
{
    Integer me = NODEID_();
    Integer master = SR_clus_info[SR_clus_id].masterid;
    Integer nslave = SR_clus_info[SR_clus_id].nslave;
    Integer slaveid = me - master;
    Integer synch = 1;
    Integer type = (*ptype & MSGINT) ? *ptype : *ptype + MSGINT;
    Integer nleft = *pn;
    Integer zero = 0;
    Integer *tmp = x;
    Integer *work;
    Integer nb, ndo, lenmes, from, up, left, right;

#ifdef USE_VAMPIR
    vampir_begin(TCGMSG_IGOP,__FILE__,__LINE__);
#endif

    if (!(work = (Integer *) 
                malloc((unsigned) (TCG_MIN(nleft,GOP_BUF_SIZE)*sizeof(Integer))))) {
        Error("IGOP: failed to malloc workspace", nleft);
    }

    /* This loop for pipelining and to avoid caller
       having to provide workspace */

    while (nleft) {
        ndo = TCG_MIN(nleft, GOP_BUF_SIZE);
        nb  = ndo * sizeof(Integer);
        /* Do summation amoung slaves in a cluster */

        up    = master + (slaveid-1)/2;
        left  = master + 2*slaveid + 1;
        right = master + 2*slaveid + 2;

        if (left < (master+nslave)) {
            RCV_(&type, (char *) work, &nb, &lenmes, &left, &from, &synch);
            idoop(ndo, op, x, work);
        }
        if (right < (master+nslave)) {
            RCV_(&type, (char *) work, &nb, &lenmes, &right, &from, &synch);
            idoop(ndo, op, x, work);
        }
        if (me != master) {
            SND_(&type, (char *) x, &nb, &up, &synch);
        }

        /* Do summation amoung masters */

        if (me == master) {
            up    = (SR_clus_id-1)/2;
            left  = 2*SR_clus_id + 1;
            right = 2*SR_clus_id + 2;
            up = SR_clus_info[up].masterid;
            left = (left < SR_n_clus) ? SR_clus_info[left].masterid : -1;
            right = (right < SR_n_clus) ? SR_clus_info[right].masterid : -1;

            if (left > 0) {
                RCV_(&type, (char *) work, &nb, &lenmes, &left, &from, &synch);
                idoop(ndo, op, x, work);
            }
            if (right > 0) {
                RCV_(&type, (char *) work, &nb, &lenmes, &right, &from, &synch);
                idoop(ndo, op, x, work);
            }
            if (me != 0)
                SND_(&type, (char *) x, &nb, &up, &synch);
        }
        nleft -= ndo;
        x     += ndo;
        type  += 13;   /* Temporary hack for hippi switch */
    }
    (void) free((char *) work);

    /* Zero has the results ... broadcast them back */
    nb = *pn * sizeof(Integer);
    BRDCST_(&type, (char *) tmp, &nb, &zero);

#ifdef USE_VAMPIR
    vampir_end(TCGMSG_IGOP,__FILE__,__LINE__);
#endif
}
