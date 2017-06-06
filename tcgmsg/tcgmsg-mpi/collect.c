#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include <mpi.h>

#include "tcgmsgP.h"

/* size of internal buffer for global ops */
#define DGOP_BUF_SIZE 65536 
#define IGOP_BUF_SIZE 65536 
/*#define IGOP_BUF_SIZE (sizeof(double)/sizeof(long))*DGOP_BUF_SIZE */

static double dgop_work[DGOP_BUF_SIZE]; /**< global ops buffer */
static long   igop_work[IGOP_BUF_SIZE]; /**< global ops buffer */

/**
 * global operations -- integer version 
 */
void IGOP_(long *ptype, long *x, long *pn, char *op, int oplen)
{
    long *work  = (long *) igop_work;
    long nleft  = *pn;
    long buflen = TCG_MIN(nleft,IGOP_BUF_SIZE); /**< Try to get even sized buffers */
    long nbuf   = (nleft-1) / buflen + 1;
    long n;

/* #ifdef ARMCI */
    if(!_tcg_initialized){
        PBEGINF_();
    }
/* #endif */

    buflen = (nleft-1) / nbuf + 1;

    if (strncmp(op,"abs",3) == 0) {
        n = *pn;
        while(n--) {
            x[n] = TCG_ABS(x[n]);
        }
    }

    while (nleft) {
        int ierr = MPI_SUCCESS;
        int ndo = TCG_MIN(nleft, buflen);
        MPI_Op mop;

        if (strncmp(op,"+",1) == 0) {
            mop = MPI_SUM;
        }
        else if (strncmp(op,"*",1) == 0) {
            mop = MPI_PROD;
        }
        else if (strncmp(op,"max",3) == 0 || strncmp(op,"absmax",6) == 0) {
            mop = MPI_MAX;
        }
        else if (strncmp(op,"min",3) == 0 || strncmp(op,"absmin",6) == 0) {
            mop = MPI_MIN;
        }
        else if (strncmp(op,"or",2) == 0) {
            mop = MPI_BOR;
        }
        /* these are new */
        else if ((strncmp(op, "&&", 2) == 0) || (strncmp(op, "land", 4) == 0)) {
            mop = MPI_LAND;
        }
        else if ((strncmp(op, "||", 2) == 0) || (strncmp(op, "lor", 3) == 0)) {
            mop = MPI_LOR;
        }
        else if ((strncmp(op, "&", 1) == 0) || (strncmp(op, "band", 4) == 0)) {
            mop = MPI_BAND;
        }
        else if ((strncmp(op, "|", 1) == 0) || (strncmp(op, "bor", 3) == 0)) {
            mop = MPI_BOR;
        }
        else {
            Error("IGOP: unknown operation requested", (long) *pn);
        }

        ierr = MPI_Allreduce(x, work, ndo, TCG_INT, mop, TCGMSG_Comm);
        tcgmsg_test_statusM("IGOP: MPI_Allreduce:", ierr  );

        n = ndo;
        while(n--) {
            x[n] = work[n];
        }

        nleft -= ndo; x+= ndo;
    }
}



/**
 * global operations -- double version 
 */
void DGOP_(long *ptype, double *x, long *pn, char *op, int oplen)
{
    double *work=  dgop_work;
    long nleft  = *pn;
    long buflen = TCG_MIN(nleft,DGOP_BUF_SIZE); /**< Try to get even sized buffers */
    long nbuf   = (nleft-1) / buflen + 1;
    long n;

    buflen = (nleft-1) / nbuf + 1;

    if (strncmp(op,"abs",3) == 0) {
        n = *pn;
        while(n--) {
            x[n] = TCG_ABS(x[n]);
        }
    }

    while (nleft) {
        int ierr = MPI_SUCCESS;
        int ndo = TCG_MIN(nleft, buflen);
        MPI_Op mop;

        if (strncmp(op,"+",1) == 0) {
            mop = MPI_SUM;
        }
        else if (strncmp(op,"*",1) == 0) {
            mop = MPI_PROD;
        }
        else if (strncmp(op,"max",3) == 0 || strncmp(op,"absmax",6) == 0) {
            mop = MPI_MAX;
        }
        else if (strncmp(op,"min",3) == 0 || strncmp(op,"absmin",6) == 0) {
            mop = MPI_MIN;
        }
        else if (strncmp(op,"or",2) == 0) {
            mop = MPI_BOR;
        }
        /* these are new */
        else if ((strncmp(op, "&&", 2) == 0) || (strncmp(op, "land", 4) == 0)) {
            mop = MPI_LAND;
        }
        else if ((strncmp(op, "||", 2) == 0) || (strncmp(op, "lor", 3) == 0)) {
            mop = MPI_LOR;
        }
        else if ((strncmp(op, "&", 1) == 0) || (strncmp(op, "band", 4) == 0)) {
            mop = MPI_BAND;
        }
        else if ((strncmp(op, "|", 1) == 0) || (strncmp(op, "bor", 3) == 0)) {
            mop = MPI_BOR;
        }
        else {
            Error("DGOP: unknown operation requested", (long) *pn);
        }
        ierr = MPI_Allreduce(x, work, ndo, TCG_DBL, mop, TCGMSG_Comm);
        tcgmsg_test_statusM("DGOP: MPI_Allreduce:", ierr  );

        n = ndo;
        while(n--) {
            x[n] = work[n];
        }

        nleft -= ndo; x+= ndo;
    }
}


/**
 * Synchronize processes
 */
void SYNCH_(long *type)
{
/* #ifdef ARMCI */
    if(!_tcg_initialized){
        PBEGINF_();
    }
/* #endif */
    MPI_Barrier(TCGMSG_Comm);
}



/**
 * broadcast buffer to all other processes from process originator
 */
void BRDCST_(long *type, void *buf, long *lenbuf, long *originator)
{
    /*  hope that MPI int is large enough to store value in lenbuf */
    int count = (int)*lenbuf, root = (int)*originator;

    MPI_Bcast(buf, count, MPI_CHAR, root, TCGMSG_Comm);
}
