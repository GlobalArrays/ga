#include <mpi.h>
#include "tcgmsgP.h"


/* size of internal buffer for global ops */
#define DGOP_BUF_SIZE 65536 
#define IGOP_BUF_SIZE (sizeof(Double)/sizeof(Int))*DGOP_BUF_SIZE 

static Double gop_work[DGOP_BUF_SIZE];              /* global ops buffer */


/*\ global operations -- integer version 
\*/
void IGOP_(ptype, x, pn, op)
     Int  *x;
     Int  *ptype, *pn;
     char *op;
{
Int *work   = (Int *) gop_work;
long nleft  = *pn;
long buflen = MIN(nleft,IGOP_BUF_SIZE); /* Try to get even sized buffers */
long nbuf   = (nleft-1) / buflen + 1;
long n;

  buflen = (nleft-1) / nbuf + 1;

  if (strncmp(op,"abs",3) == 0) {
    n = *pn;
    while(n--) x[n] = ABS(x[n]);
  }

  while (nleft) {
    int root = 0; 
    int ierr  ;
    int ndo = MIN(nleft, buflen);

    if (strncmp(op,"+",1) == 0)
      ierr   = MPI_Reduce(x, work, ndo, TCG_INT, MPI_SUM, root, TCGMSG_Comm);
    else if (strncmp(op,"*",1) == 0)
      ierr   = MPI_Reduce(x, work, ndo, TCG_INT, MPI_PROD, root, TCGMSG_Comm);
    else if (strncmp(op,"max",3) == 0 || strncmp(op,"absmax",6) == 0)
      ierr   = MPI_Reduce(x, work, ndo, TCG_INT, MPI_MAX, root, TCGMSG_Comm);
    else if (strncmp(op,"min",3) == 0 || strncmp(op,"absmin",6) == 0)
      ierr   = MPI_Reduce(x, work, ndo, TCG_INT, MPI_MIN, root, TCGMSG_Comm);
    else if (strncmp(op,"or",2) == 0)
      ierr   = MPI_Reduce(x, work, ndo, TCG_INT, MPI_BOR, root, TCGMSG_Comm);
    else
      Error("IGOP: unknown operation requested", (Int) *pn);
    tcgmsg_test_statusM("IGOP: MPI_Reduce:", ierr  );

    ierr   = MPI_Bcast(work, ndo, TCG_INT, root, TCGMSG_Comm);
    tcgmsg_test_statusM("IGOP: MPI_Bcast:", ierr  );

    n = ndo;
    while(n--) x[n] = work[n];

    nleft -= ndo; x+= ndo;
  }
}



/*\ global operations -- double version 
\*/
void DGOP_(ptype, x, pn, op)
     Double  *x;
     Int     *ptype, *pn;
     char    *op;
{
Double *work=  gop_work;
long nleft  = *pn;
long buflen = MIN(nleft,DGOP_BUF_SIZE); /* Try to get even sized buffers */
long nbuf   = (nleft-1) / buflen + 1;
long n;

  buflen = (nleft-1) / nbuf + 1;

  if (strncmp(op,"abs",3) == 0) {
    n = *pn;
    while(n--) x[n] = ABS(x[n]);
  }

  while (nleft) {
    int root = 0; 
    int ierr  ;
    int ndo = MIN(nleft, buflen);

    if (strncmp(op,"+",1) == 0)
      ierr   = MPI_Reduce(x, work, ndo, TCG_DBL, MPI_SUM, root, TCGMSG_Comm);
    else if (strncmp(op,"*",1) == 0)
      ierr   = MPI_Reduce(x, work, ndo, TCG_DBL, MPI_PROD, root, TCGMSG_Comm);
    else if (strncmp(op,"max",3) == 0 || strncmp(op,"absmax",6) == 0)
      ierr   = MPI_Reduce(x, work, ndo, TCG_DBL, MPI_MAX, root, TCGMSG_Comm);
    else if (strncmp(op,"min",3) == 0 || strncmp(op,"absmin",6) == 0)
      ierr   = MPI_Reduce(x, work, ndo, TCG_DBL, MPI_MIN, root, TCGMSG_Comm);
    else
      Error("DGOP: unknown operation requested", (Int) *pn);
    tcgmsg_test_statusM("DGOP: MPI_Reduce:", ierr  );

    ierr   = MPI_Bcast(work, ndo, TCG_DBL, root, TCGMSG_Comm);
    tcgmsg_test_statusM("DGOP: MPI_Bcast:", ierr  );

    n = ndo;
    while(n--) x[n] = work[n];

    nleft -= ndo; x+= ndo;
  }
}


/*\ Synchronize processes
\*/
void SYNCH_(type)
     Int *type;
{
     MPI_Barrier(TCGMSG_Comm);
}



/*\ broadcast buffer to all other processes from process originator
\*/
void BRDCST_(type, buf, lenbuf, originator)
     Int  *type;
     char *buf;
     Int  *lenbuf;
     Int  *originator;
{
/*  hope that MPI int is large enough to store value in lenbuf */
int count = (int)*lenbuf, root = (int)*originator;

     MPI_Bcast(buf, count, MPI_CHAR, root, TCGMSG_Comm);
}


/* Wrapper for fortran interface ... UGH ... note that
   string comparisons above do NOT rely on NULL termination
   of the operation character string */

#ifdef CRAY
#  include <fortran.h>
#endif

/* This crap to handle FORTRAN character strings */


#if defined(CRAY)
void dgop_(ptype, x, pn, arg)
     Int *ptype, *pn;
     Double *x;
     _fcd arg;
{
  char *op = _fcdtocp(arg);
  int len_op = _fcdlen(arg);
#else
void dgop_(ptype, x, pn, op, len_op)
     Int *ptype, *pn;
     Double *x;
     char *op;
     int len_op;
{
#endif
  DGOP_(ptype, x, pn, op);
}

#if defined(CRAY)
void igop_(ptype, x, pn, arg)
     Int *ptype, *pn;
     Int *x;
     _fcd arg;
{
  char *op = _fcdtocp(arg);
  int len_op = _fcdlen(arg);
#else
void igop_(ptype, x, pn, op, len_op)
     Int *ptype, *pn;
     Int *x;
     char *op;
     int len_op;
{
#endif
  IGOP_(ptype, x, pn, op);
}
