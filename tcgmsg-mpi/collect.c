#include <mpi.h>
#include "tcgmsgP.h"


/* size of internal buffer for global ops */
#define DGOP_BUF_SIZE 65536 
#define IGOP_BUF_SIZE (sizeof(double)/sizeof(Integer))*DGOP_BUF_SIZE 

static double gop_work[DGOP_BUF_SIZE];              /* global ops buffer */


/*\ global operations -- integer version 
\*/
void FATR IGOP_(ptype, x, pn, op)
     Integer  *x;
     Integer  *ptype, *pn;
     char *op;
{
Integer *work   = (Integer *) gop_work;
long nleft  = *pn;
long buflen = MIN(nleft,IGOP_BUF_SIZE); /* Try to get even sized buffers */
long nbuf   = (nleft-1) / buflen + 1;
long n;

#ifdef ARMCI
     if(!_tcg_initialized){
         TCGMSG_Comm = MPI_COMM_WORLD;
         _tcg_initialized = 1;
     }
#endif

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
      Error("IGOP: unknown operation requested", (Integer) *pn);
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
void FATR DGOP_(ptype, x, pn, op)
     double  *x;
     Integer     *ptype, *pn;
     char    *op;
{
double *work=  gop_work;
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
      Error("DGOP: unknown operation requested", (Integer) *pn);
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
void FATR SYNCH_(type)
     Integer *type;
{
#ifdef ARMCI
     if(!_tcg_initialized){
         TCGMSG_Comm = MPI_COMM_WORLD;
         _tcg_initialized = 1;
     }
#endif
     MPI_Barrier(TCGMSG_Comm);
}



/*\ broadcast buffer to all other processes from process originator
\*/
void FATR BRDCST_(type, buf, lenbuf, originator)
     Integer  *type;
     char *buf;
     Integer  *lenbuf;
     Integer  *originator;
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


#if defined(CRAY) || defined(WIN32)
void FATR dgop_(ptype, x, pn, arg)
     Integer *ptype, *pn;
     double *x;
     _fcd arg;
{
  char *op = _fcdtocp(arg);
  int len_op = _fcdlen(arg);
#else
void FATR dgop_(ptype, x, pn, op, len_op)
     Integer *ptype, *pn;
     double *x;
     char *op;
     int len_op;
{
#endif
  DGOP_(ptype, x, pn, op);
}

#if defined(CRAY) || defined(WIN32)
void FATR igop_(ptype, x, pn, arg)
     Integer *ptype, *pn;
     Integer *x;
     _fcd arg;
{
  char *op = _fcdtocp(arg);
  int len_op = _fcdlen(arg);
#else
void FATR igop_(ptype, x, pn, op, len_op)
     Integer *ptype, *pn;
     Integer *x;
     char *op;
     int len_op;
{
#endif
  IGOP_(ptype, x, pn, op);
}
