/* $Header: /tmp/hpctools/ga/tcgmsg-mpi/wrap.c,v 1.4 2002-01-30 01:15:36 d3h325 Exp $ */
#include <stdlib.h>
#include <mpi.h>
#ifdef CRAY
#include <fortran.h>
#endif
#include "sndrcv.h"
#include "tcgmsgP.h"
#include "typesf2c.h"
#include "msgtypesc.h"

#define DGOP_BUF_SIZE 65536 
#define IGOP_BUF_SIZE (sizeof(double)/sizeof(long))*DGOP_BUF_SIZE 
static double gop_work[DGOP_BUF_SIZE];              /* global ops buffer */

#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define ABS(a) (((a) >= 0) ? (a) : (-(a)))

static void idoop(n, op, x, work)
     long n;
     char *op;
     Integer *x, *work;
{
  if (strncmp(op,"+",1) == 0)
    while(n--)
      *x++ += *work++;
  else if (strncmp(op,"*",1) == 0)
    while(n--)
      *x++ *= *work++;
  else if (strncmp(op,"max",3) == 0)
    while(n--) {
      *x = MAX(*x, *work);
      x++; work++;
    }
  else if (strncmp(op,"min",3) == 0)
    while(n--) {
      *x = MIN(*x, *work);
      x++; work++;
    }
  else if (strncmp(op,"absmax",6) == 0)
    while(n--) {
      register long x1 = ABS(*x), x2 = ABS(*work);
      *x = MAX(x1, x2);
      x++; work++;
    }
  else if (strncmp(op,"absmin",6) == 0)
    while(n--) {
      register long x1 = ABS(*x), x2 = ABS(*work);
      *x = MIN(x1, x2);
      x++; work++;
    }
  else if (strncmp(op,"or",2) == 0) 
    while(n--) {
      *x |= *work;
      x++; work++;
    }
  else
    Error("idoop: unknown operation requested", (long) n);
}


void FATR wrap_snd(wrap_type, buf, wrap_lenbuf, wrap_node, wrap_sync)

     Integer *wrap_type;
     Integer *wrap_lenbuf;
     Integer *wrap_node;
     char *buf;
     Integer *wrap_sync;
{
     long type;
     long lenbuf;
     long node;
     long sync;

     type=  (long) *wrap_type;
     lenbuf= (long) *wrap_lenbuf;
     node= (long) *wrap_node;
     sync= (long) *wrap_sync;
     
     (void)  SND_(&type, buf, &lenbuf, &node, &sync);
     return;
     }

void FATR wrap_rcv(wrap_type, buf, wrap_lenbuf, wrap_lenmes, wrap_nodeselect, wrap_nodefrom, wrap_sync)

     Integer *wrap_type;
     Integer *wrap_lenbuf;
     Integer *wrap_lenmes;
     Integer *wrap_nodeselect;
     Integer *wrap_nodefrom;
     char *buf;
     Integer *wrap_sync;
{
     long type;
     long lenbuf, lenmes;
     long nodeselect;
     long nodefrom;
     long sync;

     type=  (long) *wrap_type;
     lenbuf= (long) *wrap_lenbuf;
     lenmes= (long) *wrap_lenmes;
     nodeselect= (long) *wrap_nodeselect;
     nodefrom= (long) *wrap_nodefrom;
     sync= (long) *wrap_sync;
     
     (void)  RCV_(&type, buf, &lenbuf, &lenmes, &nodeselect, &nodefrom, &sync);
     return;
     }

Integer FATR wrap_probe(wrap_type, wrap_node)
     Integer *wrap_type, *wrap_node;
{
  long type, node;
     type=  (long) *wrap_type;
     node=  (long) *wrap_node;

  return (Integer) PROBE_(&type, &node);
}

Integer FATR wrap_mitod(wrap_n)
     Integer *wrap_n;
{
  long n=  (long) *wrap_n;

  return (Integer) MITOD_(&n);
}

Integer FATR wrap_mdtob(wrap_n)
     Integer *wrap_n;
{
  long n=  (long) *wrap_n;

  return (Integer) MDTOB_(&n);
}

Integer FATR wrap_mitob(wrap_n)
     Integer *wrap_n;
{
  long n=  (long) *wrap_n;

  return (Integer) MITOB_(&n);
}

Integer FATR wrap_mdtoi(wrap_n)
     Integer *wrap_n;
{
  long n=  (long) *wrap_n;

  return (Integer) MDTOI_(&n);
}

void FATR wrap_brdcst(wrap_type, buf, wrap_lenbuf, wrap_originator)
     Integer *wrap_type;
     char *buf;
     Integer *wrap_lenbuf;
     Integer *wrap_originator;
{
     long type;
     long lenbuf;
     long originator;

     type = (long) *wrap_type;
     lenbuf = (long) *wrap_lenbuf;
     originator =  (long) *wrap_originator;

     (void) BRDCST_(&type, buf, &lenbuf, &originator);
     return;
}
void FATR wrap_synch( wrap_type)
     Integer *wrap_type;
{
  long type;
  type=  (long) *wrap_type;
  (void )SYNCH_(&type);
  return;
}
void FATR wrap_setdbg( wrap_value)
     Integer *wrap_value;
{
  long value=  (long) *wrap_value;
  (void )SETDBG_(&value);
  return;
}
void FATR wrap_parerr( wrap_code)
     Integer *wrap_code;
{
  long code=  (long) *wrap_code;
  (void )Error("User detected error in FORTRAN",code);
  return;
}
void FATR wrap_waitcom(wrap_node)
     Integer *wrap_node;
{
  long node=  (long) *wrap_node;
  (void )WAITCOM_(&node);
  return;
}

Integer FATR wrap_mtime()
{

  return (Integer) MTIME_();
}
Integer FATR wrap_nodeid()
{

  return (Integer) NODEID_();
}
Integer FATR wrap_nnodes()
{

  return (Integer) NNODES_();
}
Integer FATR wrap_nxtval( wrap_mproc)
     Integer *wrap_mproc;
{
  long mproc;
  mproc = (long) *wrap_mproc;

  return (Integer)  NXTVAL_(&mproc);
}

#if defined(CRAY) || defined(WIN32)
void FATR dgop_(wrap_ptype, x, wrap_pn, arg)
     Integer *wrap_ptype, *wrap_pn;
     double *x;
     _fcd arg;
{
  char *op = _fcdtocp(arg);
  int len_op = _fcdlen(arg);
  long ptype, pn;
#else
void FATR dgop_(wrap_ptype, x, wrap_pn, op, len_op)
     Integer *wrap_ptype, *wrap_pn;
     double *x;
     char *op;
     int len_op;
{
     long ptype, pn;
#endif
     ptype = (long) *wrap_ptype;
     pn = (long) *wrap_pn;
     (void) DGOP_(&ptype, x, &pn, op);
}
#if defined(CRAY) || defined(WIN32)
void FATR igop_(ptype, x, pn, arg)
     Integer *x;
     Integer *ptype, *pn;
     _fcd arg;
{
  char *op = _fcdtocp(arg);
  int len_op = _fcdlen(arg);
#else
void FATR igop_(ptype, x, pn, op)
     Integer *x;
     Integer *ptype, *pn;
     char *op;
{
#endif
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
      Error("IGOP: unknown operation requested", (long) *pn);
    tcgmsg_test_statusM("IGOP: MPI_Reduce:", ierr  );

    ierr   = MPI_Bcast(work, ndo, TCG_INT, root, TCGMSG_Comm);
    tcgmsg_test_statusM("IGOP: MPI_Bcast:", ierr  );

    n = ndo;
    while(n--) x[n] = work[n];

    nleft -= ndo; x+= ndo;
  }
}

void FATR wrap_pfcopy(wrap_type, wrap_node0, fname, len)
  Integer *wrap_type;
  Integer *wrap_node0;
  char *fname;
  int   len;
{
  long type = (long) *wrap_type;
  long node0 = (long) *wrap_node0;
  (void) PFCOPY_(&type, &node0, fname, len);
}
