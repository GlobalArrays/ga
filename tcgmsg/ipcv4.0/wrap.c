/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/wrap.c,v 1.5 2002-02-14 20:59:07 edo Exp $ */
#include <stdlib.h>
#include "sndrcv.h"
#include "sndrcvP.h"
#include "typesf2c.h"
#include "msgtypesc.h"

#define GOP_BUF_SIZE 81920

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


void wrap_snd(wrap_type, buf, wrap_lenbuf, wrap_node, wrap_sync)

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
void wrap_rcv(wrap_type, buf, wrap_lenbuf, wrap_lenmes, wrap_nodeselect, wrap_nodefrom, wrap_sync)

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
     nodeselect= (long) *wrap_nodeselect;
     nodefrom= (long) *wrap_nodefrom;
     sync= (long) *wrap_sync;
     
     (void)  RCV_(&type, buf, &lenbuf, &lenmes, &nodeselect, &nodefrom, &sync);
     lenmes= (Integer) *wrap_lenmes;
     return;
     }

Integer wrap_probe(wrap_type, wrap_node)
     Integer *wrap_type, *wrap_node;
{
  long type, node;
     type=  (long) *wrap_type;
     node=  (long) *wrap_node;

  return (Integer) PROBE_(type, node);
}

Integer wrap_mitod(wrap_n)
     Integer *wrap_n;
{
  long n=  (long) *wrap_n;

  return (Integer) MITOD_(&n);
}

Integer wrap_mdtob(wrap_n)
     Integer *wrap_n;
{
  long n=  (long) *wrap_n;

  return (Integer) MDTOB_(&n);
}

Integer wrap_mitob(wrap_n)
     Integer *wrap_n;
{
  long n=  (long) *wrap_n;

  return (Integer) MITOB_(&n);
}

Integer wrap_mdtoi(wrap_n)
     Integer *wrap_n;
{
  long n=  (long) *wrap_n;

  return (Integer) MDTOI_(&n);
}

void wrap_brdcst(wrap_type, buf, wrap_lenbuf, wrap_originator)
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
void wrap_synch( wrap_type)
     Integer *wrap_type;
{
  long type;
  type=  (long) *wrap_type;
  (void )SYNCH_(&type);
  return;
}
void wrap_setdbg( wrap_value)
     Integer *wrap_value;
{
  long value=  (long) *wrap_value;
  (void )SETDBG_(&value);
  return;
}
void wrap_parerr( wrap_code)
     Integer *wrap_code;
{
  long code=  (long) *wrap_code;
  (void )PARERR_(&code);
  return;
}
void wrap_waitcom(wrap_node)
     Integer *wrap_node;
{
  long node=  (long) *wrap_node;
  (void )WAITCOM_(&node);
  return;
}

Integer wrap_mtime()
{

  return (Integer) MTIME_();
}
Integer wrap_nodeid()
{

  return (Integer) NODEID_();
}
Integer wrap_nnodes()
{

  return (Integer) NNODES_();
}
Integer wrap_nxtval( wrap_mproc)
     Integer *wrap_mproc;
{
  long mproc;
  mproc = (long) *wrap_mproc;

  return (Integer)  NXTVAL_(&mproc);
}

void dgop_(wrap_ptype, x, wrap_pn, op, len_op)
     Integer *wrap_ptype, *wrap_pn;
     double *x;
     char *op;
     int len_op;
{
     long ptype, pn;
     ptype = (long) *wrap_ptype;
     pn = (long) *wrap_pn;
  (void) DGOP_(&ptype, x, &pn, op);
}

void igop_(ptype, x, pn, op)
     Integer *x;
     Integer *ptype, *pn;
     char *op;
/*
  Global summation optimized for networks of clusters of processes.

  This routine is directly callable from C only.  There is a
  wrapper that makes fortran work (see the bottom of this file).
*/
{
  long me = NODEID_();
  long master = SR_clus_info[SR_clus_id].masterid;
  long nslave = SR_clus_info[SR_clus_id].nslave;
  long slaveid = me - master;
  long synch = 1;
  long type = (*ptype & MSGINT) ? *ptype : *ptype + MSGINT;
  long nleft = *pn;
  long zero = 0;
  Integer *tmp = x;
  Integer *work;
  long nb, ndo, lenmes, from, up, left, right;
  if (!(work = (Integer *) 
	malloc((unsigned) (MIN(nleft,GOP_BUF_SIZE)*sizeof(Integer)))))
     Error("IGOP: failed to malloc workspace", nleft);

  /* This loop for pipelining and to avoid caller
     having to provide workspace */

  while (nleft) {
    ndo = MIN(nleft, GOP_BUF_SIZE);
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
    if (me != master)
      SND_(&type, (char *) x, &nb, &up, &synch);

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
}

void wrap_pfcopy(wrap_type, wrap_node0, fname, len)
  Integer *wrap_type;
  Integer *wrap_node0;
  char *fname;
  int   len;
{
  long type = (long) *wrap_type;
  long node0 = (long) *wrap_node0;
  (void) PFCOPY_(&type, &node0, fname, len);
}
