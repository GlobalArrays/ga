/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv5.0/wrap.c,v 1.1 2001-05-08 17:42:12 edo Exp $ */

#include "sndrcv.h"
#include "typesf2c.h"
#include "msgtypesc.h"


#define BUF_SIZE  10000
#define IBUF_SIZE (BUF_SIZE * sizeof(DoublePrecision)/sizeof(Integer)) 
DoublePrecision _gops_work[BUF_SIZE];
long one=1;

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
     lenmes= (long) *wrap_lenmes;
     nodeselect= (long) *wrap_nodeselect;
     nodefrom= (long) *wrap_nodefrom;
     sync= (long) *wrap_sync;
     
     (void)  RCV_(&type, buf, &lenbuf, &lenmes, &nodeselect, &nodefrom, &sync);
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
void wrap_igop(type, x, n, op)
     Integer *type, *n;
     Integer *x;
     char *op;
{
     long me=NODEID_(), nproc=NNODES_(), len, lenmes, from, root=0;
     Integer *work = (Integer*)_gops_work, *origx = x;
     long ndo, up, left, right, np=*n, orign =*n;

     /* determine location in the binary tree */
     up    = (me-1)/2;    if(up >= nproc)       up = -1;
     left  =  2* me + 1;  if(left >= nproc)   left = -1;
     right =  2* me + 2;  if(right >= nproc) right = -1;

     while ((ndo = (np<=IBUF_SIZE) ? np : IBUF_SIZE)) {
	 len = lenmes = ndo*sizeof(Integer);

         if (left > -1) {
           RCV_(type, (char *) work, &len, &lenmes, &left, &from, &one);
           idoop(ndo, op, x, work);
         }
         if (right > -1) {
           RCV_(type, (char *) work, &len, &lenmes, &right, &from, &one);
           idoop(ndo, op, x, work);
         }
         if (me != root) SND_(type, x, &len, &up, &one); 

         np -=ndo;
         x  +=ndo;
     }

     /* Now, root broadcasts the result down the binary tree */
     len = orign*sizeof(Integer);
     BRDCST_(type, (char *) origx, &len, &root);
}
