#if HAVE_CONFIG_H
#   include "config.h"
#endif

/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/globalop.c,v 1.8 2004-04-01 02:04:56 manoj Exp $ */
#include <stdlib.h>
#include <string.h>
#include "sndrcv.h"
#include "msgtypesc.h"

#define TCG_MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define TCG_MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define TCG_ABS(a) (((a) >= 0) ? (a) : (-(a)))

extern void free();

#include "sndrcvP.h"

#define GOP_BUF_SIZE 81920

/*\ reduce operation for int
\*/
static void idoop(long n, char * op, long * x, long * work)
{
  if (strncmp(op,"+",1) == 0) {
    while(n--) {
      *x++ += *work++;
    }
  }
  else if (strncmp(op,"*",1) == 0) {
    while(n--) {
      *x++ *= *work++;
    }
  }
  else if (strncmp(op,"max",3) == 0) {
    while(n--) {
      *x = TCG_MAX(*x, *work);
      x++; work++;
    }
  }
  else if (strncmp(op,"min",3) == 0) {
    while(n--) {
      *x = TCG_MIN(*x, *work);
      x++; work++;
    }
  }
  else if (strncmp(op,"absmax",6) == 0) {
    while(n--) {
      register long x1 = TCG_ABS(*x), x2 = TCG_ABS(*work);
      *x = TCG_MAX(x1, x2);
      x++; work++;
    }
  }
  else if (strncmp(op,"absmin",6) == 0) {
    while(n--) {
      register long x1 = TCG_ABS(*x), x2 = TCG_ABS(*work);
      *x = TCG_MIN(x1, x2);
      x++; work++;
    }
  }
  else if (strncmp(op,"or",2) == 0) {
    while(n--) {
      *x |= *work;
      x++; work++;
    }
  }
  /* these are new */
  else if ((strncmp(op, "&&", 2) == 0) || (strncmp(op, "land", 4) == 0)) {
    while(n--) {
      *x = *x && *work;
      x++; work++;
    }
  }
  else if ((strncmp(op, "||", 2) == 0) || (strncmp(op, "lor", 3) == 0)) {
    while(n--) {
      *x = *x || *work;
      x++; work++;
    }
  }
  else if ((strncmp(op, "&", 1) == 0) || (strncmp(op, "band", 4) == 0)) {
    while(n--) {
      *x &= *work;
      x++; work++;
    }
  }
  else if ((strncmp(op, "|", 1) == 0) || (strncmp(op, "bor", 3) == 0)) {
    while(n--) {
      *x |= *work;
      x++; work++;
    }
  }
  else {
    Error("idoop: unknown operation requested", n);
  }
}

static void ddoop(long n, char * op, double * x, double * work)
{
  if (strncmp(op,"+",1) == 0) {
    while(n--) {
      *x++ += *work++;
    }
  }
  else if (strncmp(op,"*",1) == 0) {
    while(n--) {
      *x++ *= *work++;
    }
  }
  else if (strncmp(op,"max",3) == 0) {
    while(n--) {
      *x = TCG_MAX(*x, *work);
      x++; work++;
    }
  }
  else if (strncmp(op,"min",3) == 0) {
    while(n--) {
      *x = TCG_MIN(*x, *work);
      x++; work++;
    }
  }
  else if (strncmp(op,"absmax",6) == 0) {
    while(n--) {
      register double x1 = TCG_ABS(*x), x2 = TCG_ABS(*work);
      *x = TCG_MAX(x1, x2);
      x++; work++;
    }
  }
  else if (strncmp(op,"absmin",6) == 0) {
    while(n--) {
      register double x1 = TCG_ABS(*x), x2 = TCG_ABS(*work);
      *x = TCG_MIN(x1, x2);
      x++; work++;
    }
  }
  else {
    Error("ddoop: unknown operation requested", (long) n);
  }
}

/*
  Global summation optimized for networks of clusters of processes.

  This routine is directly callable from C only.  There is a
  wrapper that makes fortran work (see bottom of this file).
*/
void DGOP_(long * ptype, double * x, long * pn, char * op, int len)
{
  long me = NODEID_();
  long master = SR_clus_info[SR_clus_id].masterid;
  long nslave = SR_clus_info[SR_clus_id].nslave;
  long slaveid = me - master;
  long synch = 1;
  long type = (*ptype & MSGDBL) ? *ptype : *ptype + MSGDBL;
  long nleft = *pn;
  long buflen = TCG_MIN(nleft,GOP_BUF_SIZE); /* Try to get even sized buffers */
  long nbuf   = (nleft-1) / buflen + 1;
  long zero = 0;
  double *tmp = x;
  double *work;
  long nb, ndo, lenmes, from, up, left, right;

  buflen = (nleft-1) / nbuf + 1;
  if (!(work = (double *) malloc((unsigned) (buflen*sizeof(double)))))
     Error("DGOP: failed to malloc workspace", nleft);

  /* This loop for pipelining and to avoid caller
     having to provide workspace */

  while (nleft) {
    ndo = TCG_MIN(nleft, buflen);
    nb  = ndo * sizeof(double);

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
  nb = *pn * sizeof(double);
  BRDCST_(&type, (char *) tmp, &nb, &zero);
}

/*
  Global summation optimized for networks of clusters of processes.

  This routine is directly callable from C only.  There is a
  wrapper that makes fortran work (see the bottom of this file).
*/
void IGOP_(long * ptype, long * x, long * pn, char * op, int len)
{
  long me = NODEID_();
  long master = SR_clus_info[SR_clus_id].masterid;
  long nslave = SR_clus_info[SR_clus_id].nslave;
  long slaveid = me - master;
  long synch = 1;
  long type = (*ptype & MSGINT) ? *ptype : *ptype + MSGINT;
  long nleft = *pn;
  long zero = 0;
  long *tmp = x;
  long *work;
  long nb, ndo, lenmes, from, up, left, right;

  if (!(work = (long *) 
	malloc((unsigned) (TCG_MIN(nleft,GOP_BUF_SIZE)*sizeof(long)))))
     Error("IGOP: failed to malloc workspace", nleft);

  /* This loop for pipelining and to avoid caller
     having to provide workspace */

  while (nleft) {
    ndo = TCG_MIN(nleft, GOP_BUF_SIZE);
    nb  = ndo * sizeof(long);
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
  nb = *pn * sizeof(long);
  BRDCST_(&type, (char *) tmp, &nb, &zero);
}
