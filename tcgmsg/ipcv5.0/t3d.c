/*\
 *       TCGMSG INTERFACE FOR THE CRAY T3D/E
 *           Jarek Nieplocha, 10.12.1994
\*/

#include <stdio.h>
#include <stdlib.h>
#include <mpp/shmem.h>
#include "srftoc.h"

#ifdef GA_USE_VAMPIR
#include "tcgmsg_vampir.h"
#endif

#ifdef EVENTLOG
#include "evlog.h"
#endif

#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define ABS(a) (((a) >= 0) ? (a) : (-(a)))
#define MEMCPY(dst, src, n) memcpy(dst, src, (size_t) n)

#define INCR 1                 /* increment for NXTVAL */
#define BUSY -1L               /* indicates somebody else updating counter*/


/* Global variables */

long pSync[_SHMEM_BCAST_SYNC_SIZE];   /* workspace needed for shmem routines */
long rSync[_SHMEM_REDUCE_SYNC_SIZE];  /* workspace needed for shmem routines */
#pragma _CRI cache_align pSync,rSync

static volatile long n_in_msg_q = 0;    /* No. in the message q */

#define MAX_Q_LEN 2048         /* Maximum no. of outstanding messages */
static struct msg_q_struct{
  long   msg_id;
  long   node;
  long   type;
  long   lenbuf;
  long   snd;
  long   from;
} msg_q[MAX_Q_LEN];


static long me, procs;
long nxtval_counter=0;
extern long DEBUG_;


/***********************************************************/


/*\ Return number of the calling process ...
\*/
long NODEID_()
{
  return ((long)_my_pe());
}



/*\ Return number of USER tasks/processes
\*/
long NNODES_()
{
  return ((long)_num_pes());
}



/*\ Error handler
\*/
void Error(string, code)
     char *string;
     long code;
{

  (void) fflush(stdout);

  (void) fprintf(stdout, "%3d:%s %ld(%x)\n", NODEID_(), string, code, code);
  (void) fflush(stdout);
  (void) fprintf(stderr, "%3d:%s %ld(%x)\n", NODEID_(), string, code, code);
  (void) perror("system message");

  (void) fflush(stdout);
  (void) fflush(stderr);

  globalexit(1);
}


/*\ Synchronize processes
\*/
void SYNCH_(type)
     long *type;
{
#ifdef GA_USE_VAMPIR
     vampir_begin(TCGMSG_SYNCH,__FILE__,__LINE__);
#endif
     barrier();
#ifdef GA_USE_VAMPIR
     vampir_end(TCGMSG_SYNCH,__FILE__,__LINE__);
#endif
}


/*\ initialize gops work array
\*/
void t3d_gops_init()
{
  int node;

  for(node=0;node<_SHMEM_BCAST_SYNC_SIZE;node++)pSync[node] = _SHMEM_SYNC_VALUE;
  for(node=0;node<_SHMEM_REDUCE_SYNC_SIZE;node++)rSync[node]= _SHMEM_SYNC_VALUE;
}


void PBEGINF_()
{
  PBEGIN_();
}



/*\ Define value of debug flag
\*/
void SETDBG_(onoff)
     long *onoff;
{
  DEBUG_ = *onoff;
}



/*\ Get next value of shared counter.
\*/
long NXTVAL_(mproc)
     long *mproc;
/*
  mproc > 0 ... returns requested value
  mproc < 0 ... server blocks until abs(mproc) processes are queued
                and returns junk
  mproc = 0 ... indicates to server that I am about to terminate
*/
{
  int server = MAX(0, (int) procs - 1);
  long local; 

#ifdef GA_USE_VAMPIR
  vampir_begin(TCGMSG_NXTVAL,__FILE__,__LINE__);
#endif

  me = NODEID_();
  procs = NNODES_();

  if (*mproc == 0)

    Error("NVS: invalid mproc ", *mproc);

  else if (*mproc > 0) {

     /* use atomic swap operation to increment nxtval counter */

     while((local = shmem_swap(&nxtval_counter, BUSY, server)) == BUSY);
     shmem_swap(&nxtval_counter, (local+INCR), server);

  } else if (*mproc < 0) {
     
    barrier();
 
    /* reset the counter value to zero */

    if( NODEID_() == server){

        nxtval_counter = 0;

    }
    barrier();
  }
  if (DEBUG_) {
        printf("NVS: from=%d  mproc=%d value=%d \n", me, *mproc, local );
        (void) fflush(stdout);
  }

#ifdef GA_USE_VAMPIR
  vampir_end(TCGMSG_NXTVAL,__FILE__,__LINE__);
#endif
  return(local);
}




void PBFTOC_()
{
  Error("PBFTOC_: what the hell are we doing here?",(long) -1);
}


/*\ Handle request for application error termination
\*/
void PARERR_(code)
  long *code;
{
  Error("FORTRAN error detected", *code);
}


/*\ Print out statistics for communications ... not yet implemented
\*/
void STATS_()
{
  (void) fprintf(stderr,"STATS_ not yet supported\n");
  (void) fflush(stderr);
}


/* global operation stuff */
#define GPSZ 10000
#define GOP_WORK_SIZE (MAX(GPSZ, _SHMEM_REDUCE_MIN_WRKDATA_SIZE))
#define GOP_BUF_SIZE  (3*GOP_WORK_SIZE)

/* gop_work buffer is partitioned into 3 even chunks in IGOP/DGOP */
/* BRDCST uses full buffer space                                  */
static double gop_work[GOP_BUF_SIZE];
static double *gop_target_buf=gop_work+GOP_WORK_SIZE;
static double *gop_source_buf=gop_work+2*GOP_WORK_SIZE;



void BRDCST_(type, x, bytes, originator)
     long *type;
     char *x;
     long *bytes;
     long *originator;
/*
 * Broadcast buffer to all other processes from process originator:
 * all processes call this routine specifying the same orginating process.
 */
{
  long buflen = GOP_BUF_SIZE;
  int words   = (*bytes+sizeof(long)-1)/sizeof(long);
  long nleft  = words;
  long nbuf   = (nleft-1) / buflen + 1;
  long bytes_to_copy;
  char *start = x;
  long *brdcst_buf = (long*)gop_work;

#ifdef GA_USE_VAMPIR
  vampir_begin(TCGMSG_BRDCST,__FILE__,__LINE__);
#endif

  me = NODEID_();
  procs = NNODES_();

  if (DEBUG_) {
        printf("BRDCST: me=%2ld  originator=%2ld  %d bytes\n", me, *originator,
               *bytes );
        (void) fflush(stdout);
  }


  buflen = (nleft-1) / nbuf + 1;

  if(((long)x)%sizeof(long))
     Error("Broadcast: buffer must be alligned on 8-byte boundary: ",(long) x);


  while (nleft) {
    long ndo = MIN(nleft, buflen);

    barrier(); /* synchronize to make sure that nobody is accessing pSync */
    shmem_broadcast(brdcst_buf, (long*)x, ndo, (int)*originator, 0, 0, 
                   (int)procs, pSync);
    if(me!= *originator){
          bytes_to_copy =  (long)( (start + *bytes >= x + ndo*sizeof(long)) ? 
                                    ndo*sizeof(long) : start + *bytes - x);
          
          MEMCPY(x,brdcst_buf,bytes_to_copy);
    }
    nleft -= ndo; x+= ndo*sizeof(long);
  }
  if (DEBUG_) {
        printf("BRDCST: me=%2ld  done,   long value=%ld \n", me,*(long*)start);
        (void) fflush(stdout);
  }
#ifdef GA_USE_VAMPIR
  vampir_end(TCGMSG_BRDCST,__FILE__,__LINE__);
#endif
}



void DGOP_(ptype, x, pn, op)
     double *x;
     long *ptype, *pn;
     char *op;
{
  double *work = gop_work;
  double *target    = gop_target_buf;
  double *source    = gop_source_buf;
  long nleft  = *pn;
  long buflen = MIN(nleft,GOP_WORK_SIZE); /* Try to get even sized buffers */
  long nbuf   = (nleft-1) / buflen + 1;
  long n;

#ifdef GA_USE_VAMPIR
  vampir_begin(TCGMSG_DGOP,__FILE__,__LINE__);
#endif

  me = NODEID_();
  procs = NNODES_();

  buflen = (nleft-1) / nbuf + 1;

  barrier();

  if (strncmp(op,"abs",3) == 0) {
    n = *pn;
    while(n--) x[n] = ABS(x[n]);
  } 
  
  while (nleft) {
    long ndo = MIN(nleft, buflen);

    MEMCPY(source,x,ndo*sizeof(double)); 
    barrier();

    if (strncmp(op,"+",1) == 0)
      shmem_double_sum_to_all(target, source, ndo, 0, 0, procs, work, rSync);
    else if (strncmp(op,"*",1) == 0)
      shmem_double_prod_to_all(target, source, ndo, 0, 0, procs, work, rSync);
    else if (strncmp(op,"max",3) == 0 || strncmp(op,"absmax",6) == 0)
      shmem_double_max_to_all(target, source, ndo, 0, 0, procs, work, rSync);
    else if (strncmp(op,"min",3) == 0 || strncmp(op,"absmin",6) == 0)
      shmem_double_min_to_all(target, source, ndo, 0, 0, procs, work, rSync);
    else
      Error("DGOP: unknown operation requested", (long) *pn);

    MEMCPY(x,target,ndo*sizeof(double));

    nleft -= ndo; x+= ndo;
  }

#ifdef GA_USE_VAMPIR
  vampir_end(TCGMSG_DGOP,__FILE__,__LINE__);
#endif
}


void IGOP_(ptype, x, pn, op)
     int *x;
     int *ptype, *pn;
     char *op;
{
  int *work  = (int*) gop_work;
  int *target     = (int*) gop_target_buf;
  int *source     = (int*) gop_source_buf;
  int nleft  = *pn;
  int buflen = MIN(nleft,GOP_WORK_SIZE); /* Try to get even sized buffers */
  int nbuf   = (nleft-1) / buflen + 1;
  int n;

#ifdef GA_USE_VAMPIR
  vampir_begin(TCGMSG_IGOP,__FILE__,__LINE__);
#endif

  me = NODEID_();
  procs = NNODES_();

  buflen = (nleft-1) / nbuf + 1;

  barrier();

  if (strncmp(op,"abs",3) == 0) {
    n = *pn;
    while(n--) x[n] = ABS(x[n]);
  } 
  
  while (nleft) {
    int ndo = MIN(nleft, buflen);

    MEMCPY(source,x,ndo*sizeof(int)); 
    barrier();

    if (strncmp(op,"+",1) == 0)
      shmem_int_sum_to_all(target, source, ndo, 0, 0, procs, work, rSync);
    else if (strncmp(op,"*",1) == 0)
      shmem_int_prod_to_all(target, source, ndo, 0, 0, procs, work, rSync);
    else if (strncmp(op,"max",3) == 0 || strncmp(op,"absmax",6) == 0)
      shmem_int_max_to_all(target, source, ndo, 0, 0, procs, work, rSync);
    else if (strncmp(op,"min",3) == 0 || strncmp(op,"absmax",6) == 0)
      shmem_int_min_to_all(target, source, ndo, 0, 0, procs, work, rSync);
    else if (strncmp(op,"or",2) == 0)
      shmem_int_or_to_all(target, source, ndo, 0, 0, procs, work, rSync);
    else
      Error("IGOP: unknown operation requested", (long) *pn);

    MEMCPY(x,target,ndo*sizeof(int));
    nleft -= ndo; x+= ndo;
  }
#ifdef GA_USE_VAMPIR
  vampir_end(TCGMSG_IGOP,__FILE__,__LINE__);
#endif
}


