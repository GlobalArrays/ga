/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/sndrcv.h,v 1.10 2004-04-01 02:04:57 manoj Exp $ */

#include "srftoc.h"


#include "msgtypesc.h"

#ifndef SNDRCV
#define SNDRCV 1
/*
  This header file declares stubs and show prototypes of the 
  public sndrcv calls

  srftoc.h contains macros which define the names of c routines
  accessible from FORTRAN and vice versa
*/

#ifdef __cplusplus
extern "C" {

extern long NODEID_();
extern long NNODES_();
extern long MTIME_();
extern double TCGTIME_();
extern void SND_(long *type, char *buf, long *lenbuf, long *node, long *sync);
extern void RCV_(long *type, char *buf, long *lenbuf, long *lenmes, 
                 long *nodeselect, long * nodefrom, long *sync);

extern long PROBE_(long *type, long *node);
extern void BRDCST_(long *type, char *buf, long *lenbuf, long *originator);
extern void DGOP_(long *type, double *x, long *n, char *op);
extern void IGOP_(long *type,   long *x, long *n, char *op);
extern void PBEGIN_(int argc, char **argv);
extern void PEND_();
extern void Error(char* str, long code);
extern void SETDBG_(long *value);
extern long NXTVAL_(long *mproc);
extern long MDTOB_(long *n);
extern long MITOB_(long *n);
extern long MDTOI_(long *n);
extern long MITOD_(long *n);
extern void PFILECOPY_(long *type, long *node0, char *filename);
extern long TCGREADY_();
extern double DRAND48_();
extern void SRAND48_(long *seed);
extern void LLOG_();
extern void STATS_();
extern void SYNCH_(long *type);
extern void WAITCOM_(long *node);
extern void ALT_PBEGIN_();
}
#else

/*
  long NODEID_() returns logical node no. of current process.
  This is 0,1,...,NNODES_()-1
*/
extern long NODEID_();

/*
  long NNODES_() returns total no. of processes
*/
extern long NNODES_();

/*
  MTIME_() return wall clock time in centiseconds
*/
extern long MTIME_();

/*
  TCGTIME_() returns wall clock time in seconds as accurately as possible
*/
extern double TCGTIME_();

/*
  void SND_(long *type, char *buf, long *lenbuf, long *node, long *sync)
  send message of type *type, from address buf, length *lenbuf bytes
  to node *node.
  Specify *sync as 1 for (mostly) synchronous, 0 for asynchronous.
*/
extern void SND_();
  
/*
  void RCV_(long *type, char *buf, long *lenbuf, long *lenmes, 
       long *nodeselect, long * nodefrom, long *sync)
  receive a message of type *type in buffer buf, size of buf is *lenbuf
  bytes. The actual length returned in lenmes, the sending node in 
  nodefrom. If *nodeselect is a positve integer a message is sought
  from that node. If it is -1 the next pending message is read.
  Specify sync as 1 for synchronous, 0 for asynchronous.
*/
extern void RCV_();

/*
  long PROBE_(long *type, long *node)
  Return TRUE/FALSE (1/0) if a message of the specified type is
  available from the specified node (-1 == any node)
*/
extern long PROBE_();

/*
  void BRDCST_(long *type, char *buf, long *lenbuf, long *originator)
  Broadcast to all other nodes the contents of buf, length *lenbuf
  bytes, type *type, with the info originating from node *originator.
*/
extern void BRDCST_();

/*
  void DGOP_(long *type, double *x, long *n, char *op)
  void IGOP_(long *type,   long *x, long *n, char *op)
  Apply commutative global operation to elements of x on an element
  by element basis.
*/
extern void DGOP_();
extern void IGOP_();

/*
  void PBEGINF_()
  This interfaces FORTRAN to the C routine pbegin. This is the first
  thing that should be called on entering the FORTRAN main program.
  The last thing done is to call PEND_()
*/
extern void PBEGINF_();

/*
  void PBEGIN_(int argc, char **argv)
  Initialize the parallel environment ... argc and argv contain the
  arguments in the usual C fashion. pbegin is only called from C.
  FORTRAN should call pbeginf which has no arguments.
*/
extern void PBEGIN_();

/*
  void PEND_()
  call to tidy up and signal master that have finished
*/
extern void PEND_();

/*
  void SETDBG_(long *value)
  set internal debug flag on this process to value (TRUE or FALSE)
*/
extern void SETDBG_();

/*
  long NXTVAL_(long *mproc)
  This call communicates with a dedicated server process and returns the 
  next counter (value 0, 1, ...) associated with a single active loop.
  mproc is the number of processes actively requesting values. 
  This is used for load balancing.
  It is used as follows:

  mproc = nproc;
  while ( (i=nxtval(&mproc)) < top ) {
    do work for iteration i;
  }
  mproc = -mproc;
  (void) nxtval(&mproc);

  Clearly the value from nxtval() can be used to indicate that some
  locally determined no. of iterations should be done as the overhead
  of nxtval() may be large (approx 0.05-0.5s per call ... so each process
  should do about 5-50s of work per call for a 1% overhead).
*/
extern long NXTVAL_();

/*
  void LLOG_() reopens stdin and stderr as log.<process no.>
*/
extern void LLOG_();

/*
  void SYNCH_(long *type)
  Synchronize processes with messages of given type.
*/
extern void SYNCH_();

/*
  void STATS_() print out communication statitics for the current process
*/
extern void STATS_();

/*
  void WAITCOM_(long *node)
  Wait for completion of all asynchronous send/recieve to node *node
*/
extern void WAITCOM_();
extern void ALT_PBEGIN_();

/*
  void Error(char *string, long integer) 
  Prints error message and terminates after cleaning up as much as possible. 
  Called only from C. FORTRAN should call the routine PARERR.
*/
extern void Error();
#define ERROR_ Error

/*
  void PARERR_(long *code)
  FORTRAN interface to Error which is called as
  Error("User detected error in FORTRAN", *code);
*/
extern void PARERR_();

/*
  double DRAND48_()  returns double precision random no. in [0.0,1.0]
  void SRAND48_(long *seed) sets seed of DRAND48 ... seed a positive integer
*/
extern double DRAND48_();
extern void SRAND48_();

/*
  long MDTOB_(long *n) returns no. of bytes that *n double occupy
  long MITOB_(long *n) returns no. of bytes that *n longs occupy
  long MDTOI_(long *n) returns minimum no. of longs that can hold n doubles
  long MITOD_(long *n) returns minimum no. of doubles that can hold b longs
*/
extern long MDTOB_();
extern long MITOB_();
extern long MDTOI_();
extern long MITOD_();

/*
  void PFILECOPY_(long *type, long *node0, char *filename)
  ... C interface
  void PFCOPY_(long *type, long *node0, FORT CHAR *filename)
  ... FORTRAN interface

  All processes call this simultaneously ... the file (unopened) on
  process node0 is copied to all the other nodes.  Since processes
  may be in the same directory it is recommended that distinct
  filenames are used.
*/
extern void PFILECOPY_();
extern void PFCOPY_();

/*
 TCGREADY tells if TCGMSG was already initialized (1) or not (0) 
*/
extern long TCGREADY_();

/*
  Miscellaneous routines for internal use only?
*/

extern void RemoteConnect();
extern void PrintProcInfo();
extern void MtimeReset();
extern void USleep();

#endif

#endif
