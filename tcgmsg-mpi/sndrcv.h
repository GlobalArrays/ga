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
   extern long FATR NODEID_();
   extern long FATR NNODES_();
   extern long FATR MTIME_();
   extern double FATR TCGTIME_();
   extern void FATR SND_();
   extern void FATR RCV_();
   extern long FATR PROBE_();
   extern void FATR BRDCST_();
   extern void FATR DGOP_();
   extern void FATR IGOP_();
   extern void FATR PBEGINF_();
   extern void PBEGIN_();
   extern void FATR PEND_();
   extern void FATR SETDBG_();
   extern long FATR NXTVAL_();
   extern void FATR LLOG_();
   extern void FATR SYNCH_();
   extern void FATR STATS_();
   extern void FATR WAITCOM_();
   extern void Error();
   extern void FATR PARERR_();
   extern double FATR DRAND48_();
   extern void FATR SRAND48_();
   extern long TCGREADY_();
   extern long FATR MDTOB_();
   extern long FATR MITOB_();
   extern long FATR MDTOI_();
   extern long FATR MITOD_();
   extern void FATR PFILECOPY_();
   extern void FATR PFCOPY_();
}
#endif

/*
  long NODEID_() returns logical node no. of current process.
  This is 0,1,...,NNODES_()-1
*/
extern long FATR NODEID_();

/*
  long NNODES_() returns total no. of processes
*/
extern long FATR NNODES_();

/*
  MTIME_() return wall clock time in centiseconds
*/
extern long FATR MTIME_();

/*
  TCGTIME_() returns wall clock time in seconds as accurately as possible
*/
extern double FATR TCGTIME_();

/*
  void SND_(long *type, char *buf, long *lenbuf, long *node, long *sync)
  send message of type *type, from address buf, length *lenbuf bytes
  to node *node.
  Specify *sync as 1 for (mostly) synchronous, 0 for asynchronous.
*/
extern void FATR SND_();
  
/*
  void RCV_(long *type, char *buf, long *lenbuf, long *lenmes, 
  long *nodeselect, long * nodefrom, long *sync)
  receive a message of type *type in buffer buf, size of buf is *lenbuf
  bytes. The actual length returned in lenmes, the sending node in 
  nodefrom. If *nodeselect is a positve integer a message is sought
  from that node. If it is -1 the next pending message is read.
  Specify sync as 1 for synchronous, 0 for asynchronous.
*/
extern void FATR RCV_();

/*
  long PROBE_(long *type, long *node)
  Return TRUE/FALSE (1/0) if a message of the specified type is
  available from the specified node (-1 == any node)
*/
extern long FATR PROBE_();

/*
  void BRDCST_(long *type, char *buf, long *lenbuf, long *originator)
  Broadcast to all other nodes the contents of buf, length *lenbuf
  bytes, type *type, with the info originating from node *originator.
*/
extern void FATR BRDCST_();

/*
  void DGOP_(long *type, double *x, long *n, char *op)
  void IGOP_(long *type,   long *x, long *n, char *op)
  Apply commutative global operation to elements of x on an element
  by element basis.
*/
extern void FATR DGOP_();
extern void FATR IGOP_();

/*
  void PBEGINF_()
  This interfaces FORTRAN to the C routine pbegin. This is the first
  thing that should be called on entering the FORTRAN main program.
  The last thing done is to call PEND_()
*/
extern void FATR PBEGINF_();

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
extern void FATR PEND_();

/*
  void SETDBG_(long *value)
  set internal debug flag on this process to value (TRUE or FALSE)
*/
extern void FATR SETDBG_();

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
extern long FATR NXTVAL_();

/*
  void LLOG_() reopens stdin and stderr as log.<process no.>
*/
extern void FATR LLOG_();

/*
  void SYNCH_(long *type)
  Synchronize processes with messages of given type.
*/
extern void FATR SYNCH_();

/*
  void STATS_() print out communication statitics for the current process
*/
extern void FATR STATS_();

/*
  void WAITCOM_(long *node)
  Wait for completion of all asynchronous send/recieve to node *node
*/
extern void FATR WAITCOM_();

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
extern void FATR PARERR_();

/*
  double DRAND48_()  returns double precision random no. in [0.0,1.0]
  void SRAND48_(long *seed) sets seed of DRAND48 ... seed a positive integer
*/
extern double FATR DRAND48_();
extern void FATR SRAND48_();

/*
 TCGREADY tells if TCGMSG was already initialized (1) or not (0)
*/
extern long TCGREADY_();

/*
  long MDTOB_(long *n) returns no. of bytes that *n double occupy
  long MITOB_(long *n) returns no. of bytes that *n Ints occupy
  long MDTOI_(long *n) returns minimum no. of longs that can hold n doubles
  long MITOD_(long *n) returns minimum no. of doubles that can hold b Ints
*/
extern long FATR MDTOB_();
extern long FATR MITOB_();
extern long FATR MDTOI_();
extern long FATR MITOD_();

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
extern void FATR PFILECOPY_();
extern void FATR PFCOPY_();

#endif
