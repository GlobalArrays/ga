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

/*
  Int NODEID_() returns logical node no. of current process.
  This is 0,1,...,NNODES_()-1
*/
extern Int NODEID_();

/*
  Int NNODES_() returns total no. of processes
*/
extern Int NNODES_();

/*
  MTIME_() return wall clock time in centiseconds
*/
extern Int MTIME_();

/*
  TCGTIME_() returns wall clock time in seconds as accurately as possible
*/
extern Double TCGTIME_();

/*
  void SND_(Int *type, char *buf, Int *lenbuf, Int *node, Int *sync)
  send message of type *type, from address buf, length *lenbuf bytes
  to node *node.
  Specify *sync as 1 for (mostly) synchronous, 0 for asynchronous.
*/
extern void SND_();
  
/*
  void RCV_(Int *type, char *buf, Int *lenbuf, Int *lenmes, 
       Int *nodeselect, Int * nodefrom, Int *sync)
  receive a message of type *type in buffer buf, size of buf is *lenbuf
  bytes. The actual length returned in lenmes, the sending node in 
  nodefrom. If *nodeselect is a positve integer a message is sought
  from that node. If it is -1 the next pending message is read.
  Specify sync as 1 for synchronous, 0 for asynchronous.
*/
extern void RCV_();

/*
  Int PROBE_(Int *type, Int *node)
  Return TRUE/FALSE (1/0) if a message of the specified type is
  available from the specified node (-1 == any node)
*/
extern Int PROBE_();

/*
  void BRDCST_(Int *type, char *buf, Int *lenbuf, Int *originator)
  Broadcast to all other nodes the contents of buf, length *lenbuf
  bytes, type *type, with the info originating from node *originator.
*/
extern void BRDCST_();

/*
  void DGOP_(Int *type, Double *x, Int *n, char *op)
  void IGOP_(Int *type,   Int *x, Int *n, char *op)
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
  void SETDBG_(Int *value)
  set internal debug flag on this process to value (TRUE or FALSE)
*/
extern void SETDBG_();

/*
  Int NXTVAL_(Int *mproc)
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
extern Int NXTVAL_();

/*
  void LLOG_() reopens stdin and stderr as log.<process no.>
*/
extern void LLOG_();

/*
  void SYNCH_(Int *type)
  Synchronize processes with messages of given type.
*/
extern void SYNCH_();

/*
  void STATS_() print out communication statitics for the current process
*/
extern void STATS_();

/*
  void WAITCOM_(Int *node)
  Wait for completion of all asynchronous send/recieve to node *node
*/
extern void WAITCOM_();

/*
  void Error(char *string, Int integer) 
  Prints error message and terminates after cleaning up as much as possible. 
  Called only from C. FORTRAN should call the routine PARERR.
*/
extern void Error();
#define ERROR_ Error

/*
  void PARERR_(Int *code)
  FORTRAN interface to Error which is called as
  Error("User detected error in FORTRAN", *code);
*/
extern void PARERR_();

/*
  Double DRAND48_()  returns Double precision random no. in [0.0,1.0]
  void SRAND48_(Int *seed) sets seed of DRAND48 ... seed a positive integer
*/
extern Double DRAND48_();
extern void SRAND48_();

/*
  Int MDTOB_(Int *n) returns no. of bytes that *n Double occupy
  Int MITOB_(Int *n) returns no. of bytes that *n Ints occupy
  Int MDTOI_(Int *n) returns minimum no. of Ints that can hold n Doubles
  Int MITOD_(Int *n) returns minimum no. of Doubles that can hold b Ints
*/
extern Int MDTOB_();
extern Int MITOB_();
extern Int MDTOI_();
extern Int MITOD_();

/*
  void PFILECOPY_(Int *type, Int *node0, char *filename)
  ... C interface
  void PFCOPY_(Int *type, Int *node0, FORT CHAR *filename)
  ... FORTRAN interface

  All processes call this simultaneously ... the file (unopened) on
  process node0 is copied to all the other nodes.  Since processes
  may be in the same directory it is recommended that distinct
  filenames are used.
*/
extern void PFILECOPY_();
extern void PFCOPY_();

#endif
