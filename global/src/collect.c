#include "typesf2c.h"
#include "globalp.h"

#if defined(CRAY)
#  include <fortran.h>
#endif

/*
#if defined(CRAY) || defined(WIN32)
#  define igop_ IGOP 
#  define dgop_ DGOP 
#  define synch_ SYNCH 
#  define brdcst_  BRDCST
#  define ga_brdcst_ GA_BRDCST
#  define nodeid_ NODEID
#  define nnodes_ NNODES
#endif

void FATR dgop_();
void FATR igop_();
*/

#ifndef TCGMSG
#   define MPI 
#endif

#ifdef MPI
#  include <mpi.h>
#else
#  include "sndrcv.h"
#endif

void ga_msg_brdcst(type, buffer, len, root)
Integer type, len, root;
Void*   buffer;
{
#  ifdef MPI
      MPI_Bcast(buffer, (int)len, MPI_CHAR, (int)root, MPI_COMM_WORLD);
#  else
      brdcst_(&type, buffer, &len, &root);
#  endif
}

/*\ BROADCAST
\*/
void FATR ga_brdcst_(type, buf, len, originator)
     Integer *type, *len, *originator;
     Void *buf;
{
    ga_msg_brdcst(*type,buf,*len,*originator);
}

#ifdef MPI
void ga_mpi_communicator(GA_COMM)
MPI_Comm *GA_COMM;
{
       *GA_COMM = MPI_COMM_WORLD;
}
#endif


void ga_msg_sync_()
{
#ifdef MPI
     MPI_Barrier(MPI_COMM_WORLD);
#else
     long type=GA_TYPE_SYN;
     synch_(&type);
#endif
}

void ga_dgop(type, x, n, op)
     Integer type, n;
     DoublePrecision *x;
     char *op;
{
dgop_(&type, x, &n, op);
}


/*\ GLOBAL OPERATIONS
 *  Fortran
\*/
#if defined(CRAY) || defined(WIN32)
void FATR GA_DGOP(type, x, n, op)
     _fcd op;
#else
void ga_dgop_(type, x, n, op, len)
     char *op;
     int len;
#endif
     Integer *type, *n;
     DoublePrecision *x;
{
long gtype,gn;
     gtype = (long)*type; gn = (long)*n;

#if defined(CRAY) || defined(WIN32)
     ga_dgop(gtype, x, gn, _fcdtocp(op));
#else
     ga_dgop(gtype, x, gn, op);
#endif
}


void ga_igop(type, x, n, op)
     Integer type, n, *x;
     char *op;
{
            igop_(&type, x, &n, op);
}


/*\ GLOBAL OPERATIONS
 *  Fortran
\*/
#if defined(CRAY) || defined(WIN32)
void FATR GA_IGOP(type, x, n, op)
     _fcd op;
#else
void ga_igop_(type, x, n, op, len)
     char *op;
     int  len;
#endif
     Integer *type, *n;
     Integer *x;
{
long gtype,gn;
     gtype = (long)*type; gn = (long)*n;

#if defined(CRAY) || defined(WIN32)
     ga_igop(gtype, x, gn, _fcdtocp(op));
#else
     ga_igop(gtype, x, gn, op);
#endif
}


Integer ga_msg_nnodes_()
{     
#ifdef MPI
     int numprocs;
     MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
     return((Integer)numprocs);
#else
     return nnodes_();
#endif
}


Integer ga_msg_nodeid_()
{     
#ifdef MPI
     int myid;

     MPI_Comm_rank(MPI_COMM_WORLD,&myid);
     return((Integer)myid);
#else
     return nodeid_();
#endif
}



