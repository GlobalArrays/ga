/* $Id: collect.c,v 1.13 2002-01-18 19:52:12 vinod Exp $ */
#include "typesf2c.h"
#include "globalp.h"
#include "global.h"
#include "message.h"

/* can handle ga_brdcst/igop/dgop via ARMCI or native message-passing library
 * uncomment line below to use the ARMCI version */
#define  ARMCI_COLLECTIVES 

#if defined(CRAY)
#  include <fortran.h>
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
#ifdef ARMCI_COLLECTIVES
   armci_msg_bcast(buffer, (int)len, (int)root);
#else
#  ifdef MPI
      MPI_Bcast(buffer, (int)len, MPI_CHAR, (int)root, MPI_COMM_WORLD);
#  else
      BRDCST_(&type, buffer, &len, &root);
#  endif
#endif
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
     SYNCH_(&type);
#endif
}


void ga_type_gop(int Type, void *x, int n, char* op)
{
int atype, type=ga_type_f2c(Type);
    switch(type){
    case MT_FLOAT: atype=ARMCI_FLOAT; break;
    case MT_DBL: atype=ARMCI_DOUBLE; break;
    case MT_LONGINT: atype=ARMCI_LONG; break;
    case MT_INT: atype=ARMCI_INT; break;
    default: ga_error("ga_type_gop: type not supported",Type);
    }
    armci_msg_gop_scope(SCOPE_ALL, x, n, op, atype);   
}


void ga_dgop(type, x, n, op)
     Integer type, n;
     DoublePrecision *x;
     char *op;
{
#if defined(ARMCI_COLLECTIVES) || defined(MPI)
            armci_msg_dgop(x, (int)n, op);
#else
            DGOP_(&type, x, &n, op);
#endif
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

void ga_lgop(type, x, n, op)
     Integer type, n;
     long *x;
     char *op;
{
	armci_msg_lgop(x, (int)n, op);
}

void ga_igop(type, x, n, op)
     Integer type, n, *x;
     char *op;
{
#if defined(ARMCI_COLLECTIVES) || defined(MPI)
#   ifdef EXT_INT
            armci_msg_lgop(x, (int)n, op);
#   else
            armci_msg_igop(x, (int)n, op);
#   endif
#else
            IGOP_(&type, x, &n, op);
#endif
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


void ga_fgop(type, x, n, op)
     Integer type, n;
     float *x;
     char *op;
{
            armci_msg_fgop(x, (int)n, op);
}
 
 
/*\ GLOBAL OPERATIONS
 *  Fortran
\*/
#if defined(CRAY) || defined(WIN32)
void FATR GA_SGOP(type, x, n, op)
     _fcd op;
#else
void ga_sgop_(type, x, n, op, len)
     char *op;
     int  len;
#endif
     Integer *type, *n;
     float *x;
{
long gtype,gn;
     gtype = (long)*type; gn = (long)*n;
 
#if defined(CRAY) || defined(WIN32)
     ga_fgop(gtype, x, gn, _fcdtocp(op));
#else
     ga_fgop(gtype, x, gn, op);
#endif
}                                   


#if 0
Integer ga_msg_nnodes_()
{     
#ifdef MPI
     int numprocs;
     MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
     return((Integer)numprocs);
#else
     return NNODES_();
#endif
}


Integer ga_msg_nodeid_()
{     
#ifdef MPI
     int myid;

     MPI_Comm_rank(MPI_COMM_WORLD,&myid);
     return((Integer)myid);
#else
     return NODEID_();
#endif
}
#endif
