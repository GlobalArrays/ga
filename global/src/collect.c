/* $Id: collect.c,v 1.18 2004-06-28 17:47:53 manoj Exp $ */
#include "typesf2c.h"
#include "globalp.h"
#include "global.h"
#include "message.h"
#include "base.h"

/* can handle ga_brdcst/igop/dgop via ARMCI or native message-passing library
 * uncomment line below to use the ARMCI version */
#ifndef NEC
#define  ARMCI_COLLECTIVES 
#endif

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
    int p_grp = (int)ga_pgroup_get_default_();
    if (p_grp > 0) {
       int aroot = PGRP_LIST[p_grp].inv_map_proc_list[root];
       armci_msg_group_bcast_scope(SCOPE_ALL,buffer, (int)len, aroot,(&(PGRP_LIST[p_grp].group)));
    } else {
       armci_msg_bcast(buffer, (int)len, (int)root);
    }
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
    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
    ga_msg_brdcst(*type,buf,*len,*originator);
}

void FATR ga_pgroup_brdcst_(grp_id, type, buf, len, originator)
     Integer *type, *len, *originator, *grp_id;
     Void *buf;
{
    int p_grp = (int)*grp_id;
    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
    if (p_grp > 0) {
       int aroot = PGRP_LIST[p_grp].inv_map_proc_list[*originator];
       armci_msg_group_bcast_scope(SCOPE_ALL,buf,(int)*len,aroot,(&(PGRP_LIST[p_grp].group)));
    } else {
       int aroot = (int)*originator;
       armci_msg_bcast(buf, (int)len, (int)aroot);
    }
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
    int p_grp = (int)ga_pgroup_get_default_(); 
#ifdef MPI
    if(p_grp>0)
       armci_msg_group_barrier(&(PGRP_LIST[p_grp].group));
    else
       MPI_Barrier(MPI_COMM_WORLD);
#else
     long type=GA_TYPE_SYN;
#  ifdef LAPI
     armci_msg_barrier();
#  else
     SYNCH_(&type);
#  endif
#endif
}

void ga_msg_pgroup_sync_(Integer *grp_id)
{
    int p_grp = (int)(*grp_id);
#ifdef MPI
    armci_msg_group_barrier(&(PGRP_LIST[p_grp].group));
#else
    ga_error("ga_msg_pgroup_sync not implemented",0);
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

#if defined(CRAY) || defined(WIN32)
void FATR GA_PGROUP_DGOP(grp_id, type, x, n, op)
     _fcd op;
#else
void ga_pgroup_dgop_(grp_id, type, x, n, op, len)
     char *op;
     int len;
#endif
     Integer *type, *n, *grp_id;
     DoublePrecision *x;
{
long gtype,gn,grp;
     gtype = (long)*type; gn = (long)*n;
     grp = (long)*grp_id;

#if defined(CRAY) || defined(WIN32)
     ga_pgroup_dgop(grp, gtype, x, gn, _fcdtocp(op));
#else
     ga_pgroup_dgop(grp, gtype, x, gn, op);
#endif
}

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

#if defined(CRAY) || defined(WIN32)
void FATR GA_PGROUP_SGOP(grp_id, type, x, n, op)
     _fcd op;
#else
void ga_pgroup_sgop_(grp_id, type, x, n, op, len)
     char *op;
     int  len;
#endif
     Integer *type, *n, *grp_id;
     float *x;
{
long gtype,gn,grp;
     gtype = (long)*type; gn = (long)*n;
     grp = (long)*grp_id;
 
#if defined(CRAY) || defined(WIN32)
     ga_pgroup_fgop(grp, gtype, x, gn, _fcdtocp(op));
#else
     ga_pgroup_fgop(grp, gtype, x, gn, op);
#endif
}                                   

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

#if defined(CRAY) || defined(WIN32)
void FATR GA_PGROUP_IGOP(grp_id, type, x, n, op)
     _fcd op;
#else
void ga_pgroup_igop_(grp_id, type, x, n, op, len)
     char *op;
     int  len;
#endif
     Integer *type, *n, *grp_id;
     Integer *x;
{
long gtype,gn,grp;
     gtype = (long)*type; gn = (long)*n;
     grp = (long)*grp_id;

#if defined(CRAY) || defined(WIN32)
     ga_pgroup_igop(grp, gtype, x, gn, _fcdtocp(op));
#else
     ga_pgroup_igop(grp, gtype, x, gn, op);
#endif
}

void ga_type_gop(int Type, void *x, int n, char* op)
{
int atype, type=ga_type_f2c(Type);
    switch(type){
    case MT_REAL: atype=ARMCI_FLOAT; break;
    case MT_DBL: atype=ARMCI_DOUBLE; break;
    case MT_LONGINT: atype=ARMCI_LONG; break;
    case MT_INT: atype=ARMCI_INT; break;
    default: ga_error("ga_type_gop: type not supported",Type);
    }
    armci_msg_gop_scope(SCOPE_ALL, x, n, op, atype);   
}

void ga_pgroup_dgop(p_grp, type, x, n, op)
     Integer type, n, p_grp;
     DoublePrecision *x;
     char *op;
{
    int group = (int)p_grp;
    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
#if defined(ARMCI_COLLECTIVES) || defined(MPI)
     if (group > 0) {
       armci_msg_group_dgop(x, (int)n, op,(&(PGRP_LIST[group].group)));
     } else {
       armci_msg_dgop(x, (int)n, op);
     }
#else
       ga_error("Groups not implemented for system",0);
#endif
}

void ga_dgop(type, x, n, op)
     Integer type, n;
     DoublePrecision *x;
     char *op;
{
     Integer p_grp = ga_pgroup_get_default_();
    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
     if ((int)p_grp > 0) {
       ga_pgroup_dgop(p_grp, type, x, n, op);
     } else {
#if defined(ARMCI_COLLECTIVES) || defined(MPI)
       armci_msg_dgop(x, (int)n, op);
#else
            DGOP_(&type, x, &n, op);
#endif
     }
}

void ga_pgroup_lgop(p_grp,type, x, n, op)
     Integer p_grp,type, n;
     long *x;
     char *op;
{
        int group = (int)p_grp;
        _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
#if defined(ARMCI_COLLECTIVES) || defined(MPI)
        if (group > 0)
	  armci_msg_group_lgop(x, (int)n, op,(&(PGRP_LIST[group].group)));
        else
	  armci_msg_lgop(x, (int)n, op);
#else
            ga_error("Groups not implemented for system",0);
#endif
}
void ga_lgop(type, x, n, op)
     Integer type, n;
     long *x;
     char *op;
{
        Integer p_grp = ga_pgroup_get_default_();
        _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
#if defined(ARMCI_COLLECTIVES) || defined(MPI)
        if ((int)p_grp > 0) 
          ga_pgroup_lgop(p_grp,type,x,n,op);
        else 
#endif
	  armci_msg_lgop(x, (int)n, op);
}

void ga_pgroup_igop(p_grp, type, x, n, op)
     Integer p_grp, type, n, *x;
     char *op;
{
            int group = (int) p_grp;
            int me = (int)ga_nodeid_();
            _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
#if defined(ARMCI_COLLECTIVES) || defined(MPI)
#   ifdef EXT_INT
            if (group > 0) {
              armci_msg_group_lgop(x, (int)n, op,(&(PGRP_LIST[group].group)));
            } else {
              armci_msg_lgop(x, (int)n, op);
            }
#   else
            if (group > 0)
              armci_msg_group_igop(x, (int)n, op,(&(PGRP_LIST[group].group)));
            else
              armci_msg_igop(x, (int)n, op);
#   endif
#else
            ga_error("Groups not implemented for system",0);
#endif
}

void ga_igop(type, x, n, op)
     Integer type, n, *x;
     char *op;
{
     Integer p_grp = ga_pgroup_get_default_();
            _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
#if defined(ARMCI_COLLECTIVES) || defined(MPI)
            if (p_grp > 0) {
              ga_pgroup_igop(p_grp,type,x,n,op);
            } else {
#   ifdef EXT_INT
            armci_msg_lgop(x, (int)n, op);
#   else
            armci_msg_igop(x, (int)n, op);
#   endif
            }
#else
            IGOP_(&type, x, &n, op);
#endif
}


void ga_pgroup_fgop(p_grp, type, x, n, op)
     Integer type, n, p_grp;
     float *x;
     char *op;
{
    int group = (int)p_grp;
    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
     if (p_grp > 0) {
#if defined(ARMCI_COLLECTIVES) || defined(MPI)
       if (group > 0) {
         armci_msg_group_fgop(x, (int)n, op, (&(PGRP_LIST[group].group)));
       } else {
         armci_msg_fgop(x, (int)n, op);
       }
#else
       ga_error("Groups not implemented for system",0);
#endif
     }
}

void ga_fgop(type, x, n, op)
     Integer type, n;
     float *x;
     char *op;
{
     Integer p_grp = ga_pgroup_get_default_();
    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
     if (p_grp > 0) {
       ga_pgroup_fgop(p_grp, type, x, n, op);
     } else {
#if defined(ARMCI_COLLECTIVES) || defined(MPI)
       armci_msg_fgop(x, (int)n, op);
#else
       ga_error("Operation not defined for system",0);
#endif
     }
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
