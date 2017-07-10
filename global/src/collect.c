#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STRING_H
#   include "string.h"
#endif
#if HAVE_MATH_H
#   include <math.h>
#endif

/* $Id: collect.c,v 1.23.2.5 2007-08-03 19:52:28 manoj Exp $ */
#include "typesf2c.h"
#include "globalp.h"
#include "message.h"
#include "base.h"
#include "ga-papi.h"
#include "ga-wapi.h"

/* can handle ga_brdcst/igop/dgop via ARMCI or native message-passing library
 * uncomment line below to use the ARMCI version */

#ifdef MSG_COMMS_MPI
#   include <mpi.h>
extern MPI_Comm ARMCI_COMM_WORLD;
#   include "ga-mpi.h"
#   if HAVE_ARMCI_GROUP_COMM
extern MPI_Comm armci_group_comm(ARMCI_Group *group);
#   endif
#else
#  include <tcgmsg.h>
#endif


#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_msg_brdcst = pnga_msg_brdcst
#endif
void pnga_msg_brdcst(Integer type, void *buffer, Integer len, Integer root)
{
  long bigint = 2147483647L;
  int i, nsteps;//= (int) ceil(((double)len)/((double)bigint));
  long len_small;
  void *buffer_ptr;
  long istart=0;
  int p_grp;
  /*          printf("%ld len %ld bigint %ld  \n",GAme,len,bigint); */
    nsteps = (int) ceil(((double)len)/((double)bigint));
    /*          printf("%ld len %ld bigint %ld  nsteps %d \n",GAme,len,bigint,nsteps); */
  for (i=0; i < nsteps; i++){
    len_small=bigint;
    buffer_ptr=(char *)buffer+istart;
    if (istart+len_small > len) len_small=((long)(len - istart));
    /*               printf("%ld step %d of %d  len= %d total=%ld istart= %ld\n",GAme,(i+1),nsteps,len_small,len,istart); */
    p_grp = (int)pnga_pgroup_get_default();
    if (p_grp > 0) {
#   ifdef MSG_COMMS_MPI
        int aroot = PGRP_LIST[p_grp].inv_map_proc_list[root];
        armci_msg_group_bcast_scope(SCOPE_ALL,buffer_ptr, (int)len_small, aroot,(&(PGRP_LIST[p_grp].group)));
#   endif
    } else {
      armci_msg_bcast(buffer_ptr, (int)len_small, (int)root);
    }
          istart+=len_small;
  }
}


/*\ BROADCAST
\*/
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_brdcst = pnga_brdcst
#endif
void pnga_brdcst(Integer type, void *buf, Integer len, Integer originator)
{
    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
    pnga_msg_brdcst(type,buf,len,originator);
}


#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_pgroup_brdcst = pnga_pgroup_brdcst
#endif
void pnga_pgroup_brdcst(Integer grp_id, Integer type, void *buf,
                             Integer len, Integer originator)
{
    int p_grp = (int)grp_id;
    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
    if (p_grp > 0) {
#ifdef MSG_COMMS_MPI
       int aroot = PGRP_LIST[p_grp].inv_map_proc_list[originator];
       armci_msg_group_bcast_scope(SCOPE_ALL,buf,(int)len,aroot,(&(PGRP_LIST[p_grp].group)));
#endif
    } else {
       int aroot = (int)originator;
       armci_msg_bcast(buf, (int)len, (int)aroot);
    }
}


#ifdef MSG_COMMS_MPI
MPI_Comm GA_MPI_Comm()
{
    return GA_MPI_Comm_pgroup(-1);
}
MPI_Comm GA_MPI_Comm_pgroup_default()
{
    return GA_MPI_Comm_pgroup(pnga_pgroup_get_default());
}
MPI_Comm GA_MPI_Comm_pgroup(int p_grp)
{
    ARMCI_Group group;
    if (p_grp > 0) {
        group = PGRP_LIST[p_grp].group;
    }
    else {
        ARMCI_Group_get_world(&group);
    }
#   if HAVE_ARMCI_GROUP_COMM_MEMBER
    return group.comm;
#   else
    return armci_group_comm(&group);
#   endif
}
#endif


#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_msg_sync = pnga_msg_sync
#endif
void pnga_msg_sync()
{
#ifdef MSG_COMMS_MPI
    int p_grp = (int)pnga_pgroup_get_default(); 
    if(p_grp>0)
       armci_msg_group_barrier(&(PGRP_LIST[p_grp].group));
    else
       armci_msg_barrier();
#else
     tcg_synch(GA_TYPE_SYN);
#endif
}


#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_msg_pgroup_sync = pnga_msg_pgroup_sync
#endif
void pnga_msg_pgroup_sync(Integer grp_id)
{
    int p_grp = (int)(grp_id);
    if(p_grp>0) {
#     ifdef MSG_COMMS_MPI       
        armci_msg_group_barrier(&(PGRP_LIST[p_grp].group));
#     else
        pnga_error("ga_msg_pgroup_sync not implemented",0);
#     endif
    }
    else {
#     if defined(MSG_COMMS_MPI)
       armci_msg_barrier();
#     else
       tcg_synch(GA_TYPE_SYN);
#    endif
    }
}


#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_pgroup_gop = pnga_pgroup_gop
#endif
void pnga_pgroup_gop(Integer p_grp, Integer type, void *x, Integer n, char *op)
{
    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
    if (p_grp > 0) {
#if  defined(MSG_COMMS_MPI)
        int group = (int)p_grp;
        switch (type){
            case C_INT:
                armci_msg_group_igop((int*)x, n, op, (&(PGRP_LIST[group].group)));
                break;
            case C_LONG:
                armci_msg_group_lgop((long*)x, n, op, (&(PGRP_LIST[group].group)));
                break;
            case C_LONGLONG:
                armci_msg_group_llgop((long long*)x, n, op, (&(PGRP_LIST[group].group)));
                break;
            case C_FLOAT:
                armci_msg_group_fgop((float*)x, n, op, (&(PGRP_LIST[group].group)));
                break;
            case C_DBL:
                armci_msg_group_dgop((double*)x, n, op, (&(PGRP_LIST[group].group)));
                break;
            case C_SCPL:
                armci_msg_group_fgop((float*)x, 2*n, op, (&(PGRP_LIST[group].group)));
                break;
            case C_DCPL:
                armci_msg_group_dgop((double*)x, 2*n, op, (&(PGRP_LIST[group].group)));
                break;
            default: pnga_error(" wrong data type ",type);
        }
#else
        pnga_error("Groups not implemented for system",0);
#endif
    } else {
        pnga_gop(type, x, n, op);
    }
}


#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_gop = pnga_gop
#endif
void pnga_gop(Integer type, void *x, Integer n, char *op)
{
    Integer p_grp = pnga_pgroup_get_default();

    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
    if (p_grp > 0) {
        pnga_pgroup_gop(p_grp, type, x, n, op);
    } else {
        switch (type){
            case C_INT:
                armci_msg_igop((int*)x, n, op);
                break;
            case C_LONG:
                armci_msg_lgop((long*)x, n, op);
                break;
            case C_LONGLONG:
                armci_msg_llgop((long long*)x, n, op);
                break;
            case C_FLOAT:
                armci_msg_fgop((float*)x, n, op);
                break;
            case C_DBL:
                armci_msg_dgop((double*)x, n, op);
                break;
            case C_SCPL:
                armci_msg_fgop((float*)x, 2*n, op);
                break;
            case C_DCPL:
                armci_msg_dgop((double*)x, 2*n, op);
                break;
            default:
                pnga_error(" wrong data type ",type);
        }
    }
}
