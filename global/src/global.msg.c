/*
 * module: global.msg.c
 * author: Jarek Nieplocha
 * date:  Wed Sep 27 09:40:10 PDT 1995
 * description: internal GA message-passing communication routines 
                implemented as wrappers to MPI/NX/MPL/TCGMSG libraries
 * notes:       non-reentrant MPICH code must be avoided in the context
 *              of interrupt-driven communication; 
 *              MPICH might be also invoked through TCGMSG interface therefore
 *              for that reason need to avoid TCGMSG on Intel and SP machines 
 *
 */



#if defined SP1 
#  include <mpproto.h>
#elif defined(NX) 
#  if defined(PARAGON)
#     include <nx.h>
#  elif defined(DELTA)
#     include <mesh.h>
#  else
#     include <cube.h>
#  endif
#endif

#include "global.h"
#include "globalp.h"
#include "message.h"
#include <stdio.h>


/*\ wrapper to a BLOCKING MESSAGE SEND operation
\*/
void ga_msg_snd(type, buffer, bytes, to)
     Integer type, bytes, to;
     Void    *buffer;
{
#    if defined(NX) 

        csend(type, buffer, bytes, to, 0);

#    elif defined(SP1)

        /* need to avoid blocking calls that disable interrupts */
        int status, msgid;

        status = mpc_send(buffer, bytes, to, type, &msgid);
        if(status == -1) ga_error("ga_msg_snd: error sending ", type);
        while((status=mpc_status(msgid)) == -1); /* nonblocking probe */
        if(status < -1) ga_error("ga_msg_snd: invalid message ID ", msgid );

#    elif defined(MPI)

        int ierr;
        ierr = MPI_Send(buffer, (int)bytes, MPI_CHAR, (int)to, (int)type,
                        MPI_COMM_WORLD);
        if(ierr != MPI_SUCCESS) ga_error("ga_msg_snd: failed ", type);

#    else
        Integer sync=SYNC;
        snd_(&type, buffer, &bytes, &to, &sync);
#    endif
}



/*\ wrapper to a BLOCKING MESSAGE RECEIVE operation
\*/
void ga_msg_rcv(type, buffer, buflen, msglen, from, whofrom)
     Integer type, buflen, *msglen, from, *whofrom;
     Void    *buffer;
{
#    if defined(NX) 

#       ifdef  PARAGON
           long info[8], ptype=0;
           crecvx(type, buffer, buflen, from, ptype, info); 
           *msglen = info[1];
           *whofrom = info[2];
#       else
           crecv(type, buffer, buflen); /* cannot receive by sender */ 
           *msglen = infocount();
           *whofrom = infonode();
           if(from!=-1 &&  *whofrom != from) {
             fprintf(stderr,"ga_msg_rcv: from %d expected %d\n",*whofrom,from);
             ga_error("ga_msg_rcv: error receiving",from);
           }
#       endif

#    elif defined(SP1)

        /* need to avoid blocking calls that disable interrupts */
        int status, msgid, ffrom, ttype=type; 
 
        ffrom = (from == -1)? DONTCARE: from;
        status = mpc_recv(buffer, buflen, &ffrom, &ttype, &msgid);
        if(status == -1) ga_error("ga_msg_rcv: error receiving", type);

        while((status=mpc_status(msgid)) == -1); /* nonblocking probe */
        if(status < -1) ga_error("ga_msg_rcv: invalid message ID ", msgid );
        *msglen = status;

#    elif defined(MPI)

        int ierr, count, ffrom;
        MPI_Status status;

        ffrom = (from == -1)? MPI_ANY_SOURCE : (int)from;
        ierr = MPI_Recv(buffer, (int)buflen, MPI_CHAR, ffrom, (int)type,
               MPI_COMM_WORLD, &status);
        if(ierr != MPI_SUCCESS) ga_error("ga_msg_rcv: Recv failed ", type);

        ierr = MPI_Get_count(&status, MPI_CHAR, &count);
        if(ierr != MPI_SUCCESS) ga_error("ga_msg_rcv: Get_count failed ", type);
        *whofrom = (Integer)status.MPI_SOURCE;
        *msglen  = (Integer)count;

#    else
        Integer sync=SYNC;
        rcv_(&type, buffer, &buflen, msglen, &from, whofrom, &sync);
#    endif
}



/*\ wrapper to a NONBLOCKING MESSAGE RECEIVE
\*/
msgid_t ga_msg_ircv(type, buffer, buflen, from)
     Integer type, buflen, from;
     Void    *buffer;
{
msgid_t msgid;

#    if defined(NX) 

#       ifdef  PARAGON
           long ptype=0;
           /*msginfo is NX internal */
           msgid = irecvx(type, buffer, buflen, from, ptype, msginfo);
#       else
           msgid = irecv(type, buffer, buflen); /* cannot receive by sender */
#       endif

#    elif defined(SP1)

        int status;
        static int ffrom, ttype; /*  MPL writes upon message arrival */
        ttype = type;
        ffrom = (from == -1)? DONTCARE: from;
        status = mpc_recv(buffer, buflen, &ffrom, &ttype, &msgid);
        if(status == -1) ga_error("ga_msg_ircv: error receiving", type);

#    elif defined(MPI)

        int ierr, count, ffrom;
        ffrom = (from == -1)? MPI_ANY_SOURCE : (int)from;
        ierr = MPI_Irecv(buffer, (int)buflen, MPI_CHAR, ffrom, (int)type,
               MPI_COMM_WORLD, &msgid);
        if(ierr != MPI_SUCCESS) ga_error("ga_msg_ircv: Recv failed ", type);

#    else

       Integer sync=ASYNC, msglen, whofrom;
       rcv_(&type, buffer, &buflen, &msglen, &from, &whofrom, &sync);
       msgid = from; /*TCGMSG waits for all comms to/from node */

#    endif

     return(msgid);
}



/*\ wrapper to BLOCKING MESSAGE WAIT operation  
 *  Note: values returned in whofrom and msglen might not be reliable (updated)
\*/
void ga_msg_wait(msgid, whofrom, msglen)
msgid_t msgid;
Integer *whofrom, *msglen;
{
#    if defined(NX) 

        msgwait(msgid);
/*        *msglen = infocount();*/
/*        *whofrom = infonode();*/

#    elif defined(SP1)

        int status;
        while((status=mpc_status(msgid)) == -1); /* nonblocking probe */
        if(status < -1) ga_error("ga_wait_msg: invalid message ID ", msgid);
        *msglen = status;
        /* whofrom is currently not retrieved from MPL */

#    elif defined(MPI)

        int ierr, count;
        MPI_Status status;

        ierr = MPI_Wait(&msgid, &status);
        if(ierr != MPI_SUCCESS) ga_error("ga_msg_wait: failed ", 0);
        ierr = MPI_Get_count(&status, MPI_CHAR, &count);
        if(ierr != MPI_SUCCESS) ga_error("ga_msg_wait: Get_count failed", 0);
        *whofrom = (Integer)status.MPI_SOURCE;
        *msglen  = (Integer)count;

#    else
        waitcom_(&msgid); /* cannot get whofrom and msglen from TCGMSG */
#    endif
} 


/*\ total NUMBER OF PROCESSES that can communicate using message-passing
 *  Note: might be larger than the value returned by ga_nnodes_()
\*/
Integer ga_msg_nnodes_()
{
#  ifdef MPI
     int numprocs;
     MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
     return((Integer)numprocs);
#  else
     return (nnodes_());
#  endif
}


/*\ message-passing RANK/NUMBER of current PROCESS
 *  Note: might be different than the value returned by ga_nodeid_()
\*/
Integer ga_msg_nodeid_()
{
#  ifdef MPI
     int myid;

     MPI_Comm_rank(MPI_COMM_WORLD,&myid);
     return((Integer)myid);
#  else
     return (nodeid_());
#  endif
}


void ga_msg_brdcst(type, buffer, len, root)
Integer type, len, root;
Void*   buffer; 
{
#  ifdef MPI
      MPI_Bcast(buffer, (int)len, MPI_CHAR, (int)root, MPI_COMM_WORLD);
#  else
      void brdcst_();
      brdcst_(&type, buffer, &len, &root);
#  endif
}



#ifdef SP1

/* This paranoia is required to assure that there is always posted receive 
 * for synchronization message. MPL (and EUIH) "in order message delivery"
 * rule causes that synchronization message arriving from another node
 * must be received before any following request (rcvncall) shows up.
 * 
 * MPL synchronization must be avoided as it is not compatible with rcvncall
 */

Integer syn_me, syn_up, syn_left, syn_right, syn_root=0;
static msgid_t syn_up_id, syn_left_id, syn_right_id;
Integer syn_type1=37772, syn_type2=37773, syn_type3=37774;
Integer first_time=1, xsyn;

void sp_init_sync()
{
  if (syn_me != syn_root) syn_up_id = ga_msg_ircv(syn_type1, &xsyn, 1, syn_up);
  if (syn_left > -1 ) syn_left_id = ga_msg_ircv(syn_type2, &xsyn, 1, syn_left);
  if (syn_right > -1) syn_right_id = ga_msg_ircv(syn_type2, &xsyn, 1,syn_right);
}


void sp_sync()
{
Integer len, from, xsyn;
  /* from root down */
  if (syn_me != syn_root){
      ga_msg_wait(syn_up_id, &from, &len );
      syn_up_id = ga_msg_ircv(syn_type1, &xsyn, 0, syn_up);
  }
  if (syn_left > -1)  ga_msg_snd(syn_type1, &xsyn, 0, syn_left);
  if (syn_right > -1) ga_msg_snd(syn_type1, &xsyn, 0, syn_right);

  /* up to root */
  if (syn_left > -1 ){
      ga_msg_wait(syn_left_id, &from, &len );
      syn_left_id = ga_msg_ircv(syn_type2, &xsyn, 0, syn_left);
  }
  if (syn_right > -1){
      ga_msg_wait(syn_right_id, &from, &len );
      syn_right_id = ga_msg_ircv(syn_type2, &xsyn, 0, syn_right);
  }
  if (syn_me != syn_root) ga_msg_snd(syn_type2, &xsyn, 0, syn_up);

}
#endif



void ga_msg_sync_()
{
#  ifdef SP1
     Integer group_participate();
     if(first_time){
        group_participate(&syn_me, &syn_root, &syn_up, &syn_left, &syn_right, ALL_GRP);
        sp_init_sync();
        first_time =0;
     }
     sp_sync(); 
     sp_sync(); /* one sync should be enogh -- this code needs more work */

#  elif defined(MPI)
      MPI_Barrier(MPI_COMM_WORLD);
#  else
      Integer type = GA_TYPE_SYN;
      synch_(&type);
#  endif
}


/* Note: collective comms  below could be implemented as trivial wrappers
 * to MPI if it was reentrant
 */


/*\ determines if calling process participates in collective comm
 *  also, parameters for a given communicator are found
\*/
Integer group_participate(me, root, up, left, right, group)
     Integer *me, *root, *up, *left, *right, group;
{
     Integer nproc, index;

     switch(group){
                           /* all message-passing processes */
           case  ALL_GRP:  *root = 0;
                           *me = ga_msg_nodeid_(); nproc = ga_msg_nnodes_();
                           *up = (*me-1)/2;       if(*up >= nproc) *up = -1;
                           *left  =  2* *me + 1;  if(*left >= nproc) *left = -1;
                           *right =  2* *me + 2;  if(*right >= nproc)*right =-1;

                           break;
                           /* all GA (compute) processes in cluster */
          case CLUST_GRP:  *root = cluster_master;
                           *me = ga_msg_nodeid_(); nproc =cluster_compute_nodes;
                           if(*me < *root || *me >= *root +nproc)
                                                return 0; /*does not*/ 

                           index  = *me  - *root;
                           *up    = (index-1)/2 + *root;  
                           *left  = 2*index + 1 + *root; 
                                    if(*left >= *root+nproc) *left = -1;
                           *right = 2*index + 2 + *root; 
                                    if(*right >= *root+nproc) *right = -1;

                           break;
                           /* all msg-passing processes in cluster */
      case ALL_CLUST_GRP:  *root = cluster_master;
                           *me = ga_msg_nodeid_(); 
                           nproc = cluster_nodes; /* +server*/
                           if(*me < *root || *me >= *root +nproc)
                                                return 0; /*does not*/

                           index  = *me  - *root;
                           *up    = (index-1)/2 + *root;   
                                    if( *up < *root) *up = -1;
                           *left  = 2*index + 1 + *root;
                                    if(*left >= *root+nproc) *left = -1;
                           *right = 2*index + 2 + *root;
                                    if(*right >= *root+nproc) *right = -1;

                           break;
                           /* cluster masters (designated process in cluster) */
    case INTER_CLUST_GRP:  *root = GA_clus_info[0].masterid;
                           *me = ga_msg_nodeid_();  nproc = num_clusters;

                           if(*me != cluster_master) return 0; /*does not*/

                           *up    = (cluster_id-1)/2;
                           if(*up >= nproc) *up = -1;
                             else *up = GA_clus_info[*up].masterid;

                           *left  = 2*cluster_id+ 1;
                           if(*left >= nproc) *left = -1;
                             else *left = GA_clus_info[*left].masterid;

                           *right = 2*cluster_id+ 2;
                           if(*right >= nproc) *right = -1;
                             else *right = GA_clus_info[*right].masterid;

                           break;
                 default:  ga_error("group_participate: wrong group ", group);
     }
     return (1);
}




/*\ BROADCAST 
 *  internal GA routine that is used in data server mode
 *  with predefined communicators
\*/
void ga_brdcst_clust(type, buf, len, originator, group)
     Integer type, len, originator, group;
     Void *buf;
{
     Integer me, lenmes, from, root=0;
     Integer up, left, right, participate;

     participate = group_participate(&me, &root, &up, &left, &right, group);

     /*  cannot exit just yet -->  send the data to root */

     if (originator != root ){
       if(me == originator) ga_msg_snd(type, buf, len, root); 
       if(me == root) ga_msg_rcv(type, buf, len, &lenmes, originator, &from); 
     }

     if( ! participate) return;

     if (me != root) ga_msg_rcv(type, buf, len, &lenmes, up, &from);
     if (left > -1)  ga_msg_snd(type, buf, len, left);
     if (right > -1) ga_msg_snd(type, buf, len, right);
}



/*\ BROADCAST
\*/
void ga_brdcst_(type, buf, len, originator)
     Integer *type, *len, *originator;
     Void *buf;
{
     void brdcst_();
     Integer orig_clust, tcg_orig_node, tcg_orig_master; 

     if(ClusterMode){
        /* originator is GA node --> need to transform it into TCGMSG */
        orig_clust = ClusterID(*originator); 
        tcg_orig_master =  GA_clus_info[orig_clust].masterid;
        tcg_orig_node   =  *originator + orig_clust;
        if(orig_clust == cluster_id){
           ga_brdcst_clust(*type, buf, *len, tcg_orig_node, CLUST_GRP);
           ga_brdcst_clust(*type, buf, *len, tcg_orig_master, INTER_CLUST_GRP);
        }else{
           ga_brdcst_clust(*type, buf, *len, tcg_orig_master, INTER_CLUST_GRP);
           ga_brdcst_clust(*type, buf, *len, cluster_master, CLUST_GRP);
        }
     } else {
        /* use TCGMSG as a wrapper to native implementation of broadcast */
        Integer gtype,gfrom,glen;
        gtype =(long) *type; gfrom =(long) *originator; glen =(long) *len;
#       ifdef SP1
            ga_sync_();
            /*            brdcst_(&gtype,buf,&glen,&gfrom);*/
            ga_msg_brdcst(gtype, buf, glen, gfrom);
            ga_sync_();
#       else
            ga_msg_brdcst(gtype, buf, glen, gfrom);
#       endif
     }
}


/*\  global operations:
 *     . all processors participate  
 *     . all processors in the cluster participate  
 *     . master processors in each cluster participate  
\*/
void ga_dgop_clust(type, x, n, op, group)
     Integer type, n, group;
     DoublePrecision *x;
     char *op;
{
#    define BUF_SIZE 10000
     Integer  me, lenmes, from, len, root;
     DoublePrecision work[BUF_SIZE], *origx = x;
     static void ddoop();
     Integer ndo, up, left, right, orign = n;

     if( ! group_participate(&me, &root, &up, &left, &right, group)) return;

     while ((ndo = (n<=BUF_SIZE) ? n : BUF_SIZE)) {
	 len = lenmes = ndo*sizeof(DoublePrecision);

         if (left > -1) {
           ga_msg_rcv(type, (char *) work, len, &lenmes, left, &from);
           ddoop(ndo, op, x, work);
         }
         if (right > -1) {
           ga_msg_rcv(type, (char *) work, len, &lenmes, right, &from);
           ddoop(ndo, op, x, work);
         }
         if (me != root) ga_msg_snd(type, x, len, up); 

       n -=ndo;
       x +=ndo;
     }
     /* Now, root broadcasts the result down the binary tree */
     len = orign*sizeof(DoublePrecision);
     ga_brdcst_clust(type, (char *) origx, len, root, group);
}


/*\ GLOBAL OPERATIONS
 *  (C)
 *  We cannot use TCGMSG in data-server mode
 *  where only compute processes participate
\*/
void ga_dgop(type, x, n, op)
     Integer type, n;
     DoublePrecision *x;
     char *op;
{
     void dgop_();

     if(ClusterMode){
        ga_dgop_clust(type, x, n, op, CLUST_GRP);
        ga_dgop_clust(type, x, n, op, INTER_CLUST_GRP);
        ga_brdcst_clust(type, x, n*sizeof(DoublePrecision), cluster_master,
                        CLUST_GRP);
     } else {
        /* use TCGMSG as a wrapper to native implementation of global ops */
#       ifdef SP1
            ga_msg_sync_();
#       endif
#       ifdef MPI
            ga_dgop_clust(type, x, n, op, ALL_GRP);
#       else
            dgop_(&type, x, &n, op, (Integer)strlen(op));
#       endif
#       ifdef SP1
            ga_msg_sync_();
#       endif
     }
}


/*\ GLOBAL OPERATIONS 
 *  Fortran
\*/
#ifdef CRAY_T3D
void ga_dgop_(type, x, n, op)
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

#ifdef CRAY_T3D
     ga_dgop(gtype, x, gn, _fcdtocp(op));
#else
     ga_dgop(gtype, x, gn, op);
#endif
}


/*\  global operations:
 *     . all processors participate
 *     . all processors in the cluster participate
 *     . master processors in each cluster participate
\*/
void ga_igop_clust(type, x, n, op, group)
     Integer type, n, group;
     Integer *x;
     char *op;
{
#    define BUF_SIZE 10000
     Integer  me, lenmes,  from, len, root=0 ;
     Integer work[BUF_SIZE], *origx = x;
     static void idoop();
     Integer ndo, up, left, right, orign =n;

     if( ! group_participate(&me, &root, &up, &left, &right, group)) return;

     while ((ndo = (n<=BUF_SIZE) ? n : BUF_SIZE)) {
	 len = lenmes = ndo*sizeof(Integer);

         if (left > -1) {
           ga_msg_rcv(type, (char *) work, len, &lenmes, left, &from);
	   idoop(ndo, op, x, work); 
         }
         if (right > -1) {
           ga_msg_rcv(type, (char *) work, len, &lenmes, right, &from);
	   idoop(ndo, op, x, work); 
         }
         if (me != root) ga_msg_snd(type, x, len, up);

       n -=ndo;
       x +=ndo;
     }
     /* Now, root broadcasts the result down the binary tree */
     len = orign*sizeof(Integer);
     ga_brdcst_clust(type, (char *) origx, len, root, group);
}


/*\ GLOBAL OPERATIONS
 *  (C)
 *  We cannot use TCGMSG in data-server mode
 *  where only compute processes participate
\*/
void ga_igop(type, x, n, op)
     Integer type, n, *x;
     char *op;
{
     void igop_();

     if(ClusterMode){
        ga_igop_clust(type, x, n, op, CLUST_GRP);
        ga_igop_clust(type, x, n, op, INTER_CLUST_GRP);
        ga_brdcst_clust(type, x, n*sizeof(Integer), cluster_master, CLUST_GRP);
     } else {
        /* use TCGMSG as a wrapper to native implementation of global ops */
#       ifdef SP1
            ga_msg_sync_();
#       endif
#       ifdef MPI
            ga_igop_clust(type, x, n, op, ALL_GRP);
#       else
            igop_(&type, x, &n, op, (Integer)strlen(op));
#       endif
#       ifdef SP1
            ga_msg_sync_();
#       endif
     }
}




/*\ GLOBAL OPERATIONS 
 *  Fortran
\*/
#ifdef CRAY_T3D
void ga_igop_(type, x, n, op)
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

#ifdef CRAY_T3D
     ga_igop(gtype, x, gn, _fcdtocp(op));
#else
     ga_igop(gtype, x, gn, op);
#endif
}



static void ddoop(n, op, x, work)
     long n;
     char *op;
     double *x, *work;
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
      register double x1 = ABS(*x), x2 = ABS(*work);
      *x = MAX(x1, x2);
      x++; work++;
    }
  else if (strncmp(op,"absmin",6) == 0)
    while(n--) {
      register double x1 = ABS(*x), x2 = ABS(*work);
      *x = MIN(x1, x2);
      x++; work++;
    }
  else
    ga_error("ga_ddoop: unknown operation requested", (long) n);
}


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
      register Integer x1 = ABS(*x), x2 = ABS(*work);
      *x = MAX(x1, x2);
      x++; work++;
    }
  else if (strncmp(op,"absmin",6) == 0)
    while(n--) {
      register Integer x1 = ABS(*x), x2 = ABS(*work);
      *x = MIN(x1, x2);
      x++; work++;
    }
  else if (strncmp(op,"or",2) == 0) 
    while(n--) {
      *x |= *work;
      x++; work++;
    }
  else
    ga_error("ga_idoop: unknown operation requested", (long) n);
}


