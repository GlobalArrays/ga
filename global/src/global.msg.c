/*
 * module: global.msg.c
 * author: Jarek Nieplocha
 * date: Mon Dec 19 19:06:18 CST 1994
 * description: data server and message-passing communication routines 
 *
 *
 * DISCLAIMER
 *
 * This material was prepared as an account of work sponsored by an
 * agency of the United States Government.  Neither the United States
 * Government nor the United States Department of Energy, nor Battelle,
 * nor any of their employees, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY,
 * COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT,
 * SOFTWARE, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
 * INFRINGE PRIVATELY OWNED RIGHTS.
 *
 *
 * ACKNOWLEDGMENT
 *
 * This software and its documentation were produced with United States
 * Government support under Contract Number DE-AC06-76RLO-1830 awarded by
 * the United States Department of Energy.  The United States Government
 * retains a paid-up non-exclusive, irrevocable worldwide license to
 * reproduce, prepare derivative works, perform publicly and display
 * publicly by or for the US Government, including the right to
 * distribute to other US Government contractors.
 */


#include "global.h"
#include "globalp.h"
#include "tcgmsg.h"
#include "message.h"
#include <stdio.h>
#ifdef CRAY_T3D
#  include <fortran.h>
#endif

#define DEBUG 0 
#define ACK   0
   
/* send & receive buffers alligned on sizeof(double) boundary  */
double _snd_dbl_buf[MSG_BUF_DBL_SIZE];
double _rcv_dbl_buf[MSG_BUF_DBL_SIZE];
struct message_struct *MessageSnd = (struct message_struct*)_snd_dbl_buf,
                      *MessageRcv = (struct message_struct*)_rcv_dbl_buf;


Integer in_handler = 0;
Integer NumSndReq=0;


Integer cluster_master;
Integer num_clusters;
Integer cluster_id;
Integer cluster_nodes=1;
Integer cluster_compute_nodes;
Integer ClusterMode=0;


/*\ determines cluster structure according to *.p file
 *  through TCGMSG SR_clus_info
\*/
void    ClustInfoInit()
{
#ifndef SYSV
    num_clusters = nnodes_();
    cluster_id = cluster_master = nodeid_();
    cluster_compute_nodes  = cluster_nodes = 1;
#else
    void PrintClusInfo();
    if(nodeid_()==0 && DEBUG) PrintClusInfo();
    if(nnodes_()==1){
       num_clusters = 1;
       cluster_id =  0;
       cluster_master = 0;
       cluster_nodes  = 1;
    }else{
       num_clusters = (Integer)SR_n_clus;
       cluster_id =  SR_clus_id;
       cluster_master = SR_clus_info[cluster_id].masterid; 
       cluster_nodes  = SR_clus_info[cluster_id].nslave; 
    } 
    if(num_clusters>1 && cluster_nodes <2)
           ga_error("ClustInfoInit: need at least 2 processes on cluster: ",
                    cluster_id);
    /*####*/
    if(num_clusters>1) {
       ClusterMode = 1;
       cluster_compute_nodes  = cluster_nodes-1; 
    }else
       cluster_compute_nodes  = cluster_nodes;
#endif
}


Integer ClusterID(proc)
    Integer proc;
{
#ifdef SYSV
    Integer clust, compute_nodes=0;

    if(ClusterMode){
       /* start adding compute nodes in each cluster
        * when > proc => we found cluster => we know server
        * WARNING: works only for single data server per cluster !
        */
       for(clust=0; clust< num_clusters; clust++){
           compute_nodes += SR_clus_info[clust].nslave -1;
           if(compute_nodes > proc) break;
       }
       return(clust);
    }else
#endif
       return(proc);
}


/*\ determines to which process (data_server) we need to send request for proc
\*/
Integer DataServer(proc)
    Integer proc;
{
#ifdef SYSV
    Integer clust, server;
    if(ClusterMode){
       clust = ClusterID(proc);
       /* server is the last tcgmsg process in the cluster */
       server = SR_clus_info[clust].masterid+ SR_clus_info[clust].nslave-1;
       return(server);
    }else
#endif
       return(proc);
}
       

/*\ returns the number of data_servers to the application
\*/
void ga_num_data_servers_(num)
     Integer *num;
{
    if(ClusterMode) *num = num_clusters;
    else *num = 0;
}



/*\ returns nodeid for all the data servers in user provided array "list"
 *  dimension of list has to be >= number returned from ga_num_data_servers()
\*/
void ga_list_data_servers_(list)
     Integer *list;
{
#ifdef SYSV
   int clust;
   if(ClusterMode)
     for(clust = 0; clust < num_clusters; clust ++)
       list[clust] = SR_clus_info[clust].masterid+ SR_clus_info[clust].nslave-1;
#endif
}


/*\ determine TCGMSG nodeid for the first GA <num_procs> processes 
\*/
void ga_list_nodeid_(list, num_procs)
     Integer *list, *num_procs;
{
    Integer proc, tcgnode, server;

    if(*num_procs < 1 || *num_procs > ga_nnodes_())
      ga_error("ga_list_nodeid: invalid number of GA num_procs ",*num_procs);

    if(ClusterMode){
       proc = tcgnode = 0;
       server = DataServer(proc);
       for( proc = 0; proc < *num_procs; proc++){
          while(tcgnode == server){
             tcgnode ++;
             server = DataServer(proc); /* server id for GA proc */
          }
          list[proc] = tcgnode;
          tcgnode ++;
       }

    }else for( proc = 0; proc < *num_procs; proc++) list[proc]=proc;
}



/*\ wrapper to a MESSAGE SEND operation
\*/
void ga_snd_msg(type, buffer, bytes, to, sync)
     Integer type, bytes, to, sync;
     Void    *buffer;
{
   snd_(&type, buffer, &bytes, &to, &sync);
}


/*\ wrapper to a MESSAGE RECEIVE operation
\*/
void ga_rcv_msg(type, buffer, buflen, msglen, from, whofrom, sync)
     Integer type, buflen, *msglen, from, *whofrom, sync;
     Void    *buffer;
{
   rcv_(&type, buffer, &buflen, msglen, &from, whofrom, &sync);
}


/*\  SEND REQUEST MESSAGE to the owner/data_server 
 *   to perform operation "oper" on g_a[ilo:ihi,jlo:jhi]
\*/
void ga_snd_req(g_a, ilo,ihi,jlo,jhi, nbytes, data_type, oper, proc, to)
     Integer g_a, ilo,ihi,jlo,jhi,nbytes,  data_type, oper, to, proc; 
{
    Integer  len, ack, from;

    MessageSnd->g_a = g_a;
    MessageSnd->ilo = ilo;
    MessageSnd->ihi = ihi;
    MessageSnd->jlo = jlo;
    MessageSnd->jhi = jhi;
    MessageSnd->to     = proc;
    MessageSnd->type   = data_type;
    MessageSnd->tag    = 77;
    MessageSnd->operation  = oper;
    
    len = nbytes + MSG_HEADER_SIZE;

    /* fprintf(stderr, "sending request %d to server %d\n",oper, to);*/
#   ifdef NX
      /*  on Intel machines we have to use csend because TCGMSG transforms
       *  user type and we have interrupt receive posted for GA_TYPE_REQ message
       */
       csend(GA_TYPE_REQ, (char*)MessageSnd, len, to, 0);
#   else
       ga_snd_msg(GA_TYPE_REQ, (char*)MessageSnd, len, to, SYNC);
#   endif

#   if defined(DATA_SERVER)
    if(ACK)
       ga_rcv_msg(GA_TYPE_ACK, (char*)&ack, sizeof(ack), &len, to, &from, SYNC);
#   endif
    if(DEBUG)fprintf(stderr,"sending request %d to server %d done \n",oper, to);
    NumSndReq++; /* count requests sent */
}



/*\ DATA SERVER services remote requests
 *       . invoked by interrupt-receive message as a server thread in addition 
 *         to the application thread which it might be suspended or run
 *         concurrently (Paragon), or
 *       . dedicated data-server node that loops here servicing requests
 *         since this routine is called in ga_initialize() until terminate
 *         request GA_OP_END (sent in ga_terminate() ) is received
\*/
void ga_SERVER(from)
     Integer from;
{
Integer msglen, ld, offset = 0, rdi_val, elem_size, nelem, toproc;
char    *piindex, *pjindex, *pvalue;

void    ga_get_local(), ga_put_local(), ga_acc_local(), ga_scatter_local(),
        ga_gather_local();
Integer ga_read_inc_local();

#ifdef DATA_SERVER
   if(DEBUG) fprintf(stderr, "data server %d ready\n",nodeid_());
   do {
      Integer len, ack;
      len = TOT_MSG_SIZE; /* MSG_BUF_SIZE + MSG_HEADER_SIZE */ 
      ga_rcv_msg(GA_TYPE_REQ, (char*)MessageRcv, len, &msglen, -1, &from,SYNC);
      if(ACK) ga_snd_msg(GA_TYPE_ACK, &ack, sizeof(ack), from, SYNC);
#else
      extern Integer in_handler;
      in_handler = 1; /*distinguish cases when GA ops are called by the server*/
#endif

      if(DEBUG) fprintf(stderr, "server got request %d from %d\n",
                                MessageRcv->operation, from);
      /* fprintf(stderr, "server %d ready\n",nodeid_()); */

      elem_size = GAsizeof(MessageRcv->type);
      toproc =  MessageRcv->to;

      if(MessageRcv->operation == GA_OP_GET || 
         MessageRcv->operation == GA_OP_PUT ||
         MessageRcv->operation == GA_OP_ACC){
           msglen = (MessageRcv->ihi - MessageRcv->ilo +1)
                  * (MessageRcv->jhi - MessageRcv->jlo +1) * elem_size;
           if(msglen > MSG_BUF_SIZE) 
                ga_error("ga_server: msgbuf overflow ", msglen);

           /* leading dimension for buf */
           ld = MessageRcv->ihi - MessageRcv->ilo +1;
      }

      GA_PUSH_NAME("ga_server");
      switch ( MessageRcv->operation) {
          case GA_OP_GET:   /* get */
                            ga_check_handle(&MessageRcv->g_a,"server: ga_get");
                            ga_get_local( MessageRcv->g_a,
                               MessageRcv->ilo, MessageRcv->ihi,
                               MessageRcv->jlo, MessageRcv->jhi,  
                               MessageRcv->buffer, offset, ld, toproc);
                            ga_snd_msg(GA_TYPE_GET, MessageRcv->buffer, msglen,
                               from, SYNC);
                            break;

          case GA_OP_PUT:   /* put */
                            ga_check_handle(&MessageRcv->g_a,"server: ga_put");
                            ga_put_local( MessageRcv->g_a,
                               MessageRcv->ilo, MessageRcv->ihi,
                               MessageRcv->jlo, MessageRcv->jhi,
                               MessageRcv->buffer, offset, ld, toproc);
                            break;

          case GA_OP_ACC:   /* accumulate */
                            ga_check_handle(&MessageRcv->g_a,"server: ga_acc");
                            ga_acc_local( MessageRcv->g_a,
                               MessageRcv->ilo, MessageRcv->ihi,
                               MessageRcv->jlo, MessageRcv->jhi,
                               MessageRcv->buffer, offset, ld, toproc,
                               *(DoublePrecision*)(MessageRcv->buffer+msglen)); 
                               /* alpha is at the end*/
                            break;

          case GA_OP_RDI:   /* read and increment */
                            {
                              Integer inc = MessageRcv->ihi;
                              ga_check_handle(&MessageRcv->g_a,"server:ga_rdi");
                              rdi_val = ga_read_inc_local( MessageRcv->g_a,
                                 MessageRcv->ilo, MessageRcv->jlo, inc, toproc);
                              ga_snd_msg(GA_TYPE_RDI, &rdi_val, sizeof(rdi_val),
                                         from, SYNC);
                            }
                            break;
          case GA_OP_DST:   /* scatter */
                            ga_check_handle(&MessageRcv->g_a,"server:ga_scat");

                            /* buffer contains (val,i,j) */
                            nelem = MessageRcv->ilo;
                            if (nelem > MSG_BUF_SIZE/
                               (elem_size+2*sizeof(Integer)) )
                                  ga_error("ga_server: scatter overflows buf ",
                                            nelem);
                            pvalue  = (char*)MessageRcv->buffer;
                            piindex = ((char*)MessageRcv->buffer) + 
                                      nelem*elem_size;
                            pjindex = piindex + nelem*sizeof(Integer);
                            ga_scatter_local(MessageRcv->g_a, pvalue,
                                             piindex, pjindex, nelem, toproc);
                            break;

          case GA_OP_DGT:   /* gather */
                            ga_check_handle(&MessageRcv->g_a,"server:ga_gath");

                            /* rcv buffer contains (i,j) only but we also
                             * need space in the same buffer for v
                             * value will be sent by server in rcv buffer 
                             */
                            nelem = MessageRcv->ilo;
                            if (nelem > MSG_BUF_SIZE/
                               (elem_size+2*sizeof(Integer)) )
                                  ga_error("ga_server: scatter overflows buf ",
                                            nelem);

                            piindex = (char*)MessageRcv->buffer;  
                            pjindex = piindex + nelem*sizeof(Integer);
                            pvalue  = pjindex + nelem*sizeof(Integer);
                            ga_gather_local(MessageRcv->g_a, pvalue,
                                            piindex, pjindex, nelem, toproc);
                            ga_snd_msg(GA_TYPE_DGT, pvalue, 
                                       elem_size*nelem, from, SYNC);
                            break;                          

          case GA_OP_CRE:   /* create an array */
                            {
                              Integer dim1   = MessageRcv->ilo;
                              Integer nblock1 = MessageRcv->ihi;
                              Integer dim2 = MessageRcv->jlo;
                              Integer nblock2 = MessageRcv->jhi;
                              char *map1 = MessageRcv->buffer+SHMID_BUF_SIZE;
                              char *map2 = map1 + nblock1*sizeof(Integer);
                              char *array_name = "server_created";
                              Integer g_a;

                              /* create can fail due to memory limits */
                              if(!ga_create_irreg(& MessageRcv->type, 
                                  &dim1, &dim2, array_name, (Integer*) map1, 
                                  &nblock1, (Integer*) map2, &nblock2, &g_a))
                                     fprintf(stderr,"ga_server:create failed\n",
                                             ga_nodeid_());
                            }
                            break;                          

          case GA_OP_DUP:   /* duplicate an array */
                            {
                              Integer g_a = MessageRcv->g_a , g_b;
                              char *array_name = "server_created";

                              /* duplicate can fail due to memory limits */
                              if(!ga_duplicate(&g_a, &g_b, array_name))
                                  fprintf(stderr,"ga_server:duplicate failed\n",
                                           ga_nodeid_());
                            }
                            break;                          

          case GA_OP_DES:   /* destroy an array */
                            if (! ga_destroy_(&MessageRcv->g_a))
                                  ga_error("ga_server: destroy failed", 
                                            MessageRcv->g_a);
                            break;                          

          case GA_OP_END:   /* terminate */
                            fprintf(stderr,"GA data server terminating\n");
                            ga_terminate_();
                            pend_();
                            exit(0);
                 default:   ga_error("ga_server: unknown request",ga_nodeid_());
      }

      (*NumRecReq)++;  /* increment Counter of Requests received and serviced */
      GA_POP_NAME;

#ifdef DATA_SERVER
   }while (MessageRcv->operation != GA_OP_END); 
#else
   in_handler = 0;
#endif
   /* fprintf(stderr,"leaving handler %d\n",nodeid_()); */
}



/*\ determines if calling process participates in collective comm
 *  also, parameters for a given communicator are found
\*/
Integer group_participate(me, root, up, left, right, group)
     Integer *me, *root, *up, *left, *right, group;
{
     Integer nproc, index;

     switch(group){
           case  ALL_GRP:  *root = 0;
                           *me = ga_nodeid_(); nproc = ga_nnodes_();
                           *up = (*me-1)/2;       if(*up >= nproc) *up = -1;
                           *left  =  2* *me + 1;  if(*left >= nproc) *left = -1;
                           *right =  2* *me + 2;  if(*right >= nproc)*right =-1;

                           break;
          case CLUST_GRP:  *root = cluster_master;
                           *me = nodeid_(); nproc = cluster_compute_nodes;
                           if(*me < *root || *me >= *root +nproc)
                                                return 0; /*does not*/ 

                           index  = *me  - *root;
                           *up    = (index-1)/2 + *root;  
                           *left  = 2*index + 1 + *root; 
                                    if(*left >= *root+nproc) *left = -1;
                           *right = 2*index + 2 + *root; 
                                    if(*right >= *root+nproc) *right = -1;

                           break;
      case ALL_CLUST_GRP:  *root = cluster_master;
                           *me = nodeid_(); nproc = cluster_nodes; /* +server*/
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
    case INTER_CLUST_GRP:  *root = SR_clus_info[0].masterid;
                           *me = nodeid_();  nproc = num_clusters;

                           if(*me != cluster_master) return 0; /*does not*/

                           *up    = (cluster_id-1)/2;
                           if(*up >= nproc) *up = -1;
                             else *up = SR_clus_info[*up].masterid;

                           *left  = 2*cluster_id+ 1;
                           if(*left >= nproc) *left = -1;
                             else *left = SR_clus_info[*left].masterid;

                           *right = 2*cluster_id+ 2;
                           if(*right >= nproc) *right = -1;
                             else *right = SR_clus_info[*right].masterid;

                           break;
                 default:  ga_error("group_participate: wrong group ", group);
     }
     return (1);
}




/*\ BROADCAST 
\*/
void ga_brdcst_clust(type, buf, len, originator, group)
     Integer type, len, originator, group;
     Void *buf;
{
#ifdef SYSV 
     Integer me, lenmes, sync=1, from, root=0;
     Integer up, left, right, participate;

     participate = group_participate(&me, &root, &up, &left, &right, group);

     /*  cannot exit just yet -->  send the data to root */

     if (originator != root ){
       if(me == originator) snd_(&type, buf, &len, &root, &sync); 
       if(me == root) rcv_(&type, buf, &len, &lenmes, &originator, &sync); 
     }

     if( ! participate) return;

     if (me != root)
       rcv_(&type, buf, &len, &lenmes, &up, &from, &sync);
     if (left > -1)
       snd_(&type, buf, &len, &left, &sync);
     if (right > -1)
       snd_(&type, buf, &len, &right, &sync);
#endif
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
        tcg_orig_master =  SR_clus_info[orig_clust].masterid;
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
        long gtype,gfrom,glen;
        gtype =(long) *type; gfrom =(long) *originator; glen =(long) *len;
#       ifdef SP1
            ga_sync_();
#       endif
            brdcst_(&gtype,buf,&glen,&gfrom);
#       ifdef SP1
            ga_sync_();
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
#ifdef SYSV 
#    define BUF_SIZE 10000
     Integer  me, lenmes, sync=1, from, lenbuf, root;
     DoublePrecision work[BUF_SIZE], *origx = x;
     static void ddoop();
     Integer ndo, up, left, right, orign = n;

     if( ! group_participate(&me, &root, &up, &left, &right, group)) return;

     while ((ndo = (n<=BUF_SIZE) ? n : BUF_SIZE)) {
	 lenbuf = lenmes = ndo*sizeof(DoublePrecision);

         if (left > -1) {
           rcv_(&type, (char *) work, &lenmes, &lenbuf, &left, &from, &sync);
           ddoop(ndo, op, x, work);
         }
         if (right > -1) {
           rcv_(&type, (char *) work, &lenmes, &lenbuf, &right, &from, &sync);
           ddoop(ndo, op, x, work);
         }
         if (me != root)
	   snd_(&type, x, &lenmes, &up, &sync); 

       n -=ndo;
       x +=ndo;
     }
     /* Now, root broadcasts the result down the binary tree */
     lenmes = orign*sizeof(DoublePrecision);
     ga_brdcst_clust(type, (char *) origx, lenmes, root, group);
#endif
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
            ga_sync_();
#       endif
        dgop_(&type, x, &n, op, (Integer)strlen(op));
#       ifdef SP1
            ga_sync_();
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
#ifdef SYSV
#    define BUF_SIZE 10000
     Integer  me, lenmes, sync=1, from, lenbuf, root=0 ;
     Integer work[BUF_SIZE], *origx = x;
     static void idoop();
     Integer ndo, up, left, right, orign =n;

     if( ! group_participate(&me, &root, &up, &left, &right, group)) return;

     while ((ndo = (n<=BUF_SIZE) ? n : BUF_SIZE)) {
	 lenbuf = lenmes = ndo*sizeof(Integer);

         if (left > -1) {
           rcv_(&type, (char *) work, &lenmes, &lenbuf, &left, &from, &sync);
	   idoop(ndo, op, x, work); 
         }
         if (right > -1) {
           rcv_(&type, (char *) work, &lenmes, &lenbuf, &right, &from, &sync);
	   idoop(ndo, op, x, work); 
         }
         if (me != root)
           snd_(&type, x, &lenmes, &up, &sync);

       n -=ndo;
       x +=ndo;
     }
     /* Now, root broadcasts the result down the binary tree */
     lenmes = orign*sizeof(Integer);
     ga_brdcst_clust(type, (char *) origx, lenmes, root, group);
#endif
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
            ga_sync_();
#       endif
        igop_(&type, x, &n, op, (Integer)strlen(op));
#       ifdef SP1
            ga_sync_();
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


/*\ WAIT until all request are serviced 
\*/
void ga_wait_server()
{
Integer outstanding, local_req; 
char sum='+';
   
   local_req = NumSndReq;

#  ifdef DATA_SERVER 
     /* add all requests sent by ga nodes in this cluster */
     ga_igop_clust(GA_TYPE_SYN, &local_req, 1, &sum, CLUST_GRP);

     if(nodeid_() == cluster_master){
#  endif
        do {

           /* *NumRecReq has the number of requests received by local server */
           outstanding = local_req - *NumRecReq; 

#          ifdef DATA_SERVER 
             ga_igop_clust(GA_TYPE_SYN, &outstanding, 1, &sum, INTER_CLUST_GRP);
#          else
             ga_igop(GA_TYPE_SYN, &outstanding, 1, &sum);
#          endif
        } while (outstanding != 0);

#  ifdef DATA_SERVER 
        /* now cluster master knows that there are no outstanding requests */
   }
#  endif
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


