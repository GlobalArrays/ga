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
#include "message.h"
#include <stdio.h>
#ifdef CRAY_T3D
#  include <fortran.h>
#endif


#define DEBUG 0 
#define ACK   0
   
/* send & receive buffers alligned on sizeof(double) boundary  */
#define ALGN_EXTRA PAGE_SIZE/sizeof(double) -1
double _snd_dbl_buf[MSG_BUF_DBL_SIZE+ALGN_EXTRA];
double _rcv_dbl_buf[MSG_BUF_DBL_SIZE+ALGN_EXTRA];
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
 *  through TCGMSG GA_clus_info
\*/
void    ClustInfoInit()
{
#ifndef SYSV
    num_clusters = ga_msg_nnodes_();
    cluster_id = cluster_master = ga_msg_nodeid_();
    cluster_compute_nodes  = cluster_nodes = 1;
#else
/*    void PrintClusInfo();*/
/*    if(ga_msg_nodeid_()==0 && DEBUG) PrintClusInfo();*/
    void init_message_interface();
    
    init_msg_interface();
    if(ga_msg_nnodes_()==1){
       num_clusters = 1;
       cluster_id =  0;
       cluster_master = 0;
       cluster_nodes  = 1;
    }else{
       num_clusters = (Integer)GA_n_clus;
       cluster_id =  GA_clus_id;
       cluster_master = GA_clus_info[cluster_id].masterid; 
       cluster_nodes  = GA_clus_info[cluster_id].nslave; 
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
           compute_nodes += GA_clus_info[clust].nslave -1;
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
       server = GA_clus_info[clust].masterid+ GA_clus_info[clust].nslave-1;
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
       list[clust] = GA_clus_info[clust].masterid+ GA_clus_info[clust].nslave-1;
#endif
}


/*\ determine msg-passing nodeid for the first GA <num_procs> processes 
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
    ga_msg_snd(GA_TYPE_REQ, (char*)MessageSnd, len, to);

#   if defined(DATA_SERVER)
     if(ACK) ga_msg_rcv(GA_TYPE_ACK, (char*)&ack, sizeof(ack), &len, to, &from);
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
   if(DEBUG) fprintf(stderr, "data server %d ready\n",ga_msg_nodeid_());
   do {
      Integer len, ack;
      len = TOT_MSG_SIZE; /* MSG_BUF_SIZE + MSG_HEADER_SIZE */ 
      ga_msg_rcv(GA_TYPE_REQ, (char*)MessageRcv, len, &msglen, -1, &from);
      if(ACK) ga_msg_snd(GA_TYPE_ACK, &ack, sizeof(ack), from);
#else
      extern Integer in_handler;
      in_handler = 1; /*distinguish cases when GA ops are called by the server*/
#endif

      if(DEBUG) fprintf(stderr, "server got request %d from %d\n",
                                MessageRcv->operation, from);
      /* fprintf(stderr, "server %d ready\n",ga_msg_nodeid_()); */

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
                            ga_msg_snd(GA_TYPE_GET, MessageRcv->buffer, msglen,
                               from);
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
                              ga_msg_snd(GA_TYPE_RDI, &rdi_val, sizeof(rdi_val),
                                         from);
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
                            ga_msg_snd(GA_TYPE_DGT, pvalue, 
                                       elem_size*nelem, from);
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
#                           ifdef MPI
                              MPI_Finalize();
#                           else
                              pend_();
#                           endif
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
   /* fprintf(stderr,"leaving handler %d\n",ga_msg_nodeid_()); */
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

     if(ga_msg_nodeid_() == cluster_master){
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



