/*
 * module: global.server.c
 * author: Jarek Nieplocha
 * date: Mon Dec 19 19:06:18 CST 1994
 * description: data server handler and configuration code
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
#include "macommon.h"
#include <stdio.h>
#ifdef CRAY_T3D
#  include <fortran.h>
#endif

Integer NumSndReq=0;
int in_handler =0;

#ifdef LAPI
#  include "lapidefs.h"
   extern void ga_header_handler();

   void ga_wait_cmpl()
   { 
/*
   int val;
            if(LAPI_Getcntr(lapi_handle, &cmpl_cntr, &val)) 
                                         ga_error("LAPI_Getcntr failed",0);
   fprintf(stderr," %d ga_wait_cmpl (%d, %d)\n",ga_nodeid_(),NumSndReq, val);

         do {
            if(LAPI_Getcntr(lapi_handle, &cmpl_cntr, &val)) 
                                         ga_error("LAPI_Getcntr failed",0);
         }while (NumSndReq > val);
*/
    int p;

    for (p=0;p<ga_nnodes_();p++) FENCE_NODE(p);
   }
#endif


#define DEBUG  0
#define DEBUG0 0
#define DEBUG1 0

#define ACK   0
   
/* send & receive buffers alligned on sizeof(double) boundary  */
#define ALGN_EXTRA PAGE_SIZE/sizeof(double) -1
double _snd_dbl_buf[MSG_BUF_DBL_SIZE+ALGN_EXTRA];
double _rcv_dbl_buf[MSG_BUF_DBL_SIZE+ALGN_EXTRA];
struct message_struct *MessageSnd = (struct message_struct*)_snd_dbl_buf,
                      *MessageRcv = (struct message_struct*)_rcv_dbl_buf;



Integer cluster_master;
Integer cluster_hidden_nodes;
Integer cluster_compute_nodes;
Integer cluster_server=-1;
Integer ClusterMode=0;
Integer cluster_nodes=0;

int GA_fence_set=0;
#if !(defined(CRAY_T3D) || defined(CONVEX) || defined(KSR))
    char fence_array[MAX_NPROC];
#   define  RECORD_FENCE(to, oper)\
      if(GA_fence_set && (oper == GA_OP_PUT || oper == GA_OP_ACC ||\
             oper == GA_OP_SCT || oper == GA_OP_RDI)) fence_array[to]=1
#else
#   define  RECORD_FENCE(to, oper)
#endif



/*\ determines cluster structure according to *.p file
 *  through TCGMSG GA_clus_info
\*/
void    ClustInfoInit()
{
    void init_message_interface();
    cluster_hidden_nodes=1;
    GA_n_clus = 1;

    init_msg_interface();
    
    if( GA_n_proc ==1){
       GA_clus_id =  0;
       cluster_master = 0;
       cluster_nodes  = 1;
    }else{
       cluster_master = GA_clus_info[GA_clus_id].masterid; 
       cluster_nodes  = GA_clus_info[GA_clus_id].nslave; 
    } 
    cluster_compute_nodes  = cluster_nodes;

    /*####*/
#   if defined(DATA_SERVER) || defined(IWAY) || defined(SOCKCONNECT)
       if(GA_n_clus>1 && cluster_nodes < cluster_hidden_nodes +1)
           ga_error("ClustInfoInit: minimum number of processes on cluster: ",
                    cluster_hidden_nodes+1);
       if(GA_n_clus > 1) {
          ClusterMode = 1;
          cluster_compute_nodes  -= cluster_hidden_nodes; 
          cluster_server = cluster_master + cluster_nodes-1;
       }
#   endif
    if(!ClusterMode) cluster_master = 0;

    if(DEBUG0){
       printf("me=%d master=%d server=%d cluster_nodes=%d\n", ga_msg_nodeid_(),
              cluster_master, cluster_server, cluster_nodes);
       fflush(stdout);
    }
}


Integer ClusterID(proc)
    Integer proc;
{
    Integer clust, compute_nodes=0;

    if(ClusterMode){
       /* start adding compute nodes in each cluster
        * when > proc => we found cluster => we know server
        */
       for(clust=0; clust< GA_n_clus; clust++){
           compute_nodes += GA_clus_info[clust].nslave - cluster_hidden_nodes;
           if(compute_nodes > proc) break;
       }
       return(clust);
    }else
       return(proc);
}


/*\ determines to which process (data server) we need to send request for proc
\*/
Integer DataServer(proc)
    Integer proc;
{
    Integer clust, server;
    if(ClusterMode){

       clust = ClusterID(proc);

       /* request goes directly to local node skipping server (SP+Intel) */
#      ifdef IWAY
          if(clust == GA_clus_id){
             /* GAid to MSGid works only for 2 clusters */
             if(clust ==1) proc++;
             return(proc);
          }
#      endif

#      ifndef DATA_SERVER
             /* on IWAY requests to remote cluster go through local server */
             clust =  GA_clus_id;
#      endif

       /* server is the last tcgmsg process in the cluster */
       server = GA_clus_info[clust].masterid + 
                GA_clus_info[clust].nslave   - cluster_hidden_nodes;
       return(server);
    }else
       return(proc);
}
       

/*\ returns the number of data_servers to the application
\*/
void ga_num_data_servers_(num)
     Integer *num;
{
    if(ClusterMode) *num = GA_n_clus;
    else *num = 0;
}



/*\ returns nodeid for all the data servers in user provided array "list"
 *  dimension of list has to be >= number returned from ga_num_data_servers()
\*/
void ga_list_data_servers_(list)
     Integer *list;
{
   int clust;
   if(ClusterMode)
     for(clust = 0; clust < GA_n_clus; clust ++)
       list[clust] = GA_clus_info[clust].masterid + 
                     GA_clus_info[clust].nslave   - cluster_hidden_nodes;
}


/*\ determine msg-passing nodeid for the first GA <num_procs> processes 
\*/
void ga_list_nodeid_(list, num_procs)
     Integer *list, *num_procs;
{
    Integer proc, msg_node, server;

    if(*num_procs < 1 || *num_procs > ga_nnodes_())
      ga_error("ga_list_nodeid: invalid number of GA num_procs ",*num_procs);

    if(ClusterMode){
       proc = msg_node = 0;
       server = DataServer(proc);
       for( proc = 0; proc < *num_procs; proc++){
          while(msg_node == server){
             msg_node ++;
             server = DataServer(proc); /* server id for GA proc */
          }
          list[proc] = msg_node;
          msg_node ++;
       }

    }else for( proc = 0; proc < *num_procs; proc++) list[proc]=proc;
}


void ga_checksum(void* data, int bytes, double *sum)
{
    int i;
    *sum =0;
    for(i=0;i<bytes;i+=sizeof(double))
        *sum += *(double*)(i + (char*)data);
}

int snum =0;


#define FILL_MSG_HEADER {\
        MessageSnd->g_a = g_a;\
        MessageSnd->ilo = ilo;\
        MessageSnd->ihi = ihi;\
        MessageSnd->jlo = jlo;\
        MessageSnd->jhi = jhi;\
        MessageSnd->to     = proc;\
        MessageSnd->from   = ga_nodeid_();\
        MessageSnd->type   = type;\
        MessageSnd->req_tag    = 77;\
        MessageSnd->operation  = oper;\
}


/*\  SEND REQUEST MESSAGE to the owner/data_server 
 *   to perform operation "oper" on g_a[ilo:ihi,jlo:jhi]
\*/
void ga_snd_req(g_a, ilo,ihi,jlo,jhi, nbytes, type, oper, proc, to)
     Integer g_a, ilo,ihi,jlo,jhi,nbytes,  type, oper, to, proc; 
{
    Integer  len, from;
#ifdef LAPI
    int rc, val;
    lapi_cntr_t *pcntr, *pcmpl_cntr;
#if defined(LAPI_SPLIT)
    MessageSnd->bytes = nbytes;
    MessageSnd->tag.cntr = &buf_cntr.cntr;
    MessageSnd->tag.buf  = MessageSnd->buffer;
#endif
#endif

    FILL_MSG_HEADER; 

#ifdef LAPI
      len = MSG_HEADER_SIZE;
      
      SET_COUNTER(buf_cntr,1); 
#     ifdef LAPI_SPLIT
            if(nbytes <= lapi_max_uhdr_data_sz ){
               len += nbytes;
               pcntr = &buf_cntr.cntr;
            } else
#     endif
               pcntr = NULL;

      /* completion counter should be incremented only for store ops */
      if(oper == GA_OP_GET || oper == GA_OP_GAT || 
         oper == GA_OP_LCK || oper == GA_OP_UNL  ) pcmpl_cntr=NULL;
      else pcmpl_cntr = &cmpl_arr[to].cntr; 

      if((rc=LAPI_Amsend(lapi_handle,(uint)to,ga_header_handler,MessageSnd,len, 
#     if defined(LAPI_MPI) || defined(LAPI_SPLIT)
              NULL, 0, 
#     else
              MessageSnd->buffer, nbytes,
#     endif
              NULL, pcntr, pcmpl_cntr )))
              ga_error("ga_snd_req: LAPI_Amsend failed",rc); 
#     if defined(LAPI_MPI)
              ga_msg_snd(GA_TYPE_REQ, MessageSnd->buffer, nbytes, to);
#     endif
#else
    len = nbytes + MSG_HEADER_SIZE;
    ga_msg_snd(GA_TYPE_REQ, (char*)MessageSnd, len, to);
#endif

    if(DEBUG0)fprintf(stderr,"GAme=%d request=%d to%d\n",ga_nodeid_(),oper,to);
    NumSndReq++; /* count requests sent */
    RECORD_FENCE(to, oper);
}


#include "mem.ops.h"

void ga_snd_req2D(g_a, ilo,ihi,jlo,jhi, type, oper, ptr_src, ld, proc, to)
     Integer g_a, ilo,ihi,jlo,jhi, type, oper, proc, to, ld;
     char *ptr_src;
{
    Integer  len, from, rows, cols, nbytes;
#ifdef LAPI
    int rc, val;
    lapi_cntr_t *pcntr,*pcmpl_cntr;
#endif

    rows = ihi-ilo+1;
    cols = jhi-jlo+1;
    nbytes = (rows*cols)*GAsizeofM(type);

#if defined(LAPI_SPLIT)
    MessageSnd->bytes = nbytes;
    MessageSnd->tag.cntr = &buf_cntr.cntr;
    MessageSnd->tag.buf  = MessageSnd->buffer;
#endif

    FILL_MSG_HEADER;

    Copy2D(type, &rows, &cols, ptr_src, &ld, (char *)MessageSnd->buffer,&rows);

#ifdef CHECKSUM
    ga_checksum(MessageSnd->buffer, nbytes, &(MessageSnd->checksum));
    if(DEBUG) fprintf(stderr,"%d checksum=%f (%d)\n",ga_nodeid_(),MessageSnd->checksum,snum++);
#endif

#ifdef LAPI
      len = MSG_HEADER_SIZE;
      SET_COUNTER(buf_cntr,1);
#     ifdef LAPI_SPLIT
            if(nbytes <= lapi_max_uhdr_data_sz ){
               len += nbytes;
               pcntr = &buf_cntr.cntr;
               if(DEBUG) fprintf(stderr,"short AM len=%d\n",len);
            } else
#     endif
               pcntr = NULL;

      /* completion counter should be incremented only for store ops */
      if(oper == GA_OP_GET || oper == GA_OP_GAT) pcmpl_cntr=NULL;
      else pcmpl_cntr = &cmpl_arr[to].cntr; 

      if((rc=LAPI_Amsend(lapi_handle,(uint)to,ga_header_handler,MessageSnd,len,
#     if defined(LAPI_MPI) || defined(LAPI_SPLIT)
              NULL, 0,
#     else
              MessageSnd->buffer, nbytes,
#     endif
              NULL, pcntr, pcmpl_cntr )))
              ga_error("ga_snd_req: LAPI_Amsend failed",rc);
#     if defined(LAPI_MPI)
              ga_msg_snd(GA_TYPE_REQ, MessageSnd->buffer, nbytes, to);
#     endif
#else
    len = nbytes + MSG_HEADER_SIZE;
    ga_msg_snd(GA_TYPE_REQ, (char*)MessageSnd, len, to);
#endif

    if(DEBUG0)fprintf(stderr,"GAme=%d request=%d to%d\n",ga_nodeid_(),oper,to);
    NumSndReq++; /* count requests sent */
    RECORD_FENCE(to, oper);
}


void ga_update_serv_num()
{
   GAstat.numser ++;
}


/*\ DATA SERVER services remote requests
 *       . invoked by interrupt-receive message as a server thread in addition 
 *         to the application thread which it might be suspended or run
 *         concurrently (Paragon), or
 *       . dedicated data-server node that loops here servicing requests
 *         since this routine is called in ga_initialize() until terminate
 *         request GA_OP_END (sent in ga_terminate() ) is received
\*/
void ga_SERVER(from, message)
     Integer from;
     struct message_struct *message;
{
Integer msglen, ld, offset = 0, rdi_val, elem_size, nelem, toproc,ack;
char    *piindex, *pjindex, *pvalue;
void ga_server_lock(Integer handle, Integer id, Integer node, msg_tag_t tag);
void ga_server_unlock(Integer handle, Integer id, Integer node, Integer next);
void    ga_acc_local(), ga_scatter_local(), ga_gather_local();
Integer ga_read_inc_local();

#     ifndef LAPI
         GA_PUSH_NAME("ga_server"); /*not thread-safe */
#     endif

#     if   defined(SOCKCONNECT) && defined(IWAY) && !defined(SHMEM)
          /* adjust process number for msg sender -- from is set by NX/MPL */
          from += cluster_master;
#     endif

#ifdef DATA_SERVER
   if(DEBUG) fprintf(stderr, "data server %d ready\n",ga_msg_nodeid_());
   do {
      Integer len;
      len = TOT_MSG_SIZE; /* MSG_BUF_SIZE + MSG_HEADER_SIZE */ 
      ga_msg_rcv(GA_TYPE_REQ, (char*)message, len, &msglen, -1, &from);
      if(ACK) ga_msg_snd(GA_TYPE_ACK, &ack, sizeof(ack), from);
#endif

      if(DEBUG0) fprintf(stderr,"%d>ga_server got request %d from %d GAto=%d\n",
                                ga_msg_nodeid_(),message->operation, from,
                                message->to);

      elem_size = GAsizeof(message->type);
      toproc =  message->to;

#ifndef DATA_SERVER
      if(from != message->from){
         printf("from=%d message->from=%d\n",from,message->from);
         ga_error("server error: from=", message->from);
      }
#endif

#     ifdef IWAY
        /* set up header for the response message */
        message->to = message->from;
        message->from = toproc;
#     endif

      if(message->operation == GA_OP_GET || message->operation == GA_OP_PUT ||
         message->operation == GA_OP_ACC){
           msglen = (message->ihi - message->ilo +1)
                  * (message->jhi - message->jlo +1) * elem_size;
           if(msglen > MSG_BUF_SIZE) 
                ga_error("ga_server: msgbuf overflow ", msglen);

           /* leading dimension for buf */
           ld = message->ihi - message->ilo +1;
      }

      switch ( message->operation) {
          case GA_OP_GET:   /* get */
                            ga_check_handle(&message->g_a,"server: ga_get");
                            ga_get_local( message->g_a,
                               message->ilo, message->ihi,
                               message->jlo, message->jhi,  
                               message->buffer, offset, ld, toproc);

#                           ifdef LAPI
                               ga_lapi_send(message->tag, 
                                            message->buffer, msglen, from);
#                           elif defined(IWAY)
                               /* need header */
                               ga_msg_snd(GA_TYPE_GET, message, 
                                       msglen+MSG_HEADER_SIZE, from);
#                           else
                               ga_msg_snd(GA_TYPE_GET, message->buffer, 
                                       msglen, from);
#                           endif
                            break;

          case GA_OP_PUT:   /* put */
                            ga_check_handle(&message->g_a,"server: ga_put");
                            ga_put_local( message->g_a,
                               message->ilo, message->ihi,
                               message->jlo, message->jhi,
                               message->buffer, offset, ld, toproc);
                            break;

          case GA_OP_ACC:   /* accumulate */
                            ga_check_handle(&message->g_a,"server: ga_acc");
                            ga_acc_local( message->g_a,
                               message->ilo, message->ihi,
                               message->jlo, message->jhi,
                               message->buffer, offset, ld, toproc,
                               message->alpha);
                            break;

          case GA_OP_RDI:   /* read and increment */
                            {
                              Integer inc = message->ihi;
                              ga_check_handle(&message->g_a,"server:ga_rdi");
                              rdi_val = ga_read_inc_local( message->g_a,
                                 message->ilo, message->jlo, inc, toproc);

                              message->ilo = rdi_val;
                              ga_msg_snd(GA_TYPE_RDI, message, 
                                         MSG_HEADER_SIZE, from);
                            }
                            break;
          case GA_OP_SCT:   /* scatter */
                            ga_check_handle(&message->g_a,"server:ga_scat");

                            /* buffer contains (val,i,j) */
                            nelem = message->ilo;
                            if (nelem > MSG_BUF_SIZE/
                               (elem_size+2*sizeof(Integer)) )
                                  ga_error("ga_server: scatter overflows buf ",
                                            nelem);
                            pvalue  = (char*)message->buffer;
                            piindex = ((char*)message->buffer) + 
                                      nelem*elem_size;
                            pjindex = piindex + nelem*sizeof(Integer);
                            ga_scatter_local(message->g_a, pvalue,
                                             piindex, pjindex, nelem, toproc);
                            break;

          case GA_OP_GAT:   /* gather */
                            ga_check_handle(&message->g_a,"server:ga_gath");

                            /* rcv buffer contains (i,j) only but we also
                             * need space in the same buffer for v
                             * value will be sent by server in rcv buffer 
                             */
                            nelem = message->ilo;
                            if (nelem > MSG_BUF_SIZE/
                               (elem_size+2*sizeof(Integer)) )
                                  ga_error("ga_server: gather overflows buf ",
                                            nelem);

                            piindex = (char*)message->buffer;  
                            pjindex = piindex + nelem*sizeof(Integer);
                            pvalue  = pjindex + nelem*sizeof(Integer);

                            ga_gather_local(message->g_a, pvalue,
                                            piindex, pjindex, nelem, toproc);

#                           ifdef LAPI
                               ga_lapi_send(message->tag, 
                                            pvalue, elem_size*nelem,from); 
#                           elif defined(IWAY)
                              /* redundant copy: for IWAY need header */
                              Copy(pvalue, message->buffer, elem_size*nelem);
                              ga_msg_snd(GA_TYPE_GAT, message,
                                       elem_size*nelem+MSG_HEADER_SIZE, from);
#                           else
                              ga_msg_snd(GA_TYPE_GAT, pvalue,
                                       elem_size*nelem, from);
#                           endif
                            break;                          

          case GA_OP_CRE:   /* create an array */
                            {
                              Integer dim1   = message->ilo;
                              Integer nblock1 = message->ihi;
                              Integer dim2 = message->jlo;
                              Integer nblock2 = message->jhi;
                              char *map1 = message->buffer+SHMID_BUF_SIZE;
                              char *map2 = map1 + nblock1*sizeof(Integer);
                              char *array_name = "server_created";
                              Integer g_a;

                              /* create can fail due to memory limits */
                              if(!ga_create_irreg(& message->type, 
                                  &dim1, &dim2, array_name, (Integer*) map1, 
                                  &nblock1, (Integer*) map2, &nblock2, &g_a))
                                  fprintf(stderr,"%d ga_server:create failed\n",
                                             ga_nodeid_());
                            }
                            break;                          

          case GA_OP_DUP:   /* duplicate an array */
                            {
                              Integer g_a = message->g_a , g_b;
                              char *array_name = "server_created";

                              /* duplicate can fail due to memory limits */
                              if(!ga_duplicate(&g_a, &g_b, array_name))
                                  fprintf(stderr,"%d duplicate failed\n",
                                           ga_nodeid_());
                            }
                            break;                          

          case GA_OP_DES:   /* destroy an array */
                            if (! ga_destroy_(&message->g_a))
                                  fprintf(stderr,"ga_server:destroy failed%d\n",
                                            message->g_a);
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

          case GA_OP_ACK:   /* acknowledge completion of previous requests */
                            /* Note that since messages are not overtaking,
                             * all requests from the client must be already done
                             * It does not apply to LAPI
                             */
                            ga_msg_snd(GA_TYPE_ACK, &ack, 0, from);
                            break;                          

          case GA_OP_LCK:   /* acquire lock on behalf client node*/
                            ga_server_lock(message->g_a,message->ilo,from,
#                              ifdef LAPI
                                           message->tag); 
#                              else
                                           GA_TYPE_LCK);  
#                              endif
                            break;

          case GA_OP_UNL:   /* release lock held  the node; g_a = id */
                            ga_server_unlock(message->g_a,message->ilo,
                                             from,message->ihi);
                            break;


                 default:   ga_error("ga_server: unknown request",ga_nodeid_());
      }

#ifndef LAPI
      (*NumRecReq)++;  /*increment Counter of Requests serviced--thread-unsafe*/
      if(DEBUG)fprintf(stderr,"%d received %d\n", ga_nodeid_(), *NumRecReq);
#endif

#ifdef DATA_SERVER
   }while (message->operation != GA_OP_END); 
#endif

#ifndef LAPI
    GA_POP_NAME;
#endif

    if(DEBUG)fprintf(stderr,"leaving handler %d\n",ga_msg_nodeid_()); 
}


/*\ initialize tracing of request completion
\*/
void ga_init_fence_()
{
    Integer proc;
    GA_fence_set++;
#if defined(KSR) || defined(CONVEX) || defined(CRAY_T3D) || defined(CRAY_T3E)
#else
# ifdef SYSV
       proc = ga_msg_nnodes_()-1;
#   else
       proc = ga_nnodes_()-1;
# endif
    /* initialize array if fence has not been initialized  */
    if(GA_fence_set==1)for(;proc>=0; proc --) fence_array[proc]=0;
#endif
}


/*\ wait until requests intiated by calling process are completed
\*/
void ga_fence_()
{
    Integer proc;
    if(GA_fence_set<1)ga_error("ga_fence: fence not initialized",0);
    GA_fence_set--;
#if defined(CRAY_T3D) || defined(CRAY_T3E)
    shmem_quiet();
#elif defined(CONVEX) || defined(KSR)
    return;
#else
#   if defined(SYSV)
       proc = ga_msg_nnodes_()-1;
#   else
       proc = ga_nnodes_()-1;
#   endif
    for(;proc >= 0; proc--) if(fence_array[proc]){
#      ifdef LAPI
         FENCE_NODE(proc);
#      else
         Integer dummy;
         fence_array[proc]=0;
         ga_snd_req(0, 0, 0, 0, 0, 0, 0, GA_OP_ACK, 0, proc);
         ga_msg_rcv(GA_TYPE_ACK, &dummy, 0, &dummy, proc, &dummy);
#      endif
    }
#endif
}



#ifdef IWAY

#  define SERVER_TYPES 8
Integer server_type_array[SERVER_TYPES] = {
        GA_TYPE_REQ, GA_TYPE_GET, GA_TYPE_RDI,
        GA_TYPE_GAT, GA_TYPE_SYN,
        GA_TYPE_GOP, GA_TYPE_BRD};

void ga_server_handler()
{
  Integer msglen, len, to_proc, to_clust, from, type;
  int from_server;
  int msg_available;
  Integer other_server = GA_clus_info[1 - GA_clus_id].masterid +
                         GA_clus_info[1 - GA_clus_id].nslave -1;
 
  Integer ONE=1;
  static Integer  other_server_sync=0, sync_msg_received=0, sync_msglen=0;

  printf("GA:%d MSG:%d running server handler\n",ga_nodeid_(),ga_msg_nodeid_());
  fflush(stdout);
  

  
/*     setdbg_(&ONE);*/

     /* wait for any message */
     while(1) { 
        if(DEBUG){
             printf("%s:%d> starting cycle\n",GA_clus_info[GA_clus_id].hostname,
                    ga_msg_nodeid_());
             fflush(stdout);
        }

        for(msg_available=0; !msg_available; ){
             int i;
             /* some msg-passing libs do not support probing for "any type" */
             for(i=0; i< SERVER_TYPES; i++){ 
                 type = server_type_array[i];
                 if(ga_msg_probe(type, -1)){ 
                   len = (Integer)TOT_MSG_SIZE;
                   ga_msg_rcv(type, (char*)MessageRcv, len, &msglen, -1, &from);
                   msg_available=1;
                   break; 
                 }
             }
        }
    
        if(type != GA_TYPE_SYN && type != GA_TYPE_GOP && type != GA_TYPE_BRD){ 
                                  to_clust=  ClusterID(MessageRcv->to);
                                  to_proc =  MessageRcv->to; 
/*                                  from = MessageRcv->from;*/
             if(to_clust==1) to_proc++; /* adjust GA to MSG numbering*/ 
             if(DEBUG0){
                   printf("cluster_server: GAto=%d ADJto=%d op=%d GAfrom=%d\n",
                           MessageRcv->to, to_proc,  MessageRcv->operation,  
                           MessageRcv->from);
                   fflush(stdout);
             }

             if(to_proc<0 || to_proc >= GA_n_proc )
                          ga_error("ga_server_handler: wrong to_proc",to_proc);

             if(from<0 || from >= GA_n_proc )
                          ga_error("ga_server_handler: wrong msg from",from);
        }else{

             if(from == other_server) to_proc = cluster_master;
             else if(from == cluster_master) to_proc = other_server;
             else ga_error("ga_server_handler: Unexpected msg sender",from);
        }

        if(DEBUG0){
           printf("%s:%d>server got message type=%d from=%d to=%d other_s=%d\n",
                 GA_clus_info[GA_clus_id].hostname, ga_msg_nodeid_(), 
                 type, from, to_proc,other_server);
           fflush(stdout);
        }

        switch (type) {

        case GA_TYPE_REQ: 

             if (to_clust == GA_clus_id){
                /* to_proc is in my cluster ==> must service the request */ 

                if(DEBUG0){
                      printf("%s:%d> server EXEC request %d GAto=%d op=%d\n",
                      GA_clus_info[GA_clus_id].hostname, ga_msg_nodeid_(),type,
                      MessageRcv->to, MessageRcv->operation);
                      fflush(stdout);
                }

#               ifdef SHMEM
                      /* service the request */
                      ga_SERVER(other_server); /* only for shared memory */
#               else
                      if( MessageRcv->operation == GA_OP_END){
                          void server_close();
                          sleep(5);
                          server_close();
                          ga_SERVER(0); /* terminate */
                      }
                      /* forward the message to the process it is intended */
                      ga_msg_snd(type, (char*)MessageRcv, msglen, to_proc);
#                     ifdef SP1
                        /* place to fix get/rdi/gath */
#                     endif
#               endif

             } else {

                /* to_proc is not in my cluster ==> must forward the request */ 
                if(DEBUG0){
                      printf("%s:%d> server forwarding request to remote t%d\n",
                      GA_clus_info[GA_clus_id].hostname, ga_msg_nodeid_(),type);
                      fflush(stdout);
                }
                ga_msg_snd(type, (char*)MessageRcv, msglen, other_server);

             }
             break;

        case GA_TYPE_GET:
        case GA_TYPE_RDI:
        case GA_TYPE_GAT:
             if(DEBUG0){
                printf("%s:%d> server forwarding response %d from=%d\n",
                GA_clus_info[GA_clus_id].hostname,ga_msg_nodeid_(),type,from);
                fflush(stdout);
             }

             if (to_clust == GA_clus_id)
                 ga_msg_snd(type, (char*)MessageRcv, msglen, to_proc);
             else
                 ga_msg_snd(type, (char*)MessageRcv, msglen, other_server);

             break;

        case GA_TYPE_SYN:
             if(DEBUG){
                printf("%s:%d> server received SYN %d from=%d\n",
                GA_clus_info[GA_clus_id].hostname,ga_msg_nodeid_(),type,from);
                fflush(stdout);
             }

             if(from == cluster_master){
                if(sync_msg_received)
                   ga_error("ga_server_handler: cluster double synch ",0); 

                /* do not respond yet !!! */ 
                ga_msg_snd(type, (char*)MessageRcv, msglen, other_server);
                sync_msg_received = 1; 
                sync_msglen = msglen;
             }else{
                if(other_server_sync)
                   ga_error("ga_server_handler: server double synch ",0); 

                other_server_sync = 1;
             }

             /* notify node that everybody synchronized */
             if(sync_msg_received && other_server_sync){
                  ga_msg_snd(type,(char*)MessageRcv,sync_msglen,cluster_master);
                  sync_msg_received = 0; 
                  other_server_sync = 0;
             }
             break;

        case GA_TYPE_GOP: 
        case GA_TYPE_BRD:

             if(DEBUG){
                printf("%s:%d> server forwarding GOP/BRD %d to=%d\n",
                GA_clus_info[GA_clus_id].hostname,ga_msg_nodeid_(),type,to_proc);
                fflush(stdout);
             }
             ga_msg_snd(type, (char*)MessageRcv, msglen, to_proc);
             break;

        default: 
             ga_error("ga_server_handler: Received unexpected message", type);
        }

        if(DEBUG){
             printf("%s:%d> cycle done sync: master=%d other=%d\n", 
                    GA_clus_info[GA_clus_id].hostname,
                    ga_msg_nodeid_(),sync_msg_received, other_server_sync);
             fflush(stdout);
        }
     }
}
     
            
#endif          





/*\ WAIT until all requests are serviced 
\*/
void ga_check_req_balance()
{
Integer outstanding, local_req=NumSndReq; 
char sum='+';

#  if defined(LAPI) || defined(IWAY)
     ga_error("ga_check_req_balance: does not work on this platform",0);
#  endif
   
#  if defined(DATA_SERVER)
       /* add all requests sent by ga nodes in this cluster */
       ga_igop_clust(GA_TYPE_SYN, &local_req, 1, &sum, CLUST_GRP);

       if(ga_msg_nodeid_() == cluster_master){
#  endif

     /* *NumRecReq has the number of requests received by local server */
      outstanding = local_req - *NumRecReq; 

#     if defined(DATA_SERVER)
             ga_igop_clust(GA_TYPE_SYN, &outstanding, 1, &sum, INTER_CLUST_GRP);
#     else
             ga_igop(GA_TYPE_SYN, &outstanding, 1, &sum);
#     endif

      if(outstanding != 0) 
         ga_error("ERROR:ga_check_req_balance: mismatch detected!",outstanding);

#  if defined(DATA_SERVER)
        /* now cluster master knows that there are no outstanding requests */
     }
#  endif


}




/*\ WAIT until all requests are serviced 
\*/
void ga_wait_server()
{
Integer outstanding, local_req=NumSndReq; 
char sum='+';

#  ifdef LAPI
   ga_error("ga_wait_server: does not work under LAPI -relies on NumRecReq",0);
#  endif
   
#  if defined(IWAY)
      Integer msglen, from;

      /* notify server that everbody in cluster ready to synchronize */
      if(ClusterMode && ga_msg_nodeid_() == cluster_master){
         ga_msg_snd(GA_TYPE_SYN,&local_req, 1, cluster_server);

         /* wait for signal that other cluster is also synchronizing */
         ga_msg_rcv(GA_TYPE_SYN,&local_req, 1,&msglen,cluster_server,&from);
      }

#  else

#    if defined(DATA_SERVER)
       /* add all requests sent by ga nodes in this cluster */
       ga_igop_clust(GA_TYPE_SYN, &local_req, 1, &sum, CLUST_GRP);

       if(ga_msg_nodeid_() == cluster_master){
#    endif

        do {

           /* *NumRecReq has the number of requests received by local server */
           outstanding = local_req - *NumRecReq; 

#          if defined(DATA_SERVER)
             ga_igop_clust(GA_TYPE_SYN, &outstanding, 1, &sum, INTER_CLUST_GRP);
#          else
             ga_igop(GA_TYPE_SYN, &outstanding, 1, &sum);
#          endif
        } while (outstanding != 0);

#    if defined(DATA_SERVER)
        /* now cluster master knows that there are no outstanding requests */
     }
#    endif
#  endif


}

