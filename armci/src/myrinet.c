
/* DISCLAIMER
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


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "myrinet.h"
#include "armcip.h"

#define DEBUG_ 0
#define DEBUG_INIT_ 0

armci_gm_proc_t *proc_gm;
armci_gm_serv_t *serv_gm;

char *MessageSndBuffer;
char *MessageRcvBuffer;

armci_gm_context_t *armci_gm_context, *armci_gm_serv_context;

/*********************************************************************
                        UTILITY FUNCTIONS                            
 *********************************************************************/

/* check memory */
void wait_flag_updated(long *buf, int val)
{
    long res;

    res = check_flag(buf);
    while(res != (long)val) {
        res = check_flag(buf);
    }
    
}

/*********************************************************************
                           COMPUTING PROCESS                            
 *********************************************************************/

/* pre-allocate required memory at start up*/
int armci_gm_proc_mem_alloc()
{
    /* allocate buf keeping the pointers of server ack buf */
    proc_gm->serv_ack_ptr = (long *)calloc(armci_nclus, sizeof(long));
    if(!proc_gm->serv_ack_ptr) return FALSE;

    /* allocate buf keeping the pointers of server MessageRcvBuffer */
    proc_gm->serv_buf_ptr = (long *)calloc(armci_nclus, sizeof(long));
    if(!proc_gm->serv_buf_ptr) return FALSE;
    
    /* allocate send call back context */
    armci_gm_context = (armci_gm_context_t *)malloc(1 * 
                       sizeof(armci_gm_context_t));
    if(armci_gm_context == NULL) return FALSE;

    /* allocate send buffer */
    MessageSndBuffer = (char *)gm_dma_malloc(proc_gm->port, MSG_BUFLEN);
    if(MessageSndBuffer == 0) 
         return FALSE;
    
    return TRUE;
}

/* deallocate the preallocated memory used by gm */
int armci_gm_proc_mem_free()
{
    free(proc_gm->serv_ack_ptr);
    free(proc_gm->serv_buf_ptr);

    free(armci_gm_context);

    gm_dma_free(proc_gm->port, MessageSndBuffer);
        
    return TRUE;
}

/* initialization of computing process */
int armci_gm_proc_init()
{
    int i;
    int status;
    
    /* allocate gm data structure for computing process */
    proc_gm = (armci_gm_proc_t *)malloc(1 * sizeof(armci_gm_proc_t));
    if(proc_gm == NULL) {
        fprintf(stderr, "%d: Error allocate proc data structure.\n",
                armci_me);
        return FALSE;
    }
    proc_gm->node_map = (int *)calloc(armci_nproc, sizeof(int));
    if(proc_gm->node_map == NULL) {
        fprintf(stderr, "%d: Error allocate proc data structure.\n",
                armci_me);
        return FALSE;
    }

    /* use existing MPI port */
    proc_gm->port = gmpi_gm_port;

    /* get my node id */
    status = gm_get_node_id(proc_gm->port, &(proc_gm->node_id));
    if(status != GM_SUCCESS) {
        fprintf(stderr, "%d: Could not get node id\n", armci_me);
        return FALSE;
    }
    if(DEBUG_INIT_) fprintf(stdout, "%d: PROC node id is %d\n",
                            armci_me, proc_gm->node_id);

    /* broadcasting my node id to other processes */
    proc_gm->node_map[armci_me] = proc_gm->node_id;
    armci_msg_barrier();
    armci_msg_igop(proc_gm->node_map, armci_nproc, "+");

    /* allow direct send */
    status = gm_allow_remote_memory_access(proc_gm->port);
    if(status != GM_SUCCESS) {
        fprintf(stderr, "%d: PROC could not enable direct sends\n",
                armci_me);
        return FALSE;
    }

    /* memory preallocation for computing process */
    if(!armci_gm_proc_mem_alloc()) {
        fprintf(stderr, "%d: PROC failed allocating memory\n",
                armci_me);
        return FALSE;
    }
    
    return TRUE;
}

/* call back func of gm_send_with_callback */
void armci_proc_callback(struct gm_port *port, void *context,
                         gm_status_t status)
{
    if(status == GM_SUCCESS)
        ((armci_gm_context_t *)context)->done = ARMCI_GM_SENT;
    else ((armci_gm_context_t *)context)->done = ARMCI_GM_FAILED;
}

/* trigue the gm_unknown, so that callback func can be executed */
void armci_proc_call_gm_unknown()
{
    MPI_Status status;
    int flag;
    
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
}

/* computing process start initial communication with all the servers */
void armci_client_create_connection_gm()
{
    int i;
    int server_mpi_id, size;

    /* make initial conection to the server, not the server in this node */
    for(i=0; i<armci_nclus; i++) {
        if(armci_clus_me != i) {
            server_mpi_id = armci_clus_info[i].master;
            ((long *)(MessageSndBuffer))[0] = ARMCI_GM_CLEAR;
            ((long *)(MessageSndBuffer))[1] = armci_me;
            ((long *)(MessageSndBuffer))[2] = (long)(MessageSndBuffer);
            ((long *)(MessageSndBuffer))[3] = ARMCI_GM_CLEAR;

            /* currently the tag is not used, just set some dummy number */
            /* armci_gm_context->tag = 100 + armci_me; */
            
            /* wait til the last sending done, either successful or failed */
            while(armci_gm_context->done == ARMCI_GM_SENDING);
            armci_gm_context->done = ARMCI_GM_SENDING;

            size = gm_min_size_for_length(4*sizeof(long));
            
            gm_send_with_callback(proc_gm->port,
                    MessageSndBuffer+sizeof(long), size, 2*sizeof(long),
                    GM_LOW_PRIORITY, proc_gm->node_map[server_mpi_id],
                    ARMCI_GM_SERVER_RCV_PORT, armci_proc_callback, 
                    armci_gm_context);

            /* blocking: wait til the send is done by calling the callback */
            while(armci_gm_context->done == ARMCI_GM_SENDING)
                armci_proc_call_gm_unknown();

            if(armci_gm_context->done == ARMCI_GM_FAILED)
                armci_die(" failed to make connection with server",
                          server_mpi_id);

            if(DEBUG_INIT_)
                fprintf(stdout,
                        "%d: sent 1st msg to server %d waiting reply at %d\n",
                        armci_me, server_mpi_id, MessageSndBuffer);

            /* wait til the serv_ack_ptr has been updated */
            wait_flag_updated((long *)MessageSndBuffer, ARMCI_GM_COMPLETE);
            wait_flag_updated((long *)MessageSndBuffer+3, ARMCI_GM_COMPLETE);
            
            proc_gm->serv_ack_ptr[i] = ((long *)MessageSndBuffer)[1];
            proc_gm->serv_buf_ptr[i] = ((long *)MessageSndBuffer)[2];
            
            /* send back the ack to server */
            ((long *)MessageSndBuffer)[0] = ARMCI_GM_ACK;
            armci_gm_context->done = ARMCI_GM_SENDING;
            if(DEBUG_INIT_) {
                fprintf(stdout, "%d: rcvd first msg from server %d.\n",
                        armci_me, server_mpi_id);
                fprintf(stdout, "%d: sending back ack to server %d at %d\n",
                        armci_me, server_mpi_id, proc_gm->serv_ack_ptr[i]);
                fflush(stdout);
            }
            
            gm_directed_send_with_callback(proc_gm->port, MessageSndBuffer,
                     (gm_remote_ptr_t)(gm_up_t)(proc_gm->serv_ack_ptr[i]),
                     sizeof(long), GM_LOW_PRIORITY,
                     proc_gm->node_map[server_mpi_id], 
                     ARMCI_GM_SERVER_RCV_PORT, armci_proc_callback, 
                     armci_gm_context);

            /* blocking: wait til the send is done by calling the callback */
            while(armci_gm_context->done == ARMCI_GM_SENDING)
                armci_proc_call_gm_unknown();
            
            if(armci_gm_context->done == ARMCI_GM_FAILED)
                armci_die(" failed sending ack to server", server_mpi_id);

            if(DEBUG_INIT_)
                fprintf(stdout, "%d: connected to server %d\n",
                        armci_me, server_mpi_id);
        }
    }
}

/* used regular gm_send_with_call_back to send message
 * assumption: the buffer is pinned and most probably is MessageSndBuffer
 */
int armci_dma_send_gm(int dst, char *buf, int len)
{
    int size;
    request_header_t *msginfo = (request_header_t *)buf;
    
    size = gm_min_size_for_length(len);
    
    /* prepare the callback context */
    /* currently the tag is not used */
    /* armci_gm_context->tag = 1000 + armci_me; */

    armci_gm_context->done = ARMCI_GM_SENDING;

    /* set the message tag */
    msginfo->tag.data_ptr = (void *)(buf + sizeof(request_header_t)
                                     - sizeof(long));
    msginfo->tag.ack = ARMCI_GM_CLEAR;
    
    gm_send_with_callback(proc_gm->port, buf, size, len, GM_LOW_PRIORITY,
                          proc_gm->node_map[dst], ARMCI_GM_SERVER_RCV_PORT,
                          armci_proc_callback, armci_gm_context);
    
    /* blocking: wait til the send is done by calling the callback */
    while(armci_gm_context->done == ARMCI_GM_SENDING) 
        armci_proc_call_gm_unknown();
    return(armci_gm_context->done);
}

/* check if data is available in the buffer
 * assume the buf is pinned and is inside MessageSndBuffer
 * format buf = hdr ack + data + tail ack
 */
char *armci_ReadFromDirect(char *buf, int len)
{
    int msglen;    
    request_header_t  *msginfo = (request_header_t *)buf;

    /* check the header ack */
    wait_flag_updated(&(msginfo->tag.ack), ARMCI_GM_COMPLETE);
    /* reset header ack */
    msginfo->tag.ack = ARMCI_GM_CLEAR;

    buf += sizeof(request_header_t);
    
    /* check the tail ack */
    wait_flag_updated((long *)(buf + len), ARMCI_GM_COMPLETE);
    /* reset tail ack */
    *(long *)(buf + len) = ARMCI_GM_CLEAR;

    return(buf);
}

#if 0
/* send notice to server the message has been received */
/* not used currently */
void armci_NotifyDirect(int dst)
{
    int srid = armci_clus_id(dst);
    
    ((long *)MessageSndBuffer)[0] = ARMCI_GM_ACK;
    armci_gm_context->done = ARMCI_GM_SENDING;
    if(DEBUG_) {
        fprintf(stdout, "%d: sending ack to server %d (proc %d) @%d\n",
                armci_me, srid, dst, proc_gm->serv_ack_ptr[srid]);
        fflush(stdout);
    }
    
    gm_directed_send_with_callback(proc_gm->port, MessageSndBuffer,
                     (gm_remote_ptr_t)(gm_up_t)(proc_gm->serv_ack_ptr[srid]),
                     sizeof(long), GM_LOW_PRIORITY,
                     proc_gm->node_map[dst], ARMCI_GM_SERVER_RCV_PORT,
                     armci_proc_callback, armci_gm_context);

    /* blocking: wait til the send is done by calling the callback */
    while(armci_gm_context->done == ARMCI_GM_SENDING) 
        armci_proc_call_gm_unknown();
            
    if(armci_gm_context->done == ARMCI_GM_FAILED)
        armci_die(" failed sending ack to server", srid);
}
#endif

/*********************************************************************
                           SERVER SIDE                            
 *********************************************************************/

/* preallocate required memory at the startup */
int armci_gm_serv_mem_alloc()
{
    int i;
    
    /* allocate dma buffer for low priority */
    for(i=ARMCI_GM_MIN_MESG_SIZE; i<=ARMCI_GM_MAX_MESG_SIZE; i++) {
        serv_gm->dma_buf[i] = (char *)gm_dma_malloc(serv_gm->rcv_port,
                                        gm_max_length_for_size(i));
        if(serv_gm->dma_buf[i] == 0) return FALSE;
    }

    /* allocate ack buffer for each server process */
    serv_gm->ack_buf = (long *)gm_dma_malloc(serv_gm->rcv_port,
                                               armci_nproc*sizeof(long));
    if(serv_gm->ack_buf == 0) return FALSE;

    /* allocate recv buffer */
    MessageRcvBuffer = (char *)gm_dma_malloc(serv_gm->snd_port, MSG_BUFLEN);
    if(MessageRcvBuffer == 0) return FALSE;

    /* allocate buf for keeping the pointers of client MessageSndbuffer */
    serv_gm->proc_buf_ptr = (long *)calloc(armci_nproc, sizeof(long));
    if(!serv_gm->proc_buf_ptr) return FALSE;

    /* allocate server send call back context */
    armci_gm_serv_context = (armci_gm_context_t *)malloc(1 * 
					 sizeof(armci_gm_context_t));
    if(armci_gm_serv_context == NULL) return FALSE;
    
    return TRUE;
}

/* deallocate the preallocated memory used by gm */
int armci_gm_serv_mem_free()
{
    int i;
    
    gm_dma_free(serv_gm->rcv_port, serv_gm->ack_buf);
    
    free(serv_gm->ack_buf);
    free(serv_gm->proc_buf_ptr);

    for(i=ARMCI_GM_MIN_MESG_SIZE; i<=ARMCI_GM_MAX_MESG_SIZE; i++) {
        gm_dma_free(serv_gm->rcv_port, serv_gm->dma_buf[i]);
    }

    gm_dma_free(serv_gm->snd_port, MessageRcvBuffer);

    free(armci_gm_serv_context);
    
    return TRUE;
}

/* server side call back func */
void armci_serv_callback(struct gm_port *port, void *context, 
			 gm_status_t status)
{
    if(status == GM_SUCCESS)
        ((armci_gm_context_t *)context)->done = ARMCI_GM_SENT;
    else ((armci_gm_context_t *)context)->done = ARMCI_GM_FAILED;
}

/* server trigues gm_unknown, so that callback func can be executed */
int armci_serv_send_complete()
{
    gm_recv_event_t *event;

    while(armci_gm_serv_context->done == ARMCI_GM_SENDING) {
        event = gm_blocking_receive_no_spin(serv_gm->snd_port);
        gm_unknown(serv_gm->snd_port, event);
    }
    
    return(armci_gm_serv_context->done);
}

/* initialization of server thread */
int armci_gm_server_init() 
{
    int i;
    int status;
    
    unsigned long size_mask;
    unsigned int min_mesg_size, min_mesg_length;
    unsigned int max_mesg_size, max_mesg_length;
 
    /* allocate gm data structure for server */
    serv_gm = (armci_gm_serv_t *)malloc(1 * sizeof(armci_gm_serv_t));
    if(serv_gm == NULL) {
        fprintf(stderr, "%d: Error allocate server data structure.\n",
                armci_me);
        return FALSE;
    }
    serv_gm->node_map = (int *)malloc(armci_nproc * sizeof(int));
    serv_gm->port_map = (int *)calloc(armci_nproc, sizeof(int));
    if((serv_gm->node_map == NULL) || (serv_gm->port_map == NULL)) {
        fprintf(stderr, "%d: Error allocate server data structure.\n",
                armci_me);
        return FALSE;
    }

    /* opening gm port */
    if(DEBUG_) 
        fprintf(stdout,
            "%d(server):opening gm port %d(rcv)dev=%d and %d(snd)dev=%d\n",
                armci_me, ARMCI_GM_SERVER_RCV_PORT, ARMCI_GM_SERVER_RCV_DEV,
                          ARMCI_GM_SERVER_SND_PORT, ARMCI_GM_SERVER_SND_DEV);

    serv_gm->rcv_port = NULL; serv_gm->snd_port = NULL;
    status = gm_open(&(serv_gm->rcv_port), ARMCI_GM_SERVER_RCV_DEV,
                     ARMCI_GM_SERVER_RCV_PORT, "gm_pt", GM_API_VERSION_1_1);
    if(status != GM_SUCCESS) {
        fprintf(stderr, "%d: Could not open rcv port, status: %d\n",
                armci_me, status);
        return FALSE;
    }
    status = gm_open(&(serv_gm->snd_port), ARMCI_GM_SERVER_SND_DEV,
                     ARMCI_GM_SERVER_SND_PORT, "gm_pt", GM_API_VERSION_1_1);
    if(status != GM_SUCCESS) {
        fprintf(stderr, "%d: Could not open snd port, status: %d\n",
                armci_me, status);
        return FALSE;
    }

    /* get my node id */
    /* server node id should be the same as the master process.
     * at this point the initialization of master process should
     * have been done. so copy directly
     */
    serv_gm->node_id = proc_gm->node_id;
    if(DEBUG_) fprintf(stdout, "%d(server): node id is %d\n",
                       armci_me, proc_gm->node_id);
    for(i=0; i<armci_nproc; i++)
        serv_gm->node_map[i] = proc_gm->node_map[i];

    /* allow direct send */
    status = gm_allow_remote_memory_access(serv_gm->rcv_port);
    if(status != GM_SUCCESS) {
        fprintf(stderr, "%d(server): could not enable direct sends\n",
                armci_me);
        return FALSE;
    }
    
    /* memory preallocation for server */
    if(!armci_gm_serv_mem_alloc()) {
        fprintf(stderr, "%d(server): failed allocating memory\n",
                armci_me);
        return FALSE;
    }

    /* set message size on server */
    min_mesg_size = ARMCI_GM_MIN_MESG_SIZE;
    min_mesg_length = gm_max_length_for_size(min_mesg_size);
    max_mesg_size = ARMCI_GM_MAX_MESG_SIZE;
    max_mesg_length = gm_max_length_for_size(max_mesg_size);
    if(DEBUG_INIT_) {
        printf("%d: SERVER min_mesg_size = %d, max_mesg_size = %d\n",
               armci_me, min_mesg_size, max_mesg_size);
        printf("%d: SERVER min_mesg_length = %d, max_mesg_length = %d\n",
               armci_me, min_mesg_length, max_mesg_length);
    }
    
    /* accept only the smallest size messages */
    size_mask = (2 << max_mesg_size) - 1;
    status = gm_set_acceptable_sizes(serv_gm->rcv_port, GM_LOW_PRIORITY,
                                     size_mask);
    if (status != GM_SUCCESS) {
        fprintf(stderr, "%d(server): error setting acceptable sizes",
                armci_me);
        return FALSE;
    }

    /* provide the buffers initially create a size mask and set */
    for(i=ARMCI_GM_MIN_MESG_SIZE; i<=ARMCI_GM_MAX_MESG_SIZE; i++)
        gm_provide_receive_buffer_with_tag(serv_gm->rcv_port,
               serv_gm->dma_buf[i], i, GM_LOW_PRIORITY, 0);

    return TRUE;
}

/* server start communication with all the computing processes */
void armci_server_initial_connection_gm()
{
    int i;
    gm_recv_event_t *event;
    unsigned int size, length;
    char *buf;

    int rid;

    int procs_in_clus = armci_clus_info[armci_clus_me].nslave;
    int iexit;


    /* receive the initial connection from all computing processes,
     * except those from the same node
     */
    iexit = armci_nproc - procs_in_clus;
    while(iexit) {
        event = gm_blocking_receive_no_spin(serv_gm->rcv_port);
        
        switch (event->recv.type) {
          case GM_RECV_EVENT:
          case GM_PEER_RECV_EVENT:
              iexit--;

              size = gm_ntohc(event->recv.size);
              length = gm_ntohl(event->recv.length);
              buf = gm_ntohp(event->recv.buffer);
              
              

              /* receiving the remote mpi id and addr of serv_ack_ptr */
              rid = (int)(((long *)buf)[0]);
              if(DEBUG_INIT_) 
                  fprintf(stdout,
                   "%d(server): received init mesg from %d, size=%d, len=%d\n",
                          armci_me, rid, size, length);
              
              serv_gm->proc_buf_ptr[rid] = ((long *)buf)[1];
              
              serv_gm->port_map[rid] = gm_ntohc(event->recv.sender_port_id);

              /* send server ack buffer and MessageRcvBuffer ptr to client */
              serv_gm->ack_buf[rid] != ARMCI_GM_CLEAR;
              ((long *)MessageRcvBuffer)[0] = ARMCI_GM_COMPLETE;
              ((long *)MessageRcvBuffer)[1] =
                  (long)(&(serv_gm->ack_buf[rid]));
              ((long *)MessageRcvBuffer)[2] = (long)(MessageRcvBuffer);
              ((long *)MessageRcvBuffer)[3] = ARMCI_GM_COMPLETE;
              
              armci_gm_serv_context->done = ARMCI_GM_SENDING;
              gm_directed_send_with_callback(serv_gm->snd_port,
                   MessageRcvBuffer,
                   (gm_remote_ptr_t)(gm_up_t)(serv_gm->proc_buf_ptr[rid]),
                   4*sizeof(long), GM_LOW_PRIORITY, serv_gm->node_map[rid],
                   serv_gm->port_map[rid], armci_serv_callback,
                   armci_gm_serv_context);

              /* blocking: wait til the send is complete */
              if(armci_serv_send_complete() == ARMCI_GM_FAILED)
                  armci_die(" Init: server could not send msg to client", rid);

              /* wait the client send back the ack */
              if(DEBUG_INIT_)
                  fprintf(stdout,
                     "%d(server): sent msg to %d (@%d), expecting ack at %d\n",
                          armci_me, rid, serv_gm->proc_buf_ptr[rid],
                          &(serv_gm->ack_buf[rid]));

              wait_flag_updated(&(serv_gm->ack_buf[rid]), ARMCI_GM_ACK);
              if(DEBUG_INIT_) {
                fprintf(stdout, "%d(server): connected to client %d\n",
                        armci_me, rid);
                fprintf(stdout, "%d(server): expecting %d more connections\n",
                        armci_me, iexit);
              }
              
              buf = serv_gm->dma_buf[size];
              gm_provide_receive_buffer_with_tag(serv_gm->rcv_port, buf,
                                                 size, GM_LOW_PRIORITY, 0);
              break;
          default:
              gm_unknown(serv_gm->rcv_port, event);
              break;
        }
    }
}

/* server direct send to the client
 * assume buf is pinned and using MessageRcvBuffer
 * MessageRcvBuffer: .... + hdr ack + data + tail ack
 *                                         ^
 *                                         buf (= len)
 */
void armci_WriteToDirect(int dst, request_header_t *msginfo, char *buf)
{
    int status;
    
    char *ptr = buf - sizeof(long);

    /* adjust the dst pointer */
    void *dst_addr = msginfo->tag.data_ptr;
    
    /* set head ack */
    *(long *)ptr = ARMCI_GM_COMPLETE;
   
    /* set tail ack */
    *(long *)(buf + msginfo->datalen) = ARMCI_GM_COMPLETE;

    /* serv_gm->ack_buf[dst] = ARMCI_GM_CLEAR; */

    armci_gm_serv_context->done = ARMCI_GM_SENDING;
    gm_directed_send_with_callback(serv_gm->snd_port, ptr,
                     (gm_remote_ptr_t)(gm_up_t)(dst_addr),
                     msginfo->datalen+2*sizeof(long), GM_LOW_PRIORITY,
                     serv_gm->node_map[dst], serv_gm->port_map[dst],
                     armci_serv_callback, armci_gm_serv_context);
}

#if 0
/* wait the client send back the ack */
/* currently not used */
void armci_WaitAckDirect(int from)
{
    /* wait the client send back the ack */
    if(DEBUG_) {
        fprintf(stdout, "%d(server): expecting ack @%d from %d\n",
                armci_me, (long)(&(serv_gm->ack_buf[from])), from);
        fflush(stdout);
    }
    
    wait_flag_updated(&(serv_gm->ack_buf[from]), ARMCI_GM_ACK);
    if(DEBUG_) {
        fprintf(stdout, "%d(server): mesg sent successful to %d\n",
                armci_me, from);
        fflush(stdout);    
    }
    serv_gm->ack_buf[from] = ARMCI_GM_CLEAR;
}

#endif 

/* the main data server loop: accepting events and pass it to data server
 * code to be porcessed
 */
void armci_data_server_gm()
{
    int iexit = FALSE;

    unsigned int size, length;
    char *buf;
    
    /* gm event */
    gm_recv_event_t *event;
    
    if(DEBUG_){
        fprintf(stdout, "%d(server): waiting for request\n",armci_me);
        fflush(stdout);
    }

    /* server main loop; wait for and service requests until QUIT requested */
    while(!iexit) {        
        event = gm_blocking_receive_no_spin(serv_gm->rcv_port);
        if(DEBUG_INIT_) {
            fprintf(stdout, "%d(server): receive event type %d\n",
                    armci_me, event->recv.type);
            fflush(stdout);     
        }
        
        switch(event->recv.type) {
          case GM_RECV_EVENT:
          case GM_PEER_RECV_EVENT:
              size = gm_ntohc(event->recv.size);
              length = gm_ntohl(event->recv.length);
              buf = (char *)gm_ntohp(event->recv.buffer);

              armci_data_server(buf);
              
              buf = serv_gm->dma_buf[size];
              gm_provide_receive_buffer_with_tag(serv_gm->rcv_port, buf,
                                                 size, GM_LOW_PRIORITY, 0);
    
              break;
          default:
              gm_unknown(serv_gm->rcv_port, event);
              break;
        }
    }
    
    if(DEBUG_) {
        fprintf(stdout, "%d(server): done! closing ...\n");
        fflush(stdout);
    }
}



/* cleanup of gm: applies to either server or client */
void armci_transport_cleanup()
{    
    /* deallocate the gm data structure */
    if(SERVER_CONTEXT) {
/*
        if(!armci_gm_serv_mem_free()) 
            armci_die("server memory deallocate memory failed", armci_me);
            
*/
        gm_close(serv_gm->rcv_port);
        gm_close(serv_gm->snd_port);
        free(serv_gm->node_map); free(serv_gm->port_map);
        free(serv_gm);
    }
    else {
/*
        if(!armci_gm_proc_mem_free()) 
            armci_die("computing process  memory deallocate memory failed",
                      armci_me);
*/
        
        free(proc_gm->node_map);
        free(proc_gm);
    }
}
 
