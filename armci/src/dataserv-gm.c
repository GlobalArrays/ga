#include "armcip.h"
#include "copy.h"
#include "myrinet.h"
#include <stdio.h>
#include <errno.h>

#define DEBUG_ 0


/*********************************************************************
                        UTILITY FUNCTIONS                            
 *********************************************************************/

long check_flag(long *buf)
{
    return(*buf);
}

/*******************************************************************
 *                   CLIENT SIDE                                   *
 *******************************************************************/


void armci_rcv_strided_data_bypass(int proc, int datalen,
                                   void *ptr, int stride_levels)
{
    int i;
    int len;

    if(DEBUG_){
        fprintf(stdout,
                "%d: armci_rcv_strided_data: expecting datalen %d from %d\n",
                armci_me, datalen, proc);
        fflush(stdout);
    }

    /* wait message arrives */
    wait_flag_updated((long *)(proc_gm->ack_buf), ARMCI_GM_COMPLETE);

    if(DEBUG_){
        fprintf(stdout, "%d: armci_rcv_strided_data: got %d bytes from %d\n",
                armci_me, datalen, proc);
        fflush(stdout);
    }
}



void armci_wait_for_server()
{
}


/*******************************************************************
 *                   SERVER SIDE                                   *
 *******************************************************************/

/* server receives request */
void armci_rcv_req(void *mesg,
                   void *phdr, void *pdescr, void *pdata, int *buflen)
{
    int stat;
    request_header_t *msginfo = (request_header_t *)mesg;
    
    if(DEBUG_) {
        printf("%d(server): got %d req (dscrlen=%d datalen=%d) from %d\n",
               armci_me, msginfo->operation, msginfo->dscrlen,
               msginfo->datalen, msginfo->from);
        fflush(stdout);
    }

    *(void **)phdr = msginfo;

    if(msginfo->bypass) {
        *(void**)pdata  = MessageRcvBuffer;
        *buflen = MSG_BUFLEN;
    } else {
        /* leave space for header ack */
        *(void**)pdata  = MessageRcvBuffer + sizeof(long);
        *buflen = MSG_BUFLEN - sizeof(request_header_t) - sizeof(long);
        /* tail acks -^ */
    }
    
    if(msginfo->bytes) {
        *(void **)pdescr = msginfo+1;
        if(msginfo->operation != GET)
            *(void **)pdata = msginfo->dscrlen + (char*)(msginfo+1);
    }else 
        *(void**)pdescr = NULL;

    /* make sure last send is complete and the buffer is available */
    stat = armci_serv_send_complete();
    if(stat == ARMCI_GM_FAILED)
        armci_die(" last armci_send_(strided_)data: write failed", stat);  
}


void armci_send_strided_data_bypass(int proc, request_header_t *msginfo,
                                    void *loc_buf, int msg_buflen,
                                    void *loc_ptr, int *loc_stride_arr,
                                    void *rem_ptr, int *rem_stride_arr,
                                    int *count, int stride_levels)
{
    int i, j, stat;
    long loc_idx, rem_idx;
    int n1dim;  /* number of 1 dim block */
    int bvalue[MAX_STRIDE_LEVEL], bunit[MAX_STRIDE_LEVEL]; 

    int to = msginfo->from;
    void *buf;
    int buflen;
    int msg_threshold;

    msg_threshold = MIN(msg_buflen, INTERLEAVE_GET_THRESHOLD);
    
    buf = loc_buf; buflen = msg_buflen;
    
    if(DEBUG_){
        fprintf(stdout, "%d(server): sending data to %d\n", armci_me, to);
        fflush(stdout);
    }
 
    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++)
        n1dim *= count[i];

    /* calculate the destination indices */
    bvalue[0] = 0; bvalue[1] = 0; bunit[0] = 1; bunit[1] = 1;
    for(i=2; i<=stride_levels; i++) {
        bvalue[i] = 0;
        bunit[i] = bunit[i-1] * count[i-1];
    }

    for(i=0; i<n1dim; i++) {
        loc_idx = 0; rem_idx = 0;
        for(j=1; j<=stride_levels; j++) {
            loc_idx += bvalue[j] * loc_stride_arr[j-1];
            rem_idx += bvalue[j] * rem_stride_arr[j-1];
            if((i+1) % bunit[j] == 0) bvalue[j]++;
            if(bvalue[j] > (count[j]-1)) bvalue[j] = 0;
        }

        /* segment is larger than INTERLEAVE_GET_THRESHOLD,
         * but less than msg_buflen
         */
        if((count[0] > INTERLEAVE_GET_THRESHOLD) && (count[0] <= msg_buflen)) {
            char *src_ptr = (char *)loc_ptr+loc_idx;
            char *dst_ptr = (char *)rem_ptr+rem_idx;
            int msg_size = count[0] /2;

            armci_serv_send_nonblocking_complete(0);
            
            armci_copy(src_ptr, buf, msg_size);
            armci_server_direct_send(to, buf, dst_ptr, msg_size,
                                     ARMCI_GM_NONBLOCKING);
            
            src_ptr += msg_size; dst_ptr += msg_size;
            buf += msg_size;
            
            armci_copy(src_ptr, buf, msg_size);
            armci_server_direct_send(to, buf, dst_ptr, msg_size,
                                     ARMCI_GM_NONBLOCKING);
            
            buf = loc_buf;
        }
        
        /* if each segment is larger than the buffer size */
        else if(count[0] > msg_buflen) {
            int msglen = count[0];
            char *src_ptr = (char *)loc_ptr+loc_idx;
            char *dst_ptr = (char *)rem_ptr+rem_idx;

            /* the message buffer is divided into two */
            int msg_size = msg_buflen/2;
            while(msglen > 0) {
                int len;
                if(msglen > msg_size) len = msg_size;
                else len = msglen;
                
                armci_copy(src_ptr, buf, len);

                armci_server_direct_send(to, buf, dst_ptr, len,
                                         ARMCI_GM_NONBLOCKING);
                
                msglen -= len;
                src_ptr += len; dst_ptr += len;

                if(buf == loc_buf) buf += msg_size; else buf = loc_buf;

                /* at any time, there can be only one outstanding
                 * callback not called
                 */
                armci_serv_send_nonblocking_complete(1);
            }

            buf = loc_buf;
        }
        /* if each segment is less than the buffer size */
        else {
            armci_copy((char *)loc_ptr+loc_idx, buf, count[0]);

            /* if the this is the last segment to fit into the buffer in
             * this round
             */
            if((buflen - count[0]) < count[0]) {
                armci_server_direct_send(to, buf, (char*)rem_ptr+rem_idx,
                                         count[0], ARMCI_GM_NONBLOCKING);
                buf = loc_buf; buflen = msg_buflen;
                
                /* check if the buffer is ready, prepare for next round */
                /* if(stride_levels == 0) */
                armci_serv_send_nonblocking_complete(0);
            }
            else{ 
                armci_server_direct_send(to, buf, (char*)rem_ptr+rem_idx,
                                         count[0], ARMCI_GM_NONBLOCKING);

                 buf += count[0]; buflen -= count[0];
                 
                 armci_serv_send_nonblocking_complete(5);
            }
        }
    }

    armci_serv_send_nonblocking_complete(0);
    
    /* inform client the send is over */
    armci_InformClient(to, serv_gm->direct_ack, ARMCI_GM_COMPLETE);
    
    if(DEBUG_){
        fprintf(stdout, "%d(server): sent data to %d\n", armci_me, to);
        fflush(stdout);
    }
}


void *armci_server_code(void *data)
{
    if(DEBUG_)
        printf("%d: in server after creating thread.\n",armci_me);
    
    /* make initial contact with all the computing process
     * get the port id of all computing process
     */
    armci_server_initial_connection_gm();
    
    if(DEBUG_) {
        printf("%d(server): connected to all computing processes\n",armci_me);
        fflush(stdout);
    }

    armci_data_server_gm();

    armci_transport_cleanup();
    
    return(NULL);
}
