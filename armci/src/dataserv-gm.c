#include "armcip.h"
#include "copy.h"
#include "myrinet.h"
#include <stdio.h>
#include <errno.h>

#define DEBUG_ 0
#define DEBUG1 0


/*********************************************************************
                        UTILITY FUNCTIONS                            
 *********************************************************************/

long check_flag(long *buf)
{
    return(*buf);
}

/*\ wait for strided data to arrive
\*/
void armci_rcv_strided_data_bypass(int proc, int datalen,
                                   void *ptr, int stride_levels)
{

    if(DEBUG_){ printf("%d:rcv_strided_data:expecting datalen %d from %d\n",
                armci_me, datalen, proc); fflush(stdout);
    }

    armci_wait_for_data_bypass(); /* wait until data arrives */

    if(DEBUG1){ printf("%d:rcv_strided_data bypass: got %d bytes from %d\n",
                armci_me, datalen, proc); fflush(stdout);
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
    request_header_t *msginfo = (request_header_t *)mesg;
    
    if(DEBUG_) {
        printf("%d(server): got %d req (dscrlen=%d datalen=%d) from %d\n",
               armci_me, msginfo->operation, msginfo->dscrlen,
               msginfo->datalen, msginfo->from);
        fflush(stdout);
    }

    *(void **)phdr = msginfo;

#ifdef CLIENT_BUF_BYPASS
    if(msginfo->bypass) {
        *(void**)pdata  = MessageRcvBuffer;
        *buflen = MSG_BUFLEN;
    } else 
#endif
    {
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
    if(armci_serv_send_complete()==ARMCI_GM_FAILED)
        armci_die("armci_send_(strided_)data: write failed",msginfo->from);  
}



void armci_send_contig_bypass(int proc, request_header_t *msginfo,
                              void *src_ptr, void *rem_ptr, int bytes)
{
     int to = msginfo->from;

     if(armci_pin_contig(src_ptr,bytes)){
       armci_server_direct_send(to,src_ptr,rem_ptr,bytes,ARMCI_GM_NONBLOCKING);
       armci_server_send_ack(to);
       armci_unpin_contig(src_ptr, bytes);
     }else armci_die("send_contig_bypass failed", bytes);
}
       


void armci_send_strided_data_bypass(int proc, request_header_t *msginfo,
                                    void *loc_buf, int msg_buflen,
                                    void *loc_ptr, int *loc_stride_arr,
                                    void *rem_ptr, int *rem_stride_arr,
                                    int *count, int stride_levels)
{
    int i, j;
    long loc_idx, rem_idx;
    int n1dim;  /* number of 1 dim block */
    int bvalue[MAX_STRIDE_LEVEL], bunit[MAX_STRIDE_LEVEL]; 

    int to = msginfo->from;
    char *buf;
    int buflen;
    int msg_threshold;

    msg_threshold = MIN(msg_buflen, INTERLEAVE_GET_THRESHOLD);
    
    buf = loc_buf; buflen = msg_buflen;
    
    if(DEBUG1){
        printf("%d(server): sending data bypass to %d\n", armci_me, to);
        fflush(stdout);
    }
 
#if 0
    if(stride_levels==0 && count[0] >800000){
       armci_send_contig_bypass(proc,msginfo, loc_ptr, rem_ptr, count[0]);
       return;
    }
#endif


    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++) n1dim *= count[i];

    /* calculate the destination indices */
    bvalue[0] = 0; bvalue[1] = 0; bunit[0] = 1; bunit[1] = 1;
    for(i=2; i<=stride_levels; i++) {
        bvalue[i] = 0;
        bunit[i] = bunit[i-1] * count[i-1];
    }

    for(i=0; i<n1dim; i++) {
        char *src_ptr, *dst_ptr;

        loc_idx = 0; rem_idx = 0;

        for(j=1; j<=stride_levels; j++) {
            loc_idx += bvalue[j] * loc_stride_arr[j-1];
            rem_idx += bvalue[j] * rem_stride_arr[j-1];
            if((i+1) % bunit[j] == 0) bvalue[j]++;
            if(bvalue[j] > (count[j]-1)) bvalue[j] = 0;
        }

        src_ptr = (char *)loc_ptr+loc_idx;
        dst_ptr = (char *)rem_ptr+rem_idx;

        /*  SEGMENT LARGER than the BUFFER size */
        if(count[0] > msg_buflen) {
            int msglen = count[0];

            /* the message buffer is divided into two */
            int msg_size = msg_buflen/2;
            while(msglen > 0) {
                int len;
                if(msglen > msg_size) len = msg_size;
                else len = msglen;
                
                armci_copy(src_ptr, buf, len);
                armci_server_direct_send(to, buf, dst_ptr, len, ARMCI_GM_NONBLOCKING);
                
                msglen -= len;
                src_ptr += len; dst_ptr += len;

                if(buf == loc_buf) buf += msg_size; else buf = loc_buf;

                /* at any time, there can be only one outstanding send */
                armci_serv_send_nonblocking_complete(1);
            }

            buf = loc_buf;
        } else

        /* MEDIUM SEGMENTS */
        if( count[0] > INTERLEAVE_GET_THRESHOLD) {
            int msg_size = count[0] /2;

            armci_serv_send_nonblocking_complete(0);
            
            armci_copy(src_ptr, buf, msg_size);
            armci_server_direct_send(to, buf, dst_ptr, msg_size, ARMCI_GM_NONBLOCKING);
            src_ptr += msg_size; dst_ptr += msg_size; buf += msg_size;
            
            armci_copy(src_ptr, buf, msg_size);
            armci_server_direct_send(to, buf, dst_ptr, msg_size, ARMCI_GM_NONBLOCKING);
            buf = loc_buf;
        } else

        /* SMALL SEGMENTS */
        {
            armci_copy(src_ptr, buf, count[0]);
            armci_server_direct_send(to, buf, dst_ptr,
                                         count[0], ARMCI_GM_NONBLOCKING);

            /* if the this is the last segment in this round */
            if((buflen - count[0]) < count[0]) {

                /* prepare for next round */
                buf = loc_buf; buflen = msg_buflen;
                
                /* complete all outstanding sends using this buffer */
                armci_serv_send_nonblocking_complete(0);

            } else{ 

                 buf += count[0]; buflen -= count[0];
                 
                 armci_serv_send_nonblocking_complete(5);
            }
        }
    }

    
    /* inform client all data was sent */
    armci_server_send_ack(to);
    
    armci_serv_send_nonblocking_complete(0);

    if(DEBUG_){
        fprintf(stdout, "%d(server): sent data to %d\n", armci_me, to);
        fflush(stdout);
    }
}


