#include "armcip.h"
#include "memlock.h"
#include "copy.h"
#include <stdio.h>

double _armci_snd_buf[MSG_BUFLEN_DBL], _armci_rcv_buf[MSG_BUFLEN_DBL];
char* MessageRcvBuffer = (char*)_armci_rcv_buf;
char* MessageSndBuffer = (char*)_armci_snd_buf;

void armci_send_req_(int proc)
{
void armci_server_vector();
int hdrlen = sizeof(request_header_t);
int dscrlen = ((request_header_t*)MessageSndBuffer)->dscrlen;

    armci_copy(MessageSndBuffer, MessageRcvBuffer, MSG_BUFLEN);
    armci_server_vector(MessageRcvBuffer, MessageRcvBuffer + hdrlen,
                 MessageRcvBuffer +hdrlen+dscrlen, MSG_BUFLEN-hdrlen-dscrlen);
/*
*/
}


void armci_rcv_data_(int proc)
{
}


void armci_send_data_(request_header_t* msginfo, char *data)
{
int hdrlen = sizeof(request_header_t);
armci_copy(data,MessageSndBuffer,msginfo->datalen);
}


#define ADDBUF(buf,type,val) *(type*)(buf) = (val); (buf) += sizeof(type)
#define GETBUF(buf,type,var) (var) = *(type*)(buf); (buf) += sizeof(type)


int armci_rem_vector(int op, void *scale, armci_giov_t darr[],int len,int proc)
{
    char *buf = MessageSndBuffer;
    request_header_t *msginfo = (request_header_t*)MessageSndBuffer;
    int bytes =0, s, slen=0;
    void *rem_ptr;

    GET_SEND_BUFFER;

    /* fill vector descriptor */
    buf += sizeof(request_header_t);
    ADDBUF(buf,int,len); /* number of sets */
    for(s=0;s<len;s++){

        bytes += darr[s].ptr_array_len * darr[s].bytes;
        ADDBUF(buf,int,darr[s].ptr_array_len); /* number of elements */
        ADDBUF(buf,int,darr[s].bytes);         /* sizeof element */

        if(op == GET) rem_ptr = darr[s].src_ptr_array;   
        else rem_ptr = darr[s].dst_ptr_array;        
        armci_copy(rem_ptr,buf, darr[s].ptr_array_len * sizeof(void*)); 
        buf += darr[s].ptr_array_len*sizeof(void*);
    }

    /* fill message header */
    msginfo->dscrlen = buf - MessageSndBuffer - sizeof(request_header_t);
/*    printf("len=%d dscrlen=%d\n",len, msginfo->dscrlen);*/
    msginfo->bytes = bytes;
    msginfo->from  = armci_me;
    msginfo->to    = proc;
    msginfo->operation  = op;
    msginfo->format  = VECTOR;

    msginfo->datalen = bytes;

    /* put scale for accumulate */
    switch(op){
    case ARMCI_ACC_INT:
               *(int*)buf = *(int*)scale; slen= sizeof(int); break;
    case ARMCI_ACC_DCP:
               ((double*)buf)[0] = ((double*)scale)[0];
               ((double*)buf)[1] = ((double*)scale)[1];
               slen=2*sizeof(double);break;
    case ARMCI_ACC_DBL:
               *(double*)buf = *(double*)scale; slen = sizeof(double); break;
    case ARMCI_ACC_CPL:
               ((float*)buf)[0] = ((float*)scale)[0];
               ((float*)buf)[1] = ((float*)scale)[1];
               slen=2*sizeof(float);break; 
    case ARMCI_ACC_FLT:
               *(float*)buf = *(float*)scale; slen = sizeof(float); break;
    default: slen=0;
    }
    buf += slen;
    msginfo->datalen += slen;

    /* for put and accumulate copy data into buffer */
    if(op != GET){
/*       fprintf(stderr,"sending %lf\n",*(double*)darr[0].src_ptr_array[0]);*/
/*       fprintf(stderr,"in buffer %lf\n",*(double*)buf);*/

       armci_vector_to_buf(darr, len, buf);
    }

    armci_send_req(proc);

    if(op == GET){
       armci_rcv_data(proc);
       armci_vector_from_buf(darr, len, MessageSndBuffer);
    }
    return 0;
}



/*\ client version of remote strided operation
\*/
int armci_rem_strided(int op, void* scale, int proc,
                       void *src_ptr, int src_stride_arr[],
                       void* dst_ptr, int dst_stride_arr[],
                       int count[], int stride_levels, int lockit)
{
    int  buf_stride_arr[MAX_STRIDE_LEVEL+1];
    char *buf = MessageSndBuffer;
    request_header_t *msginfo = (request_header_t*)MessageSndBuffer;
    int  rc, i;
    size_t adr;
    int slen=0;
    void *rem_ptr;
    int  *rem_stride_arr;

    GET_SEND_BUFFER;

    if(op == GET){
       rem_ptr = src_ptr;
       rem_stride_arr = src_stride_arr;
    }else{
       rem_ptr = dst_ptr;
       rem_stride_arr = dst_stride_arr;
    }

    buf_stride_arr[0]=count[0];
    for(i=0; i< stride_levels; i++) 
        buf_stride_arr[i+1]= buf_stride_arr[i]*count[i+1];

    for(i=0, msginfo->datalen=1;i<=stride_levels;i++)msginfo->datalen*=count[i];
    
    /* fill strided descriptor */
                                       buf += sizeof(request_header_t);
    *(void**)buf = rem_ptr;            buf += sizeof(void*);
    *(int*)buf = stride_levels;        buf += sizeof(int);
    for(i=0;i<stride_levels;i++)((int*)buf)[i] = rem_stride_arr[i];
                                       buf += stride_levels*sizeof(int);
    for(i=0;i< stride_levels+1;i++)((int*)buf)[i] = count[i];
                                       buf += (1+stride_levels)*sizeof(int);

    /* align buf for doubles (8-bytes) before copying data */
    adr = (size_t)buf;
    adr >>=3; 
    adr <<=3;
    adr +=8; 
    buf = (char*)adr;

    /* fill message header */
    msginfo->dscrlen = buf - MessageSndBuffer - sizeof(request_header_t);
    msginfo->bytes = buf_stride_arr[stride_levels];
    msginfo->from  = armci_me;
    msginfo->to    = proc;
    msginfo->operation  = op;
    msginfo->format  = STRIDED;

    /* put scale for accumulate */
    switch(op){
    case ARMCI_ACC_INT: 
               *(int*)buf = *(int*)scale; slen= sizeof(int); break;
    case ARMCI_ACC_DCP:
               ((double*)buf)[0] = ((double*)scale)[0];
               ((double*)buf)[1] = ((double*)scale)[1];
               slen=2*sizeof(double);break; 
    case ARMCI_ACC_DBL: 
               *(double*)buf = *(double*)scale; slen = sizeof(double); break;
    case ARMCI_ACC_CPL:
               ((float*)buf)[0] = ((float*)scale)[0];
               ((float*)buf)[1] = ((float*)scale)[1];
               slen=2*sizeof(float);break; 
    case ARMCI_ACC_FLT:
               *(float*)buf = *(float*)scale; slen = sizeof(float); break;
    default: slen=0;
    }
/*    if(ACC(op))*/
/*      fprintf(stderr,"%d in server len=%d alpha=(%d,%d)\n",*/
/*              armci_me, slen, ((double*)buf)[0],((double*)buf)[1]); */
    buf += slen;
    msginfo->datalen += slen;

    /* for put and accumulate copy data into buffer */
    if(op != GET)
       if(rc = armci_op_strided(GET, scale, armci_me, src_ptr, src_stride_arr,
               buf, buf_stride_arr, count, stride_levels, 0))
               armci_die("rem_strided: put to buf failed",rc);
       
    armci_send_req(proc);

    if(op == GET){
       armci_rcv_data(proc);
       if(rc = armci_op_strided(GET, scale, armci_me, 
               MessageSndBuffer, buf_stride_arr, 
               dst_ptr, dst_stride_arr, count, stride_levels, 0))
               armci_die("rem_strided: get from buf failed",rc);
    }
    return 0;
}




void armci_server(request_header_t *msginfo, char *dscr, char* buf, int buflen)
{
    int  buf_stride_arr[MAX_STRIDE_LEVEL+1];
    int  *loc_stride_arr; 
    int  *count, stride_levels;
    void *buf_ptr, *loc_ptr;
    void *scale;
    int  rc, i;

    /* unpack descriptor record */
    loc_ptr = *(void**)dscr;           dscr += sizeof(void*);
    stride_levels = *(int*)dscr;       dscr += sizeof(int);
    loc_stride_arr = (int*)dscr;       dscr += stride_levels*sizeof(int);
    count = (int*)dscr;

    /* compute stride array for buffer */
    buf_stride_arr[0]=count[0];
    for(i=0; i< stride_levels; i++)
        buf_stride_arr[i+1]= buf_stride_arr[i]*count[i+1];
    
    /* get scale for accumulate, adjust buf to point to data */
    scale = buf;
    switch(msginfo->operation){
    case ARMCI_ACC_INT:     buf += sizeof(int); break;
    case ARMCI_ACC_DCP:     buf += 2*sizeof(double); break;
    case ARMCI_ACC_DBL:     buf += sizeof(double); break;
    case ARMCI_ACC_CPL:     buf += 2*sizeof(float); break;
    case ARMCI_ACC_FLT:     buf += sizeof(float); break;
    }

    buf_ptr = buf; /*  data in buffer */

    if(msginfo->operation == GET){
    
       if(rc = armci_op_strided(GET, scale, armci_me, loc_ptr, loc_stride_arr,
               buf_ptr, buf_stride_arr, count, stride_levels, 0))
               armci_die("server_strided: get to buf failed",rc);

       armci_send_data(msginfo, buf);

    } else{

       if(rc = armci_op_strided(msginfo->operation, scale, armci_me,
               buf_ptr, buf_stride_arr, loc_ptr, loc_stride_arr,
               count, stride_levels, 1))
               armci_die("server_strided: op from buf failed",rc);
    }
}



void armci_server_vector( request_header_t *msginfo, 
                          char *dscr, char* buf, int buflen)
{
    int  len;
    void *buf_ptr, *loc_ptr;
    void *scale;
    int  rc, i,s;
    armci_riov_t *rdarr;
    char *sbuf = buf;

    /* unpack descriptor record */
    GETBUF(dscr, int, len);
    rdarr = (armci_riov_t*)dscr;
    
    /* get scale for accumulate, adjust buf to point to data */
    scale = buf;
    switch(msginfo->operation){
    case ARMCI_ACC_INT:     buf += sizeof(int); break;
    case ARMCI_ACC_DCP:     buf += 2*sizeof(double); break;
    case ARMCI_ACC_DBL:     buf += sizeof(double); break;
    case ARMCI_ACC_CPL:     buf += 2*sizeof(float); break;
    case ARMCI_ACC_FLT:     buf += sizeof(float); break;
    }


    /* execute the operation */

    switch(msginfo->operation) {
    case GET:
 
      for(i = 0; i< len; i++){
        int parlen, bytes;
        void **ptr;
        GETBUF(dscr, int, parlen);
        GETBUF(dscr, int, bytes);
/*        fprintf(stderr,"len=%d bytes=%d parlen=%d\n",len,bytes,parlen);*/
        ptr = (void**)dscr; dscr += parlen*sizeof(char*);
        for(s=0; s< parlen; s++){
          armci_copy(ptr[s], buf, bytes);
          buf += bytes;
        }
      }
    
/*      fprintf(stderr,"server sending buffer %lf\n",*(double*)sbuf);*/
      armci_send_data(msginfo, sbuf);
      break;

    case PUT:

/*    fprintf(stderr,"received in buffer %lf\n",*(double*)buf);*/
      for(i = 0; i< len; i++){
        int parlen, bytes;
        void **ptr;
        GETBUF(dscr, int, parlen);
        GETBUF(dscr, int, bytes);
        ptr = (void**)dscr; dscr += parlen*sizeof(char*);
        for(s=0; s< parlen; s++){
          armci_copy(buf, ptr[s], bytes);
          buf += bytes;
        }
      }
      break;

     default:

      /* this should be accumulate */
      if(!ACC(msginfo->operation))
               armci_die("v server: wrong op code",msginfo->operation);

      for(i = 0; i< len; i++){
        int parlen, bytes;
        void **ptr;
        GETBUF(dscr, int, parlen);
        GETBUF(dscr, int, bytes);
        ptr = (void**)dscr; dscr += parlen*sizeof(char*);
        armci_lockmem_scatter(ptr, parlen, bytes, armci_me); 
        for(s=0; s< parlen; s++){
          armci_acc_2D(msginfo->operation, scale, armci_me, buf, ptr[s],
                       bytes, 1, bytes, bytes, 0);
          buf += bytes;
        }
        ARMCI_UNLOCKMEM();
      }
    }
}
