#include "armcip.h"
#include "request.h"
#include "memlock.h"
#include "copy.h"
#include <stdio.h>

#define DEBUG_ 0

double _armci_snd_buf[MSG_BUFLEN_DBL], _armci_rcv_buf[MSG_BUFLEN_DBL];
char* MessageRcvBuffer = (char*)_armci_rcv_buf;
char* MessageSndBuffer = (char*)_armci_snd_buf;


#define ADDBUF(buf,type,val) *(type*)(buf) = (val); (buf) += sizeof(type)
#define GETBUF(buf,type,var) (var) = *(type*)(buf); (buf) += sizeof(type)



/*\  request RMW from server
\*/
void armci_rem_rmw(int op, int *ploc, int *prem, int extra, int proc)
{
request_header_t *msginfo = (request_header_t*)MessageSndBuffer;
char *buf = (char*)(msginfo+1);

    msginfo->dscrlen = sizeof(void*);
    msginfo->from  = armci_me;
    msginfo->to    = proc; 
    msginfo->format  = msginfo->operation = op;
    msginfo->datalen =sizeof(int); /* extra */
    msginfo->bytes   =msginfo->datalen+msginfo->dscrlen ;

    ADDBUF(buf,void* ,prem);
    ADDBUF(buf,int,extra); 

    armci_send_req(proc);

    /* need to adjust datalen for long datatype version */
    msginfo->datalen = (op==ARMCI_FETCH_AND_ADD)? sizeof(int): sizeof(long);

    armci_rcv_data(proc);  /* receive response */
    if(op==ARMCI_FETCH_AND_ADD)
      *ploc = *(int*)MessageSndBuffer;
    else
      *(long*)ploc = *(long*)MessageSndBuffer;
}


/*\ server response to RMW 
\*/
void armci_server_rmw(request_header_t* msginfo,void* ptr, void* pextra)
{
     long lold;
     int iold;
     void *pold;
     int op = msginfo->operation;

     if(DEBUG_){
        printf("%d server: executing RMW from %d\n",armci_me,msginfo->from);
        fflush(stdout);
     }

     switch(op){
     case ARMCI_FETCH_AND_ADD:
        if(msginfo->datalen != sizeof(int))
          armci_die("armci_server_rmw: bad datalen=",msginfo->datalen);
        pold = &iold;
        msginfo->datalen = sizeof(int);
        break;

     case ARMCI_FETCH_AND_ADD_LONG:
        if(msginfo->datalen != sizeof(int))
          armci_die("armci_server_rmw: long bad datalen=",msginfo->datalen);
        pold = &lold;
        msginfo->datalen = sizeof(long);
        break;

     default:
          armci_die("armci_server_rmw: bad operation code=",op);
     }

     armci_generic_rmw(op, pold, *(int**)ptr, *(int*) pextra, msginfo->to);

     armci_send_data(msginfo, pold);
}


int armci_rem_vector(int op, void *scale, armci_giov_t darr[],int len,int proc)
{
    char *buf = MessageSndBuffer;
    request_header_t *msginfo = (request_header_t*)MessageSndBuffer;
    int bytes =0, s, slen=0;
    void *rem_ptr;
    size_t adr;

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

    /* align buf for doubles (8-bytes) before copying data */
    adr = (size_t)buf;
    adr >>=3;
    adr <<=3;
    adr +=8;
    buf = (char*)adr;

    /* fill message header */
    msginfo->dscrlen = buf - MessageSndBuffer - sizeof(request_header_t);
    /*printf("VECTOR len=%d dscrlen=%d\n",len, msginfo->dscrlen);*/
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
    msginfo->bytes = msginfo->datalen+msginfo->dscrlen;

    /* for put and accumulate copy data into buffer */
    if(op != GET){
/*       fprintf(stderr,"sending %lf\n",*(double*)darr[0].src_ptr_array[0]);*/

       armci_vector_to_buf(darr, len, buf);
/*       fprintf(stderr,"sending first=%lf last =%lf in buffer\n",*/
/*                     *((double*)buf),((double*)buf)[99]);*/
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
    msginfo->bytes = msginfo->datalen+msginfo->dscrlen;

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
    int  rc, i,proc;

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

    proc = msginfo->to;

    if(msginfo->operation == GET){
    
       if(rc = armci_op_strided(GET, scale, proc, loc_ptr, loc_stride_arr,
               buf_ptr, buf_stride_arr, count, stride_levels, 0))
               armci_die("server_strided: get to buf failed",rc);

       armci_send_data(msginfo, buf);

    } else{

       if(rc = armci_op_strided(msginfo->operation, scale, proc,
               buf_ptr, buf_stride_arr, loc_ptr, loc_stride_arr,
               count, stride_levels, 1))
               armci_die("server_strided: op from buf failed",rc);
    }
}



void armci_server_vector( request_header_t *msginfo, 
                          char *dscr, char* buf, int buflen)
{
    int  len,proc;
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


    proc = msginfo->to;

    /*fprintf(stderr,"scale=%lf\n",*(double*)scale);*/
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

/*      fprintf(stderr,"received first=%lf last =%lf in buffer\n",*/
/*                     *((double*)buf),((double*)buf)[99]);*/

      for(i = 0; i< len; i++){
        int parlen, bytes;
        void **ptr;
        GETBUF(dscr, int, parlen);
        GETBUF(dscr, int, bytes);
        ptr = (void**)dscr; dscr += parlen*sizeof(char*);
        armci_lockmem_scatter(ptr, parlen, bytes, proc); 
        for(s=0; s< parlen; s++){
          armci_acc_2D(msginfo->operation, scale, proc, buf, ptr[s],
                       bytes, 1, bytes, bytes, 0);
          buf += bytes;
        }
        ARMCI_UNLOCKMEM(proc);
      }
    }
}
