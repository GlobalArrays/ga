/* $Id: request.c,v 1.33 2002-02-26 15:29:19 vinod Exp $ */
#include "armcip.h"
#include "request.h"
#include "memlock.h"
#include "shmem.h"
#include "copy.h"
#include <stdio.h>

#define DEBUG_ 0

#if !defined(GM) && !defined(VIA) && !defined(LAPI)
  double _armci_rcv_buf[MSG_BUFLEN_DBL];
  double _armci_snd_buf[MSG_BUFLEN_DBL]; 
  char* MessageSndBuffer = (char*)_armci_snd_buf;
  char* MessageRcvBuffer = (char*)_armci_rcv_buf;
#endif


#define ADDBUF(buf,type,val) *(type*)(buf) = (val); (buf) += sizeof(type)
#define GETBUF(buf,type,var) (var) = *(type*)(buf); (buf) += sizeof(type)



/*\ send request to server to LOCK MUTEX
\*/
void armci_rem_lock(int mutex, int proc, int *ticket)      
{
request_header_t *msginfo;
int *ibuf;
int bufsize = sizeof(request_header_t)+sizeof(int);
 
    msginfo = (request_header_t*)GET_SEND_BUFFER(bufsize,LOCK,proc);

    msginfo->datalen = sizeof(int);
    msginfo->dscrlen = 0;
    msginfo->from  = armci_me;
    msginfo->to    = proc;
    msginfo->operation = LOCK;
    msginfo->format  = mutex;
    msginfo->bytes = msginfo->datalen + msginfo->dscrlen;

    ibuf = (int*)(msginfo+1);
    *ibuf = mutex;

    armci_send_req(proc, msginfo, bufsize);

    /* receive ticket from server */
    *ticket = *(int*)armci_rcv_data(proc,msginfo);
    FREE_SEND_BUFFER(msginfo);
    
    if(DEBUG_)fprintf(stderr,"%d receiving ticket %d\n",armci_me, *ticket);
}




void armci_server_lock(request_header_t *msginfo)
{
int *ibuf = (int*)(msginfo+1);
int proc  = msginfo->from;
int mutex;
int ticket;

    mutex = *(int*)ibuf;

    /* acquire lock on behalf of requesting process */
    ticket = armci_server_lock_mutex(mutex, proc, msginfo->tag);
    
    if(ticket >-1){
       /* got lock */
       msginfo->datalen = sizeof(int);
       armci_send_data(msginfo, &ticket);
    }
}
       
    


/*\ send request to server to UNLOCK MUTEX
\*/
void armci_rem_unlock(int mutex, int proc, int ticket)
{
request_header_t *msginfo;
int *ibuf;
int bufsize = sizeof(request_header_t)+sizeof(ticket);

    msginfo = (request_header_t*)GET_SEND_BUFFER(bufsize,UNLOCK,proc);

    msginfo->dscrlen = msginfo->bytes = sizeof(ticket); 
    msginfo->datalen = 0; 
    msginfo->from  = armci_me;
    msginfo->to    = proc;
    msginfo->operation = UNLOCK;
    msginfo->format  = mutex;

    ibuf = (int*)(msginfo+1);
    *ibuf = ticket;

    if(DEBUG_)fprintf(stderr,"%d sending unlock\n",armci_me);
    armci_send_req(proc, msginfo, bufsize);
}
    


/*\ server unlocks mutex and passes lock to the next waiting process
\*/
void armci_server_unlock(request_header_t *msginfo, char* dscr)
{
    int ticket = *(int*)dscr;
    int mutex  = msginfo->format;
    int proc   = msginfo->to;
    int waiting;
    
    waiting = armci_server_unlock_mutex(mutex,proc,ticket,&msginfo->tag);

    if(waiting >-1){ /* -1 means that nobody is waiting */

       ticket++;
       /* pass ticket to the waiting process */
       msginfo->from = waiting;
       msginfo->datalen = sizeof(ticket);
       armci_send_data(msginfo, &ticket);

    }
}


void armci_unlock_waiting_process(msg_tag_t tag, int proc, int ticket)
{
request_header_t header;
request_header_t *msginfo = &header;

       msginfo->datalen = sizeof(int);
       msginfo->tag     = tag;
       msginfo->from      = proc;
       msginfo->to    = armci_me;
       armci_send_data(msginfo, &ticket); 
}


/*\ control message to the server, e.g.: ATTACH to shmem, return ptr etc.
\*/
void armci_serv_attach_req(void *info, int ilen, long size, void* resp,int rlen)
{
char *buf;
int bufsize = sizeof(request_header_t)+ilen + sizeof(long)+sizeof(rlen);
request_header_t *msginfo = (request_header_t*)GET_SEND_BUFFER(bufsize,ATTACH,armci_me);

    msginfo->from  = armci_me;
    msginfo->to    = SERVER_NODE(armci_clus_me);
    msginfo->dscrlen   = ilen;
    msginfo->datalen = sizeof(long)+sizeof(rlen);
    msginfo->operation =  ATTACH;
    msginfo->bytes = msginfo->dscrlen+ msginfo->datalen;

    armci_copy(info, msginfo +1, ilen);
    buf = ((char*)msginfo) + ilen + sizeof(request_header_t);
    *((long*)buf) =size;
    *(int*)(buf+ sizeof(long)) =rlen;
    armci_send_req(armci_master, msginfo, bufsize);
    if(rlen){
      msginfo->datalen = rlen;
      buf= armci_rcv_data(armci_master, msginfo);  /* receive response */
      armci_copy(buf, resp, rlen);
      FREE_SEND_BUFFER(msginfo);

      if(DEBUG_){printf("%d:client attaching got ptr=%p %d bytes\n",armci_me,buf,rlen);
         fflush(stdout);
      }
    }
}


/*\ server initializes its copy of the memory lock data structures
\*/
static void server_alloc_memlock(void *ptr_myclus)
{
int i;

    /* for protection, set pointers for processes outside local node NULL */
    memlock_table_array = calloc(armci_nproc,sizeof(void*));
    if(!memlock_table_array) armci_die("malloc failed for ARMCI lock array",0);

    /* set pointers for processes on local cluster node
     * ptr_myclus - corresponds to the master process
     */
    for(i=0; i< armci_clus_info[armci_clus_me].nslave; i++){
        memlock_table_array[armci_master +i] = ((char*)ptr_myclus)
                + MAX_SLOTS*sizeof(memlock_t)*i;
    }

    /* set pointer to the use flag */
#ifdef MEMLOCK_SHMEM_FLAG
    armci_use_memlock_table = (int*) (MAX_SLOTS*sizeof(memlock_t) +
                      (char*) memlock_table_array[armci_clus_last]);
    
    if(DEBUG_)
      fprintf(stderr,"server initialized memlock %p\n",armci_use_memlock_table);
#endif
}


static int allocate_memlock=1;

/*\ server actions triggered by client request to ATTACH
\*/
void armci_server_ipc(request_header_t* msginfo, void* descr,
                      void* buffer, int buflen)
{
   double *ptr;
   long *idlist = (long*)descr;
   long size = *(long*)buffer;
   int rlen = *(int*)(sizeof(long)+(char*)buffer);

   if(size<0) armci_die("armci_server_ipc: size<0",(int)size);
	ptr=(double*)Attach_Shared_Region(idlist+1,size,idlist[0]);
   if(!ptr)armci_die("armci_server_ipc: failed to attach",0);
   /* provide data server with access to the memory lock data structures */
   if(allocate_memlock){
      allocate_memlock = 0;
      server_alloc_memlock(ptr);
   }
#  ifndef HITACHI
   if(size>0)armci_set_mem_offset(ptr);
#endif
   if(msginfo->datalen != sizeof(long)+sizeof(int))
      armci_die("armci_server_ipc: bad msginfo->datalen ",msginfo->datalen);

   if(rlen==sizeof(ptr)){
     msginfo->datalen = rlen;
     armci_send_data(msginfo, &ptr);
   }else armci_die("armci_server_ipc: bad rlen",rlen);
}


/*\ send RMW request to server
\*/
void armci_rem_rmw(int op, int *ploc, int *prem, int extra, int proc)
{
request_header_t *msginfo;
char *buf;
void *buffer;
int bufsize = sizeof(request_header_t)+sizeof(long)+sizeof(void*);
 
    msginfo = (request_header_t*)GET_SEND_BUFFER(bufsize,op,proc);

    msginfo->dscrlen = sizeof(void*);
    msginfo->from  = armci_me;
    msginfo->to    = proc; 
    msginfo->operation = op;
    msginfo->datalen = sizeof(long);

    buf = (char*)(msginfo+1);
    ADDBUF(buf, void*, prem); /* pointer is shipped as descriptor */

    /* data field: extra argument in fetch&add and local value in swap */
    if(op==ARMCI_SWAP){
       ADDBUF(buf, int, *ploc); 
    }else if(op==ARMCI_SWAP_LONG) {
       ADDBUF(buf, long, *((long*)ploc) ); 
       msginfo->datalen = sizeof(long);
    }else {
       ADDBUF(buf, int, extra);
    }

    msginfo->bytes   = msginfo->datalen+msginfo->dscrlen ;

    if(DEBUG_){
        printf("%d sending RMW request %d to %d\n",armci_me,op,proc);
        fflush(stdout);
    }
    armci_send_req(proc, msginfo, bufsize);

    buffer = armci_rcv_data(proc,msginfo);  /* receive response */

    if(op==ARMCI_FETCH_AND_ADD || op== ARMCI_SWAP)
        *ploc = *(int*)buffer;
    else
        *(long*)ploc = *(long*)buffer;

    FREE_SEND_BUFFER(msginfo);
}


/*\ server response to RMW 
\*/
void armci_server_rmw(request_header_t* msginfo,void* ptr, void* pextra)
{
     long lold;
     int iold;
     void *pold=0;
     int op = msginfo->operation;

     if(DEBUG_){
        printf("%d server: executing RMW from %d\n",armci_me,msginfo->from);
        fflush(stdout);
     }
     if(msginfo->datalen != sizeof(long))
          armci_die2("armci_server_rmw: bad datalen=",msginfo->datalen,op);

     /* for swap operations *pextra has the  value to swap
      * for fetc&add it carries the increment argument
      */
     switch(op){
     case ARMCI_SWAP:
        iold = *(int*) pextra;
     case ARMCI_FETCH_AND_ADD:
        pold = &iold;
        break;

     case ARMCI_SWAP_LONG:
        lold = *(long*) pextra;
     case ARMCI_FETCH_AND_ADD_LONG:
        pold = &lold;
        break;

     default:
          armci_die("armci_server_rmw: bad operation code=",op);
     }

     armci_generic_rmw(op, pold, *(int**)ptr, *(int*) pextra, msginfo->to);

     armci_send_data(msginfo, pold);
}

extern int armci_direct_vector(request_header_t *msginfo , armci_giov_t darr[], int len, int proc);
int armci_rem_vector(int op, void *scale, armci_giov_t darr[],int len,int proc,int flag)
{
    char *buf,*buf0;
    request_header_t *msginfo;
    int bytes =0, s, slen=0;
    void *rem_ptr;
    size_t adr;
    int bufsize = sizeof(request_header_t);


    /* compute size of the buffer needed */
    for(s=0; s<len; s++){
        bytes   += darr[s].ptr_array_len * darr[s].bytes; /* data */
        bufsize += darr[s].ptr_array_len *sizeof(void*)+2*sizeof(int); /*descr*/
    }
            
    bufsize += bytes + sizeof(long) +2*sizeof(double) +8; /*+scale+allignment*/
#if defined(USE_SOCKET_VECTOR_API) 
    if(flag){
        int totaliovecs=0;
        /*if(op==PUT)*/bufsize-=bytes; 
        for(s=0; s<len; s++)
	    totaliovecs+=darr[s].ptr_array_len;
        buf = buf0= GET_SEND_BUFFER((bufsize+sizeof(struct iovec)*totaliovecs),op,proc);
    }
    else
#endif
    {
        buf = buf0= GET_SEND_BUFFER(bufsize,op,proc);
    }
    msginfo = (request_header_t*)buf;

    /* fill vector descriptor */
    buf += sizeof(request_header_t);
    ADDBUF(buf,long,len); /* number of sets */

    for(s=0;s<len;s++){

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
    msginfo->dscrlen = buf - buf0 - sizeof(request_header_t);
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

#if defined(USE_SOCKET_VECTOR_API) 
    if(flag&&(op==GET||op==PUT)){
    	armci_direct_vector(msginfo,darr,len,proc);
        return 0;
    }   
#endif 
    /* for put and accumulate copy data into buffer */
    if(op != GET){
/*       fprintf(stderr,"sending %lf\n",*(double*)darr[0].src_ptr_array[0]);*/
       
       armci_vector_to_buf(darr, len, buf);
    }

    armci_send_req(proc, msginfo, bufsize);

    if(op == GET){
       armci_rcv_vector_data(proc, msginfo, darr, len);
       FREE_SEND_BUFFER(msginfo);
    }
    return 0;
}


#define CHUN_ (8*8096)
#define CHUN 200000

/*\ client version of remote strided operation
\*/
int armci_rem_strided(int op, void* scale, int proc,
                       void *src_ptr, int src_stride_arr[],
                       void* dst_ptr, int dst_stride_arr[],
                       int count[], int stride_levels, int flag)
{
    char *buf, *buf0;
    request_header_t *msginfo;
    int  i, slen=0, bytes;
    size_t adr;
    void *rem_ptr;
    int  *rem_stride_arr;
    int bufsize = sizeof(request_header_t);

    /* calculate size of the buffer needed */
    for(i=0, bytes=1;i<=stride_levels;i++)bytes*=count[i];
    bufsize += bytes+sizeof(void*)+2*sizeof(int)*(stride_levels+1)
               +2*sizeof(double) + 8; /* +scale+alignment */
#   ifdef CLIENT_BUF_BYPASS
      if(flag && _armci_bypass) bufsize -=bytes; /* we are not sending data*/
#   endif
    

    if(flag){
#if defined(USE_SOCKET_VECTOR_API) 
	bufsize = sizeof(request_header_t)+sizeof(void*)+2*sizeof(int)*(stride_levels+1)+2*sizeof(double) + 8;
	
        buf = buf0= GET_SEND_BUFFER((bufsize+sizeof(struct iovec)*bytes/count[0]),op,proc);
#else
        if(op==GET)bufsize -=bytes;
         buf = buf0= GET_SEND_BUFFER(bufsize,op,proc);
#endif
    }
    else
    
    buf = buf0= GET_SEND_BUFFER(bufsize,op,proc);
    msginfo = (request_header_t*)buf;

    if(op == GET){
       rem_ptr = src_ptr;
       rem_stride_arr = src_stride_arr;
    }else{
       rem_ptr = dst_ptr;
       rem_stride_arr = dst_stride_arr;
    }
     
    msginfo->datalen=bytes;  
#if defined(USE_SOCKET_VECTOR_API) 
    /*****for making put use readv/writev is sockets*****/
    if(op==PUT && flag)
       msginfo->datalen=0;
    /* fill strided descriptor */
#endif
                                       buf += sizeof(request_header_t);
    *(void**)buf = rem_ptr;            buf += sizeof(void*);
    *(int*)buf = stride_levels;        buf += sizeof(int);
    for(i=0;i<stride_levels;i++)((int*)buf)[i] = rem_stride_arr[i];
                                       buf += stride_levels*sizeof(int);
    for(i=0;i< stride_levels+1;i++)((int*)buf)[i] = count[i];
                                       buf += (1+stride_levels)*sizeof(int);

#   ifdef CLIENT_BUF_BYPASS
      if(flag && _armci_bypass){
         /* to bypass the client MessageSnd buffer in get we need to add source
            pointer and stride info - server will put data directly there */
         ADDBUF(buf,void*,dst_ptr);
         for(i=0;i<stride_levels;i++)((int*)buf)[i] = dst_stride_arr[i];
                                       buf += stride_levels*sizeof(int);
         msginfo->bypass=1;
         msginfo->pinned=0; /* if set then pin is done before sending req*/
      }else{
         msginfo->bypass=0;
         msginfo->pinned=0;
      }
#   endif


    /* align buf for doubles (8-bytes) before copying data */
    adr = (size_t)buf;
    adr >>=3; 
    adr <<=3;
    adr +=8; 
    buf = (char*)adr;

    /* fill message header */
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
	
	/*
	if(ACC(op)) fprintf(stderr,"%d in client len=%d alpha=%lf)\n",
	             armci_me, buf - (char*)msginfo , ((double*)buf)[0]); 
	*/

    buf += slen;
    msginfo->dscrlen = buf - buf0 - sizeof(request_header_t);
    msginfo->bytes = msginfo->datalen+msginfo->dscrlen;

    if(op == GET){
#      ifdef CLIENT_BUF_BYPASS
         if(msginfo->bypass){

#ifdef MULTISTEP_PIN
          if(stride_levels==0 && !msginfo->pinned && count[0]>=400000){
              int seq=1;
              armci_send_req(proc,msginfo,bufsize);
              for(i=0; i< bytes; i+=CHUN){
                  int len= MIN(CHUN,(bytes-i));
                  char *p = i +(char*)dst_ptr;
  
#if 0
                  armci_pin_contig(p, len);
                  armci_client_send_ack(proc, seq);
#endif
                  seq++;
              }
                  armci_pin_contig(dst_ptr,CHUN);
                  armci_client_send_ack(proc, 1);
                  armci_pin_contig(CHUN+(char*)dst_ptr,count[0]-CHUN);
                  armci_client_send_ack(proc, seq-1);
             armci_rcv_strided_data_bypass(proc, msginfo->datalen,
                                           dst_ptr, stride_levels);
                  armci_unpin_contig(dst_ptr,CHUN);
                  armci_unpin_contig(CHUN+(char*)dst_ptr,count[0]-CHUN);
#if 0
                  armci_unpin_contig(dst_ptr,count[0]);
             for(i=0; i< bytes; i+=CHUN){
                  int len= MIN(CHUN,(bytes-i));
                  char *p = i +(char*)dst_ptr;
                  armci_unpin_contig(p, len);
             }
#endif
          }else
#endif
          {
             if(!msginfo->pinned) armci_send_req(proc,msginfo,bufsize);

             if(!armci_pin_memory(dst_ptr,dst_stride_arr,count, stride_levels))
                                         return 1; /* failed:cannot do bypass */

             if(msginfo->pinned) armci_send_req(proc,msginfo,bufsize);
             else armci_client_send_ack(proc, 1);
             armci_rcv_strided_data_bypass(proc, msginfo,dst_ptr,stride_levels);
             armci_unpin_memory(dst_ptr,dst_stride_arr,count, stride_levels);
          }

         }else
#      endif             
       {
          armci_send_req(proc, msginfo, bufsize);
          armci_rcv_strided_data(proc, msginfo, msginfo->datalen,
                                 dst_ptr, stride_levels, dst_stride_arr, count);
       }

       FREE_SEND_BUFFER(msginfo);

    } else{
       /* for put and accumulate send data */
       armci_send_strided(proc,msginfo, buf, 
                          src_ptr, stride_levels, src_stride_arr, count); 
    }

    return 0;
}




void armci_server(request_header_t *msginfo, char *dscr, char* buf, int buflen)
{
    int  buf_stride_arr[MAX_STRIDE_LEVEL+1];
    int  *loc_stride_arr,slen; 
    int  *count, stride_levels;
    void *buf_ptr, *loc_ptr;
    void *scale;
    char *dscr_save = dscr;
    int  rc, i,proc;
#   ifdef CLIENT_BUF_BYPASS
      int  *client_stride_arr=0; 
      void *client_ptr=0;
#   endif

    if(msginfo->operation==PUT && msginfo->datalen==0)return;/*return if using readv/socket for put*/
    /* unpack descriptor record */
    loc_ptr = *(void**)dscr;           dscr += sizeof(void*);
    stride_levels = *(int*)dscr;       dscr += sizeof(int);
    loc_stride_arr = (int*)dscr;       dscr += stride_levels*sizeof(int);
    count = (int*)dscr;                

    /* compute stride array for buffer */
    buf_stride_arr[0]=count[0];
    for(i=0; i< stride_levels; i++)
        buf_stride_arr[i+1]= buf_stride_arr[i]*count[i+1];

#   ifdef CLIENT_BUF_BYPASS
       if(msginfo->bypass){
          dscr += (1+stride_levels)*sizeof(int); /* move past count */
          GETBUF(dscr,void*,client_ptr);
          client_stride_arr = (int*)dscr; dscr += stride_levels*sizeof(int);
        }
#   endif

    /* get scale for accumulate, adjust buf to point to data */
    switch(msginfo->operation){
    case ARMCI_ACC_INT:     slen = sizeof(int); break;
    case ARMCI_ACC_DCP:     slen = 2*sizeof(double); break;
    case ARMCI_ACC_DBL:     slen = sizeof(double); break;
    case ARMCI_ACC_CPL:     slen = 2*sizeof(float); break;
    case ARMCI_ACC_FLT:     slen = sizeof(float); break;
	default:				slen=0;
    }

	scale = dscr_save+ (msginfo->dscrlen - slen);
/*
    if(ACC(msginfo->operation))
      fprintf(stderr,"%d in server len=%d slen=%d alpha=%lf\n", armci_me,
				 msginfo->dscrlen, slen, *(double*)scale); 
*/

    buf_ptr = buf; /*  data in buffer */

    proc = msginfo->to;

    if(msginfo->operation == GET){
    
#      ifdef CLIENT_BUF_BYPASS
         if(msginfo->bypass)
             armci_send_strided_data_bypass(proc, msginfo, buf, buflen,
                       loc_ptr, loc_stride_arr, 
                       client_ptr, client_stride_arr, count, stride_levels);

         else
#      endif

       armci_send_strided_data(proc, msginfo, buf,
                               loc_ptr, stride_levels, loc_stride_arr, count); 

    } else{

#ifdef PIPE_BUFSIZE
       if((msginfo->bytes==0) && (msginfo->operation==PUT)){
         armci_pipe_prep_receive_strided(msginfo,buf_ptr,stride_levels,
                    loc_stride_arr, count, buflen); 
         armci_pipe_receive_strided(msginfo,loc_ptr,loc_stride_arr, count,
                    stride_levels);
           
       } else
#endif
       if((rc = armci_op_strided(msginfo->operation, scale, proc,
               buf_ptr, buf_stride_arr, loc_ptr, loc_stride_arr,
               count, stride_levels, 1)))
               armci_die("server_strided: op from buf failed",rc);
    }
}



void armci_server_vector( request_header_t *msginfo, 
                          char *dscr, char* buf, int buflen)
{
    int  proc;
    long  len;
    void *scale;
    int  i,s;
    char *sbuf = buf;
    if(msginfo->operation==PUT && msginfo->datalen==0)return;/*return if using readv/socket for put*/
    /* unpack descriptor record */
    GETBUF(dscr, long ,len);
    
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
/*
          armci_copy(buf, ptr[s], bytes);
*/
          bcopy(buf, ptr[s], (size_t)bytes);
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

