/* $Id: buffers.c,v 1.7 2002-03-13 18:10:48 vinod Exp $    **/
#define SIXTYFOUR 64
#define DEBUG_  0
#define DEBUG2_ 0
#define EXTRA_ERR_CHECK     

/**********************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include "armcip.h"
#include "request.h"
#ifdef WIN32
#  include <windows.h>
   typedef unsigned long ssize_t;
#else
#  include <unistd.h>
#endif

#define ALIGN64ADD(buf) (SIXTYFOUR-(((ssize_t)(buf))%SIXTYFOUR))
/* the following symbols should be defined if needed in protocol specific
   header file:  BUF_EXTRA_FIELD, BUFID_PAD_T, BUF_ALLOCATE 
*/


#ifndef BUFID_PAD_T
#   define BUFID_PAD_T double
#endif
#ifndef  BUF_EXTRA_FIELD_T
#  define        SIZE_BUF_EXTRA_FIELD 0 
#  define BUF_TO_EBUF(buf) (buf_ext_t*)(((char*)buf) - sizeof(BUFID_PAD_T) -\
                                      SIZE_BUF_EXTRA_FIELD)
#else
#  define BUF_TO_EBUF(buf) (buf_ext_t*)(((char*)buf) - sizeof(BUFID_PAD_T) -\
				      sizeof(BUF_EXTRA_FIELD_T))
#endif
#ifndef BUF_ALLOCATE
#   define BUF_ALLOCATE malloc
#endif
#if defined(DATA_SERVER) && defined(SOCKETS)  
#define MAX_BUFS  1
#else
#define MAX_BUFS  4
#endif

#ifdef HITACHI
int numofbuffers=MAX_BUFS;
#endif
#ifndef MSG_BUFLEN_SMALL
#define MSG_BUFLEN_SMALL (MSG_BUFLEN >>0) 
#endif
#define LEFT_GUARD  11.11e11
#define RIGHT_GUARD 22.22e22
#define CLEAR_TABLE_SLOT(idx) *((int*)(_armci_buf_state->table+(idx))) =0

#ifdef STORE_BUFID
#   define BUF_TO_BUFINDEX(buf) (BUF_TO_EBUF((buf)))->id.bufid
#else
#  define BUF_TO_BUFINDEX(buf)\
          ((BUF_TO_EBUF((buf)))- _armci_buffers)/sizeof(buf_ext_t)
#endif

/* we allow multiple buffers (up to 15) per single request
 * adjacent buffers can be coalesced into a large one
 */
typedef struct {
  unsigned int op:8;     /* pending operation code */
  unsigned int snd:1;    /* if 1 then buffer is used for sending request */
  unsigned int rcv:1;    /* if 1 then buffer is used for receiving data */
  unsigned int async:1;  /* if 1 then request is nonblocking */
  unsigned int first:5;  /* id of the 1st buffer in the set in same request */
  unsigned int count:4;  /* how many buffers used for this request 8 possible*/
  unsigned int to:12;    /* serv/proc to which request was sent, 4096 possible*/
}buf_state_t;

/* message send buffer data structure */
typedef struct {
#ifdef STORE_BUFID
  union {
    int bufid;
    BUFID_PAD_T pad;
  }id;
#endif
# ifdef BUF_EXTRA_FIELD_T
        BUF_EXTRA_FIELD_T field;
# endif
  char buffer[MSG_BUFLEN_SMALL];
} buf_ext_t;

/* we keep table and buffer pointer together for better locality */
typedef struct {
  double left_guard;        /* stamp to verify if array was corrupted */
  buf_state_t table[MAX_BUFS]; /* array with state of buffer usage */
  buf_ext_t *buf;           /* address of buffer pool */
  buf_ext_t *largebuf;      /* address of the large buffer pool */
  int avail;
  int pad;
  double right_guard;       /* stamp to verify if array was corrupted */
} reqbuf_pool_t;            


buf_ext_t *_armci_buffers;        /* these are the actual buffers */
reqbuf_pool_t* _armci_buf_state;  /* array that describes state of each buf */ 


/*\ we allocate alligned buffer space
 *  this operation can be implemented in platform specific files
\*/ 
void _armci_buf_init()
{
char *tmp = BUF_ALLOCATE(MAX_BUFS*sizeof(buf_ext_t) + 64);
int  extra= ALIGN64ADD(tmp);

     if(sizeof(buf_state_t) != sizeof(int)) 
        armci_die("armci_buf_init size buf_state_t!=int",sizeof(buf_state_t));
                   
     _armci_buffers = (buf_ext_t *) (tmp + extra); 

     if(DEBUG2_){
	printf("%d:armci_init_bufs: pointer %p, before align ptr=%p bufptr=%p end of region is %p  size=%d extra=%d\n",
               armci_me,_armci_buffers,tmp,_armci_buffers->buffer,(_armci_buffers+MAX_BUFS),
               MAX_BUFS*sizeof(buf_ext_t),extra);
	fflush(stdout);
     }
     /* now allocate state array */
     tmp  = calloc(1, sizeof(reqbuf_pool_t) + 64);
	if(!tmp)armci_die("_armci_buf_init calloc failed",0);
     extra= ALIGN64ADD(tmp);
     _armci_buf_state = (reqbuf_pool_t*)(tmp + extra); 

     /* initialize it */
     _armci_buf_state->left_guard  = LEFT_GUARD;
     _armci_buf_state->right_guard = RIGHT_GUARD;
     _armci_buf_state->avail =0;
     _armci_buf_state->buf = _armci_buffers; 
}


/*\ convert buffer pointer to index (in state array)
\*/
int _armci_buf_to_index(void *buf)
{
int index;
char *ptr = (char*)buf;

   if(DEBUG2_){
     printf("%d: in _armci_buf_to_index %p\n",armci_me, buf);
     fflush(stdout);
   }

   index = BUF_TO_BUFINDEX(ptr);
   if((index >= MAX_BUFS)|| (index<0)) 
      armci_die2("armci_buf_to_index: bad index:",index,MAX_BUFS);
   
   return(index);
}



/*\  complete outstanding operation that uses the specified buffer
\*/
void _armci_buf_complete_index(int idx, int called)
{
int count;
buf_state_t *buf_state = _armci_buf_state->table +idx;
extern void _armci_asyn_complete_strided_get(int dsc_id, void *buf);

    count = buf_state->count;
    if(DEBUG_ ){
       printf("%d:complete_buf_index:%d op=%d first=%d count=%d called=%d\n",
              armci_me,idx,buf_state->op,buf_state->first,buf_state->count,
              called); 
       fflush(stdout);
    }

    if(buf_state->first != (unsigned int)idx){ 
      armci_die2("complete_buf_index:inconsistent index:",idx,buf_state->first);
    }

    if(buf_state->async){
      /* completion of strided get should release that buffer */
      if(buf_state->op == GET)
        _armci_asyn_complete_strided_get(idx,_armci_buf_state->buf[idx].buffer);
      else
         armci_die2("buf_complete_index: async mode not avail for this op",
                     buf_state->op,idx);
    }
#   ifdef BUF_EXTRA_FIELD_T
      else{
       /* need to call platform specific function */
       CLEAR_SEND_BUF_FIELD(_armci_buf_state->buf[idx].field,buf_state->snd,buf_state->rcv,buf_state->to);
      }
#   endif

    /* clear table slots for all the buffers in the set for this request */
    for(; count; count--, buf_state++) *(int*)buf_state = 0;
}


/*\ make sure that there are no other pending operations to that smp node
 *  this operation is called from platforms specific routine that sends
 *  request
 *  we could have accomplished the same in armci_buf_get but as Vinod
 *  is pointing out, it is better to delay completing outstanding
 *  calls to overlap memcpy for the current buffer with communication
\*/
void _armci_buf_ensure_one_outstanding_op_per_node(void *buf, int node)
{
    int i;
    char *ptr = (char*)buf;
    int index = BUF_TO_BUFINDEX(ptr);
    int this = _armci_buf_state->table[index].first;
    int nfirst, nlast;

    nfirst=armci_clus_info[node].master;
    nlast = nfirst+armci_clus_info[node].nslave-1;

    if((_armci_buf_state->table[index].to<(unsigned int) nfirst) || 
       (_armci_buf_state->table[index].to>(unsigned int) nlast))
        armci_die2("_armci_buf_ensure_one_outstanding_op_per_node: bad to",node,
                (int)_armci_buf_state->table[index].to);

    for(i=0;i<MAX_BUFS;i++){
        buf_state_t *buf_state = _armci_buf_state->table +i;
        if((buf_state->to >= nfirst) && (buf_state->to<= (unsigned int) nlast))
          if((buf_state->first != (unsigned int) this)&&(buf_state->first==(unsigned int) i) && buf_state->op)
                _armci_buf_complete_index(i,0);
    }
}

/*\ same as above but for process
\*/
void _armci_buf_ensure_one_outstanding_op_per_proc(void *buf, int proc)
{
    int i;
    char *ptr = (char*)buf;
    int index = BUF_TO_BUFINDEX(ptr);
    int this = _armci_buf_state->table[index].first;

    if(_armci_buf_state->table[index].to !=(unsigned int)  proc )
       armci_die2("_armci_buf_ensure_one_outstanding_op_per_proc: bad to", proc,
                (int)_armci_buf_state->table[index].to);

    for(i=0;i<MAX_BUFS;i++){
        buf_state_t *buf_state = _armci_buf_state->table +i;
        if(buf_state->to == (unsigned int) proc)
          if((buf_state->first != (unsigned int) this)&&(buf_state->first==(unsigned int) i) && buf_state->op)
                _armci_buf_complete_index(i,0);
    }
}


#define HISTORY__ 
#ifdef HISTORY
typedef struct{ int size; int op; int count; int id; } history_t;
history_t history[100];
int h=0;

void print_history()
{
int i;
    fflush(stdout);
    printf("%d records\n",h);
    for(i=0; i<h;i++) printf("size=%d id=%d ptr=%p count=%d op=%d\n",
        history[i].size, history[i].id,
       _armci_buf_state->buf[history[i].id].buffer, history[i].count,
        history[i].op);

    fflush(stdout);
}
#endif


/*\  call corresponding to GET_SEND_BUF
\*/
char *_armci_buf_get(int size, int operation, int to)
{
int avail=_armci_buf_state->avail;
int count=1, i;
   
    /* compute number of buffers needed (count) to satisfy the request */
    if((size > MSG_BUFLEN_SMALL) ){ 
       double val = (double)size;  /* use double due to a bug in gcc */
       val /= MSG_BUFLEN_SMALL;
       count=(int)val;
       if(size%MSG_BUFLEN_SMALL) count++; 
    }
    /* start from 0 if there is not enough bufs available from here */
    if((avail+count) > MAX_BUFS)avail = 0;

    /* avail should never point to buffer in a middle of a set of used bufs */
    if(_armci_buf_state->table[avail].op && 
      (_armci_buf_state->table[avail].first != (unsigned int) avail)){ sleep(1); 
              armci_die2("armci_buf_get: inconsistent first", avail,
                         _armci_buf_state->table[avail].first);
      }
 
    /* we need complete "count" number of buffers */
    for(i=0;i<count;i++){
        int cur = i +avail;
        if(_armci_buf_state->table[cur].op &&
           _armci_buf_state->table[cur].first==(unsigned int) cur)
                                   _armci_buf_complete_index(cur,1);
    }

    for(i=0; i<count; i++){
       _armci_buf_state->table[avail+i].op = operation;
       _armci_buf_state->table[avail+i].to = to;
       _armci_buf_state->table[avail+i].count=  count;
       _armci_buf_state->table[avail+i].first = avail;
    }

#ifdef STORE_BUFID
    _armci_buf_state->buf[avail].id.bufid=avail; 
#endif

# ifdef BUF_EXTRA_FIELD_T
    INIT_SEND_BUF(_armci_buf_state->buf[avail].field,_armci_buf_state->table[avail].snd,_armci_buf_state->table[avail].rcv);
#endif

#ifdef HITACHI
	PASSBUFID(_armci_buf_state->buf[avail].id.bufid);
#endif

#ifdef HISTORY
    history[h].size=size;
    history[h].op=operation;
    history[h].count=count;
    history[h].id = avail;
    h++;
#endif

    if(DEBUG_ || 0){
      printf("%d:buf_get:size=%d max=%d got %d ptr=%p count=%d op=%d to=%d\n",
             armci_me,size,MSG_BUFLEN_SMALL,avail,
            _armci_buf_state->buf[avail].buffer, count,operation,to);
      fflush(stdout);
    }

    /* select candidate buffer for next allocation request */
    _armci_buf_state->avail = avail+count;
    _armci_buf_state->avail %= MAX_BUFS;

    return(_armci_buf_state->buf[avail].buffer); 
}



/*\ release buffer when it becomes free
\*/
void _armci_buf_release(void *buf)
{
char *ptr = (char*)buf;
int  count, index = BUF_TO_BUFINDEX(ptr);
buf_state_t *buf_state = _armci_buf_state->table +index;
   if((index >= MAX_BUFS)|| (index<0))
      armci_die2("armci_buf_release: bad index:",index,MAX_BUFS);

   count =  _armci_buf_state->table[index].count;

   if(DEBUG_){
     printf("%d:_armci_buf_release %d ptr=%p count=%d op=%d\n",
            armci_me,index,buf,count, _armci_buf_state->table[index].op);
     fflush(stdout);
   }

   /* clear table slots for all the buffers in the set for this request */
   for(; count; count--, buf_state++) *(int*)buf_state = 0;

   /* the current buffer is prime candidate to satisfy next buffer request */
   _armci_buf_state->avail = index;
}


/*\ return pointer to buffer number id
\*/
char *_armci_buf_ptr_from_id(int id)
{
  if(id <0 || id >=MAX_BUFS) 
              armci_die2("armci_buf_ptr_from_id: bad id",id,MAX_BUFS);

  return(_armci_buf_state->buf[id].buffer);
}
