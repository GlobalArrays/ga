

#include "armcip.h"
#include <string.h> /* memcpy */
#include <stdio.h>

#define _MAX_AGG_BUFFERS   64   /* Maximum # of aggregation buffers available */
#define _MAX_AGG_BUFSIZE   2048 /* size of each buffer */
#define _MAX_AGG_HANDLE    _MAX_AGG_BUFFERS /* Max # of aggregation handles */

/* aggregate request handle */
typedef struct {
  unsigned int tag;      /* non-blocking request tag */
  short int proc;        /* remote process id */
  short int request_len; /* number of requests */
  short int buf_pos;     /* current position of the agg buffer */
  short int buf_pos_end; /* current ending position of the buffer */
  armci_giov_t *darr;    /* giov vectors */
}agg_req_t;
static agg_req_t *aggr[_MAX_AGG_HANDLE];  /* aggregate request handle */


/* data structure for dynamic buffer management */
typedef struct {
  int size; /* represents the size of the list (not linked list) */
  int index[_MAX_AGG_HANDLE];
} agg_list_t;
static agg_list_t ulist, alist;/*in-use & available aggr buffer index list*/


/* aggregation buffer */
static char agg_buf[_MAX_AGG_BUFFERS][_MAX_AGG_BUFSIZE];

/**
 * ---------------------------------------------------------------------
 * fill descriptor from this side (left to right)
 *            --->
 *             _______________________________________________
 *            |  |  |  |. .  .  .  .  .  .  .   .  . |  |  |  |
 *            |__|__|__|_____________________________|__|__|__|
 *	                                      
 *	                                             <---
 *	             fill src and dst pointer (arrays) in this direction
 *		     (right to left)
 *			  
 * Once they are about to cross each other (implies buffer is full), 
 * complete the data transfer.
 * ---------------------------------------------------------------------
 */

/* initialize/set the fields in the buffer*/
#define _armci_agg_set_buffer(index, tag, proc, len) { \
    aggr[(index)]->tag = (tag);       \
    aggr[(index)]->proc = (proc);     \
    aggr[(index)]->request_len = (len); \
    ulist.index[ulist.size++] = (index);/* add the new index to the in-use list and increment it's size*/ \
}

static int _armci_agg_get_buffer_index(int tag, int proc) {
    int i, index;
    
    /* check if there is an entry for this handle in the existing list*/
    for(i=ulist.size-1; i>=0; i--) {
      index = ulist.index[i];
      if(aggr[index]->tag == tag && aggr[index]->proc == proc)	return index;
    }
    
    /* else it is a new handle, so get a aggr buffer from either 
       of the lists. ???? don't throw exception here */
    if(ulist.size >= _MAX_AGG_BUFFERS && alist.size == 0)
      armci_die("_armci_agg_get_index: Too many outstanding aggregation requests\n", ulist.size);
    
    /*If there is a buffer in readily available list,use it*/
    if(alist.size > 0) index = alist.index[alist.size--];
    else { /* else use/get a buffer from the main list */
      index = ulist.size; 
      
      /* allocate memory for aggregate request handle */
      aggr[index] = (agg_req_t *)agg_buf[index];
      aggr[index]->buf_pos = sizeof(agg_req_t); /* update position of buffer */
      
      /* allocate memory for giov vector field in aggregate request handler */
      aggr[index]->darr = (armci_giov_t *)(agg_buf[index] +
					   aggr[index]->buf_pos);
      aggr[index]->buf_pos_end = _MAX_AGG_BUFSIZE;
      aggr[index]->request_len = 0;
    }
    
    _armci_agg_set_buffer(index, tag, proc, 0); 
    return index;
}

static void _armci_agg_update_lists(int index) {
    int i;
    /* remove that index from the in-use list and bring the last element 
       in the in-use list to the position of the removed one. */
    for(i=0; i<ulist.size; i++)
      if(ulist.index[i] == index) {
	ulist.index[i] = ulist.index[ulist.size-1];
	--(ulist.size);
	break;
      }
    
    /* and add the removed index to the available list and increment */
    alist.index[alist.size++] = index;
}

/* replace with macro later */
static int _armci_agg_get_buffer_position(int index, int *ptr_array_len, 
					  int is_registered_put, int bytes,
					  int *rid, armci_ihdl_t nb_handle) {
  int bytes_needed;
    
    /* memory required for this request */
    bytes_needed = 2 * (*ptr_array_len) * sizeof(void **);
    if(is_registered_put)  bytes_needed += bytes;

    aggr[index]->buf_pos += sizeof(armci_giov_t);
    
#if 0
    if(armci_me == 0) {
      printf("%d: %d %d %d\n", aggr[index]->request_len, bytes_needed, 
	     aggr[index]->buf_pos, aggr[index]->buf_pos_end);
    }
#endif
    
    /* If buffer is full, then complete data transfer */
    if(aggr[index]->buf_pos + bytes_needed > aggr[index]->buf_pos_end) {
      
      armci_agg_complete(nb_handle, SET);
      aggr[index]->buf_pos += sizeof(armci_giov_t);
          
      /* if buffer is still not big enough to hold all pointer array, then
	 do it by parts. determine a new ptr_array_len that fits buffer */
      if(bytes_needed >= aggr[index]->buf_pos_end - aggr[index]->buf_pos) {
        bytes_needed   = aggr[index]->buf_pos_end - aggr[index]->buf_pos;
        *ptr_array_len = bytes_needed/(2*sizeof(void **));
      }
    }
    
    *rid=aggr[index]->request_len++;/*get new request id*/

    /* update end position of buffer, according to usage of memory required 
       to store descriptor, src and dst pointer array */
    return (aggr[index]->buf_pos_end -= bytes_needed);
}


int armci_agg_save_descriptor(void *src, void *dst, int bytes, int proc, int op,
			      int is_registered_put, armci_ihdl_t nb_handle) {
    int pos; /* current position of the buffer */
    int rid; /* request id */
    int index, one=1, vptr_size = sizeof(void **);
    char *buf;

    /* set up the handle if it is a new aggregation request */
    AGG_INIT_NB_HANDLE(op, proc, nb_handle);
    
    /* get the index of the aggregation buffer to be used */
    index = _armci_agg_get_buffer_index(nb_handle->tag, nb_handle->proc);
    buf = agg_buf[index]; /* buffer used for this handle */
    
    /* get the current end position of the buffer and request id */
    pos=_armci_agg_get_buffer_position(index, &one, is_registered_put, 
				       bytes, &rid, nb_handle);

    if(is_registered_put) { /* if it is registered put, copy data into buffer */
      memcpy(&buf[pos], src, bytes);
      src = &buf[pos];
      pos += bytes;
    }
    
    /* malloc and, save source & destination pointers in descriptor */
    aggr[index]->darr[rid].src_ptr_array = (void **)&buf[pos]; pos+=vptr_size;
    aggr[index]->darr[rid].dst_ptr_array = (void **)&buf[pos]; pos+=vptr_size;
    
    aggr[index]->darr[rid].src_ptr_array[0] = src;
    aggr[index]->darr[rid].dst_ptr_array[0] = dst;
    aggr[index]->darr[rid].bytes = bytes;
    aggr[index]->darr[rid].ptr_array_len = 1;
    
    return 0;
}


int armci_agg_save_giov_descriptor(armci_giov_t darr[], int len, int proc, 
				   int op, armci_ihdl_t nb_handle) {  
    int pos; /* current position of the buffer */
    int rid; /* request id */
    int i, j, k, index, size, ptr_array_len;
    char *buf;

    /* set up the handle if it is a new aggregation request */
    AGG_INIT_NB_HANDLE(op, proc, nb_handle);

    /* get the index of the aggregation buffer to be used */
    index = _armci_agg_get_buffer_index(nb_handle->tag, nb_handle->proc);
    buf = agg_buf[index]; /* buffer used for this handle */

    for(i=0; i<len; i++) {
      k = 0;
      ptr_array_len = darr[i].ptr_array_len;
      do {
	/* get the current end position of the buffer */
	pos = _armci_agg_get_buffer_position(index, &ptr_array_len, 0, 0,
					     &rid, nb_handle);

	/* malloc, and save source & destination pointers in descriptor */
	size = ptr_array_len * sizeof(void **);
	aggr[index]->darr[rid].src_ptr_array = (void **)&buf[pos]; pos += size;
	aggr[index]->darr[rid].dst_ptr_array = (void **)&buf[pos]; pos += size;
	for(j=0; j<ptr_array_len; j++, k++) {
	  aggr[index]->darr[rid].src_ptr_array[j] = darr[i].src_ptr_array[k];
	  aggr[index]->darr[rid].dst_ptr_array[j] = darr[i].dst_ptr_array[k];
	}
	aggr[index]->darr[rid].bytes = darr[i].bytes;
	aggr[index]->darr[rid].ptr_array_len = ptr_array_len;
	ptr_array_len = darr[i].ptr_array_len - ptr_array_len;
      } while(k < darr[i].ptr_array_len);
    }
    return 0;
}

int armci_agg_save_strided_descriptor(void *src_ptr, int src_stride_arr[], 
				      void* dst_ptr, int dst_stride_arr[], 
				      int count[], int stride_levels, int proc,
				      int op, armci_ihdl_t nb_handle) {  
    int pos; /* current position of the buffer */
    int rid; /* request id */
    int i, j, k, index, size, ptr_array_len=1, total1D=1, num1D=0;
    int offset1, offset2, factor[MAX_STRIDE_LEVEL];
    char *buf;

    /* set up the handle if it is a new aggregation request */
    AGG_INIT_NB_HANDLE(op, proc, nb_handle);

    /* get the index of the aggregation buffer to be used */
    index = _armci_agg_get_buffer_index(nb_handle->tag, nb_handle->proc);
    buf = agg_buf[index]; /* buffer used for this handle */
    
    for(i=1; i<=stride_levels; i++) {
      total1D *= count[i]; 
      factor[i-1]=0;   
    }
      
    ptr_array_len = total1D;
    do {
      /* get the current end position of the buffer */
      pos = _armci_agg_get_buffer_position(index, &ptr_array_len, 0, 0,
					   &rid, nb_handle);
      
      /* malloc, and save source & destination pointers in descriptor */
      size = ptr_array_len * sizeof(void **);
      aggr[index]->darr[rid].src_ptr_array = (void **)&buf[pos]; pos += size;
      aggr[index]->darr[rid].dst_ptr_array = (void **)&buf[pos]; pos += size;
      
      /* converting stride into giov vector */
      for(i=0; i<ptr_array_len; i++) {
	for(j=0, offset1=0, offset2=0; j<stride_levels; j++) {
	  offset1 += src_stride_arr[j]*factor[j];
	  offset2 += dst_stride_arr[j]*factor[j];
	}
	aggr[index]->darr[rid].src_ptr_array[i] = (char *)src_ptr + offset1;
	aggr[index]->darr[rid].dst_ptr_array[i] = (char *)dst_ptr + offset2;
	++factor[0];
	++num1D;
	for(j=1; j<stride_levels; j++)
	  if(num1D%count[j]==0) { 
	    ++factor[j];
	    for(k=0; k<j;k++) factor[k]=0;
	  }
      }
      aggr[index]->darr[rid].bytes = count[0];
      aggr[index]->darr[rid].ptr_array_len = ptr_array_len;      
      ptr_array_len = total1D - ptr_array_len;
    } while(num1D < total1D);
    return 0;
}


void armci_agg_complete(armci_ihdl_t nb_handle, int condition) {
    int i, index=0, rc;
    
    /* get the buffer index for this handle */
    for(i=ulist.size-1; i>=0; i--) {
      index = ulist.index[i];
      if(aggr[index]->tag == nb_handle->tag && 
	 aggr[index]->proc == nb_handle->proc)	
	break;
    }
    if(i<0) return; /* implies this handle has no requests at all */
    
#if 0
    printf("%d: Hey Buddy! Aggregation Complete to remote process %d (%d:%d requests)\n", 
	   armci_me, nb_handle->proc, index, aggr[index]->request_len);
#endif

    /* complete the data transfer */
    if(aggr[index]->request_len) {
      switch(nb_handle->op) {
      case PUT:
	if((rc=ARMCI_PutV(aggr[index]->darr, aggr[index]->request_len, 
			  nb_handle->proc)))
	  ARMCI_Error("armci_agg_complete: putv failed",rc);
	break;
      case GET:
	if((rc=ARMCI_GetV(aggr[index]->darr, aggr[index]->request_len, 
			  nb_handle->proc)))
	  ARMCI_Error("armci_agg_complete: getv failed",rc);  
	break;
      }
    }

    /* setting request length to zero, as the requests are completed */
    aggr[index]->request_len = 0;
    aggr[index]->buf_pos = sizeof(agg_req_t);
    aggr[index]->buf_pos_end = _MAX_AGG_BUFSIZE;
    
    /* If armci_agg_complete() is called ARMCI_Wait(), then unset nb_handle*/
    if(condition==UNSET) { 
      nb_handle->proc = -1; /* nb_handle->tag  = 0; */
      _armci_agg_update_lists(index);
    }
}

