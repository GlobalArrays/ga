/* begin_generated_IBM_copyright_prolog                             */
/*                                                                  */
/* ---------------------------------------------------------------- */
/* (C)Copyright IBM Corp.  2007, 2008                               */
/* IBM CPL License                                                  */
/* ---------------------------------------------------------------- */
/*                                                                  */
/* end_generated_IBM_copyright_prolog                               */
/**
 * \file armci/src/x/dcmf/armcix_put.c
 * \brief DCMF ARMCI Extension for put operations.
 */

#include "armcix_impl.h"

typedef struct ARMCIX_DCMF_PutInfo_t
{
  void     * dst;
  unsigned   unused[3];
}
ARMCIX_DCMF_PutInfo_t __attribute__ ((__aligned__ (16)));

DCMF_Protocol_t __put_protocol;

/**
 * \brief DCMF ARMCI Extention receive short put operation callback
 *
 * \see DCMF_RecvSendShort
 */
void ARMCIX_DCMF_RecvPut1 (void           * clientdata,
                           const DCQuad   * msginfo,
                           unsigned         count,
                           unsigned         peer,
                           const char     * src,
                           unsigned         bytes)
{
  ARMCIX_DCMF_Connection_t * connection = (ARMCIX_DCMF_Connection_t *) clientdata;
  ARMCIX_DCMF_PutInfo_t * info = (ARMCIX_DCMF_PutInfo_t *) msginfo;

  memcpy (info->dst, src, bytes);

  ARMCIX_DCMF_SignalReceiveComplete (&connection[peer]);
}


/**
 * \brief DCMF ARMCI Extention receive put operation callback
 *
 * \see DCMF_RecvSend
 */
DCMF_Request_t * ARMCIX_DCMF_RecvPut2 (void             * clientdata,
                                       const DCQuad     * msginfo,
                                       unsigned           count,
                                       unsigned           peer,
                                       unsigned           sndlen,
                                       unsigned         * rcvlen,
                                       char            ** rcvbuf,
                                       DCMF_Callback_t  * cb_done)
{
  ARMCIX_DCMF_Connection_t * connection = (ARMCIX_DCMF_Connection_t *) clientdata;
  ARMCIX_DCMF_PutInfo_t * info = (ARMCIX_DCMF_PutInfo_t *) msginfo;

  *rcvlen = sndlen;
  *rcvbuf = info->dst;

  cb_done->function   = (void *)ARMCIX_DCMF_SignalReceiveComplete;
  cb_done->clientdata = (void *)&connection[peer];

  return &connection[peer].request;
}


/**
 * \brief Register the DCMF ARMCI Extention put operation.
 *
 * \param[in]  connection_array Connection array
 *
 * \see DCMF_Send_register
 */
void ARMCIX_DCMF_Put_register (ARMCIX_DCMF_Connection_t * connection_array)
{
  DCMF_CriticalSection_enter (0);

  DCMF_Send_Configuration_t configuration = {
    DCMF_DEFAULT_SEND_PROTOCOL,
    ARMCIX_DCMF_RecvPut1,
    connection_array,
    ARMCIX_DCMF_RecvPut2,
    connection_array
  };
  DCMF_Send_register (&__put_protocol, &configuration);

  DCMF_CriticalSection_exit (0);
}


/**
 * \brief ARMCI Extension blocking put operation.
 *
 * \param[in] src       Source buffer on the local node
 * \param[in] dst       Destination buffer on the remote node
 * \param[in] bytes     Number of bytes to transfer
 * \param[in] proc      Remote node rank
 *
 * \return ???
 */
int ARMCIX_Put( void * src, void * dst, int bytes, int proc)
{
  DCMF_CriticalSection_enter (0);

  volatile unsigned active = 1;
  DCMF_Callback_t cb_wait = { ARMCIX_DCMF_cb_decrement, (void *)&active };
  DCMF_Request_t request;

  ARMCIX_DCMF_PutInfo_t info;
  info.dst = dst;

  /* Must increment the origin sequence number because the target node does  */
  /* not knopw that the origin node is blocking on this operation.           */
  __connection[proc].sequence.origin++;

  DCMF_Send ( &__put_protocol,
              &request,
              cb_wait,
              DCMF_MATCH_CONSISTENCY,
              proc,
              bytes,
              (char *) src,
              (DCQuad *) &info,
              1);

#warning remove this ARMCIX_Fence() and implement some sort of ack scheme.
  ARMCIX_Fence (proc);
  while (active) DCMF_Messager_advance ();

  DCMF_CriticalSection_exit  (0);

  return 0;
}


/**
 * \brief ARMCI Extension non-blocking put operation.
 *
 * \param[in] src       Source buffer on the local node
 * \param[in] dst       Destination buffer on the remote node
 * \param[in] bytes     Number of bytes to transfer
 * \param[in] proc      Remote node rank
 * \param[in] nb_handle ARMCI non-blocking handle
 *
 * \return ???
 */
int ARMCIX_NbPut (void * src, void * dst, int bytes, int proc, armci_ihdl_t nb_handle)
{
  DCMF_CriticalSection_enter (0);

  armcix_dcmf_opaque_t * dcmf = (armcix_dcmf_opaque_t *) &nb_handle->cmpl_info;
  dcmf->active = 1;
  dcmf->connection = &__connection[proc];

  __connection[proc].sequence.origin++;
  __connection[proc].active++;
  __global_connection.active++;

  //fprintf (stderr, "ARMCIX_NbPut() -- nb_handle=%p, __connection[%d].sequence.origin=%d, __connection[%d].active=%d, __global_connection.active=%d\n", nb_handle, proc, __connection[proc].sequence.origin, proc, __connection[proc].active, __global_connection.active);

  DCMF_Callback_t cb_free = { ARMCIX_DCMF_NbOp_cb_done, nb_handle };
  DCMF_Request_t * new_request = ARMCIX_DCMF_request_allocate (cb_free);
  DCMF_Callback_t cb_done = { (void(*)(void *)) ARMCIX_DCMF_request_free, new_request };

  ARMCIX_DCMF_PutInfo_t info;
  info.dst = dst;

  DCMF_Send ( &__put_protocol,
              new_request,
              cb_done,
              DCMF_MATCH_CONSISTENCY,
              proc,
              bytes,
              (char *) src,
              (DCQuad *) &info,
              1);

  DCMF_CriticalSection_exit  (0);

  return 0;
}



/**
 * \brief ARMCI Extension blocking vector put operation.
 *
 * \param[in] darr      Descriptor array
 * \param[in] len       Length of descriptor array
 * \param[in] proc      Remote process(or) ID
 *
 * \return ???
 */
int ARMCIX_PutV (armci_giov_t * darr, int len, int proc)
{
#if 0
  DCMF_CriticalSection_enter (0);

  // Calculate the number of requests
  unsigned n = 0;
  unsigned i, j;
  for (i = 0; i < len; i++)
    for (j = 0; j < darr[i].ptr_array_len; j++)
      n++;

  /* Must increment the origin sequence number because the target node does  */
  /* not knopw that the origin node is blocking on this operation.           */
  __connection[proc].sequence.origin += n;

  volatile unsigned active = n;
  DCMF_Callback_t cb_wait = { ARMCIX_DCMF_cb_decrement, (void *)&active };
  DCMF_Request_t request[n];
  ARMCIX_DCMF_PutInfo_t info;

  for (i = 0; i < len; i++)
  {
    for (j = 0; j < darr[i].ptr_array_len; j++)
    {
      info.dst = darr[i].dst_ptr_array[j];
      DCMF_Send ( &__put_protocol,
                  &request[--n],
                  cb_wait,
                  DCMF_MATCH_CONSISTENCY,
                  proc,
                  darr[i].bytes,
                  (char *) darr[i].src_ptr_array[j],
                  (DCQuad *) &info,
                  1);
    }
  }
#warning remove this ARMCIX_Fence() and implement some sort of ack scheme.
  ARMCIX_Fence (proc);
  while (active) DCMF_Messager_advance ();

  DCMF_CriticalSection_exit  (0);
#endif

  armci_ireq_t nb_request;
  armci_ihdl_t nb_handle = (armci_ihdl_t) &nb_request;
  ARMCIX_NbPutV (darr, len, proc, nb_handle);
  ARMCIX_Wait (&nb_handle->cmpl_info);

  return 0;
}


/**
 * \brief ARMCI Extension non-blocking vector put operation.
 *
 * \param[in] darr      Descriptor array
 * \param[in] len       Length of descriptor array
 * \param[in] proc      Remote process(or) ID
 * \param[in] nb_handle ARMCI non-blocking handle
 *
 * \return ???
 */
int ARMCIX_NbPutV (armci_giov_t * darr, int len, int proc, armci_ihdl_t nb_handle)
{
  DCMF_CriticalSection_enter (0);

  //fprintf (stderr, "ARMCIX_NbPutV() >> len=%d, proc=%d\n", len, proc);

  // Calculate the number of requests
  unsigned n = 0;
  unsigned i, j;
  for (i = 0; i < len; i++)
    for (j = 0; j < darr[i].ptr_array_len; j++)
      n++;

  armcix_dcmf_opaque_t * dcmf = (armcix_dcmf_opaque_t *) &nb_handle->cmpl_info;
  dcmf->connection = &__connection[proc];
  dcmf->active = n;

  __connection[proc].sequence.origin += n;
  __connection[proc].active += n;
  __global_connection.active += n;

  //fprintf (stderr, "ARMCIX_NbPutV() -- __connection[%d].sequence.origin=%d, __connection[%d].active=%d, __global_connection.active=%d\n", proc, __connection[proc].sequence.origin, proc, __connection[proc].active, __global_connection.active);

  DCMF_Callback_t cb_free = { ARMCIX_DCMF_NbOp_cb_done, nb_handle };
  DCMF_Callback_t cb_done = { (void(*)(void *)) ARMCIX_DCMF_request_free, NULL };
  for (i = 0; i < len; i++)
  {
    for (j = 0; j < darr[i].ptr_array_len; j++)
    {
      //fprintf (stderr, "ARMCIX_NbPutV() -- src=%p, dst=%p, bytes=%d\n", darr[i].src_ptr_array[j], darr[i].dst_ptr_array[j], darr[i].bytes);
      DCMF_Request_t * new_request = ARMCIX_DCMF_request_allocate (cb_free);
      cb_done.clientdata = new_request;
      ARMCIX_DCMF_PutInfo_t info;
      info.dst = darr[i].dst_ptr_array[j];
      DCMF_Send ( &__put_protocol,
                  new_request,
                  cb_done,
                  DCMF_MATCH_CONSISTENCY,
                  proc,
                  darr[i].bytes,
                  (char *) darr[i].src_ptr_array[j],
                  (DCQuad *) &info,
                  1);
    }
  }
  //fprintf (stderr, "ARMCIX_NbPutV() <<\n");

  DCMF_CriticalSection_exit  (0);

  return 0;
}



unsigned ARMCIX_DCMF_PutS_recurse (void * src_ptr, int * src_stride_arr, 
                                   void * dst_ptr, int * dst_stride_arr, 
                                   int * seg_count, int stride_levels, int proc,
                                   armci_ihdl_t nb_handle)
{
  unsigned num_requests = 0;

  //fprintf (stderr, "ARMCIX_DCMF_PutS_recurse() >> \n");

  if (stride_levels == 0)
  {
    //fprintf (stderr, "ARMCIX_DCMF_PutS_recurse() dst=%p, src=%p, bytes=%d, nb_handle=%p, request=%p\n", dst_ptr, src_ptr, seg_count[0], nb_handle, request);

    DCMF_Callback_t cb_free = { ARMCIX_DCMF_NbOp_cb_done, nb_handle };
    DCMF_Request_t * new_request = ARMCIX_DCMF_request_allocate (cb_free);
    DCMF_Callback_t cb_done = { (void(*)(void *)) ARMCIX_DCMF_request_free, new_request };

    ARMCIX_DCMF_PutInfo_t info;
    info.dst = dst_ptr;
    DCMF_Send ( &__put_protocol,
                new_request,
                cb_done,
                DCMF_MATCH_CONSISTENCY,
                proc,
                seg_count[0],
                (char *) src_ptr,
                (DCQuad *) &info,
                1);

    num_requests++;
  }
  else
  {
    char * src_tmp = (char *) src_ptr;
    char * dst_tmp = (char *) dst_ptr;
    unsigned i;
    for (i = 0; i < seg_count[stride_levels]; i++)
    {
      num_requests += ARMCIX_DCMF_PutS_recurse (src_tmp, src_stride_arr, 
                                                dst_tmp, dst_stride_arr, 
                                                seg_count, (stride_levels-1), proc,
                                                nb_handle);

      src_tmp += src_stride_arr[(stride_levels-1)];
      dst_tmp += dst_stride_arr[(stride_levels-1)];
    }
  }

  //fprintf (stderr, "ARMCIX_DCMF_PutS_recurse() << num_requests = %d\n", num_requests);

  return num_requests;
}


/**
 * \brief ARMCI Extension blocking strided put operation.
 *
 * \param[in] src_ptr        pointer to 1st segment at source
 * \param[in] src_stride_arr array of strides at source
 * \param[in] dst_ptr        pointer to 1st segment at destination
 * \param[in] dst_stride_arr array of strides at destination
 * \param[in] seg_count      number of segments at each stride levels: count[0]=bytes
 * \param[in] stride_levels  number of stride levels
 * \param[in] proc           remote process(or) ID
 *
 * \return ???
 */
int ARMCIX_PutS (void * src_ptr, int * src_stride_arr, 
                 void * dst_ptr, int * dst_stride_arr, 
                 int * seg_count, int stride_levels, int proc)
{
  armci_ireq_t nb_request;
  armci_ihdl_t nb_handle = (armci_ihdl_t) &nb_request;
  ARMCIX_NbPutS (src_ptr, src_stride_arr, dst_ptr, dst_stride_arr,
                 seg_count, stride_levels, proc, nb_handle);
  ARMCIX_Wait (&nb_handle->cmpl_info);

  return 0;
}

/**
 * \brief ARMCI Extension non-blocking strided put operation.
 *
 * \param[in] src_ptr        pointer to 1st segment at source
 * \param[in] src_stride_arr array of strides at source
 * \param[in] dst_ptr        pointer to 1st segment at destination
 * \param[in] dst_stride_arr array of strides at destination
 * \param[in] seg_count      number of segments at each stride levels: count[0]=bytes
 * \param[in] stride_levels  number of stride levels
 * \param[in] proc           remote process(or) ID
 * \param[in] nb_handle      ARMCI non-blocking handle
 *
 * \return ???
 */
int ARMCIX_NbPutS (void * src_ptr, int * src_stride_arr, 
                   void * dst_ptr, int * dst_stride_arr, 
                   int * seg_count, int stride_levels, int proc,
                   armci_ihdl_t nb_handle)
{
  DCMF_CriticalSection_enter (0);

  // Calculate the number of requests
  unsigned i;
  unsigned n = 1;
  for (i = 0; i < stride_levels; i++) n = n * seg_count[i+1];

  armcix_dcmf_opaque_t * dcmf = (armcix_dcmf_opaque_t *) &nb_handle->cmpl_info;
  dcmf->connection = &__connection[proc];
  dcmf->active = n;

  __connection[proc].sequence.origin += n;
  __connection[proc].active += n;
  __global_connection.active += n;

  //fprintf (stderr, "ARMCIX_NbPutS() -- nb_handle=%p, &nb_handle->cmpl_info=%p, __connection[%d].sequence.origin=%d, __connection[%d].active=%d, __global_connection.active=%d\n", nb_handle, &nb_handle->cmpl_info, proc, __connection[proc].sequence.origin, proc, __connection[proc].active, __global_connection.active);

  unsigned count;
  count = ARMCIX_DCMF_PutS_recurse (src_ptr, src_stride_arr, 
                                    dst_ptr, dst_stride_arr, 
                                    seg_count, stride_levels, proc,
                                    nb_handle);

  //fprintf (stderr, "ARMCIX_NbPutS() -- n=%d == count=%d\n", n, count);
  assert (n == count);

  DCMF_CriticalSection_exit  (0);

  return 0;
}
