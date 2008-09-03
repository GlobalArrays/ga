/* begin_generated_IBM_copyright_prolog                             */
/*                                                                  */
/* ---------------------------------------------------------------- */
/* (C)Copyright IBM Corp.  2007, 2008                               */
/* IBM CPL License                                                  */
/* ---------------------------------------------------------------- */
/*                                                                  */
/* end_generated_IBM_copyright_prolog                               */
/**
 * \file armci/src/x/dcmf/armcix_impl.h
 * \brief DCMF ARMCI Extension implementation interface.
 */
#ifndef __armci_src_x_armcix_impl_h
#define __armci_src_x_armcix_impl_h

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "dcmf.h"
#include "../armcix.h"

typedef struct ARMCIX_DCMF_Connection_t
{
  DCMF_Request_t request;      /**< \todo lazy allocate the request object?  */
  unsigned active;             /**< Number of active messages to this peer   */
  unsigned peer;               /**< Maximum system size = 2^32               */
  struct
  {
    unsigned origin;
    unsigned target;
  } sequence;
  struct
  {
    unsigned watermark;
    unsigned origin     :1;
    unsigned target     :1;
    unsigned unused     :30;
  } fence;
  unsigned unused0;
  unsigned unused1;
}
ARMCIX_DCMF_Connection_t __attribute__ ((__aligned__ (16)));

typedef union ARMCIX_DCMF_Control_t
{
  DCMF_Control_t control;
  struct
  {
    unsigned isRequest;
    unsigned watermark;
  };
}
ARMCIX_DCMF_Control_t __attribute__ ((__aligned__ (16)));

typedef struct armcix_dcmf_opaque_t
{
  ARMCIX_DCMF_Connection_t * connection;
  unsigned                   active;
}
armcix_dcmf_opaque_t;


static inline armcix_dcmf_compile_time_assert ()
{
  COMPILE_TIME_ASSERT(sizeof(armcix_dcmf_opaque_t)<=sizeof(armcix_opaque_t));
}

extern ARMCIX_DCMF_Connection_t __global_connection;
extern ARMCIX_DCMF_Connection_t * __connection;

/**
 * \brief Generic decrement callback
 *
 * \param[in] clientdata The variable to decrement
 */
void ARMCIX_DCMF_cb_decrement (void * clientdata);

/**
 * \brief Callback function for non-blocking operations
 *
 * \param[in] clientdata The non-blocking handle to complete
 */
void ARMCIX_DCMF_NbOp_cb_done (void * clientdata);

/**
 * \brief Signal that a receive operation has completed.
 *
 * If there is an active fence operation this will complete the fence
 * and send an acknowledgement to the fence operation origin node.
 *
 * \param[in] connection The armci connection data for this peer.
 *
 * \see ARMCIX_DCMF_AcknowledgeFence
 * \see ARMCIX_DCMF_RecvPut1
 * \see ARMCIX_DCMF_RecvPut2
 */
void ARMCIX_DCMF_SignalReceiveComplete (ARMCIX_DCMF_Connection_t * connection);

/**
 * \brief Allocate a request from the free request pool
 *
 * Attempt to increase the size of the request pool if a free request is not
 * available. Otherwise, if the maximum request pool size has been reached,
 * block until a request completes and becomes available.
 *
 * \param[in] cb_free Callback to invoke with the request is free'd
 *
 * \return A free request
 *
 * \see ARMCIX_DCMF_request_free
 */
DCMF_Request_t * ARMCIX_DCMF_request_allocate (DCMF_Callback_t cb_free);

/**
 * \brief Release a request into the free request pool
 *
 * The callback associated with the request is invoked before the
 * request is released.
 *
 * \see ARMCIX_DCMF_request_allocate
 */
void ARMCIX_DCMF_request_free (DCMF_Request_t * request);


/**
 * \brief Register the DCMF ARMCI Extention get operation.
 *
 * \see DCMF_Get_register
 */
void ARMCIX_DCMF_Get_register ();

/**
 * \brief Register the DCMF ARMCI Extention put operation.
 *
 * \param[in]  connection_array Connection array
 *
 * \see DCMF_Send_register
 */
void ARMCIX_DCMF_Put_register (ARMCIX_DCMF_Connection_t * connection_array);

/**
 * \brief Register the DCMF ARMCI Extention accumulate operation.
 *
 * \param[in]  connection_array Connection array
 *
 * \see DCMF_Send_register
 */
void ARMCIX_DCMF_Acc_register (ARMCIX_DCMF_Connection_t * connection_array);

/**
 * \brief Register the DCMF ARMCI Extention fence operation.
 *
 * \param[in]  connection_array Connection array
 *
 * \see DCMF_Control_register
 */
void ARMCIX_DCMF_Fence_register (ARMCIX_DCMF_Connection_t * connection_array);


/**
 * \brief Register the DCMF ARMCI Extention rmw operation.
 *
 * \see DCMF_Control_register
 * \see DCMF_Send_register
 */
void ARMCIX_DCMF_Rmw_register ();







#endif
