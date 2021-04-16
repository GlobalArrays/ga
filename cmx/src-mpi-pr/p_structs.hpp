#ifndef _P_STRUCTS_H
#define _P_STRUCTS_H

#include "p_group.hpp"

/* data structures */

typedef enum {
    OP_PUT = 0,
    OP_PUT_PACKED,
    OP_PUT_DATATYPE,
    OP_PUT_IOV,
    OP_GET,
    OP_GET_PACKED,
    OP_GET_DATATYPE,
    OP_GET_IOV,
    OP_ACC_INT,
    OP_ACC_DBL,
    OP_ACC_FLT,
    OP_ACC_CPL,
    OP_ACC_DCP,
    OP_ACC_LNG,
    OP_ACC_INT_PACKED,
    OP_ACC_DBL_PACKED,
    OP_ACC_FLT_PACKED,
    OP_ACC_CPL_PACKED,
    OP_ACC_DCP_PACKED,
    OP_ACC_LNG_PACKED,
    OP_ACC_INT_IOV,
    OP_ACC_DBL_IOV,
    OP_ACC_FLT_IOV,
    OP_ACC_CPL_IOV,
    OP_ACC_DCP_IOV,
    OP_ACC_LNG_IOV,
    OP_FENCE,
    OP_FETCH_AND_ADD,
    OP_SWAP,
    OP_CREATE_MUTEXES,
    OP_DESTROY_MUTEXES,
    OP_LOCK,
    OP_UNLOCK,
    OP_QUIT,
    OP_MALLOC,
    OP_FREE,
    OP_NULL
} op_t;

/* structure to describe strided data transfers */
typedef struct {
  void *ptr;
  int stride_levels;
  cmxInt stride[CMX_MAX_STRIDE_LEVEL];
  cmxInt count[CMX_MAX_STRIDE_LEVEL+1];
} stride_t;

/* Internal struct for vector communication */
typedef struct {
  void **src; /* array of source starting addresses */
  void **dst; /* array of destination starting addresses */
  int count;  /* size of address arrays (src[count],dst[count]) */
  int bytes;  /* length in bytes for each src[i]/dst[i] pair */
} _cmx_giov_t;

typedef struct message_link {
  struct message_link *next;
  void *message;
  MPI_Request request;
  MPI_Datatype datatype;
  int need_free;
  stride_t *stride;
  _cmx_giov_t *iov;
} message_t;

typedef struct {
  int in_use;
  cmxInt send_size;
  message_t *send_head;
  message_t *send_tail;
  cmxInt recv_size;
  message_t *recv_head;
  message_t *recv_tail;
  CMX::p_Group *group;
} _cmx_request;

typedef _cmx_request cmx_request_t;

typedef struct {
  op_t operation;
  void *remote_address;
  void *local_address;
  int rank; /**< rank of target (rank of sender is iprobe_status.MPI_SOURCE */
  int length; /**< length of message/payload not including header */
} header_t;

/* keep track of all mutex requests */
typedef struct lock_link {
  struct lock_link *next;
  int rank;
} lock_t;

#endif //_P_STRUCTS_H
