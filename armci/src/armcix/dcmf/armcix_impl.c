/* begin_generated_IBM_copyright_prolog                             */
/*                                                                  */
/* ---------------------------------------------------------------- */
/* (C)Copyright IBM Corp.  2007, 2008                               */
/* IBM CPL License                                                  */
/* ---------------------------------------------------------------- */
/*                                                                  */
/* end_generated_IBM_copyright_prolog                               */
/**
 * \file armci/src/x/dcmf/armcix_impl.c
 * \brief DCMF ARMCI Extension implementation.
 */

#include "armcix_impl.h"
#include "strings.h"

ARMCIX_DCMF_Connection_t __global_connection;
ARMCIX_DCMF_Connection_t * __connection;

typedef struct ARMCIX_DCMF_Request_t
{
  DCMF_Request_t                 request;
  DCMF_Callback_t                cb_free;
  struct ARMCIX_DCMF_Request_t * next;
  unsigned                       unused;
} ARMCIX_DCMF_Request_t __attribute__ ((__aligned__ (16)));

typedef struct ARMCIX_DCMF_RequestPool_t
{
  ARMCIX_DCMF_Request_t * head;
  unsigned max;
  unsigned current;
  unsigned increment;
} ARMCIX_DCMF_RequestPool_t;


ARMCIX_DCMF_RequestPool_t __armcix_dcmf_requestpool;

void ARMCIX_DCMF_request_print (char * label)
{
  char str[1024];
  if (label == NULL) str[0] = 0;
  else snprintf (str, 1024, "[%s] ", label);

  fprintf (stderr, "%s__armcix_dcmf_requestpool { head = %p, max = %d, current = %d, increment = %d }\n", str, __armcix_dcmf_requestpool.head, __armcix_dcmf_requestpool.max, __armcix_dcmf_requestpool.current, __armcix_dcmf_requestpool.increment);

  ARMCIX_DCMF_Request_t * p = __armcix_dcmf_requestpool.head;
  while (p != NULL)
  {
    fprintf (stderr, "    (%p)->next = %p\n", p, p->next);
    p = p->next;
  }
}

void ARMCIX_DCMF_request_initialize (unsigned max, unsigned increment)
{
  unsigned count = max;
  if (increment > 0 && increment < max) count = increment;

  __armcix_dcmf_requestpool.head = (ARMCIX_DCMF_Request_t *) malloc (sizeof(ARMCIX_DCMF_Request_t) * count);
  assert (__armcix_dcmf_requestpool.head!=NULL);

  __armcix_dcmf_requestpool.max = max;
  __armcix_dcmf_requestpool.current = count;
  __armcix_dcmf_requestpool.increment = increment;

  unsigned i;
  for (i=1; i<count; i++) __armcix_dcmf_requestpool.head[i-1].next = & __armcix_dcmf_requestpool.head[i];
  __armcix_dcmf_requestpool.head[count-1].next = NULL;

  //ARMCIX_DCMF_request_print ("init");
}

DCMF_Request_t * ARMCIX_DCMF_request_allocate (DCMF_Callback_t cb_free)
{
  //ARMCIX_DCMF_request_print ("allocate");

  if (__armcix_dcmf_requestpool.head == NULL)
  {
    if (__armcix_dcmf_requestpool.current < __armcix_dcmf_requestpool.max)
    {
      unsigned previous = __armcix_dcmf_requestpool.current;
      // Allocate a new block of request objects and add them to the request pool.
      __armcix_dcmf_requestpool.head = 
        (ARMCIX_DCMF_Request_t *) malloc (sizeof(ARMCIX_DCMF_Request_t) * __armcix_dcmf_requestpool.increment);
      assert (__armcix_dcmf_requestpool.head!=NULL);

      __armcix_dcmf_requestpool.current += __armcix_dcmf_requestpool.increment;
      unsigned i;
      for (i=1; i<__armcix_dcmf_requestpool.increment; i++)
        __armcix_dcmf_requestpool.head[i-1].next = & __armcix_dcmf_requestpool.head[i];
      __armcix_dcmf_requestpool.head[__armcix_dcmf_requestpool.increment-1].next = NULL;
      //fprintf (stderr, "ARMCIX_DCMF_request_allocate() .. allocate a new block of requests (current = %d -> %d)\n", previous, __armcix_dcmf_requestpool.current);
    }
    else
    {
      // The request pool has already reached its maximum size, advance until a request is freed.
      do
      {
        DCMF_Messager_advance ();
      } while (__armcix_dcmf_requestpool.head == NULL);
    }
  }

  // Get the next free request object from the request pool, and set the
  // request pool pointer to the next available request object.
  ARMCIX_DCMF_Request_t * _request = (ARMCIX_DCMF_Request_t *) __armcix_dcmf_requestpool.head;
  __armcix_dcmf_requestpool.head = _request->next;

  // Initialize the new request object before return
  _request->cb_free = cb_free;
  _request->next = NULL;

  return (DCMF_Request_t *) _request;
}

void ARMCIX_DCMF_request_free (DCMF_Request_t * request)
{
  ARMCIX_DCMF_Request_t * _request = (ARMCIX_DCMF_Request_t *) request;

  // Invoke the "free" callback if it is specified.
  if (_request->cb_free.function != NULL)
    _request->cb_free.function (_request->cb_free.clientdata);

  // Return the request to the free request pool.
  _request->next = __armcix_dcmf_requestpool.head;
  __armcix_dcmf_requestpool.head = _request;
  //ARMCIX_DCMF_request_print ("free");
}

/**
 * \brief Generic decrement callback
 *
 * \param[in] clientdata Address of the variable to decrement
 */
void ARMCIX_DCMF_cb_decrement (void * clientdata)
{
  unsigned * value = (unsigned *) clientdata;
  (*value)--;
}

/**
 * \brief Callback function for non-blocking operations
 *
 * \param[in] clientdata The non-blocking handle to complete
 */
void ARMCIX_DCMF_NbOp_cb_done (void * clientdata)
{
  armci_ihdl_t nb_handle = (armci_ihdl_t) clientdata;

  armcix_dcmf_opaque_t * dcmf = (armcix_dcmf_opaque_t *) &nb_handle->cmpl_info;

  //fprintf (stderr, "ARMCIX_DCMF_NbOp_cb_done() >> dcmf=%p, dcmf->active=%d, dcmf->connection->active=%d, dcmf->connection->sequence.origin=%d, dcmf->tmp=%p, __global_connection.active=%d\n", dcmf, dcmf->active, dcmf->connection->active, dcmf->connection->sequence.origin, dcmf->tmp, __global_connection.active);
  
  //fprintf (stderr, "ARMCIX_DCMF_NbOp_cb_done() -- FREE MEMORY! -- dcmf=%p, dcmf->connection->sequence.origin=%d\n", dcmf, dcmf->connection->sequence.origin);
  dcmf->active--;
  dcmf->connection->active--;

  __global_connection.active--;

  //fprintf (stderr, "ARMCIX_DCMF_NbOp_cb_done() << dcmf->active=%d, dcmf->connection->active=%d, dcmf->connection->sequence.origin=%d, dcmf->tmp=%p, __global_connection.active=%d\n", dcmf->active, dcmf->connection->active, dcmf->connection->sequence.origin, dcmf->tmp, __global_connection.active);
}





static inline int
ENV_Bool(char * env, int * dval)
{
  int result = *dval;
  if(env != NULL)
    {
      if (strcmp(env, "0") == 0)
        result = 0;
      else if  (strcmp(env, "0") == 1)
        result = 1;
    }
  return *dval = result;
}

static inline int
ENV_Int(char * env, int * dval)
{
  int result = *dval;
  if(env != NULL)
    {
      result = (int) strtol((const char *)env, NULL, 10);
    }
  return *dval = result;
}


/**
 * \brief Initialize the DCMF ARMCI resources
 */
int ARMCIX_Init ()
{
  DCMF_CriticalSection_enter(0);

  DCMF_Messager_initialize ();

  unsigned size = DCMF_Messager_size ();

  posix_memalign ((void **)&__connection, 16, sizeof(ARMCIX_DCMF_Connection_t) * size);
  bzero ((void *)__connection, sizeof(ARMCIX_DCMF_Connection_t) * size);
  unsigned rank;
  for (rank = 0; rank < size; rank++) __connection[rank].peer = rank;

  __global_connection.peer = (unsigned) -1;

  /* Determine request pool defaults */
  unsigned ARMCIX_DCMF_REQUESTPOOL_MAX = 1000;
  ENV_Int (getenv ("ARMCIX_DCMF_REQUESTPOOL_MAX"), &ARMCIX_DCMF_REQUESTPOOL_MAX);
  unsigned ARMCIX_DCMF_REQUESTPOOL_INC = 0;
  ENV_Int (getenv ("ARMCIX_DCMF_REQUESTPOOL_INC"), &ARMCIX_DCMF_REQUESTPOOL_INC);
  ARMCIX_DCMF_request_initialize (ARMCIX_DCMF_REQUESTPOOL_MAX, ARMCIX_DCMF_REQUESTPOOL_INC);

  ARMCIX_DCMF_Get_register ();

  ARMCIX_DCMF_Put_register (__connection);

  ARMCIX_DCMF_Acc_register (__connection);

  ARMCIX_DCMF_Fence_register (__connection);

  ARMCIX_DCMF_Rmw_register ();

  /* Determine interrupt mode */
  int interrupts = 0;
  ENV_Bool (getenv ("DCMF_INTERRUPT"),  &interrupts);
  ENV_Bool (getenv ("DCMF_INTERRUPTS"), &interrupts);

  DCMF_Configure_t config;
  memset (&config, 0x00, sizeof(DCMF_Configure_t));
  config.interrupts = (interrupts==0)?DCMF_INTERRUPTS_OFF:DCMF_INTERRUPTS_ON;
  DCMF_Messager_configure (&config, &config);

  DCMF_Messager_configure (NULL, &config);

  //ARMCIX_DCMF_request_print ("after armcix_init");

  DCMF_CriticalSection_exit(0);
}
