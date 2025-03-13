#include <vector>
#include <set>
#include "p_environment.hpp"

/* This needs to be filled in */
#define CMX_ASSERT(WHAT)

#define CHECK_MPI_RETVAL(retval) check_mpi_retval((retval), __FILE__, __LINE__)

#if USE_MEMSET_AFTER_MALLOC
#define MAYBE_MEMSET(a,b,c) (void)memset(a,b,c)
#else
#define MAYBE_MEMSET(a,b,c) ((void)0)
#endif

CMX::p_Environment *CMX::p_Environment::p_instance = NULL;

namespace CMX {

/* world group state */
cmx_group_world_t g_state = {
    MPI_COMM_NULL,
    MPI_GROUP_NULL,
    -1,
    -1,
    NULL,
    NULL,
    MPI_COMM_NULL,
    -1,
    -1
};

typedef struct {
  int rank;
  void *ptr;
} rank_ptr_t;

/* the HEAD of the group linked list */
Group *group_list = NULL;

int _initialized = 0;

int _dbg_me;

const char *str_mpi_retval(int retval)
{
  const char *msg = NULL;

  switch(retval) {
    case MPI_SUCCESS       : msg = "MPI_SUCCESS"; break;
    case MPI_ERR_BUFFER    : msg = "MPI_ERR_BUFFER"; break;
    case MPI_ERR_COUNT     : msg = "MPI_ERR_COUNT"; break;
    case MPI_ERR_TYPE      : msg = "MPI_ERR_TYPE"; break;
    case MPI_ERR_TAG       : msg = "MPI_ERR_TAG"; break;
    case MPI_ERR_COMM      : msg = "MPI_ERR_COMM"; break;
    case MPI_ERR_RANK      : msg = "MPI_ERR_RANK"; break;
    case MPI_ERR_ROOT      : msg = "MPI_ERR_ROOT"; break;
    case MPI_ERR_GROUP     : msg = "MPI_ERR_GROUP"; break;
    case MPI_ERR_OP        : msg = "MPI_ERR_OP"; break;
    case MPI_ERR_TOPOLOGY  : msg = "MPI_ERR_TOPOLOGY"; break;
    case MPI_ERR_DIMS      : msg = "MPI_ERR_DIMS"; break;
    case MPI_ERR_ARG       : msg = "MPI_ERR_ARG"; break;
    case MPI_ERR_UNKNOWN   : msg = "MPI_ERR_UNKNOWN"; break;
    case MPI_ERR_TRUNCATE  : msg = "MPI_ERR_TRUNCATE"; break;
    case MPI_ERR_OTHER     : msg = "MPI_ERR_OTHER"; break;
    case MPI_ERR_INTERN    : msg = "MPI_ERR_INTERN"; break;
    case MPI_ERR_IN_STATUS : msg = "MPI_ERR_IN_STATUS"; break;
    case MPI_ERR_PENDING   : msg = "MPI_ERR_PENDING"; break;
    case MPI_ERR_REQUEST   : msg = "MPI_ERR_REQUEST"; break;
    case MPI_ERR_LASTCODE  : msg = "MPI_ERR_LASTCODE"; break;
    default                : msg = "DEFAULT"; break;
  }
  return msg;
}

p_Environment::p_Environment()
{
  int status = 0;
  int init_flag = 0;
  int i = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &_dbg_me);
  /* initialize many sytem state variables */
  num_mutexes = NULL;     /* (all) how many mutexes on each process */
  mutexes = NULL;         /* (masters) value is rank of lock holder */
  lq_heads.clear();        /* array of lock queues */
  semaphores = NULL;      /* semaphores for locking within SMP node */
  fence_array = NULL;

  nb_max_outstanding = CMX_MAX_NB_OUTSTANDING;
  nb_last_request = 0;
  nb_index = 0;
  nb_count_event = 0;
  nb_count_event_processed = 0;
  nb_count_send = 0;
  nb_count_send_processed = 0;
  nb_count_recv = 0;
  nb_count_recv_processed = 0;

  static_server_buffer = NULL;
  static_server_buffer_size = 0;
  eager_threshold = -1;
  max_message_size = -1;

  CMX_ENABLE_PUT_SELF = ENABLE_PUT_SELF;
  CMX_ENABLE_GET_SELF = ENABLE_GET_SELF;
  CMX_ENABLE_ACC_SELF = ENABLE_ACC_SELF;
  CMX_ENABLE_PUT_SMP = ENABLE_PUT_SMP;
  CMX_ENABLE_GET_SMP = ENABLE_GET_SMP;
  CMX_ENABLE_ACC_SMP = ENABLE_ACC_SMP;
  CMX_ENABLE_PUT_PACKED = ENABLE_PUT_PACKED;
  CMX_ENABLE_GET_PACKED = ENABLE_GET_PACKED;
  CMX_ENABLE_ACC_PACKED = ENABLE_ACC_PACKED;
  CMX_ENABLE_PUT_DATATYPE = ENABLE_PUT_DATATYPE;
  CMX_ENABLE_GET_DATATYPE = ENABLE_GET_DATATYPE;
  CMX_PUT_DATATYPE_THRESHOLD = 8192;
  CMX_GET_DATATYPE_THRESHOLD = 8192;
  CMX_ENABLE_PUT_IOV = ENABLE_PUT_IOV;
  CMX_ENABLE_GET_IOV = ENABLE_GET_IOV;
  CMX_ENABLE_ACC_IOV = ENABLE_ACC_IOV;

  if (_initialized) {
    return;
  }
  _initialized = 1;

  /* Assert MPI has been initialized */
  status = MPI_Initialized(&init_flag);
  _translate_mpi_error(status,"cmx_init");
  CHECK_MPI_RETVAL(status);
  assert(init_flag);

  /* initialize world group here */
  p_config.init(MPI_COMM_WORLD);

  /*MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);*/

  /* env vars */
  {
    char *value = NULL;
    nb_max_outstanding = CMX_MAX_NB_OUTSTANDING/2; /* default */
    value = getenv("CMX_MAX_NB_OUTSTANDING");
    if (NULL != value) {
      nb_max_outstanding = atoi(value);
    }
    CMX_ASSERT(nb_max_outstanding > 0);

    static_server_buffer_size = CMX_STATIC_BUFFER_SIZE; /* default */
    value = getenv("CMX_STATIC_BUFFER_SIZE");
    if (NULL != value) {
      static_server_buffer_size = atoi(value);
    }
    CMX_ASSERT(static_server_buffer_size > 0);

    eager_threshold = -1; /* default */
    value = getenv("CMX_EAGER_THRESHOLD");
    if (NULL != value) {
      eager_threshold = atoi(value);
    }

    CMX_ENABLE_PUT_SELF = ENABLE_PUT_SELF; /* default */
    value = getenv("CMX_ENABLE_PUT_SELF");
    if (NULL != value) {
      CMX_ENABLE_PUT_SELF = atoi(value);
    }

    CMX_ENABLE_GET_SELF = ENABLE_GET_SELF; /* default */
    value = getenv("CMX_ENABLE_GET_SELF");
    if (NULL != value) {
      CMX_ENABLE_GET_SELF = atoi(value);
    }

    CMX_ENABLE_ACC_SELF = ENABLE_ACC_SELF; /* default */
    value = getenv("CMX_ENABLE_ACC_SELF");
    if (NULL != value) {
      CMX_ENABLE_ACC_SELF = atoi(value);
    }

    CMX_ENABLE_PUT_SMP = ENABLE_PUT_SMP; /* default */
    value = getenv("CMX_ENABLE_PUT_SMP");
    if (NULL != value) {
      CMX_ENABLE_PUT_SMP = atoi(value);
    }

    CMX_ENABLE_GET_SMP = ENABLE_GET_SMP; /* default */
    value = getenv("CMX_ENABLE_GET_SMP");
    if (NULL != value) {
      CMX_ENABLE_GET_SMP = atoi(value);
    }

    CMX_ENABLE_ACC_SMP = ENABLE_ACC_SMP; /* default */
    value = getenv("CMX_ENABLE_ACC_SMP");
    if (NULL != value) {
      CMX_ENABLE_ACC_SMP = atoi(value);
    }

    CMX_ENABLE_PUT_PACKED = ENABLE_PUT_PACKED; /* default */
    value = getenv("CMX_ENABLE_PUT_PACKED");
    if (NULL != value) {
      CMX_ENABLE_PUT_PACKED = atoi(value);
    }

    CMX_ENABLE_GET_PACKED = ENABLE_GET_PACKED; /* default */
    value = getenv("CMX_ENABLE_GET_PACKED");
    if (NULL != value) {
      CMX_ENABLE_GET_PACKED = atoi(value);
    }

    CMX_ENABLE_ACC_PACKED = ENABLE_ACC_PACKED; /* default */
    value = getenv("CMX_ENABLE_ACC_PACKED");
    if (NULL != value) {
      CMX_ENABLE_ACC_PACKED = atoi(value);
    }

    CMX_ENABLE_PUT_DATATYPE = ENABLE_PUT_DATATYPE; /* default */
    value = getenv("CMX_ENABLE_PUT_DATATYPE");
    if (NULL != value) {
      CMX_ENABLE_PUT_DATATYPE = atoi(value);
    }

    CMX_ENABLE_GET_DATATYPE = ENABLE_GET_DATATYPE; /* default */
    value = getenv("CMX_ENABLE_GET_DATATYPE");
    if (NULL != value) {
      CMX_ENABLE_GET_DATATYPE = atoi(value);
    }

    CMX_PUT_DATATYPE_THRESHOLD = 8192; /* default */
    value = getenv("CMX_PUT_DATATYPE_THRESHOLD");
    if (NULL != value) {
      CMX_PUT_DATATYPE_THRESHOLD = atoi(value);
    }

    CMX_GET_DATATYPE_THRESHOLD = 8192; /* default */
    value = getenv("CMX_GET_DATATYPE_THRESHOLD");
    if (NULL != value) {
      CMX_GET_DATATYPE_THRESHOLD = atoi(value);
    }

    CMX_ENABLE_PUT_IOV = ENABLE_PUT_IOV; /* default */
    value = getenv("CMX_ENABLE_PUT_IOV");
    if (NULL != value) {
      CMX_ENABLE_PUT_IOV = atoi(value);
    }

    CMX_ENABLE_GET_IOV = ENABLE_GET_IOV; /* default */
    value = getenv("CMX_ENABLE_GET_IOV");
    if (NULL != value) {
      CMX_ENABLE_GET_IOV = atoi(value);
    }

    CMX_ENABLE_ACC_IOV = ENABLE_ACC_IOV; /* default */
    value = getenv("CMX_ENABLE_ACC_IOV");
    if (NULL != value) {
      CMX_ENABLE_ACC_IOV = atoi(value);
    }

    max_message_size = INT_MAX; /* default */
    value = getenv("CMX_MAX_MESSAGE_SIZE");
    if (NULL != value) {
      max_message_size = atoi(value);
    }

#if DEBUG
    if (0 == p_config.rank()) {
      printf("CMX_MAX_NB_OUTSTANDING=%d\n", nb_max_outstanding);
      printf("CMX_STATIC_BUFFER_SIZE=%d\n", static_server_buffer_size);
      printf("CMX_MAX_MESSAGE_SIZE=%d\n", max_message_size);
      printf("CMX_EAGER_THRESHOLD=%d\n", eager_threshold);
      printf("CMX_PUT_DATATYPE_THRESHOLD=%d\n", CMX_PUT_DATATYPE_THRESHOLD);
      printf("CMX_GET_DATATYPE_THRESHOLD=%d\n", CMX_GET_DATATYPE_THRESHOLD);
      printf("CMX_ENABLE_PUT_SELF=%d\n", CMX_ENABLE_PUT_SELF);
      printf("CMX_ENABLE_GET_SELF=%d\n", CMX_ENABLE_GET_SELF);
      printf("CMX_ENABLE_ACC_SELF=%d\n", CMX_ENABLE_ACC_SELF);
      printf("CMX_ENABLE_PUT_SMP=%d\n", CMX_ENABLE_PUT_SMP);
      printf("CMX_ENABLE_GET_SMP=%d\n", CMX_ENABLE_GET_SMP);
      printf("CMX_ENABLE_ACC_SMP=%d\n", CMX_ENABLE_ACC_SMP);
      printf("CMX_ENABLE_PUT_PACKED=%d\n", CMX_ENABLE_PUT_PACKED);
      printf("CMX_ENABLE_GET_PACKED=%d\n", CMX_ENABLE_GET_PACKED);
      printf("CMX_ENABLE_ACC_PACKED=%d\n", CMX_ENABLE_ACC_PACKED);
      printf("CMX_ENABLE_PUT_DATATYPE=%d\n", CMX_ENABLE_PUT_DATATYPE);
      printf("CMX_ENABLE_GET_DATATYPE=%d\n", CMX_ENABLE_GET_DATATYPE);
      printf("CMX_ENABLE_PUT_IOV=%d\n", CMX_ENABLE_PUT_IOV);
      printf("CMX_ENABLE_GET_IOV=%d\n", CMX_ENABLE_GET_IOV);
      printf("CMX_ENABLE_ACC_IOV=%d\n", CMX_ENABLE_ACC_IOV);
      fflush(stdout);
    }
#endif
  }

  /* mutexes */
  mutexes = NULL;
  num_mutexes = NULL;
  lq_heads.clear();

  /* reg_cache */
  /* note: every process needs a reg cache and it's always based on the
   * world rank and size */
  p_register.init(&p_config, &p_shmem);

  _malloc_semaphore();

#if DEBUG
  fprintf(stderr, "[%d] cmx_init() before progress server\n", p_config.rank());
#endif

#if PAUSE_ON_ERROR
  if ((SigSegvOrig=signal(SIGSEGV, SigSegvHandler)) == SIG_ERR) {
    p_error("signal(SIGSEGV, ...) error", -1);
  }
#endif

  status = _set_affinity(p_config.node_rank());
  CMX_ASSERT(0 == status);

  /* create a comm of only the workers */
  if (p_config.is_master()) {
    /* I'm a master */
    MPI_Comm delete_me;
    status = MPI_Comm_split(p_config.global_comm(), 0, p_config.rank(), &delete_me);
    CMX_ASSERT(MPI_SUCCESS == status);
    /* masters don't need their own comm */
    if (MPI_COMM_NULL != delete_me) {
      MPI_Comm_free(&delete_me);
    }
    p_CMX_GROUP_WORLD = NULL;
  } else {
    /* I'm a worker */
    MPI_Comm comm;
    status = MPI_Comm_split(
        p_config.global_comm(), 1, p_config.rank(), &comm);
    CMX_ASSERT(MPI_SUCCESS == status);
    int size;
    status = MPI_Comm_size(comm, &size);
    std::vector<int> ranks(size);
    for (i=0; i<size; i++) ranks[i] = i;
    p_CMX_GROUP_WORLD = new Group(size, &ranks[0], comm);
  }

  if (p_config.is_master()) {
    /* TODO: wasteful O(p) storage... */
   // mutexes = (int**)malloc(sizeof(int*) * g_state.size);
    mutexes = new int*[p_config.world_size()];
    CMX_ASSERT(mutexes);
    /* create one lock queue for each proc for each mutex */
   // lq_heads = (lock_t***)malloc(sizeof(lock_t**) * g_state.size);
    lq_heads.resize(p_config.world_size());
    /* start the server */
    _progress_server();
  }
  nb_list = new _cmx_request*[nb_max_outstanding];
  for (i=0; i<nb_max_outstanding; i++) nb_list[i] = NULL;

  _cmx_me = p_CMX_GROUP_WORLD->rank();

  /* Synch - Sanity Check */
  /* This barrier is on the world worker group */
  p_CMX_GROUP_WORLD->barrier();
  _translate_mpi_error(status, "cmx_init:MPI_Barrier");

  /* static state */
  fence_array = new char[p_config.world_size()];
  CMX_ASSERT(fence_array);
  for (i = 0; i < p_config.world_size(); ++i) {
    fence_array[i] = 0;
  }

#if DEBUG
  fprintf(stderr, "[%d] cmx_init() before barrier\n", p_config.rank());
#endif

  /* Synch - Sanity Check */
  /* This barrier is on the world worker group */
  p_CMX_GROUP_WORLD->barrier();
  _translate_mpi_error(status, "cmx_init:MPI_Barrier");

#if DEBUG
  fprintf(stderr, "[%d] cmx_init() success\n", p_config.rank());
#endif
}

/**
 * Translates the ranks of processes in one group to those in another group.  The
 * group making the call is the "from" group, the group in the argument list is
 * the "to" group.
 *
 * @param[in] n the number of ranks in the ranks_from and ranks_to arrays
 * @param[in] group_from the group to translate ranks from 
 * @param[in] ranks_from array of zero or more valid ranks in group_from
 * @param[in] group_to the group to translate ranks to 
 * @param[out] ranks_to array of corresponding ranks in group_to
 * @return CMX_SUCCESS on success
 */
int p_Environment::translateRanks(int n, Group *group_from,
    int *ranks_from, Group *group_to, int *ranks_to)
{
  MPI_Comm comm_from = group_from->MPIComm();
  MPI_Comm comm_to = group_to->MPIComm();
  MPI_Group group_f, group_t;
  int ret = CMX_SUCCESS;
  if (MPI_Comm_group(comm_from,&group_f)!=MPI_SUCCESS) ret=CMX_FAILURE;
  if (MPI_Comm_group(comm_to,&group_t)!=MPI_SUCCESS) ret=CMX_FAILURE;
  if (MPI_Group_translate_ranks(group_f,n,ranks_from,group_t,ranks_to)
      !=MPI_SUCCESS) ret=CMX_FAILURE;
  return ret;
}

/**
 * Translate the given rank from its group to its corresponding rank in the
 * world group. Convenience function for common case.
 *
 * @param[in] n the number of ranks in the group_ranks and world_ranks arrays
 * @param[in] group the group to translate ranks from 
 * @param[in] group_ranks the ranks to translate from
 * @param[out] world_ranks the corresponding world rank
 * @return CMX_SUCCESS on success
 */
int p_Environment::translateWorld(int n, Group *group, int *group_ranks,
    int *world_ranks)
{
  int i;
  for (i=0; i<n; i++) {
    world_ranks[i] = p_config.get_world_rank(group,group_ranks[i]);
  }
  return CMX_SUCCESS;
}

/**
 * Close down most environment functions. Should be called right before
 * MPI_Finalize
 */
void p_Environment::finalize()
{
  int i, ierr;

  /* it's okay to call multiple times -- extra calls are no-ops */
  if (!_initialized) {
    return;
  }

  p_CMX_GROUP_WORLD->barrier();

  _initialized = 0;

  _free_semaphore();

  p_CMX_GROUP_WORLD->barrier();

  /* clean up non-blocking calls */
  for (i=0; i<nb_max_outstanding; i++) {
    if (nb_list[i]) {
      nb_wait_for_all(nb_list[i]);
    }
  }
  delete [] nb_list;

  /* send quit message to thread */
  int my_rank_to_free;
  int is_notifier = 0; 

  my_rank_to_free = p_config.get_my_rank_to_free(p_config.rank(),
      p_config.node_size(), p_CMX_GROUP_WORLD);
  // is_notifier = p_config.rank() == smallest_rank_with_same_hostid + g_state.node_size*
  //   ((p_config.rank() - smallest_rank_with_same_hostid)/g_state.node_size);
  // if (_smallest_world_rank_with_same_hostid(group_list) == p_config.rank()) 
  if(is_notifier = my_rank_to_free == p_config.rank())
  {
    int my_master = -1;
    header_t *header = NULL;
    char *message = NULL;
    _cmx_request nb;
    nb_register_request(&nb);

    my_master = p_config.master(p_config.rank());
    message = new char[sizeof(header_t)];
    header = reinterpret_cast<header_t*>(message);
    CMX_ASSERT(header);
    MAYBE_MEMSET(header, 0, sizeof(header_t));
    header->operation = OP_QUIT;
    header->remote_address = NULL;
    header->local_address = NULL;
    header->rank = 0;
    header->length = 0;
    nb_send_header(message, sizeof(header_t), my_master, &nb);
    /* this call will free up the header allocation */
    nb_wait_for_all(&nb);
  }

  delete [] fence_array;

  ierr = MPI_Barrier(p_config.global_comm());
  _translate_mpi_error(ierr, "cmx_finalize:MPI_Barrier");

  /* reg_cache */
  p_register.destroy();

  /* destroy the groups */
}

/**
 * Return an instance of the p_Environment singleton
 * @return pointer to p_Environment singleton
 */
p_Environment *p_Environment::instance()
{
  if (p_instance == NULL) {
    p_instance = new p_Environment();
  }
  return p_instance;
}

p_Environment::~p_Environment()
{
#if 0
  int i, ierr;
#if DEBUG
  fprintf(stderr, "[%d] cmx_finalize()\n", p_config.rank());
#endif

  /* it's okay to call multiple times -- extra calls are no-ops */
  if (!_initialized) {
    return;
  }
  /* just return if this is a progress rank */
  if (p_config.rank() == g_state.master[p_config.rank()]) return;

  p_CMX_GROUP_WORLD->barrier();

  _initialized = 0;

  _free_semaphore();

#ifdef OLD_CODE
  /* Make sure that all outstanding operations are done */
  waitAll(p_CMX_GROUP_WORLD);
#endif

  p_CMX_GROUP_WORLD->barrier();

  /* clean up non-blocking calls */
  for (i=0; i<nb_max_outstanding; i++) {
    if (nb_list[i]) {
      nb_wait_for_all(nb_list[i]);
    }
  }
  free(nb_list);

  /* send quit message to thread */
  int smallest_rank_with_same_hostid, largest_rank_with_same_hostid; 
  int num_progress_ranks_per_node, is_node_ranks_packed;
  int my_rank_to_free;
  int is_notifier = 0; 

  num_progress_ranks_per_node = get_num_progress_ranks_per_node();
  is_node_ranks_packed = get_progress_rank_distribution_on_node();
  smallest_rank_with_same_hostid = _smallest_world_rank_with_same_hostid(p_CMX_GROUP_WORLD);
  largest_rank_with_same_hostid = _largest_world_rank_with_same_hostid(p_CMX_GROUP_WORLD);
  my_rank_to_free = get_my_rank_to_free(p_config.rank(),
      g_state.node_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
      num_progress_ranks_per_node, is_node_ranks_packed);
  // is_notifier = p_config.rank() == smallest_rank_with_same_hostid + g_state.node_size*
  //   ((p_config.rank() - smallest_rank_with_same_hostid)/g_state.node_size);
  // if (_smallest_world_rank_with_same_hostid(group_list) == p_config.rank()) 
  if(is_notifier = my_rank_to_free == p_config.rank())
  {
    int my_master = -1;
    header_t *header = NULL;
    _cmx_request nb;
    nb_request_init(&nb);

    my_master = g_state.master[p_config.rank()];
    header = (header_t*)malloc(sizeof(header_t));
    CMX_ASSERT(header);
    MAYBE_MEMSET(header, 0, sizeof(header_t));
    header->operation = OP_QUIT;
    header->remote_address = NULL;
    header->local_address = NULL;
    header->rank = 0;
    header->length = 0;
    nb_send_header(header, sizeof(header_t), my_master, &nb);
    /* this call will free up the header allocation */
    nb_wait_for_all(&nb);
  }

  delete [] fence_array;

  ierr = MPI_Barrier(g_state.comm);
  _translate_mpi_error(ierr, "cmx_finalize:MPI_Barrier");

  /* reg_cache */
  p_register.destroy();

  /* destroy the groups */
#ifdef OLD_CODE
  cmx_group_finalize();
#endif
#endif
}

/**
 * Fence on all processes in group
 * @param group fence all process in group
 */
void p_Environment::fence(Group *group)
{
  int p, ip;
  int count_before = 0;
  int count_after = 0;
  _cmx_request nb;
  /* NOTE: We always fence on the world group */

  /* count how many fence messagse to send */
  int size = group->size();
  /* find all processes and their masters in group */
  std::set<int> fenced_procs;
  std::set<int>::iterator it;
  for (ip=0; ip<size; ip++) {
    p = p_config.get_world_rank(group, ip);
    int master = p_config.master(p);
    if (fenced_procs.find(p) == fenced_procs.end()) fenced_procs.insert(p);
    if (fenced_procs.find(master) == fenced_procs.end())
      fenced_procs.insert(master);

  }
  it = fenced_procs.begin();
  while (it != fenced_procs.end()) {
    p = *it;
    if (fence_array[p]) {
      ++count_before;
    }
    it++;
  }

  /* check for no outstanding put/get requests */
  if (0 == count_before) {
    return;
  }

#if NEED_ASM_VOLATILE_MEMORY
#if DEBUG
  fprintf(stderr, "[%d] comex_fence_all asm volatile (\"\" : : : \"memory\"); \n",
      p_config.rank(), group);
#endif
  asm volatile ("" : : : "memory");
#endif

  /* optimize by only sending to procs which we have outstanding messages */
  nb_request_init(&nb);
  it = fenced_procs.begin();
  while (it != fenced_procs.end()) {
    p = *it;
    if (fence_array[p]) {
      int p_master = p_config.master(p);
      char *message = NULL;
      header_t *header = NULL;

      /* because we only fence to masters */
      CMX_ASSERT(p_master == p);

      /* prepost recv for acknowledgment */
      nb_recv(NULL, 0, p_master, &nb);

      /* post send of fence request */
      message = new char[sizeof(header_t)];
      header = reinterpret_cast<header_t*>(message);
      CMX_ASSERT(header);
      MAYBE_MEMSET(header, 0, sizeof(header_t));
      header->operation = OP_FENCE;
      header->remote_address = NULL;
      header->local_address = NULL;
      header->length = 0;
      header->rank = 0;
      nb_send_header(header, sizeof(header_t), p_master, &nb);
    }
    it++;
  }

  nb_wait_for_all(&nb);

  it = fenced_procs.begin();
  while (it != fenced_procs.end()) {
    p = *it;
    if (fence_array[p]) {
      fence_array[p] = 0;
      ++count_after;
    }
    it++;
  }

  CMX_ASSERT(count_before == count_after);
}

/**
 * Return a pointer to struct containing global state. This is used by
 * allocations
 * @return global state
 */
cmx_group_world_t* p_Environment::getGlobalState()
{
  return &g_state;
}


void p_Environment::p_error(const std::string msg, int code)
{
#if DEBUG
  fprintf(stderr, "[%d] Received an Error in Communication: (%d) %s\n",
      p_config.rank(), code, msg.c_str());
#if DEBUG_TO_FILE
  fclose(cmx_trace_file);
#endif
#endif
  fprintf(stderr,"[%d] Received an Error in Communication: (%d) %s\n",
      p_config.rank(), code, msg.c_str());

  MPI_Abort(p_config.global_comm(), code);
}

int p_Environment::_eager_check(int extra_bytes)
{
  return (((int)sizeof(header_t))+extra_bytes) < eager_threshold;
}


void p_Environment::_fence_master(int master_rank)
{
#if DEBUG
  printf("[%d] _fence_master(master=%d)\n", p_config.rank(), master_rank);
#endif

  if (fence_array[master_rank]) {
    header_t *header = NULL;
    char *message = NULL;
    _cmx_request nb;
    nb_request_init(&nb);

    /* prepost recv for acknowledgment */
    nb_recv(NULL, 0, master_rank, &nb);

    /* post send of fence request */
    message = new char[sizeof(header_t)];
    header = reinterpret_cast<header_t*>(message);
    CMX_ASSERT(header);
    MAYBE_MEMSET(header, 0, sizeof(header_t));
    header->operation = OP_FENCE;
    header->remote_address = NULL;
    header->local_address = NULL;
    header->length = 0;
    header->rank = 0;
    nb_send_header(header, sizeof(header_t), master_rank, &nb);
    /* this call will free up the header allocation */
    nb_wait_for_all(&nb);
    fence_array[master_rank] = 0;
    delete [] message;
  }
}

int64_t p_Environment::_packed_size(int64_t *src_stride, int64_t *count, int stride_levels)
{
  int64_t size;
  int64_t i;
  int64_t n1dim;  /* number of 1 dim block */

  CMX_ASSERT(stride_levels >= 0);
  CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);
  CMX_ASSERT(NULL != src_stride);
  CMX_ASSERT(NULL != count);
  CMX_ASSERT(count[0] > 0);

#if DEBUG
  fprintf(stderr, "[%d] _packed_size(src_stride=%p, count[0]=%d, stride_levels=%d)\n",
      p_config.rank(), src_stride, count[0], stride_levels);
#endif

  /* number of n-element of the first
   * dimension */
  n1dim = 1;
  for(i=1; i<=stride_levels; i++) {
    n1dim *= count[i];
  }

  /* allocate packed buffer now
   * that we know the size */
  size = n1dim * count[0];

  return size;
}

char* p_Environment::pack(
    char *src, int64_t *src_stride, int64_t *count, int stride_levels, int64_t *size)
{
  int64_t i, j;
  int64_t src_idx;  /* index offset of current block position to ptr */
  int64_t n1dim;  /* number of 1 dim block */
  int64_t src_bvalue[7], src_bunit[7];
  int64_t packed_index = 0;
  char *packed_buffer = NULL;

  CMX_ASSERT(stride_levels >= 0);
  CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);
  CMX_ASSERT(NULL != src);
  CMX_ASSERT(NULL != src_stride);
  CMX_ASSERT(NULL != count);
  CMX_ASSERT(count[0] > 0);
  CMX_ASSERT(NULL != size);

#if DEBUG
  fprintf(stderr, "[%d] pack(src=%p, src_stride=%p, count[0]=%d, stride_levels=%d)\n",
      p_config.rank(), src, src_stride, count[0], stride_levels);
#endif

  /* number of n-element of the first dimension */
  n1dim = 1;
  for(i=1; i<=stride_levels; i++) {
    n1dim *= count[i];
  }

  /* allocate packed buffer now that we know the size */
  packed_buffer = (char*)malloc(n1dim * count[0]);
  CMX_ASSERT(packed_buffer);

  /* calculate the destination indices */
  src_bvalue[0] = 0; src_bvalue[1] = 0; src_bunit[0] = 1; src_bunit[1] = 1;

  for(i=2; i<=stride_levels; i++) {
    src_bvalue[i] = 0;
    src_bunit[i] = src_bunit[i-1] * count[i-1];
  }

  for(i=0; i<n1dim; i++) {
    src_idx = 0;
    for(j=1; j<=stride_levels; j++) {
      src_idx += (int64_t) src_bvalue[j] * (int64_t) src_stride[j-1];
      if((i+1) % src_bunit[j] == 0) {
        src_bvalue[j]++;
      }
      if(src_bvalue[j] > (count[j]-1)) {
        src_bvalue[j] = 0;
      }
    }

    (void)memcpy(&packed_buffer[packed_index], &src[src_idx], count[0]);
    packed_index += count[0];
  }

  CMX_ASSERT(packed_index == n1dim*count[0]);
  *size = packed_index;

  return packed_buffer;
}


void p_Environment::unpack(char *packed_buffer,
    char *dst, int64_t *dst_stride, int64_t *count, int stride_levels)
{
  int64_t i, j;
  int64_t dst_idx;  /* index offset of current block position to ptr */
  int64_t n1dim;  /* number of 1 dim block */
  int64_t dst_bvalue[7], dst_bunit[7];
  int64_t packed_index = 0;

  CMX_ASSERT(stride_levels >= 0);
  CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);
  CMX_ASSERT(NULL != packed_buffer);
  CMX_ASSERT(NULL != dst);
  CMX_ASSERT(NULL != dst_stride);
  CMX_ASSERT(NULL != count);
  CMX_ASSERT(count[0] > 0);

#if DEBUG
  fprintf(stderr, "[%d] unpack(dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d)\n",
      p_config.rank(), dst, dst_stride, count[0], stride_levels);
#endif

  /* number of n-element of the first dimension */
  n1dim = 1;
  for(i=1; i<=stride_levels; i++) {
    n1dim *= count[i];
  }

  /* calculate the destination indices */
  dst_bvalue[0] = 0; dst_bvalue[1] = 0; dst_bunit[0] = 1; dst_bunit[1] = 1;

  for(i=2; i<=stride_levels; i++) {
    dst_bvalue[i] = 0;
    dst_bunit[i] = dst_bunit[i-1] * count[i-1];
  }

  for(i=0; i<n1dim; i++) {
    dst_idx = 0;
    for(j=1; j<=stride_levels; j++) {
      dst_idx += (int64_t) dst_bvalue[j] * (int64_t) dst_stride[j-1];
      if((i+1) % dst_bunit[j] == 0) {
        dst_bvalue[j]++;
      }
      if(dst_bvalue[j] > (count[j]-1)) {
        dst_bvalue[j] = 0;
      }
    }

    (void)memcpy(&dst[dst_idx], &packed_buffer[packed_index], count[0]);
    packed_index += count[0];
  }

  CMX_ASSERT(packed_index == n1dim*count[0]);
}


char* p_Environment::_generate_shm_name(int rank)
{
  int snprintf_retval = 0;
  /* /cmxUUUUUUUUUUPPPPPPPPPPCCCCCCN */
  /* 0000000001111111111222222222233 */
  /* 1234567890123456789012345678901 */
  char *name = NULL;
  static const unsigned int limit = 62;
  static const char letters[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  static unsigned int counter[6] = {0};

  CMX_ASSERT(rank >= 0);
  name = (char*)malloc(SHM_NAME_SIZE*sizeof(char));
  CMX_ASSERT(name);
  snprintf_retval = snprintf(name, SHM_NAME_SIZE,
      "/cmx%010u%010u%c%c%c%c%c%c", getuid(), getpid(),
      letters[counter[5]],
      letters[counter[4]],
      letters[counter[3]],
      letters[counter[2]],
      letters[counter[1]],
      letters[counter[0]]);
  CMX_ASSERT(snprintf_retval < (int)SHM_NAME_SIZE);
  name[SHM_NAME_SIZE-1] = '\0';
  ++counter[0];
  if (counter[0] >= limit) { ++counter[1]; counter[0] = 0; }
  if (counter[1] >= limit) { ++counter[2]; counter[1] = 0; }
  if (counter[2] >= limit) { ++counter[3]; counter[2] = 0; }
  if (counter[3] >= limit) { ++counter[4]; counter[3] = 0; }
  if (counter[4] >= limit) { ++counter[5]; counter[4] = 0; }
  if (counter[5] >= limit) {
    p_error("_generate_shm_name: too many names generated", -1);
  }
#if DEBUG
  fprintf(stderr, "[%d] _generate_shm_name(%d)=%s\n",
      p_config.rank(), rank, name);
#endif

  return name;
}


int p_Environment::dist_malloc(void **ptrs, int64_t bytes, Group *group)
{
  reg_entry_t *reg_entries = NULL;
  reg_entry_t *my_reg;
  int my_master = -1;
  int my_world_rank = -1;
  int i = 0;
  int is_notifier = 0;
  int reg_entries_local_count = 0;
  reg_entry_t *reg_entries_local = NULL;
  int status = 0;

  group->barrier();

  my_world_rank = p_config.get_world_rank(group, group->rank());
  my_master = p_config.master(my_world_rank);

  std::vector<int> world_ranks = p_config.get_world_ranks(group);
  is_notifier 
    = p_config.rank() == p_config.get_my_master_rank_with_same_hostid(
        group->rank(), p_config.node_size(),
        group->MPIComm(), my_world_rank,
        world_ranks);
  if (is_notifier) {
    reg_entries_local = new reg_entry_t[p_config.node_size()];
  }

  /* allocate space for registration cache entries */
  reg_entries = new reg_entry_t[group->size()];
  MAYBE_MEMSET(reg_entries, 0, sizeof(reg_entry_t)*group->size());

  /* allocate and register segment */
  MAYBE_MEMSET(my_reg, 0, sizeof(reg_entry_t));
  if (0 == bytes) {
    my_reg = new reg_entry_t;
    p_register.nullify(my_reg);
  }
  else {
    my_reg = p_register.malloc(sizeof(char)*bytes);
  }

  /* exchange buffer address via reg entries */
  reg_entries[group->rank()] = *my_reg;
  status = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
      reg_entries, sizeof(reg_entry_t), MPI_BYTE, group->MPIComm());
  _translate_mpi_error(status, "cmx_malloc:MPI_Allgather");
  CMX_ASSERT(MPI_SUCCESS == status);

  /* insert reg entries into local registration cache */
  for (i=0; i<group->size(); ++i) {
    if (NULL == reg_entries[i].buf) {
      /* a proc did not allocate (bytes==0) */
    }
    else if (p_config.rank() == reg_entries[i].rank) {
      /* we already registered our own memory, but PR hasn't */
      if (is_notifier) {
        /* does this need to be a memcpy?? */
        reg_entries_local[reg_entries_local_count++] = reg_entries[i];
      }
    }
    // else if (g_state.hostid[reg_entries[i].rank]
    //         == g_state.hostid[my_world_rank]) 

    else if (p_config.master(reg_entries[i].rank) == 
        p_config.master(p_config.get_my_master_rank_with_same_hostid(
        group->rank(), p_config.node_size(),
        group->MPIComm(), my_world_rank,
        world_ranks)))

    {
      /* same SMP node, need to mmap */
      /* open remote shared memory object */
      void *memory = p_shmem.attach(reg_entries[i].name, reg_entries[i].len);
      p_register.insert(
          reg_entries[i].rank,
          reg_entries[i].buf,
          reg_entries[i].len,
          reg_entries[i].name,
          memory,0
          );
      if (is_notifier) {
        /* does this need to be a memcpy?? */
        reg_entries_local[reg_entries_local_count++] = reg_entries[i];
      }
    }
    else {
    }
  }

  /* assign the cmx handle to return to caller */
  cmx_alloc_t *prev = NULL;
  for (i=0; i<group->size(); ++i) {
    ptrs[i] = reg_entries[i].buf;
  }

  /* send reg entries to my master */
  /* first non-master rank in an SMP node sends the message to master */
  if (is_notifier) {
    _cmx_request nb;
    int reg_entries_local_size = 0;
    int message_size = 0;
    char *message = NULL;
    header_t *header = NULL;

    nb_request_init(&nb);
    reg_entries_local_size = sizeof(reg_entry_t)*reg_entries_local_count;
    message_size = sizeof(header_t) + reg_entries_local_size;
    message = new char[message_size];
    CMX_ASSERT(message);
    header = reinterpret_cast<header_t*>(message);
    header->operation = OP_MALLOC;
    header->remote_address = NULL;
    header->local_address = NULL;
    header->rank = 0;
    header->length = reg_entries_local_count;
    (void)memcpy(message+sizeof(header_t), reg_entries_local, reg_entries_local_size);
    nb_recv(NULL, 0, my_master, &nb); /* prepost ack */
    nb_send_header(message, message_size, my_master, &nb);
    nb_wait_for_all(&nb);
    delete [] reg_entries_local;
  }

  delete [] reg_entries;

  group->barrier();

  return CMX_SUCCESS;
}

int p_Environment::dist_free(void *ptr, Group *group)
{
  int my_world_rank = -1;
  int my_master = -1;
  rank_ptr_t *ptrs = NULL;
  int i = 0;
  int is_notifier = 0;
  reg_entry_t *reg_entry;
  int reg_entries_local_count = 0;
  rank_ptr_t *rank_ptrs = NULL;
  int status = 0;

  my_world_rank = p_config.rank();
  my_master = p_config.master(my_world_rank);

  std::vector<int> world_ranks = p_config.get_world_ranks(group);
  is_notifier 
    = p_config.rank() == p_config.get_my_master_rank_with_same_hostid(
        group->rank(), p_config.node_size(),
        group->MPIComm(), my_world_rank,
        world_ranks);

  if (is_notifier) {
    rank_ptrs = new rank_ptr_t[p_config.node_size()];
  }

  /* allocate receive buffer for exchange of pointers */
  ptrs = new rank_ptr_t[group->size()];
  CMX_ASSERT(ptrs);
  ptrs[group->rank()].rank = my_world_rank;
  ptrs[group->rank()].ptr = ptr;

  /* exchange of pointers */
  status = MPI_Allgather(MPI_IN_PLACE, sizeof(rank_ptr_t), MPI_BYTE,
      ptrs, sizeof(rank_ptr_t), MPI_BYTE, group->MPIComm());
  _translate_mpi_error(status, "p_Environment::free:MPI_Allgather");
  CMX_ASSERT(MPI_SUCCESS == status);

  for (i=0; i<group->size(); ++i) {
    if (i == group->rank()) {
      if (is_notifier) {
        /* does this need to be a memcpy? */
        rank_ptrs[reg_entries_local_count].rank = world_ranks[i];
        rank_ptrs[reg_entries_local_count].ptr = ptrs[i].ptr;
        reg_entries_local_count++;
      }
    } else if (NULL == ptrs[i].ptr) {
    } else if (p_config.master(world_ranks[i]) ==
       p_config.master(p_config.get_my_master_rank_with_same_hostid(
           group->rank(), p_config.node_size(), group->MPIComm(),
           my_world_rank, world_ranks))) {
        /* same SMP node */
        reg_entry = NULL;
        int retval = 0;

        /* find the registered memory */
        reg_entry = p_register.find(world_ranks[i], ptrs[i].ptr, 0);
        CMX_ASSERT(reg_entry);


        p_shmem.unmap(reg_entry->mapped, reg_entry->len);
        p_register.remove(world_ranks[i], ptrs[i].ptr);

        if (is_notifier) {
          /* does this need to be a memcpy? */
          rank_ptrs[reg_entries_local_count].rank = world_ranks[i];
          rank_ptrs[reg_entries_local_count].ptr = ptrs[i].ptr;
          reg_entries_local_count++;
        }
      } else {
      }
  }

  /* send ptrs to my master */
  /* first non-master rank in an SMP node sends the message to master */
  if (is_notifier) {
    _cmx_request nb;
    int rank_ptrs_local_size = 0;
    int message_size = 0;
    char *message = NULL;
    header_t *header = NULL;

    nb_request_init(&nb);
    rank_ptrs_local_size = sizeof(rank_ptr_t) * reg_entries_local_count;
    message_size = sizeof(header_t) + rank_ptrs_local_size;
    message = new char[message_size];
    CMX_ASSERT(message);
    header = reinterpret_cast<header_t*>(message);
    header->operation = OP_FREE;
    header->remote_address = NULL;
    header->local_address = NULL;
    header->rank = 0;
    header->length = reg_entries_local_count;
    (void)memcpy(message+sizeof(header_t), rank_ptrs, rank_ptrs_local_size);
    nb_recv(NULL, 0, my_master, &nb); /* prepost ack */
    nb_send_header(message, message_size, my_master, &nb);
    nb_wait_for_all(&nb);
    delete [] rank_ptrs;
  }
  /* free ptrs array */
  delete [] ptrs;

  /* remove my ptr from reg cache and free ptr */
  reg_entry = p_register.find(my_world_rank, ptr, 0);
  p_shmem.free(reg_entry->name, reg_entry->mapped, reg_entry->len);
  p_register.remove(my_world_rank, ptr);

  /* Is this needed? */
  group->barrier();

  return CMX_SUCCESS;
}

#ifdef OLD_CODE
int p_Environment::wait_proc(int proc, Group *group)
{
  return cmx_wait_all(igroup);
}
#endif

void wait(_cmx_request *hdl);

/**
 * wait for completion of non-blocking handle
 * @param hdl non-blocking request handle
 */
void p_Environment::wait(_cmx_request* hdl)
{
  int index = 0;

  CMX_ASSERT(NULL != hdl);

#if 0
  /* this condition will likely be tripped if a blocking operation follows a
   * non-blocking operation*/
  if (0 == nb->in_use) {
    fprintf(stderr, "p[%d] cmx_wait Error: invalid handle\n",
        p_config.rank());
  }
#endif

  nb_wait_for_all(hdl);
  nb_unregister_request(hdl);
}


/**
 * test for completion of non-blocking handle. If test is true, operation has
 * completed locally
 * @param hdl non-blocking request handle
 * @return true if operation is completed locally
 */
bool p_Environment::test(_cmx_request* hdl)
{
  int index = 0;
  bool status;

  CMX_ASSERT(NULL != hdl);

#if 0
  /* this condition will likely be tripped if a blocking operation follows a
   * non-blocking operation*/
  if (0 == nb->in_use) {
    fprintf(stderr, "{%d} cmx_test Error: invalid handle\n",
        p_config.rank());
  }
#endif

  if (!nb_test_for_all(hdl)) {
    /* Completed */
    CMX_ASSERT(0 == hdl->send_size);
    CMX_ASSERT(0 == hdl->recv_size);
    status = true;
  }
  else {
    /* Not completed */
    status = false;
  }

  return status;
}


void p_Environment::waitAll(Group *group)
{
  int i;
  for (i=0; i<nb_max_outstanding; i++) {
    if (nb_list[i] != NULL && nb_list[i]->group == group) {
      nb_wait_for_all(nb_list[i]);
      nb_list[i]->in_use = 0;
      nb_list[i] = NULL;
    }
  }
}

/* The register and unregister functions are used to limit the number of
 * outstanding non-blocking handles*/
void p_Environment::nb_register_request(_cmx_request *nb)
{
  int ival = -1;
  int i;
  /* look for unused handle */
  for (i=nb_last_request; i<nb_last_request+nb_max_outstanding; i++) {
    int idx = i%nb_max_outstanding;
    if (nb_list[idx] == NULL) {
      ival = idx;
      break;
    }
  }
  if (ival < 0) {
    ival = nb_last_request;
    nb_wait_for_all(nb_list[ival]);
    nb_list[ival] == NULL;
  }
  nb_request_init(nb);
  nb_list[ival] = nb;
  nb_last_request++;
  nb_last_request = nb_last_request%nb_max_outstanding;
}

void p_Environment::nb_unregister_request(_cmx_request *nb)
{
  int i;
  for (i=0; i<nb_max_outstanding; i++) {
    if (nb = nb_list[i]) nb_list[i] = NULL;
  }
}

void p_Environment::nb_request_init(_cmx_request *nb)
{
  nb->send_size = 0;
  nb->send_head = NULL;
  nb->send_tail = NULL;
  nb->recv_size = 0;
  nb->recv_head = NULL;
  nb->recv_tail = NULL;
  nb->group = NULL;
  nb->in_use = 1;
}

/* one semaphore per world process */
void p_Environment::_malloc_semaphore()
{
  char *name;
  char *names = NULL;
  sem_t *my_sem = NULL;
  int status = 0;
  MPI_Datatype shm_name_type;
  int i = 0;

#if DEBUG
  fprintf(stderr, "[%d] _malloc_semaphore()\n", p_config.rank());
#endif

  status = MPI_Type_contiguous(SHM_NAME_SIZE, MPI_CHAR, &shm_name_type);
  _translate_mpi_error(status, "_malloc_semaphore:MPI_Type_contiguous");
  CMX_ASSERT(MPI_SUCCESS == status);
  status = MPI_Type_commit(&shm_name_type);
  _translate_mpi_error(status, "_malloc_semaphore:MPI_Type_commmit");
  CMX_ASSERT(MPI_SUCCESS == status);

  semaphores = new sem_t*[p_config.world_size()];
  CMX_ASSERT(semaphores);

  name = p_shmem.generateName(p_config.rank());
  CMX_ASSERT(name);

#if ENABLE_UNNAMED_SEM
  {
    my_sem = p_shmem.create(name, sizeof(sem_t));
    /* initialize the memory as an inter-process semaphore */
    if (0 != sem_init(my_sem, 1, 1)) {
      perror("_malloc_semaphore: sem_init");
      p_error("_malloc_semaphore: sem_init", -1);
    }
  }
#else
  {
    my_sem = sem_open(name, O_CREAT|O_EXCL, S_IRUSR|S_IWUSR, 1);
    if (SEM_FAILED == my_sem) {
      if (EEXIST == errno) {
        status = sem_unlink(name);
        if (-1 == status) {
          perror("_malloc_semaphore: sem_unlink");
          p_error("_malloc_semaphore: sem_unlink", status);
        }
      }
      /* second try */
      my_sem = sem_open(name, O_CREAT|O_EXCL, S_IRUSR|S_IWUSR, 1);
    }
    if (SEM_FAILED == my_sem) {
      perror("_malloc_semaphore: sem_open");
      p_error("_malloc_semaphore: sem_open", -1);
    }
  }
#endif

  /* store my sem in global cache */
  semaphores[p_config.rank()] = my_sem;

  //names = (char*)malloc(sizeof(char) * SHM_NAME_SIZE * g_state.size);
  names = new char[SHM_NAME_SIZE*p_config.world_size()];
  CMX_ASSERT(names);

  /* exchange names */
  (void)memcpy(&names[SHM_NAME_SIZE*p_config.rank()], name, SHM_NAME_SIZE);
  status = MPI_Allgather(MPI_IN_PLACE, 1, shm_name_type,
      names, 1, shm_name_type, p_config.global_comm());
  _translate_mpi_error(status, "_malloc_semaphore:MPI_Allgather");
  CMX_ASSERT(MPI_SUCCESS == status);

  /* create/open remote semaphores and store in cache */
  for (i=0; i<p_config.world_size(); ++i) {
    if (p_config.rank() == i) {
      continue; /* skip my own rank */
    }
    else if (p_config.hostid(p_config.rank()) == p_config.hostid(i)) {
      /* same SMP node */
#if ENABLE_UNNAMED_SEM
      semaphores[i] = p_shmem.attach(&names[SHM_NAME_SIZE*i], sizeof(sem_t));
      CMX_ASSERT(semaphores[i]);
#else
      semaphores[i] = sem_open(&names[SHM_NAME_SIZE*i], 0);
      if (SEM_FAILED == semaphores[i]) {
        perror("_malloc_semaphore: sem_open");
        p_error("_malloc_semaphore: sem_open", -2);
      }
#endif
    }
    else {
      semaphores[i] = NULL;
    }
  }

  sem_name = name;

  delete []name;
  delete [] names;

  status = MPI_Type_free(&shm_name_type);
  _translate_mpi_error(status, "_malloc_semaphore:MPI_Type_free");
  CMX_ASSERT(MPI_SUCCESS == status);
}


void p_Environment::_free_semaphore()
{
  int i;
  int retval;

#if DEBUG
  fprintf(stderr, "[%d] _free_semaphore()\n", p_config.rank());
#endif

  for (i=0; i<p_config.world_size(); ++i) {
    if (p_config.rank() == i) {
      /* me */
#if ENABLE_UNNAMED_SEM
      retval = sem_destroy(semaphores[i]);
      if (-1 == retval) {
        perror("_free_semaphore: sem_destroy");
        p_error("_free_semaphore: sem_destroy", retval);
      }
      retval = munmap(semaphores[i], sizeof(sem_t));
      if (-1 == retval) {
        perror("_free_semaphore: munmap");
        p_error("_free_semaphore: munmap", retval);
      }
      retval = shm_unlink(sem_name.c_str());
      if (-1 == retval) {
        perror("_free_semaphore: shm_unlink");
        p_error("_free_semaphore: shm_unlink", retval);
      }
#else
      retval = sem_close(semaphores[i]);
      if (-1 == retval) {
        perror("_free_semaphore: sem_close");
        p_error("_free_semaphore: sem_close", retval);
      }
      retval = sem_unlink(sem_name.c_str());
      if (-1 == retval) {
        perror("_free_semaphore: sem_unlink");
        p_error("_free_semaphore: sem_unlink", retval);
      }
#endif
    }
    else if (p_config.hostid(p_config.rank()) == p_config.hostid(i)) {
      /* same SMP node */
#if ENABLE_UNNAMED_SEM
      retval = munmap(semaphores[i], sizeof(sem_t));
      if (-1 == retval) {
        perror("_free_semaphore: munmap");
        p_error("_free_semaphore: munmap", retval);
      }
#else
      retval = sem_close(semaphores[i]);
      if (-1 == retval) {
        perror("_free_semaphore: sem_close");
        p_error("_free_semaphore: sem_close", retval);
      }
#endif
    }
  }

  delete [] semaphores;
  semaphores = NULL;
}

void p_Environment::_progress_server()
{
  int running = 0;
  char *static_header_buffer = NULL;
  int static_header_buffer_size = 0;
  int extra_size = 0;
  int ierr;

#if DEBUG
  fprintf(stderr, "[%d] _progress_server()\n", p_config.rank());
  fprintf(stderr, "[%d] _progress_server(); node_size[%d]\n", 
      p_config.rank(), g_state.node_size);
#endif

  {
    int status = _set_affinity(p_config.node_size());
    if (0 != status) {
      status = _set_affinity(p_config.node_size()-1);
      CMX_ASSERT(0 == status);
    }
  }

  /* static header buffer size must be large enough to hold the biggest
   * message that might possibly be sent using a header type message. */
  static_header_buffer_size += sizeof(header_t);
  /* extra header info could be reg entries, one per local rank */
  extra_size = sizeof(reg_entry_t)*p_config.node_size();
  /* or, extra header info could be an acc scale plus stride */
  if ((sizeof(stride_t)+sizeof(DoubleComplex)) > extra_size) {
    extra_size = sizeof(stride_t)+sizeof(DoubleComplex);
  }
  static_header_buffer_size += extra_size;
  /* after all of the above, possibly grow the size based on user request */
  if (static_header_buffer_size < eager_threshold) {
    static_header_buffer_size = eager_threshold;
  }

  /* initialize shared buffers */
  static_header_buffer = new char[static_header_buffer_size];
  CMX_ASSERT(static_header_buffer);
  static_server_buffer = new char[static_server_buffer_size];
  CMX_ASSERT(static_server_buffer);

  running = 1;
  while (running) {
    int source = 0;
    int length = 0;
    char *payload = NULL;
    header_t *header = NULL;
    MPI_Status recv_status;

    ierr = MPI_Recv(static_header_buffer, static_header_buffer_size, MPI_CHAR,
        MPI_ANY_SOURCE, CMX_TAG, p_config.global_comm(), &recv_status);
    _translate_mpi_error(ierr, "_progress_server:MPI_Recv");
    MPI_Get_count(&recv_status, MPI_CHAR, &length);
    _translate_mpi_error(ierr, "_progress_server:MPI_Get_count");
    source = recv_status.MPI_SOURCE;
#   if DEBUG
    fprintf(stderr, "[%d] progress MPI_Recv source=%d length=%d\n",
        p_config.rank(), source, length);
#   endif
    header = reinterpret_cast<header_t*>(static_header_buffer);
    payload = static_header_buffer + sizeof(header_t);
    /* dispatch message handler */
    switch (header->operation) {
      case OP_PUT:
        _put_handler(header, payload, source);
        break;
      case OP_PUT_PACKED:
        _put_packed_handler(header, payload, source);
        break;
      case OP_PUT_DATATYPE:
        _put_datatype_handler(header, payload, source);
        break;
      case OP_PUT_IOV:
        _put_iov_handler(header, source);
        break;
      case OP_GET:
        _get_handler(header, source);
        break;
      case OP_GET_PACKED:
        _get_packed_handler(header, payload, source);
        break;
      case OP_GET_DATATYPE:
        _get_datatype_handler(header, payload, source);
        break;
      case OP_GET_IOV:
        _get_iov_handler(header, source);
        break;
      case OP_ACC_INT:
      case OP_ACC_DBL:
      case OP_ACC_FLT:
      case OP_ACC_CPL:
      case OP_ACC_DCP:
      case OP_ACC_LNG:
        _acc_handler(header, payload, source);
        break;
      case OP_ACC_INT_PACKED:
      case OP_ACC_DBL_PACKED:
      case OP_ACC_FLT_PACKED:
      case OP_ACC_CPL_PACKED:
      case OP_ACC_DCP_PACKED:
      case OP_ACC_LNG_PACKED:
        _acc_packed_handler(header, payload, source);
        break;
      case OP_ACC_INT_IOV:
      case OP_ACC_DBL_IOV:
      case OP_ACC_FLT_IOV:
      case OP_ACC_CPL_IOV:
      case OP_ACC_DCP_IOV:
      case OP_ACC_LNG_IOV:
        _acc_iov_handler(header, payload, source);
        break;
      case OP_FENCE:
        _fence_handler(header, source);
        break;
      case OP_FETCH_AND_ADD:
        _fetch_and_add_handler(header, payload, source);
        break;
      case OP_SWAP:
        _swap_handler(header, payload, source);
        break;
      case OP_CREATE_MUTEXES:
        _mutex_create_handler(header, source);
        break;
      case OP_DESTROY_MUTEXES:
        _mutex_destroy_handler(header, source);
        break;
      case OP_LOCK:
        _lock_handler(header, source);
        break;
      case OP_UNLOCK:
        _unlock_handler(header, source);
        break;
      case OP_QUIT:
        running = 0;
        break;
      case OP_MALLOC:
        _malloc_handler(header, payload, source);
        break;
      case OP_FREE:
        _free_handler(header, payload, source);
        break;
      default:
        fprintf(stderr, "[%d] header operation not recognized: %d\n",
            p_config.rank(), header->operation);
        CMX_ASSERT(0);
    }
  }

  _initialized = 0;

  delete [] static_header_buffer;
  delete [] static_server_buffer;

  _free_semaphore();

  delete [] mutexes;
  lq_heads.clear();

  ierr = MPI_Barrier(p_config.global_comm());
  _translate_mpi_error(ierr, "_progress_server:MPI_Barrier");

  /* reg_cache */
  p_register.destroy();

  // destroy the communicators
#if DEBUG
  fprintf(stderr, "[%d] before cmx_group_finalize()\n", p_config.rank());
#endif
#ifdef OLD_CODE
  cmx_group_finalize();
#endif
#if DEBUG
  fprintf(stderr, "[%d] after cmx_group_finalize()\n", p_config.rank());
#endif

#if DEBUG_TO_FILE
  fclose(cmx_trace_file);
#endif

  // assume this is the end of a user's application
  MPI_Finalize();
  exit(EXIT_SUCCESS);
}


void p_Environment::_put_handler(header_t *header, char *payload, int proc)
{
  reg_entry_t *reg_entry = NULL;
  void *mapped_offset = NULL;
  int use_eager = _eager_check(header->length);

#if DEBUG
  fprintf(stderr, "[%d] _put_handler rem=%p loc=%p rem_rank=%d len=%d\n",
      p_config.rank(),
      header->remote_address,
      header->local_address,
      header->rank,
      header->length);
#endif

  reg_entry = p_register.find(
      header->rank, header->remote_address, header->length);
  CMX_ASSERT(reg_entry);
  mapped_offset = _get_offset_memory(
      reg_entry, header->remote_address);
  if (use_eager) {
    (void)memcpy(mapped_offset, payload, header->length);
  }
  else {
    char *buf = (char*)mapped_offset;
    int64_t bytes_remaining = header->length;
    do {
      int64_t size = bytes_remaining>max_message_size ?
        max_message_size : bytes_remaining;
      server_recv(buf, size, proc);
      buf += size;
      bytes_remaining -= size;
    } while (bytes_remaining > 0);
  }
}


void p_Environment::_put_packed_handler(header_t *header, char *payload, int proc)
{
  reg_entry_t *reg_entry = NULL;
  void *mapped_offset = NULL;
  char *packed_buffer = NULL;
  stride_t *stride = NULL;
  int use_eager = _eager_check(sizeof(stride_t)+header->length);
#if DEBUG
  int i=0;
#endif

#if DEBUG
  fprintf(stderr, "[%d] _put_packed_handler rem=%p loc=%p rem_rank=%d len=%d\n",
      p_config.rank(),
      header->remote_address,
      header->local_address,
      header->rank,
      header->length);
#endif

  stride = (stride_t*)payload;
  CMX_ASSERT(stride->stride_levels >= 0);
  CMX_ASSERT(stride->stride_levels < CMX_MAX_STRIDE_LEVEL);

#if DEBUG
  fprintf(stderr, "[%d] _put_packed_handler stride_levels=%d, count[0]=%d\n",
      p_config.rank(), stride->stride_levels, stride->count[0]);
  for (i=0; i<stride->stride_levels; ++i) {
    fprintf(stderr, "[%d] stride[%d]=%d count[%d+1]=%d\n",
        p_config.rank(), i, stride->stride[i], i, stride->count[i+1]);
  }
#endif

  reg_entry = p_register.find(
      header->rank, header->remote_address, stride->count[0]);
  CMX_ASSERT(reg_entry);
  mapped_offset = _get_offset_memory(
      reg_entry, header->remote_address);

  if (use_eager) {
    packed_buffer = payload+sizeof(stride_t);
    unpack(packed_buffer, (char*)mapped_offset,
        stride->stride, stride->count, stride->stride_levels);

  }
  else {
    if ((unsigned)header->length > static_server_buffer_size) {
      packed_buffer = (char*)malloc(header->length);
    }
    else {
      packed_buffer = static_server_buffer;
    }

    {
      /* we receive the buffer backwards */
      char *buf = packed_buffer + header->length;
      int64_t bytes_remaining = header->length;
      do {
        int64_t size = bytes_remaining>max_message_size ?
          max_message_size : bytes_remaining;
        buf -= size;
        server_recv(buf, size, proc);
        bytes_remaining -= size;
      } while (bytes_remaining > 0);
    }

    unpack(packed_buffer, (char*)mapped_offset,
        stride->stride, stride->count, stride->stride_levels);

    if ((unsigned)header->length > static_server_buffer_size) {
      free(packed_buffer);
    }
  }
}


void p_Environment::_put_datatype_handler(header_t *header, char *payload, int proc)
{
  MPI_Datatype dst_type;
  reg_entry_t *reg_entry = NULL;
  void *mapped_offset = NULL;
  stride_t *stride = NULL;
  int ierr;
#if DEBUG
  int i=0;
#endif

#if DEBUG
  fprintf(stderr, "[%d] _put_datatype_handler rem=%p loc=%p rem_rank=%d len=%d\n",
      p_config.rank(),
      header->remote_address,
      header->local_address,
      header->rank,
      header->length);
#endif

  stride = (stride_t*)payload;
  CMX_ASSERT(stride);
  CMX_ASSERT(stride->stride_levels >= 0);
  CMX_ASSERT(stride->stride_levels < CMX_MAX_STRIDE_LEVEL);

#if DEBUG
  fprintf(stderr, "[%d] _put_datatype_handler stride_levels=%d, count[0]=%d\n",
      p_config.rank(), stride->stride_levels, stride->count[0]);
  for (i=0; i<stride->stride_levels; ++i) {
    fprintf(stderr, "[%d] stride[%d]=%d count[%d+1]=%d\n",
        p_config.rank(), i, stride->stride[i], i, stride->count[i+1]);
  }
#endif

  reg_entry = p_register.find(
      header->rank, header->remote_address, stride->count[0]);
  CMX_ASSERT(reg_entry);
  mapped_offset = _get_offset_memory(
      reg_entry, header->remote_address);

  strided_to_subarray_dtype(stride->stride, stride->count,
      stride->stride_levels, MPI_BYTE, &dst_type);
  ierr = MPI_Type_commit(&dst_type);
  _translate_mpi_error(ierr,"_put_datatype_handler:MPI_Type_commit");

  server_recv_datatype(mapped_offset, dst_type, proc);

  ierr = MPI_Type_free(&dst_type);
  _translate_mpi_error(ierr,"_put_datatype_handler:MPI_Type_free");
}


void p_Environment::_put_iov_handler(header_t *header, int proc)
{
  reg_entry_t *reg_entry = NULL;
  void *mapped_offset = NULL;
  int i = 0;
  char *packed_buffer = NULL;
  int packed_index = 0;
  char *iov_buf = NULL;
  int iov_off = 0;
  int64_t limit = 0;
  int64_t bytes = 0;
  void **src = NULL;
  void **dst = NULL;

#if DEBUG
  fprintf(stderr, "[%d] _put_iov_handler proc=%d\n", p_config.rank(), proc);
#endif
#if DEBUG
  fprintf(stderr, "[%d] _put_iov_handler header rem=%p loc=%p rem_rank=%d len=%d\n",
      p_config.rank(),
      header->remote_address,
      header->local_address,
      header->rank,
      header->length);
#endif

  assert(OP_PUT_IOV == header->operation);

  iov_buf = (char*)malloc(header->length);
  CMX_ASSERT(iov_buf);
  server_recv(iov_buf, header->length, proc);

  limit = *((int64_t*)(&iov_buf[iov_off]));
  iov_off += sizeof(int64_t);
  CMX_ASSERT(limit > 0);

  bytes = *((int64_t*)(&iov_buf[iov_off]));
  iov_off += sizeof(int64_t);
  CMX_ASSERT(bytes > 0);

  src = (void**)&iov_buf[iov_off];
  iov_off += sizeof(void*)*limit;

  dst = (void**)&iov_buf[iov_off];
  iov_off += sizeof(void*)*limit;

  CMX_ASSERT(iov_off == header->length);

#if DEBUG
  fprintf(stderr, "[%d] _put_iov_handler limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
      p_config.rank(), limit, bytes, src[0], dst[0]);
#endif

  if ((unsigned)(bytes*limit) > static_server_buffer_size) {
    packed_buffer = (char*)malloc(bytes*limit);
    CMX_ASSERT(packed_buffer);
  }
  else {
    packed_buffer = static_server_buffer;
  }

  server_recv(packed_buffer, bytes * limit, proc);

  packed_index = 0;
  for (i=0; i<limit; ++i) {
    reg_entry = p_register.find(
        header->rank, dst[i], bytes);
    CMX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(
        reg_entry, dst[i]);

    (void)memcpy(mapped_offset, &packed_buffer[packed_index], bytes);
    packed_index += bytes;
  }
  CMX_ASSERT(packed_index == bytes*limit);

  if ((unsigned)(bytes*limit) > static_server_buffer_size) {
    free(packed_buffer);
  }

  free(iov_buf);
}


void p_Environment::_get_handler(header_t *header, int proc)
{
  reg_entry_t *reg_entry = NULL;
  void *mapped_offset = NULL;

#if DEBUG
  fprintf(stderr, "[%d] _get_handler proc=%d\n", p_config.rank(), proc);
#endif
#if DEBUG
  fprintf(stderr, "[%d] header rem=%p loc=%p rem_rank=%d len=%d\n",
      p_config.rank(),
      header->remote_address,
      header->local_address,
      header->rank,
      header->length);
#endif

  CMX_ASSERT(OP_GET == header->operation);

  reg_entry = p_register.find(
      header->rank, header->remote_address, header->length);
  CMX_ASSERT(reg_entry);
  mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

  {
    char *buf = (char*)mapped_offset;
    int64_t bytes_remaining = header->length;
    do {
      int64_t size = bytes_remaining>max_message_size ?
        max_message_size : bytes_remaining;
      server_send(buf, size, proc);
      buf += size;
      bytes_remaining -= size;
    } while (bytes_remaining > 0);
  }
}


void p_Environment::_get_packed_handler(header_t *header, char *payload, int proc)
{
  reg_entry_t *reg_entry = NULL;
  void *mapped_offset = NULL;
  char *packed_buffer = NULL;
  int64_t packed_index = 0;
  stride_t *stride_src = (stride_t*)payload;

#if DEBUG
  fprintf(stderr, "[%d] _get_packed_handler proc=%d\n", p_config.rank(), proc);
#endif
#if DEBUG
  fprintf(stderr, "[%d] header rem=%p loc=%p rem_rank=%d len=%d\n",
      p_config.rank(),
      header->remote_address,
      header->local_address,
      header->rank,
      header->length);
#endif

  assert(OP_GET_PACKED == header->operation);

  CMX_ASSERT(stride_src->stride_levels >= 0);
  CMX_ASSERT(stride_src->stride_levels < CMX_MAX_STRIDE_LEVEL);

  reg_entry = p_register.find(
      header->rank, header->remote_address, header->length);
  CMX_ASSERT(reg_entry);
  mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

  packed_buffer = pack((char*)mapped_offset,
      stride_src->stride, stride_src->count, stride_src->stride_levels,
      &packed_index);

  {
    /* we send the buffer backwards */
    char *buf = packed_buffer + packed_index;
    int64_t bytes_remaining = packed_index;
    do {
      int64_t size = bytes_remaining>max_message_size ?
        max_message_size : bytes_remaining;
      buf -= size;
      server_send(buf, size, proc);
      bytes_remaining -= size;
    } while (bytes_remaining > 0);
  }

  free(packed_buffer);
}


void p_Environment::_get_datatype_handler(header_t *header, char *payload, int proc)
{
  MPI_Datatype src_type;
  reg_entry_t *reg_entry = NULL;
  void *mapped_offset = NULL;
  stride_t *stride_src = NULL;
  int ierr;

#if DEBUG
  int i;
  fprintf(stderr, "[%d] _get_datatype_handler proc=%d\n", p_config.rank(), proc);
#endif
#if DEBUG
  fprintf(stderr, "[%d] header rem=%p loc=%p rem_rank=%d len=%d\n",
      p_config.rank(),
      header->remote_address,
      header->local_address,
      header->rank,
      header->length);
#endif

  assert(OP_GET_DATATYPE == header->operation);

  stride_src = (stride_t*)payload;
  CMX_ASSERT(stride_src);
  CMX_ASSERT(stride_src->stride_levels >= 0);
  CMX_ASSERT(stride_src->stride_levels < CMX_MAX_STRIDE_LEVEL);

#if DEBUG
  for (i=0; i<stride_src->stride_levels; ++i) {
    fprintf(stderr, "\tstride[%d]=%d\n", i, stride_src->stride[i]);
  }
  for (i=0; i<stride_src->stride_levels+1; ++i) {
    fprintf(stderr, "\tcount[%d]=%d\n", i, stride_src->count[i]);
  }
#endif

  reg_entry = p_register.find(
      header->rank, header->remote_address, header->length);
  CMX_ASSERT(reg_entry);
  mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

  strided_to_subarray_dtype(stride_src->stride, stride_src->count,
      stride_src->stride_levels, MPI_BYTE, &src_type);
  ierr = MPI_Type_commit(&src_type);
  _translate_mpi_error(ierr,"_get_datatype_handler:MPI_Type_commit");

  server_send_datatype(mapped_offset, src_type, proc);

  ierr = MPI_Type_free(&src_type);
  _translate_mpi_error(ierr,"_get_datatype_handler:MPI_Type_free");
}


void p_Environment::_get_iov_handler(header_t *header, int proc)
{
  reg_entry_t *reg_entry = NULL;
  void *mapped_offset = NULL;
  int i = 0;
  char *packed_buffer = NULL;
  int64_t packed_index = 0;
  char *iov_buf = NULL;
  int iov_off = 0;
  int64_t limit = 0;
  int64_t bytes = 0;
  void **src = NULL;
  void **dst = NULL;

#if DEBUG
  fprintf(stderr, "[%d] _get_iov_handler proc=%d\n", p_config.rank(), proc);
#endif
#if DEBUG
  fprintf(stderr, "[%d] _get_iov_handler header rem=%p loc=%p rem_rank=%d len=%d\n",
      p_config.rank(),
      header->remote_address,
      header->local_address,
      header->rank,
      header->length);
#endif

  assert(OP_GET_IOV == header->operation);

  iov_buf = (char*)malloc(header->length);
  CMX_ASSERT(iov_buf);
  server_recv(iov_buf, header->length, proc);

  limit = *((int64_t*)(&iov_buf[iov_off]));
  iov_off += sizeof(int64_t);
  CMX_ASSERT(limit > 0);

  bytes = *((int64_t*)(&iov_buf[iov_off]));
  iov_off += sizeof(int64_t);
  CMX_ASSERT(bytes > 0);

  src = (void**)&iov_buf[iov_off];
  iov_off += sizeof(void*)*limit;

  dst = (void**)&iov_buf[iov_off];
  iov_off += sizeof(void*)*limit;

  CMX_ASSERT(iov_off == header->length);

#if DEBUG
  fprintf(stderr, "[%d] _get_iov_handler limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
      p_config.rank(), limit, bytes, src[0], dst[0]);
#endif

  if ((unsigned)(bytes*limit) > static_server_buffer_size) {
    packed_buffer = (char*)malloc(bytes*limit);
    CMX_ASSERT(packed_buffer);
  }
  else {
    packed_buffer = static_server_buffer;
  }

  packed_index = 0;
  for (i=0; i<limit; ++i) {
    reg_entry = p_register.find(
        header->rank, src[i], bytes);
    CMX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(reg_entry, src[i]);

    (void)memcpy(&packed_buffer[packed_index], mapped_offset, bytes);
    packed_index += bytes;
  }
  CMX_ASSERT(packed_index == bytes*limit);

  server_send(packed_buffer, packed_index, proc);

  if ((unsigned)(bytes*limit) > static_server_buffer_size) {
    free(packed_buffer);
  }

  free(iov_buf);
}


void p_Environment::_acc_handler(header_t *header, char *scale, int proc)
{
  int sizeof_scale = 0;
  int acc_type = 0;
  reg_entry_t *reg_entry = NULL;
  void *mapped_offset = NULL;
  char *acc_buffer = NULL;
  int use_eager = 0;

#if DEBUG
  fprintf(stderr, "[%d] _acc_handler\n", p_config.rank());
#endif

  switch (header->operation) {
    case OP_ACC_INT:
      acc_type = CMX_ACC_INT;
      sizeof_scale = sizeof(int);
      break;
    case OP_ACC_DBL:
      acc_type = CMX_ACC_DBL;
      sizeof_scale = sizeof(double);
      break;
    case OP_ACC_FLT:
      acc_type = CMX_ACC_FLT;
      sizeof_scale = sizeof(float);
      break;
    case OP_ACC_LNG:
      acc_type = CMX_ACC_LNG;
      sizeof_scale = sizeof(long);
      break;
    case OP_ACC_CPL:
      acc_type = CMX_ACC_CPL;
      sizeof_scale = sizeof(SingleComplex);
      break;
    case OP_ACC_DCP:
      acc_type = CMX_ACC_DCP;
      sizeof_scale = sizeof(DoubleComplex);
      break;
    default: CMX_ASSERT(0);
  }
  use_eager = _eager_check(sizeof_scale+header->length);

  reg_entry = p_register.find(
      header->rank, header->remote_address, header->length);
  CMX_ASSERT(reg_entry);
  mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

  if (use_eager) {
    acc_buffer = scale + sizeof_scale;
  }
  else {
    if ((unsigned)header->length > static_server_buffer_size) {
      acc_buffer = (char*)malloc(header->length);
    }
    else {
      acc_buffer = static_server_buffer;
    }
    {
      char *buf = (char*)acc_buffer;
      int64_t bytes_remaining = header->length;

      do {
        int64_t size = bytes_remaining>max_message_size ?
          max_message_size : bytes_remaining;
        server_recv(buf, size, proc);
        buf += size;
        bytes_remaining -= size;
      } while (bytes_remaining > 0);
    }
  }

  if (CMX_ENABLE_ACC_SELF || CMX_ENABLE_ACC_SMP) {
    //    sem_wait(semaphores[header->rank]);
    if (sem_wait(semaphores[header->rank]) != 0) {
      if (errno == EAGAIN) {
        printf("p[%d] SEM_WAIT ERROR Operation could not be performed"
            " without blocking\n", p_config.rank());
      } else if (errno == EINTR) {
        printf("p[%d] SEM_WAIT ERROR Call interrupted\n",p_config.rank());
      } else if (errno == EINVAL) {
        printf("p[%d] SEM_WAIT ERROR Not a valid semiphore\n",p_config.rank());
      } else if (errno == ETIMEDOUT) {
        printf("p[%d] SEM_WAIT ERROR Call timed out\n",p_config.rank());
      }
      CMX_ASSERT(0);
    }
    _acc(acc_type, header->length, mapped_offset, acc_buffer, scale);
    sem_post(semaphores[header->rank]);
  }
  else {
    _acc(acc_type, header->length, mapped_offset, acc_buffer, scale);
  }

  if (use_eager) {
  }
  else {
    if ((unsigned)header->length > static_server_buffer_size) {
      free(acc_buffer);
    }
  }
}


void p_Environment::_acc_packed_handler(header_t *header, char *payload, int proc)
{
  reg_entry_t *reg_entry = NULL;
  void *mapped_offset = NULL;
  void *scale = NULL;
  int sizeof_scale = 0;
  int acc_type = 0;
  char *acc_buffer = NULL;
  stride_t *stride = NULL;
  int use_eager = 0;

#if DEBUG
  fprintf(stderr, "[%d] _acc_packed_handler\n", p_config.rank());
#endif

  switch (header->operation) {
    case OP_ACC_INT_PACKED:
      acc_type = CMX_ACC_INT;
      sizeof_scale = sizeof(int);
      break;
    case OP_ACC_DBL_PACKED:
      acc_type = CMX_ACC_DBL;
      sizeof_scale = sizeof(double);
      break;
    case OP_ACC_FLT_PACKED:
      acc_type = CMX_ACC_FLT;
      sizeof_scale = sizeof(float);
      break;
    case OP_ACC_LNG_PACKED:
      acc_type = CMX_ACC_LNG;
      sizeof_scale = sizeof(long);
      break;
    case OP_ACC_CPL_PACKED:
      acc_type = CMX_ACC_CPL;
      sizeof_scale = sizeof(SingleComplex);
      break;
    case OP_ACC_DCP_PACKED:
      acc_type = CMX_ACC_DCP;
      sizeof_scale = sizeof(DoubleComplex);
      break;
    default: CMX_ASSERT(0);
  }
  use_eager = _eager_check(sizeof_scale+sizeof(stride_t)+header->length);

  scale = payload;
  stride = (stride_t*)(payload + sizeof_scale);

  if (use_eager) {
    acc_buffer = payload+sizeof_scale+sizeof(stride_t);
  }
  else {
    if ((unsigned)header->length > static_server_buffer_size) {
      acc_buffer = (char*)malloc(header->length);
    }
    else {
      acc_buffer = static_server_buffer;
    }

    {
      /* we receive the buffer backwards */
      char *buf = acc_buffer + header->length;
      int64_t bytes_remaining = header->length;
      do {
        int64_t size = bytes_remaining>max_message_size ?
          max_message_size : bytes_remaining;
        buf -= size;
        server_recv(buf, size, proc);
        bytes_remaining -= size;
      } while (bytes_remaining > 0);
    }
  }

  reg_entry = p_register.find(
      header->rank, header->remote_address, header->length);
  CMX_ASSERT(reg_entry);
  mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

  if (CMX_ENABLE_ACC_SELF || CMX_ENABLE_ACC_SMP) {
    sem_wait(semaphores[header->rank]);
  }
  {
    char *packed_buffer = acc_buffer;
    char *dst = (char*)mapped_offset;
    int64_t *dst_stride = stride->stride;
    int64_t *count = stride->count;
    int stride_levels = stride->stride_levels;
    int64_t i, j;
    int64_t dst_idx;  /* index offset of current block position to ptr */
    int64_t n1dim;  /* number of 1 dim block */
    int64_t dst_bvalue[7], dst_bunit[7];
    int64_t packed_index = 0;

    CMX_ASSERT(stride_levels >= 0);
    CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);
    CMX_ASSERT(NULL != packed_buffer);
    CMX_ASSERT(NULL != dst);
    CMX_ASSERT(NULL != dst_stride);
    CMX_ASSERT(NULL != count);
    CMX_ASSERT(count[0] > 0);

#if DEBUG
    fprintf(stderr, "[%d] unpack(dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d)\n",
        p_config.rank(), dst, dst_stride, count[0], stride_levels);
#endif

    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++) {
      n1dim *= count[i];
    }

    /* calculate the destination indices */
    dst_bvalue[0] = 0; dst_bvalue[1] = 0; dst_bunit[0] = 1; dst_bunit[1] = 1;

    for(i=2; i<=stride_levels; i++) {
      dst_bvalue[i] = 0;
      dst_bunit[i] = dst_bunit[i-1] * count[i-1];
    }

    for(i=0; i<n1dim; i++) {
      dst_idx = 0;
      for(j=1; j<=stride_levels; j++) {
        dst_idx += (int64_t) dst_bvalue[j] * (int64_t) dst_stride[j-1];
        if((i+1) % dst_bunit[j] == 0) {
          dst_bvalue[j]++;
        }
        if(dst_bvalue[j] > (count[j]-1)) {
          dst_bvalue[j] = 0;
        }
      }

      _acc(acc_type, count[0], &dst[dst_idx], &packed_buffer[packed_index], scale);
      packed_index += count[0];
    }

    CMX_ASSERT(packed_index == n1dim*count[0]);
  }
  if (CMX_ENABLE_ACC_SELF || CMX_ENABLE_ACC_SMP) {
    sem_post(semaphores[header->rank]);
  }

  if (use_eager) {
  }
  else {
    if ((unsigned)header->length > static_server_buffer_size) {
      free(acc_buffer);
    }
  }
}


void p_Environment::_acc_iov_handler(header_t *header, char *scale, int proc)
{
  reg_entry_t *reg_entry = NULL;
  void *mapped_offset = NULL;
  int i = 0;
  char *packed_buffer = NULL;
  int packed_index = 0;
  char *iov_buf = NULL;
  int iov_off = 0;
  int64_t limit = 0;
  int64_t bytes = 0;
  void **src = NULL;
  void **dst = NULL;
  int sizeof_scale = 0;
  int acc_type = 0;

#if DEBUG
  fprintf(stderr, "[%d] _acc_iov_handler proc=%d\n", p_config.rank(), proc);
#endif
#if DEBUG
  fprintf(stderr, "[%d] _acc_iov_handler header rem=%p loc=%p rem_rank=%d len=%d\n",
      p_config.rank(),
      header->remote_address,
      header->local_address,
      header->rank,
      header->length);
#endif

#if DEBUG
  fprintf(stderr, "[%d] _acc_iov_handler limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
      p_config.rank(), limit, bytes, src[0], dst[0]);
#endif

  switch (header->operation) {
    case OP_ACC_INT_IOV:
      acc_type = CMX_ACC_INT;
      sizeof_scale = sizeof(int);
      break;
    case OP_ACC_DBL_IOV:
      acc_type = CMX_ACC_DBL;
      sizeof_scale = sizeof(double);
      break;
    case OP_ACC_FLT_IOV:
      acc_type = CMX_ACC_FLT;
      sizeof_scale = sizeof(float);
      break;
    case OP_ACC_LNG_IOV:
      acc_type = CMX_ACC_LNG;
      sizeof_scale = sizeof(long);
      break;
    case OP_ACC_CPL_IOV:
      acc_type = CMX_ACC_CPL;
      sizeof_scale = sizeof(SingleComplex);
      break;
    case OP_ACC_DCP_IOV:
      acc_type = CMX_ACC_DCP;
      sizeof_scale = sizeof(DoubleComplex);
      break;
    default: CMX_ASSERT(0);
  }

  iov_buf = (char*)malloc(header->length);
  CMX_ASSERT(iov_buf);
  server_recv(iov_buf, header->length, proc);

  limit = *((int64_t*)(&iov_buf[iov_off]));
  iov_off += sizeof(int64_t);
  CMX_ASSERT(limit > 0);

  bytes = *((int64_t*)(&iov_buf[iov_off]));
  iov_off += sizeof(int64_t);
  CMX_ASSERT(bytes > 0);

  src = (void**)&iov_buf[iov_off];
  iov_off += sizeof(void*)*limit;

  dst = (void**)&iov_buf[iov_off];
  iov_off += sizeof(void*)*limit;

  CMX_ASSERT(iov_off == header->length);

  if ((unsigned)(bytes*limit) > static_server_buffer_size) {
    packed_buffer = (char*)malloc(bytes*limit);
  }
  else {
    packed_buffer = static_server_buffer;
  }

  server_recv(packed_buffer, bytes*limit, proc);

  if (CMX_ENABLE_ACC_SELF || CMX_ENABLE_ACC_SMP) {
    sem_wait(semaphores[header->rank]);
  }
  packed_index = 0;
  for (i=0; i<limit; ++i) {
    reg_entry = p_register.find(
        header->rank, dst[i], bytes);
    CMX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(reg_entry, dst[i]);

    _acc(acc_type, bytes, mapped_offset, &packed_buffer[packed_index], scale);
    packed_index += bytes;
  }
  CMX_ASSERT(packed_index == bytes*limit);
  if (CMX_ENABLE_ACC_SELF || CMX_ENABLE_ACC_SMP) {
    sem_post(semaphores[header->rank]);
  }

  if ((unsigned)(bytes*limit) > static_server_buffer_size) {
    free(packed_buffer);
  }

  free(iov_buf);
}


void p_Environment::_fence_handler(header_t *header, int proc)
{
#if DEBUG
  fprintf(stderr, "[%d] _fence_handler proc=%d\n", p_config.rank(), proc);
#endif

  /* preconditions */
  CMX_ASSERT(header);

#if NEED_ASM_VOLATILE_MEMORY
#if DEBUG
  fprintf(stderr, "[%d] _fence_handler asm volatile (\"\" : : : \"memory\"); \n",
      p_config.rank());
#endif
  asm volatile ("" : : : "memory"); 
#endif

  /* we send the ack back to the originating proc */
  server_send(NULL, 0, proc);
}


void p_Environment::_fetch_and_add_handler(header_t *header, char *payload, int proc)
{
  reg_entry_t *reg_entry = NULL;
  void *mapped_offset = NULL;
  int *value_int = NULL;
  long *value_long = NULL;

#if DEBUG
  fprintf(stderr, "[%d] _fetch_and_add_handler proc=%d\n", p_config.rank(), proc);
#endif
#if DEBUG
  fprintf(stderr, "[%d] header rem=%p loc=%p rank=%d len=%d\n",
      p_config.rank(),
      header->remote_address,
      header->local_address,
      header->rank,
      header->length);
#endif

  CMX_ASSERT(OP_FETCH_AND_ADD == header->operation);

  reg_entry = p_register.find(
      header->rank, header->remote_address, header->length);
  CMX_ASSERT(reg_entry);
  mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

  if (sizeof(int) == header->length) {
    value_int = (int*)malloc(sizeof(int));
    *value_int = *((int*)mapped_offset); /* "fetch" */
    *((int*)mapped_offset) += *((int*)payload); /* "add" */
    server_send(value_int, sizeof(int), proc);
    free(value_int);
  }
  else if (sizeof(long) == header->length) {
    value_long = (long*)malloc(sizeof(long));
    *value_long = *((long*)mapped_offset); /* "fetch" */
    *((long*)mapped_offset) += *((long*)payload); /* "add" */
    server_send(value_long, sizeof(long), proc);
    free(value_long);
  }
  else {
    CMX_ASSERT(0);
  }
}


void p_Environment::_swap_handler(header_t *header, char *payload, int proc)
{
  reg_entry_t *reg_entry = NULL;
  void *mapped_offset = NULL;
  int *value_int = NULL;
  long *value_long = NULL;

#if DEBUG
  fprintf(stderr, "[%d] _swap_handler rem=%p loc=%p rank=%d len=%d\n",
      p_config.rank(),
      header->remote_address,
      header->local_address,
      header->rank,
      header->length);
#endif

  CMX_ASSERT(OP_SWAP == header->operation);

  reg_entry = p_register.find(
      header->rank, header->remote_address, header->length);
  CMX_ASSERT(reg_entry);
  mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

  if (sizeof(int) == header->length) {
    value_int = (int*)malloc(sizeof(int));
    *value_int = *((int*)mapped_offset); /* "fetch" */
    *((int*)mapped_offset) = *((int*)payload); /* "swap" */
    server_send(value_int, sizeof(int), proc);
    free(value_int);
  }
  else if (sizeof(long) == header->length) {
    value_long = (long*)malloc(sizeof(long));
    *value_long = *((long*)mapped_offset); /* "fetch" */
    *((long*)mapped_offset) = *((long*)payload); /* "swap" */
    server_send(value_long, sizeof(long), proc);
    free(value_long);
  }
  else {
    CMX_ASSERT(0);
  }
}


void p_Environment::_mutex_create_handler(header_t *header, int proc)
{
  int i;
  int num = header->length;

#if DEBUG
  fprintf(stderr, "[%d] _mutex_create_handler proc=%d num=%d\n",
      p_config.rank(), proc, num);
#endif

  mutexes[proc] = (int*)malloc(sizeof(int) * num);
  lq_heads[proc].resize(num);
  for (i=0; i<num; ++i) {
    mutexes[proc][i] = UNLOCKED;
    lq_heads[proc][i] = NULL;
  }

  server_send(NULL, 0, proc);
}


void p_Environment::_mutex_destroy_handler(header_t *header, int proc)
{
  int i;
  int num = header->length;

#if DEBUG
  fprintf(stderr, "[%d] _mutex_destroy_handler proc=%d\n", p_config.rank(), proc);
#endif

  for (i=0; i<num; ++i) {
    CMX_ASSERT(mutexes[proc][i] == UNLOCKED);
    CMX_ASSERT(lq_heads[proc][i] == NULL);
  }

  free(mutexes[proc]);
  lq_heads[proc].clear();

  server_send(NULL, 0, proc);
}


void p_Environment::_lock_handler(header_t *header, int proc)
{
  int id = header->length;
  int rank = header->rank;

#if DEBUG
  fprintf(stderr, "[%d] _lock_handler id=%d in rank=%d req by proc=%d\n",
      p_config.rank(), id, rank, proc);
#endif

  CMX_ASSERT(0 <= id);

  if (UNLOCKED == mutexes[rank][id]) {
    mutexes[rank][id] = proc;
    server_send(&id, sizeof(int), proc);
  }
  else {
    lock_t *lock = NULL;
#if DEBUG
    fprintf(stderr, "[%d] _lq_push rank=%d req_by=%d id=%d\n",
        p_config.rank(), rank, proc, id);
#endif
    lock = (lock_t*)malloc(sizeof(lock_t));
    lock->next = NULL;
    lock->rank = proc;

    if (lq_heads[rank][id]) {
      /* insert at tail */
      lock_t *lq = lq_heads[rank][id];
      while (lq->next) {
        lq = lq->next;
      }
      lq->next = lock;
    }
    else {
      /* new head */
      lq_heads[rank][id] = lock;
    }
  }
}


void p_Environment::_unlock_handler(header_t *header, int proc)
{
  int id = header->length;
  int rank = header->rank;

#if DEBUG
  fprintf(stderr, "[%d] _unlock_handler id=%d in rank=%d req by proc=%d\n",
      p_config.rank(), id, rank, proc);
#endif

  CMX_ASSERT(0 <= id);

  if (lq_heads[rank][id]) {
    /* a lock requester was queued */
    /* find the next lock request and update queue */
    lock_t *lock = lq_heads[rank][id];
    lq_heads[rank][id] = lq_heads[rank][id]->next;
    /* update lock */
    mutexes[rank][id] = lock->rank;
    /* notify next in line */
    server_send(&id, sizeof(int), lock->rank);
    free(lock);
  }
  else {
    /* no enqued request */
    mutexes[rank][id] = UNLOCKED;
  }
}


void p_Environment::_malloc_handler(
    header_t *header, char *payload, int proc)
{
  int i;
  int n;
  reg_entry_t *reg_entries = reinterpret_cast<reg_entry_t*>(payload);

#if DEBUG
  fprintf(stderr, "[%d] _malloc_handler proc=%d\n", p_config.rank(), proc);
#endif

  CMX_ASSERT(header);
  CMX_ASSERT(header->operation == OP_MALLOC);
  n = header->length;

#if DEBUG && DEBUG_VERBOSE
  fprintf(stderr, "[%d] _malloc_handler preconditions complete\n", p_config.rank());
#endif

  /* insert reg entries into local registration cache */
  for (i=0; i<n; ++i) {
    if (NULL == reg_entries[i].buf) {
#if DEBUG && DEBUG_VERBOSE
      fprintf(stderr, "[%d] _malloc_handler found NULL at %d\n", p_config.rank(), i);
#endif
    }
    else if (p_config.hostid(reg_entries[i].rank)
        == p_config.hostid(p_config.rank())) {
      /* same SMP node, need to mmap */
      /* attach to remote shared memory object */
      void *memory;
      memory = _shm_attach(reg_entries[i].name, reg_entries[i].len);
#if DEBUG && DEBUG_VERBOSE
      fprintf(stderr, "[%d] _malloc_handler registering "
          "rank=%d buf=%p len=%lu name=%s, mapped=%p\n",
          p_config.rank(),
          reg_entries[i].rank,
          reg_entries[i].buf,
          (unsigned long)reg_entries[i].len,
          reg_entries[i].name,
          memory);
#endif
      (void)p_register.insert(
          reg_entries[i].rank,
          reg_entries[i].buf,
          reg_entries[i].len,
          reg_entries[i].name,
          memory
          ,reg_entries[i].use_dev
          );
    }
    else {
#if 0
      /* remote SMP node */
      /* i.e. we know about the mem but don't have local shared access */
      (void)p_register.insert(
          reg_entries[i].rank,
          reg_entries[i].buf,
          reg_entries[i].len,
          reg_entries[i].name,
          NULL);
#endif
    }
  }

#if DEBUG && DEBUG_VERBOSE
  fprintf(stderr, "[%d] _malloc_handler finished registrations\n", p_config.rank());
#endif

  server_send(NULL, 0, proc); /* ack */
}


void p_Environment::_free_handler(header_t *header, char *payload, int proc)
{
  int i = 0;
  int n = header->length;
  rank_ptr_t *rank_ptrs = (rank_ptr_t*)payload;

#if DEBUG
  fprintf(stderr, "[%d] _free_handler proc=%d\n", p_config.rank(), proc);
#endif

  /* remove all pointers from registration cache */
  for (i=0; i<n; ++i) {
    if (p_config.rank() == rank_ptrs[i].rank) {
#if DEBUG && DEBUG_VERBOSE
      fprintf(stderr, "[%d] cmx_free found self at %d\n", p_config.rank(), i);
#endif
    }
    else if (NULL == rank_ptrs[i].ptr) {
#if DEBUG && DEBUG_VERBOSE
      fprintf(stderr, "[%d] _free_handler found NULL at %d\n", p_config.rank(), i);
#endif
    }
    else if (p_config.hostid(rank_ptrs[i].rank)
        == p_config.hostid(p_config.rank())) {
      /* same SMP node */
      reg_entry_t *reg_entry = NULL;
      int retval = 0;

#if DEBUG && DEBUG_VERBOSE
      fprintf(stderr, "[%d] _free_handler same hostid at %d\n", p_config.rank(), i);
#endif

      /* find the registered memory */
      reg_entry = p_register.find(rank_ptrs[i].rank, rank_ptrs[i].ptr, 0);
      CMX_ASSERT(reg_entry);

#if DEBUG && DEBUG_VERBOSE
      fprintf(stderr, "[%d] _free_handler found reg entry\n", p_config.rank());
#endif

#if 1
      p_shmem.unmap(reg_entry->mapped,reg_entry->len);
#else
      /* unmap the memory */
      retval = munmap(reg_entry->mapped, reg_entry->len);
      if (-1 == retval) {
        perror("_free_handler: munmap");
        p_error("_free_handler: munmap", retval);
      }
#endif

#if DEBUG && DEBUG_VERBOSE
      fprintf(stderr, "[%d] _free_handler unmapped mapped memory in reg entry\n",
          p_config.rank());
#endif

      p_register.remove(rank_ptrs[i].rank, rank_ptrs[i].ptr);

#if DEBUG && DEBUG_VERBOSE
      fprintf(stderr, "[%d] _free_handler deleted reg cache entry\n",
          p_config.rank());
#endif

    }
    else {
#if 0
      p_register.remove(rank_ptrs[i].rank, rank_ptrs[i].ptr);

#if DEBUG && DEBUG_VERBOSE
      fprintf(stderr, "[%d] _free_handler deleted reg cache entry\n",
          p_config.rank());
#endif
#endif
    }
  }

  server_send(NULL, 0, proc); /* ack */
}


void* p_Environment::_get_offset_memory(reg_entry_t *reg_entry, void *memory)
{
  ptrdiff_t offset = 0;

  CMX_ASSERT(reg_entry);
#if DEBUG_VERBOSE
  fprintf(stderr, "[%d] _get_offset_memory reg_entry->buf=%p memory=%p\n",
      p_config.rank(), reg_entry->buf, memory);
#endif
  offset = ((char*)memory) - ((char*)reg_entry->buf);
#if DEBUG_VERBOSE
  fprintf(stderr, "[%d] _get_offset_memory ptrdiff=%lu\n",
      p_config.rank(), (unsigned long)offset);
#endif
  return (void*)((char*)(reg_entry->mapped)+offset);
}


int p_Environment::_get_world_rank(Group *group, int rank)
{
  int ret = group->getWorldRank(rank);
  if (ret < 0) {
    MPI_Group wgrp;
    MPI_Group lgrp;
    MPI_Comm_group(p_CMX_GROUP_WORLD->MPIComm(),&wgrp);
    MPI_Comm_group(group->MPIComm(),&lgrp);
    MPI_Group_translate_ranks(lgrp, 1, &rank, wgrp, &ret);
  }
  return ret;
}


/* gets (in group order) corresponding world ranks for entire group */
int* p_Environment::_get_world_ranks(Group *group)
{
#if 0
  int i = 0;
  int *group_ranks = (int*)malloc(sizeof(int)*igroup->size);
  int *world_ranks = (int*)malloc(sizeof(int)*igroup->size);
  int status;

  for (i=0; i<igroup->size; ++i) {
    group_ranks[i] = i;
    world_ranks[i] = MPI_PROC_NULL;
  }

  status = MPI_Group_translate_ranks(
      igroup->group, igroup->size, group_ranks,
      g_state.group, world_ranks);
  CMX_ASSERT(MPI_SUCCESS == status);

  for (i=0; i<igroup->size; ++i) {
    CMX_ASSERT(MPI_PROC_NULL != world_ranks[i]);
  }

  free(group_ranks);

  return world_ranks;
#else
#if 0
  MPI_Comm comm = igroup->comm;
  int i = 0;
  int my_world_rank = p_config.rank();
  int *world_ranks = (int*)malloc(sizeof(int)*igroup->size);
  int status;

  for (i=0; i<igroup->size; ++i) {
    world_ranks[i] = MPI_PROC_NULL;
  }

  status = MPI_Allgather(&my_world_rank,1,MPI_INT,world_ranks,
      1,MPI_INT,comm);
  CMX_ASSERT(MPI_SUCCESS == status);

  for (i=0; i<igroup->size; ++i) {
    CMX_ASSERT(MPI_PROC_NULL != world_ranks[i]);
  }

  return world_ranks;
#else
  int size = group->size();
  int i = 0;
  int *world_ranks = (int*)malloc(sizeof(int)*size);
  bool ok = true;
  for (i=0; i<size; ++i) {
    world_ranks[i] = group->getWorldRank(i);
    if (world_ranks[i] < 0) ok = false;
  }
  if (!ok) {
    MPI_Group wgrp;
    MPI_Group lgrp;
    MPI_Comm_group(p_CMX_GROUP_WORLD->MPIComm(),&wgrp);
    MPI_Comm_group(group->MPIComm(),&lgrp);
    std::vector<int> lranks(size);
    for (i=0; i<size; i++) lranks[i] = i;
    MPI_Group_translate_ranks(lgrp, size, &lranks[0], wgrp, world_ranks);
  }
  return world_ranks;
#endif
#endif
}


/* we sometimes need to notify a node master of some event and the rank in
 * charge of doing that is returned by this function */
int p_Environment::_smallest_world_rank_with_same_hostid(Group *group)
{
  int i = 0;
  int smallest = p_config.rank();
  int *world_ranks = _get_world_ranks(group);

  for (i=0; i<group->size(); ++i) {
    if (g_state.hostid[world_ranks[i]] == g_state.hostid[p_config.rank()]) {
      /* found same host as me */
      if (world_ranks[i] < smallest) {
        smallest = world_ranks[i];
      }
    }
  }

  free(world_ranks);

  return smallest;
}


/* we sometimes need to notify a node master of some event and the rank in
 * charge of doing that is returned by this function */
int p_Environment::_largest_world_rank_with_same_hostid(Group *group)
{
  int i = 0;
  int largest = p_config.rank();
  int *world_ranks = _get_world_ranks(group);

  for (i=0; i<group->size(); ++i) {
    if (g_state.hostid[world_ranks[i]] == g_state.hostid[p_config.rank()]) {
      /* found same host as me */
      if (world_ranks[i] > largest) {
        largest = world_ranks[i];
      }
    }
  }

  free(world_ranks);

  return largest;
}


void* p_Environment::_shm_attach(const char *name, size_t size)
{
  void *mapped = NULL;
  int fd = 0;
  int retval = 0;

#if DEBUG
  fprintf(stderr, "[%d] _shm_attach(%s, %lu)\n",
      p_config.rank(), name, (unsigned long)size);
#endif

  /* attach to shared memory segment */
  fd = shm_open(name, O_RDWR, S_IRUSR|S_IWUSR);
  if (-1 == fd) {
    perror("_shm_attach: shm_open");
    p_error("_shm_attach: shm_open", -1);
  }

  /* map into local address space */
  mapped = _shm_map(fd, size);
  /* close file descriptor */
  retval = close(fd);
  if (-1 == retval) {
    perror("_shm_attach: close");
    p_error("_shm_attach: close", -1);
  }

  return mapped;
}

void* p_Environment::_shm_map(int fd, size_t size)
{
  void *memory  = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if (MAP_FAILED == memory) {
    perror("_shm_map: mmap");
    p_error("_shm_map: mmap", -1);
  }

  return memory;
}


int p_Environment::_set_affinity(int cpu)
{
  int status = 0;
#if CMX_SET_AFFINITY
#if HAVE_PTHREAD_SETAFFINITY_NP || HAVE_SCHED_SETAFFINITY
  cpu_set_t cpuset;

  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);
#if HAVE_PTHREAD_SETAFFINITY_NP
  status = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (0 != status) {
    perror("pthread_setaffinity_np");
  }
#elif HAVE_SCHED_SETAFFINITY
  status = sched_setaffinity(getpid(), sizeof(cpu_set_t), &cpuset);
  if (0 != status) {
    perror("sched_setaffinity");
  }
#endif
#endif
#endif

  return status;
}


void p_Environment::check_mpi_retval(int retval, const char *file, int line)
{
  if (MPI_SUCCESS != retval) {
    const char *msg = str_mpi_retval(retval);
    fprintf(stderr, "{%d} MPI Error: %s: line %d: %s\n",
        p_config.rank(), file, line, msg);
    MPI_Abort(p_config.global_comm(), retval);
  }
}


const char *p_Environment::str_mpi_retval(int retval)
{
  const char *msg = NULL;

  switch(retval) {
    case MPI_SUCCESS       : msg = "MPI_SUCCESS"; break;
    case MPI_ERR_BUFFER    : msg = "MPI_ERR_BUFFER"; break;
    case MPI_ERR_COUNT     : msg = "MPI_ERR_COUNT"; break;
    case MPI_ERR_TYPE      : msg = "MPI_ERR_TYPE"; break;
    case MPI_ERR_TAG       : msg = "MPI_ERR_TAG"; break;
    case MPI_ERR_COMM      : msg = "MPI_ERR_COMM"; break;
    case MPI_ERR_RANK      : msg = "MPI_ERR_RANK"; break;
    case MPI_ERR_ROOT      : msg = "MPI_ERR_ROOT"; break;
    case MPI_ERR_GROUP     : msg = "MPI_ERR_GROUP"; break;
    case MPI_ERR_OP        : msg = "MPI_ERR_OP"; break;
    case MPI_ERR_TOPOLOGY  : msg = "MPI_ERR_TOPOLOGY"; break;
    case MPI_ERR_DIMS      : msg = "MPI_ERR_DIMS"; break;
    case MPI_ERR_ARG       : msg = "MPI_ERR_ARG"; break;
    case MPI_ERR_UNKNOWN   : msg = "MPI_ERR_UNKNOWN"; break;
    case MPI_ERR_TRUNCATE  : msg = "MPI_ERR_TRUNCATE"; break;
    case MPI_ERR_OTHER     : msg = "MPI_ERR_OTHER"; break;
    case MPI_ERR_INTERN    : msg = "MPI_ERR_INTERN"; break;
    case MPI_ERR_IN_STATUS : msg = "MPI_ERR_IN_STATUS"; break;
    case MPI_ERR_PENDING   : msg = "MPI_ERR_PENDING"; break;
    case MPI_ERR_REQUEST   : msg = "MPI_ERR_REQUEST"; break;
    case MPI_ERR_LASTCODE  : msg = "MPI_ERR_LASTCODE"; break;
    default                : msg = "DEFAULT"; break;
  }

  return msg;
}


void p_Environment::server_send(void *buf, int64_t count, int dest)
{
  int retval = 0;

#if DEBUG
  fprintf(stderr, "[%d] server_send(buf=%p, count=%d, dest=%d)\n",
      p_config.rank(), buf, count, dest);
#endif

  retval = MPI_Send(buf, static_cast<int>(count), MPI_CHAR, dest,
      CMX_TAG, p_config.global_comm());
  _translate_mpi_error(retval,"server_send:MPI_Send");

  CHECK_MPI_RETVAL(retval);
}


void p_Environment::server_send_datatype(void *buf, MPI_Datatype dt, int dest)
{
  int retval = 0;

#if DEBUG
  fprintf(stderr, "[%d] server_send_datatype(buf=%p, ..., dest=%d)\n",
      p_config.rank(), buf, dest);
#endif

  retval = MPI_Send(buf, 1, dt, dest, CMX_TAG, p_config.global_comm());
  _translate_mpi_error(retval,"server_send_datatype:MPI_Send");

  CHECK_MPI_RETVAL(retval);
}


void p_Environment::server_recv(void *buf, int64_t count, int source)
{
  int retval = 0;
  MPI_Status status;
  int recv_count = 0;

  retval = MPI_Recv(buf, static_cast<int>(count), MPI_CHAR, source,
      CMX_TAG, p_config.global_comm(), &status);
  _translate_mpi_error(retval,"server_recv:MPI_Recv");

  CHECK_MPI_RETVAL(retval);
  CMX_ASSERT(status.MPI_SOURCE == source);
  CMX_ASSERT(status.MPI_TAG == CMX_TAG);

  retval = MPI_Get_count(&status, MPI_CHAR, &recv_count);
  _translate_mpi_error(retval,"server_recv:MPI_Get_count");
  CHECK_MPI_RETVAL(retval);
  CMX_ASSERT(recv_count == count);
}


void p_Environment::server_recv_datatype(void *buf, MPI_Datatype dt, int source)
{
  int retval = 0;
  MPI_Status status;

  printf("p[%d] (server_recv_datatype) Waiting for recv\n",p_config.rank());
  retval = MPI_Recv(buf, 1, dt, source,
      CMX_TAG, p_config.global_comm(), &status);
  printf("p[%d] (server_recv_datatype) Completed recv\n",p_config.rank());
  _translate_mpi_error(retval,"server_recv_datatype:MPI_Recv");

  CHECK_MPI_RETVAL(retval);
  CMX_ASSERT(status.MPI_SOURCE == source);
  CMX_ASSERT(status.MPI_TAG == CMX_TAG);
}


void p_Environment::nb_send_common(void *buf, int count, int dest, _cmx_request *nb, int need_free)
{
  int retval = 0;
  message_t *message = NULL;

  CMX_ASSERT(NULL != nb);

  nb->send_size += 1;
  nb_count_event += 1;
  nb_count_send += 1;

  message = new message_t;
  message->next = NULL;
  message->message = static_cast<char*>(buf);
  message->need_free = need_free;
  message->stride = NULL;
  message->iov = NULL;
  message->datatype = MPI_DATATYPE_NULL;

  if (NULL == nb->send_head) {
    nb->send_head = message;
  }
  if (NULL != nb->send_tail) {
    nb->send_tail->next = message;
  }
  nb->send_tail = message;

  retval = MPI_Isend(buf, count, MPI_CHAR, dest, CMX_TAG,
      p_config.global_comm(), &(message->request));
  _translate_mpi_error(retval,"nb_send_common:MPI_Isend");
  CHECK_MPI_RETVAL(retval);
}


void p_Environment::nb_send_datatype(void *buf, MPI_Datatype dt, int dest, _cmx_request *nb)
{
  int retval = 0;
  message_t *message = NULL;

  CMX_ASSERT(NULL != nb);

  nb->send_size += 1;
  nb_count_event += 1;
  nb_count_send += 1;

  message = new message_t;
  message->next = NULL;
  message->message = static_cast<char*>(buf);
  message->need_free = 0;
  message->stride = NULL;
  message->iov = NULL;
  message->datatype = dt;

  if (NULL == nb->send_head) {
    nb->send_head = message;
  }
  if (NULL != nb->send_tail) {
    nb->send_tail->next = message;
  }
  nb->send_tail = message;

  printf("p[%d] (nb_send_datatype) calling isend\n",p_config.rank());
  retval = MPI_Isend(buf, 1, dt, dest, CMX_TAG, p_config.global_comm(),
      &(message->request));
  printf("p[%d] (nb_send_datatype) completed isend\n",p_config.rank());
  _translate_mpi_error(retval,"nb_send_datatype:MPI_Isend");
  CHECK_MPI_RETVAL(retval);
}


void p_Environment::nb_send_header(void *buf, int count, int dest, _cmx_request *nb)
{
  nb_send_common(buf, count, dest, nb, 1);
}


void p_Environment::nb_send_buffer(void *buf, int count, int dest, _cmx_request *nb)
{
  nb_send_common(buf, count, dest, nb, 0);
}


void p_Environment::nb_recv_packed(void *buf, int count, int source, _cmx_request *nb, stride_t *stride)
{
  int retval = 0;
  message_t *message = NULL;

  CMX_ASSERT(NULL != buf);
  CMX_ASSERT(count > 0);
  CMX_ASSERT(NULL != nb);

#if DEBUG
  fprintf(stderr, "[%d] nb_recv_packed(buf=%p, count=%d, source=%d, nb=%p)\n",
      p_config.rank(), buf, count, source, nb);
#endif

  nb->recv_size += 1;
  nb_count_event += 1;
  nb_count_recv += 1;

  message = new message_t;
  message->next = NULL;
  message->message = static_cast<char*>(buf);
  message->need_free = 1;
  message->stride = stride;
  message->iov = NULL;
  message->datatype = MPI_DATATYPE_NULL;

  if (NULL == nb->recv_head) {
    nb->recv_head = message;
  }
  if (NULL != nb->recv_tail) {
    nb->recv_tail->next = message;
  }
  nb->recv_tail = message;

  retval = MPI_Irecv(buf, count, MPI_CHAR, source, CMX_TAG,
      p_config.global_comm(), &(message->request));
  _translate_mpi_error(retval,"nb_recv_packed:MPI_Irecv");
  CHECK_MPI_RETVAL(retval);
}


void p_Environment::nb_recv_datatype(void *buf, MPI_Datatype dt, int source, _cmx_request *nb)
{
  int retval = 0;
  message_t *message = NULL;

  CMX_ASSERT(NULL != buf);
  CMX_ASSERT(NULL != nb);

#if DEBUG
  fprintf(stderr, "[%d] nb_recv_datatype(buf=%p, count=%d, source=%d, nb=%p)\n",
      p_config.rank(), buf, count, source, nb);
#endif

  nb->recv_size += 1;
  nb_count_event += 1;
  nb_count_recv += 1;

  message = new message_t;
  message->next = NULL;
  message->message = static_cast<char*>(buf);
  message->need_free = 0;
  message->stride = NULL;
  message->iov = NULL;
  message->datatype = dt;

  if (NULL == nb->recv_head) {
    nb->recv_head = message;
  }
  if (NULL != nb->recv_tail) {
    nb->recv_tail->next = message;
  }
  nb->recv_tail = message;

  retval = MPI_Irecv(buf, 1, dt, source, CMX_TAG, p_config.global_comm(),
      &(message->request));
  _translate_mpi_error(retval,"nb_recv_datatype:MPI_Irecv");
  CHECK_MPI_RETVAL(retval);
}


void p_Environment::nb_recv_iov(void *buf, int count, int source, _cmx_request *nb, _cmx_giov_t *iov)
{
  int retval = 0;
  message_t *message = NULL;

  CMX_ASSERT(NULL != nb);

#if DEBUG
  fprintf(stderr, "[%d] nb_recv_iov(buf=%p, count=%d, source=%d, nb=%p)\n",
      p_config.rank(), buf, count, source, nb);
#endif

  nb->recv_size += 1;
  nb_count_event += 1;
  nb_count_recv += 1;

  message = new message_t;
  message->next = NULL;
  message->message = static_cast<char*>(buf);
  message->need_free = 1;
  message->stride = NULL;
  message->iov = iov;
  message->datatype = MPI_DATATYPE_NULL;

  if (NULL == nb->recv_head) {
    nb->recv_head = message;
    CMX_ASSERT(NULL == nb->recv_tail);
  }
  if (NULL != nb->recv_tail) {
    nb->recv_tail->next = message;
  }
  nb->recv_tail = message;

  retval = MPI_Irecv(buf, count, MPI_CHAR, source, CMX_TAG,
      p_config.global_comm(), &(message->request));
  _translate_mpi_error(retval,"nb_recv_iov:MPI_Irecv");
  CHECK_MPI_RETVAL(retval);
}


void p_Environment::nb_recv(void *buf, int count, int source, _cmx_request *nb)
{
  int retval = 0;
  message_t *message = NULL;

  CMX_ASSERT(NULL != nb);

#if DEBUG
  fprintf(stderr, "[%d] nb_recv(buf=%p, count=%d, source=%d, nb=%p)\n",
      p_config.rank(), buf, count, source, nb);
#endif

  nb->recv_size += 1;
  nb_count_event += 1;
  nb_count_recv += 1;

  message = new message_t;
  message->next = NULL;
  message->message = NULL;
  message->need_free = 0;
  message->stride = NULL;
  message->iov = NULL;
  message->datatype = MPI_DATATYPE_NULL;

  if (NULL == nb->recv_head) {
    nb->recv_head = message;
  }
  if (NULL != nb->recv_tail) {
    nb->recv_tail->next = message;
  }
  nb->recv_tail = message;

  retval = MPI_Irecv(buf, count, MPI_CHAR, source, CMX_TAG, p_config.global_comm(),
      &(message->request));
  _translate_mpi_error(retval,"nb_recv:MPI_Irecv");
  CHECK_MPI_RETVAL(retval);
}


#ifdef OLD_CODE
int p_Environment::nb_get_handle_index()
{
  int value = 0;

  if (0 == nb_index) {
    value = nb_max_outstanding-1;
  }
  else {
    value = nb_index-1;
  }

  return value;
}
#endif


void p_Environment::nb_wait_for_send1(_cmx_request *nb)
{
#if DEBUG
  fprintf(stderr, "[%d] nb_wait_for_send1(nb=%p)\n", p_config.rank(), nb);
#endif

  CMX_ASSERT(NULL != nb);
  CMX_ASSERT(NULL != nb->send_head);

  {
    MPI_Status status;
    int retval = 0;
    message_t *message_to_free = NULL;

    retval = MPI_Wait(&(nb->send_head->request), &status);
    _translate_mpi_error(retval,"nb_wait_for_send1:MPI_Wait");
    CHECK_MPI_RETVAL(retval);

    if (nb->send_head->need_free) {
      delete [] nb->send_head->message;
    }

    if (MPI_DATATYPE_NULL != nb->send_head->datatype) {
      retval = MPI_Type_free(&nb->send_head->datatype);
      _translate_mpi_error(retval,"nb_wait_for_send1:MPI_Type_free");
      CHECK_MPI_RETVAL(retval);
    }

    message_to_free = nb->send_head;
    nb->send_head = nb->send_head->next;
    delete message_to_free;

    CMX_ASSERT(nb->send_size > 0);
    nb->send_size -= 1;
    nb_count_send_processed += 1;
    nb_count_event_processed += 1;

    if (NULL == nb->send_head) {
      nb->send_tail = NULL;
    }
  }
}


/* returns true if operation has completed */
int p_Environment::nb_test_for_send1(_cmx_request *nb, message_t **save_send_head,
    message_t **prev)
{
#if DEBUG
  fprintf(stderr, "[%d] nb_test_for_send1(nb=%p)\n", p_config.rank(), nb);
#endif

  CMX_ASSERT(NULL != nb);
  CMX_ASSERT(NULL != nb->send_head);

  {
    MPI_Status status;
    int retval = 0;
    int flag;
    message_t *message_to_free = NULL;

    retval = MPI_Test(&(nb->send_head->request), &flag, &status);
    _translate_mpi_error(retval,"nb_test_for_send1:MPI_Test");
    CHECK_MPI_RETVAL(retval);

    if (flag) {
      if (nb->send_head->need_free) {
        delete [] nb->send_head->message;
      }

      if (MPI_DATATYPE_NULL != nb->send_head->datatype) {
        retval = MPI_Type_free(&nb->send_head->datatype);
        _translate_mpi_error(retval,"nb_test_for_send1:MPI_Type_free");
        CHECK_MPI_RETVAL(retval);
      }

      message_to_free = nb->send_head;
      if (*prev) (*prev)->next=nb->send_head->next;
      nb->send_head = nb->send_head->next;
      *save_send_head = NULL;
      delete message_to_free;

      CMX_ASSERT(nb->send_size > 0);
      nb->send_size -= 1;
      nb_count_send_processed += 1;
      nb_count_event_processed += 1;

      if (NULL == nb->send_head) {
        nb->send_tail = NULL;
      }
    } else {
      *prev = nb->send_head;
      *save_send_head = nb->send_head;
      nb->send_head = nb->send_head->next;
    }
    return flag;
  }
}


void p_Environment::nb_wait_for_recv1(_cmx_request *nb)
{
#if DEBUG
  fprintf(stderr, "[%d] nb_wait_for_recv1(nb=%p)\n", p_config.rank(), nb);
#endif

  CMX_ASSERT(NULL != nb);
  CMX_ASSERT(NULL != nb->recv_head);

  {
    MPI_Status status;
    int retval = 0;
    message_t *message_to_free = NULL;

    retval = MPI_Wait(&(nb->recv_head->request), &status);
    _translate_mpi_error(retval,"nb_wait_for_recv1:MPI_Wait");
    CHECK_MPI_RETVAL(retval);

    if (NULL != nb->recv_head->stride) {
      stride_t *stride = nb->recv_head->stride;
      CMX_ASSERT(nb->recv_head->message);
      CMX_ASSERT(stride);
      CMX_ASSERT(stride->ptr);
      CMX_ASSERT(stride->stride);
      CMX_ASSERT(stride->count);
      CMX_ASSERT(stride->stride_levels);
      unpack(static_cast<char*>(nb->recv_head->message),
          static_cast<char*>(stride->ptr),
          stride->stride, stride->count, stride->stride_levels);
      delete stride;
    }

    if (NULL != nb->recv_head->iov) {
      int i = 0;
      char *message = static_cast<char*>(nb->recv_head->message);
      int off = 0;
      _cmx_giov_t *iov = nb->recv_head->iov;
      for (i=0; i<iov->count; ++i) {
        (void)memcpy(iov->dst[i], &message[off], iov->bytes);
        off += iov->bytes;
      }
      delete iov->src;
      delete iov->dst;
      delete iov;
    }

    if (nb->recv_head->need_free) {
      delete [] nb->recv_head->message;
    }

    if (MPI_DATATYPE_NULL != nb->recv_head->datatype) {
      retval = MPI_Type_free(&nb->recv_head->datatype);
      _translate_mpi_error(retval,"nb_wait_for_recv1:MPI_Type_free");
      CHECK_MPI_RETVAL(retval);
    }

    message_to_free = nb->recv_head;
    nb->recv_head = nb->recv_head->next;
    delete message_to_free;

    CMX_ASSERT(nb->recv_size > 0);
    nb->recv_size -= 1;
    nb_count_recv_processed += 1;
    nb_count_event_processed += 1;

    if (NULL == nb->recv_head) {
      nb->recv_tail = NULL;
    }
  }
}


/* returns true if operation has completed */
int p_Environment::nb_test_for_recv1(_cmx_request *nb, message_t **save_recv_head,
    message_t **prev)
{
#if DEBUG
  fprintf(stderr, "[%d] nb_test_for_recv1(nb=%p)\n", p_config.rank(), nb);
#endif

  CMX_ASSERT(NULL != nb);
  CMX_ASSERT(NULL != nb->recv_head);

  {
    MPI_Status status;
    int retval = 0;
    int flag;
    message_t *message_to_free = NULL;

    retval = MPI_Test(&(nb->recv_head->request), &flag, &status);
    _translate_mpi_error(retval,"nb_test_for_recv1:MPI_Test");
    CHECK_MPI_RETVAL(retval);

    if (flag) {
      if (NULL != nb->recv_head->stride) {
        stride_t *stride = nb->recv_head->stride;
        CMX_ASSERT(nb->recv_head->message);
        CMX_ASSERT(stride);
        CMX_ASSERT(stride->ptr);
        CMX_ASSERT(stride->stride);
        CMX_ASSERT(stride->count);
        CMX_ASSERT(stride->stride_levels);
        unpack((char*)nb->recv_head->message, (char*)stride->ptr,
            stride->stride, stride->count, stride->stride_levels);
        delete stride;
      }

      if (NULL != nb->recv_head->iov) {
        int i = 0;
        char *message = (char*)nb->recv_head->message;
        int off = 0;
        _cmx_giov_t *iov = nb->recv_head->iov;
        for (i=0; i<iov->count; ++i) {
          (void)memcpy(iov->dst[i], &message[off], iov->bytes);
          off += iov->bytes;
        }
        delete iov->src;
        delete iov->dst;
        delete iov;
      }

      if (nb->recv_head->need_free) {
        delete [] nb->recv_head->message;
      }

      if (MPI_DATATYPE_NULL != nb->recv_head->datatype) {
        retval = MPI_Type_free(&nb->recv_head->datatype);
        _translate_mpi_error(retval,"nb_test_for_recv1:MPI_Type_free");
        CHECK_MPI_RETVAL(retval);
      }

      message_to_free = nb->recv_head;
      if (*prev) (*prev)->next=nb->recv_head->next;
      nb->recv_head = nb->recv_head->next;
      *save_recv_head = NULL;
      free(message_to_free);

      CMX_ASSERT(nb->recv_size > 0);
      nb->recv_size -= 1;
      nb_count_recv_processed += 1;
      nb_count_event_processed += 1;

      if (NULL == nb->recv_head) {
        nb->recv_tail = NULL;
      }
    } else {
      *prev = nb->recv_head;
      *save_recv_head = nb->recv_head;
      nb->recv_head = nb->recv_head->next;
    }
    return flag;
  }
}

/**
 * Initialize message struct before using
 * @param message struct to be initialized
 */
void p_Environment::init_message(message_t *message)
{
  message->next = NULL;
  message->message = NULL;
  message->need_free = 0;
  message->stride = NULL;
  message->iov = NULL;
}

void p_Environment::nb_wait_for_all(_cmx_request *nb)
{
#if DEBUG
  fprintf(stderr, "[%d] nb_wait_for_all(nb=%p)\n", p_config.rank(), nb);
#endif
  int world_proc = p_config.rank();

  if (nb->in_use == 0) return;

  CMX_ASSERT(NULL != nb);

  /* fair processing of requests */
  while (NULL != nb->send_head || NULL != nb->recv_head) {
    if (NULL != nb->send_head) {
      nb_wait_for_send1(nb);
    }
    if (NULL != nb->recv_head) {
      nb_wait_for_recv1(nb);
    }
  }
  if (nb->send_tail != NULL) printf("p[%d] (nb_wait_for_all) send_tail not deleted\n",p_config.rank());
  if (nb->recv_tail != NULL) printf("p[%d] (nb_wait_for_all) recv_tail not deleted\n",p_config.rank());
  nb->in_use = 0;
  nb_unregister_request(nb);
}

/* Returns 0 if no outstanding requests */

int p_Environment::nb_test_for_all(_cmx_request *nb)
{
#if DEBUG
  fprintf(stderr, "[%d] nb_test_for_all(nb=%p)\n", p_config.rank(), nb);
#endif
  int ret = 0;
  message_t *save_send_head = NULL;
  message_t *save_recv_head = NULL;
  message_t *tmp_send_head;
  message_t *tmp_recv_head;
  message_t *send_prev = NULL;
  message_t *recv_prev = NULL;

  /**
   * TODO: Determine if this condition may be true for a valid series of
   * operations. In particular, if a set of non-blocking operations follow a
   * set of blocking operations.
   CMX_ASSERT(NULL != nb);
   */
  if (nb == NULL) return 0;

  /* check for outstanding requests */
  while (NULL != nb->send_head || NULL != nb->recv_head) {
    if (NULL != nb->send_head) {
      if (!nb_test_for_send1(nb, &tmp_send_head, &send_prev)) {
        ret = 1; 
      }
      if ((NULL == save_send_head) && (ret == 1)) {
        save_send_head = tmp_send_head;
      }
    }
    if (NULL != nb->recv_head) {
      if (!nb_test_for_recv1(nb, &tmp_recv_head, &recv_prev)) {
        ret = 1;
      }
      if ((NULL == save_recv_head) && (ret == 1)) {
        save_recv_head = tmp_recv_head;
      }
    }
  }
  nb->send_head = save_send_head;
  nb->recv_head = save_recv_head;
  if (ret == 0) nb->in_use = 0;
  return ret;
}


void p_Environment::nb_put(void *src, void *dst, int64_t bytes, int proc,
    _cmx_request *nb)
{
  CMX_ASSERT(NULL != src);
  CMX_ASSERT(NULL != dst);
  CMX_ASSERT(bytes > 0);
  CMX_ASSERT(proc >= 0);
  CMX_ASSERT(proc < p_config.size());
  CMX_ASSERT(NULL != nb);

#if DEBUG
  printf("[%d] nb_put(src=%p, dst=%p, bytes=%d, proc=%d, nb=%p)\n",
      p_config.rank(), src, dst, bytes, proc, nb);
#endif

  if (CMX_ENABLE_PUT_SELF) {
    /* put to self */
    if (p_config.rank() == proc) {
      if (fence_array[p_config.master(proc)]) {
        _fence_master(p_config.master(proc));
      }
      (void)memcpy(dst, src, bytes);
      return;
    }
  }

  if (CMX_ENABLE_PUT_SMP) {
    /* put to SMP node */
    if (p_config.master(proc) == p_config.master(p_config.rank())) 
    {
      reg_entry_t *reg_entry = NULL;
      void *mapped_offset = NULL;

      if (fence_array[p_config.master(proc)]) {
        _fence_master(p_config.master(proc));
      }

      reg_entry = p_register.find(proc, dst, bytes);
      CMX_ASSERT(reg_entry);
      mapped_offset = _get_offset_memory(reg_entry, dst);
      (void)memcpy(mapped_offset, src, bytes);
      return;
    }
  }

  {
    char *message = NULL;
    size_t message_size = 0;
    header_t *header = NULL;
    int master_rank = -1;
    int use_eager = _eager_check(bytes);

    master_rank = p_config.master(proc);
    /* only fence on the master */
    fence_array[master_rank] = 1;
    if (use_eager) {
      message_size = sizeof(header_t) + bytes;
    }
    else {
      message_size = sizeof(header_t);
    }
    message = new char[message_size];
    header = reinterpret_cast<header_t*>(message);
    header->operation = OP_PUT;
    MAYBE_MEMSET(header, 0, sizeof(header_t));
    header->remote_address = static_cast<char*>(dst);
    header->local_address = static_cast<char*>(src);
    header->rank = proc;
    header->length = bytes;
    if (use_eager) {
      (void)memcpy(message+sizeof(header_t), src, bytes);
      nb_send_header(message, message_size, master_rank, nb);
    }
    else {
      char *buf = (char*)src;
      size_t bytes_remaining = bytes;
      nb_send_header(header, sizeof(header_t), master_rank, nb);
      do {
        size_t size = bytes_remaining>max_message_size ?
          max_message_size : bytes_remaining;
        nb_send_buffer(buf, size, master_rank, nb);
        buf += size;
        bytes_remaining -= size;
      } while (bytes_remaining > 0);
    }
  }
  nb->in_use = 1;
}


void p_Environment::nb_get(void *src, void *dst, int64_t bytes, int proc, _cmx_request *nb)
{
  CMX_ASSERT(NULL != src);
  CMX_ASSERT(NULL != dst);
  CMX_ASSERT(bytes > 0);
  CMX_ASSERT(proc >= 0);
  CMX_ASSERT(proc < p_config.size());
  CMX_ASSERT(NULL != nb);

  if (CMX_ENABLE_GET_SELF) {
    /* get from self */
    if (p_config.rank() == proc) {
      if (fence_array[p_config.master(proc)]) {
        _fence_master(p_config.master(proc));
      }
      (void)memcpy(dst, src, bytes);
      return;
    }
  }

  if (CMX_ENABLE_GET_SMP) {
    /* get from SMP node */
    // if (g_state.hostid[proc] == g_state.hostid[p_config.rank()]) 
    if (p_config.master(proc) == p_config.master(p_config.rank())) 
    {
      reg_entry_t *reg_entry = NULL;
      void *mapped_offset = NULL;

      if (fence_array[p_config.master(proc)]) {
        _fence_master(p_config.master(proc));
      }

      reg_entry = p_register.find(proc, src, bytes);
      CMX_ASSERT(reg_entry);
      mapped_offset = _get_offset_memory(reg_entry, src);
      (void)memcpy(dst, mapped_offset, bytes);
      return;
    }
  }

  {
    header_t *header = NULL;
    int master_rank = -1;

    master_rank = p_config.master(proc);
    char *message = new char[sizeof(header_t)];
    header = reinterpret_cast<header_t*>(message);
    CMX_ASSERT(header);
    MAYBE_MEMSET(header, 0, sizeof(header_t));
    header->operation = OP_GET;
    header->remote_address = static_cast<char*>(src);
    header->local_address = static_cast<char*>(dst);
    header->rank = proc;
    header->length = bytes;
    {
      /* prepost all receives */
      char *buf = (char*)dst;
      size_t bytes_remaining = bytes;
      do {
        size_t size = bytes_remaining>max_message_size ?
          max_message_size : bytes_remaining;
        nb_recv(buf, size, master_rank, nb);
        buf += size;
        bytes_remaining -= size;
      } while (bytes_remaining > 0);
    }
    nb_send_header(message, sizeof(header_t), master_rank, nb);
  }
  nb->in_use = 1;
}


void p_Environment::nb_acc(int datatype, void *scale,
    void *src, void *dst, int64_t bytes, int proc, _cmx_request *nb)
{
  CMX_ASSERT(NULL != src);
  CMX_ASSERT(NULL != dst);
  CMX_ASSERT(bytes > 0);
  CMX_ASSERT(proc >= 0);
  CMX_ASSERT(proc < p_config.size());
  CMX_ASSERT(NULL != nb);

  if (CMX_ENABLE_ACC_SELF) {
    /* acc to self */
    if (p_config.rank() == proc) {
      if (fence_array[p_config.master(proc)]) {
        _fence_master(p_config.master(proc));
      }
      sem_wait(semaphores[proc]);
      _acc(datatype, bytes, dst, src, scale);
      sem_post(semaphores[proc]);
      return;
    }
  }

  if (CMX_ENABLE_ACC_SMP) {
    /* acc to same SMP node */
    // if (g_state.hostid[proc] == g_state.hostid[p_config.rank()]) 
    if (p_config.master(proc) == p_config.master(p_config.rank())) 
    {
      reg_entry_t *reg_entry = NULL;
      void *mapped_offset = NULL;

      if (fence_array[p_config.master(proc)]) {
        _fence_master(p_config.master(proc));
      }

      reg_entry = p_register.find(proc, dst, bytes);
      CMX_ASSERT(reg_entry);
      mapped_offset = _get_offset_memory(reg_entry, dst);
      if (sem_wait(semaphores[proc]) != 0) {
        if (errno == EAGAIN) {
          printf("p[%d] Operation could not be performed without blocking\n",
              p_config.rank());
        } else if (errno == EINTR) {
          printf("p[%d] Call interrupted\n",p_config.rank());
        } else if (errno == EINVAL) {
          printf("p[%d] Not a valid semiphore\n",p_config.rank());
        } else if (errno == ETIMEDOUT) {
          printf("p[%d] Call timed out\n",p_config.rank());
        }
        CMX_ASSERT(0);
      }
      _acc(datatype, bytes, mapped_offset, src, scale);
      sem_post(semaphores[proc]);
      return;
    }
  }

  {
    header_t *header = NULL;
    char *message = NULL;
    int master_rank = -1;
    size_t message_size = 0;
    int scale_size = 0;
    op_t operation = OP_NULL;
    int use_eager = 0;

    switch (datatype) {
      case CMX_ACC_INT:
        operation = OP_ACC_INT;
        scale_size = sizeof(int);
        break;
      case CMX_ACC_DBL:
        operation = OP_ACC_DBL;
        scale_size = sizeof(double);
        break;
      case CMX_ACC_FLT:
        operation = OP_ACC_FLT;
        scale_size = sizeof(float);
        break;
      case CMX_ACC_CPL:
        operation = OP_ACC_CPL;
        scale_size = sizeof(SingleComplex);
        break;
      case CMX_ACC_DCP:
        operation = OP_ACC_DCP;
        scale_size = sizeof(DoubleComplex);
        break;
      case CMX_ACC_LNG:
        operation = OP_ACC_LNG;
        scale_size = sizeof(long);
        break;
      default: CMX_ASSERT(0);
    }
    use_eager = _eager_check(scale_size+bytes);

    master_rank = p_config.master(proc);

    /* only fence on the master */
    fence_array[master_rank] = 1;

    if (use_eager) {
      message_size = sizeof(header_t) + scale_size + bytes;
    }
    else {
      message_size = sizeof(header_t) + scale_size;
    }
    message = new char[message_size];
    CMX_ASSERT(message);
    header = reinterpret_cast<header_t*>(message);
    header->operation = operation;
    header->remote_address = static_cast<char*>(dst);
    header->local_address = static_cast<char*>(src);
    header->rank = proc;
    header->length = bytes;
    (void)memcpy(message+sizeof(header_t), scale, scale_size);
    if (use_eager) {
      (void)memcpy(message+sizeof(header_t)+scale_size,
          src, bytes);
      nb_send_header(message, message_size, master_rank, nb);
    }
    else {
      char *buf = static_cast<char*>(src);
      size_t bytes_remaining = bytes;
      nb_send_header(message, message_size, master_rank, nb);
      do {
        size_t size = bytes_remaining>max_message_size ?
          max_message_size : bytes_remaining;
        nb_send_buffer(buf, size, master_rank, nb);
        buf += size;
        bytes_remaining -= size;
      } while (bytes_remaining > 0);
    }
  }
  nb->in_use = 1;
}


void p_Environment::nb_puts(
    void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels, int proc, _cmx_request *nb)
{
  int i, j;
  int64_t src_idx, dst_idx;  /* index offset of current block position to ptr */
  int64_t n1dim;  /* number of 1 dim block */
  int64_t src_bvalue[7], src_bunit[7];
  int64_t dst_bvalue[7], dst_bunit[7];

#if DEBUG
  fprintf(stderr, "[%d] nb_puts(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
      p_config.rank(), src, src_stride, dst, dst_stride,
      count[0], stride_levels, proc, nb);
#endif

  /* if not actually a strided put */
  if (0 == stride_levels) {
    nb_put(src, dst, count[0], proc, nb);
    return;
  }

  /* if not a strided put to self or SMP, use datatype algorithm */
  if (CMX_ENABLE_PUT_DATATYPE
      && (!CMX_ENABLE_PUT_SELF || p_config.rank() != proc)
      && (!CMX_ENABLE_PUT_SMP
        || p_config.hostid(proc) != p_config.hostid(p_config.rank()))
      && (_packed_size(src_stride, count, stride_levels) > CMX_PUT_DATATYPE_THRESHOLD)) {
    nb_puts_datatype(src, src_stride, dst, dst_stride, count, stride_levels, proc, nb);
    return;
  }

  /* if not a strided put to self or SMP, use packed algorithm */
  if (CMX_ENABLE_PUT_PACKED
      && (!CMX_ENABLE_PUT_SELF || p_config.rank() != proc)
      && (!CMX_ENABLE_PUT_SMP
        || p_config.hostid(proc) != p_config.hostid(p_config.rank()))) {
    nb_puts_packed(src, src_stride, dst, dst_stride, count, stride_levels, proc, nb);
    return;
  }

  /* number of n-element of the first dimension */
  n1dim = 1;
  for(i=1; i<=stride_levels; i++) {
    n1dim *= count[i];
  }

  /* calculate the destination indices */
  src_bvalue[0] = 0; src_bvalue[1] = 0; src_bunit[0] = 1; src_bunit[1] = 1;
  dst_bvalue[0] = 0; dst_bvalue[1] = 0; dst_bunit[0] = 1; dst_bunit[1] = 1;

  for(i=2; i<=stride_levels; i++) {
    src_bvalue[i] = 0;
    dst_bvalue[i] = 0;
    src_bunit[i] = src_bunit[i-1] * count[i-1];
    dst_bunit[i] = dst_bunit[i-1] * count[i-1];
  }

  /* index mangling */
  for(i=0; i<n1dim; i++) {
    src_idx = 0;
    dst_idx = 0;
    for(j=1; j<=stride_levels; j++) {
      src_idx += (int64_t) src_bvalue[j] * (int64_t) src_stride[j-1];
      if((i+1) % src_bunit[j] == 0) {
        src_bvalue[j]++;
      }
      if(src_bvalue[j] > (count[j]-1)) {
        src_bvalue[j] = 0;
      }
    }

    for(j=1; j<=stride_levels; j++) {
      dst_idx += (int64_t) dst_bvalue[j] * (int64_t) dst_stride[j-1];
      if((i+1) % dst_bunit[j] == 0) {
        dst_bvalue[j]++;
      }
      if(dst_bvalue[j] > (count[j]-1)) {
        dst_bvalue[j] = 0;
      }
    }

    nb_put((char *)src + src_idx, (char *)dst + dst_idx,
        count[0], proc, nb);
  }
}


void p_Environment::nb_puts_packed(
    void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels, int proc, _cmx_request *nb)
{
  int64_t i;
  int64_t packed_index = 0;
  char *packed_buffer = NULL;
  stride_t stride;

#if DEBUG
  fprintf(stderr, "[%d] nb_puts_packed(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
      p_config.rank(), src, src_stride, dst, dst_stride,
      count[0], stride_levels, proc, nb);
#endif

  CMX_ASSERT(proc >= 0);
  CMX_ASSERT(proc < p_config.size());
  CMX_ASSERT(NULL != src);
  CMX_ASSERT(NULL != dst);
  CMX_ASSERT(NULL != count);
  CMX_ASSERT(NULL != nb);
  CMX_ASSERT(stride_levels >= 0);
  CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);

  /* copy dst info into structure */
  stride.stride_levels = stride_levels;
  stride.count[0] = count[0];
  for (i=0; i<stride_levels; ++i) {
    stride.stride[i] = dst_stride[i];
    stride.count[i+1] = count[i+1];
  }
  for (/*no init*/; i<CMX_MAX_STRIDE_LEVEL; ++i) {
    stride.stride[i] = -1;
    stride.count[i+1] = -1;
  }

  CMX_ASSERT(stride.stride_levels >= 0);
  CMX_ASSERT(stride.stride_levels < CMX_MAX_STRIDE_LEVEL);

#if DEBUG
  fprintf(stderr, "[%d] nb_puts_packed stride_levels=%d, count[0]=%d\n",
      p_config.rank(), stride_levels, count[0]);
  for (i=0; i<stride_levels; ++i) {
    printf("[%d] stride[%d]=%d count[%d+1]=%d\n",
        p_config.rank(), i, stride.stride[i], i, stride.count[i+1]);
  }
#endif

  packed_buffer = pack((char*)src, src_stride, count, stride_levels, &packed_index);

  CMX_ASSERT(NULL != packed_buffer);
  CMX_ASSERT(packed_index > 0);

  {
    char *message = NULL;
    size_t message_size = 0;
    header_t *header = NULL;
    int master_rank = -1;
    int use_eager = _eager_check(sizeof(stride_t)+packed_index);

    master_rank = p_config.master(proc);
    /* only fence on the master */
    fence_array[master_rank] = 1;
    if (use_eager) {
      message_size = sizeof(header_t)+sizeof(stride_t)+packed_index;
    }
    else {
      message_size = sizeof(header_t)+sizeof(stride_t);
    }
    message = new char[message_size];
    header = (header_t*)message;
    header->operation = OP_PUT_PACKED;
    header->remote_address = static_cast<char*>(dst);
    header->local_address = NULL;
    header->rank = proc;
    header->length = packed_index;
    (void)memcpy(message+sizeof(header_t), &stride, sizeof(stride_t));
    if (use_eager) {
      (void)memcpy(message+sizeof(header_t)+sizeof(stride_t),
          packed_buffer, packed_index);
      nb_send_header(message, message_size, master_rank, nb);
      delete [] packed_buffer;
    }
    else {
      /* we send the buffer backwards */
      char *buf = packed_buffer + packed_index;;
      size_t bytes_remaining = packed_index;
      nb_send_header(message, message_size, master_rank, nb);
      do {
        size_t size = bytes_remaining>max_message_size ?
          max_message_size : bytes_remaining;
        buf -= size;
        if (size == bytes_remaining) {
          /* on the last send, mark buffer for deletion */
          nb_send_header(buf, size, master_rank, nb);
        }
        else {
          nb_send_buffer(buf, size, master_rank, nb);
        }
        bytes_remaining -= size;
      } while (bytes_remaining > 0);
    }
  }
  nb->in_use = 1;
}


void p_Environment::nb_puts_datatype(
    void *src_ptr, int64_t *src_stride_ar,
    void *dst_ptr, int64_t *dst_stride_ar,
    int64_t *count, int stride_levels,
    int proc, _cmx_request *nb)
{
  MPI_Datatype src_type;
  int ierr;
  int64_t i;
  stride_t stride;

#if DEBUG
  fprintf(stderr, "[%d] nb_puts_datatype(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
      p_config.rank(), src_ptr, src_stride_ar, dst_ptr, dst_stride_ar,
      count[0], stride_levels, proc, nb);
#endif

  CMX_ASSERT(proc >= 0);
  CMX_ASSERT(proc < p_config.size());
  CMX_ASSERT(NULL != src_ptr);
  CMX_ASSERT(NULL != dst_ptr);
  CMX_ASSERT(NULL != count);
  CMX_ASSERT(NULL != nb);
  CMX_ASSERT(stride_levels >= 0);
  CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);

  /* copy dst info into structure */
  MAYBE_MEMSET(&stride, 0, sizeof(stride_t));
  stride.stride_levels = stride_levels;
  stride.count[0] = count[0];
  for (i=0; i<stride_levels; ++i) {
    stride.stride[i] = dst_stride_ar[i];
    stride.count[i+1] = count[i+1];
  }
  for (/*no init*/; i<CMX_MAX_STRIDE_LEVEL; ++i) {
    stride.stride[i] = -1;
    stride.count[i+1] = -1;
  }

  CMX_ASSERT(stride.stride_levels >= 0);
  CMX_ASSERT(stride.stride_levels < CMX_MAX_STRIDE_LEVEL);

#if DEBUG
  fprintf(stderr, "[%d] nb_puts_datatype stride_levels=%d, count[0]=%d\n",
      p_config.rank(), stride_levels, count[0]);
  for (i=0; i<stride_levels; ++i) {
    fprintf(stderr, "[%d] stride[%d]=%d count[%d+1]=%d\n",
        p_config.rank(), i, stride.stride[i], i, stride.count[i+1]);
  }
#endif

  strided_to_subarray_dtype(src_stride_ar, count, stride_levels,
      MPI_BYTE, &src_type);
  ierr = MPI_Type_commit(&src_type);
  _translate_mpi_error(ierr,"nb_puts_datatype:MPI_Type_commit");

  {
    char *message = NULL;
    size_t message_size = 0;
    header_t *header = NULL;
    int master_rank = -1;

    master_rank = p_config.master(proc);
    /* only fence on the master */
    fence_array[master_rank] = 1;
    message_size = sizeof(header_t) + sizeof(stride_t);
    message = new char[message_size];
    header = reinterpret_cast<header_t*>(message);
    MAYBE_MEMSET(header, 0, sizeof(header_t));
    header->operation = OP_PUT_DATATYPE;
    header->remote_address = static_cast<char*>(dst_ptr);
    header->local_address = NULL;
    header->rank = proc;
    header->length = 0;
    (void)memcpy(message+sizeof(header_t), &stride, sizeof(stride_t));
    nb_send_header(message, message_size, master_rank, nb);
    nb_send_datatype(src_ptr, src_type, master_rank, nb);
  }
  nb->in_use = 1;
}


void p_Environment::nb_gets(
    void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels, int proc, _cmx_request *nb)
{
  int i, j;
  int64_t src_idx, dst_idx;  /* index offset of current block position to ptr */
  int64_t n1dim;  /* number of 1 dim block */
  int64_t src_bvalue[7], src_bunit[7];
  int64_t dst_bvalue[7], dst_bunit[7];

  /* if not actually a strided get */
  if (0 == stride_levels) {
    nb_get(src, dst, count[0], proc, nb);
    return;
  }

  /* if not a strided get from self or SMP, use datatype algorithm */
  if (CMX_ENABLE_GET_DATATYPE
      && (!CMX_ENABLE_GET_SELF || p_config.rank() != proc)
      && (!CMX_ENABLE_GET_SMP
        || p_config.hostid(proc) != p_config.hostid(p_config.rank()))
      && (_packed_size(src_stride, count, stride_levels) > CMX_GET_DATATYPE_THRESHOLD)) {
    nb_gets_datatype(src, src_stride, dst, dst_stride, count, stride_levels, proc, nb);
    return;
  }

  /* if not a strided get from self or SMP, use packed algorithm */
  if (CMX_ENABLE_GET_PACKED
      && (!CMX_ENABLE_GET_SELF || p_config.rank() != proc)
      && (!CMX_ENABLE_GET_SMP
        || p_config.hostid(proc) != p_config.hostid(p_config.rank()))) {
    nb_gets_packed(src, src_stride, dst, dst_stride, count, stride_levels, proc, nb);
    return;
  }

  /* number of n-element of the first dimension */
  n1dim = 1;
  for(i=1; i<=stride_levels; i++) {
    n1dim *= count[i];
  }

  /* calculate the destination indices */
  src_bvalue[0] = 0; src_bvalue[1] = 0; src_bunit[0] = 1; src_bunit[1] = 1;
  dst_bvalue[0] = 0; dst_bvalue[1] = 0; dst_bunit[0] = 1; dst_bunit[1] = 1;

  for(i=2; i<=stride_levels; i++) {
    src_bvalue[i] = 0;
    dst_bvalue[i] = 0;
    src_bunit[i] = src_bunit[i-1] * count[i-1];
    dst_bunit[i] = dst_bunit[i-1] * count[i-1];
  }

  for(i=0; i<n1dim; i++) {
    src_idx = 0;
    for(j=1; j<=stride_levels; j++) {
      src_idx += (int64_t) src_bvalue[j] * (int64_t) src_stride[j-1];
      if((i+1) % src_bunit[j] == 0) {
        src_bvalue[j]++;
      }
      if(src_bvalue[j] > (count[j]-1)) {
        src_bvalue[j] = 0;
      }
    }

    dst_idx = 0;

    for(j=1; j<=stride_levels; j++) {
      dst_idx += (int64_t) dst_bvalue[j] * (int64_t) dst_stride[j-1];
      if((i+1) % dst_bunit[j] == 0) {
        dst_bvalue[j]++;
      }
      if(dst_bvalue[j] > (count[j]-1)) {
        dst_bvalue[j] = 0;
      }
    }

    nb_get((char *)src + src_idx, (char *)dst + dst_idx,
        count[0], proc, nb);
  }
}


void p_Environment::nb_gets_packed(
    void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels, int proc, _cmx_request *nb)
{
  int64_t i;
  stride_t stride_src;
  stride_t *stride_dst = NULL;

#if DEBUG
  fprintf(stderr, "[%d] nb_gets_packed(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
      p_config.rank(), src, src_stride, dst, dst_stride,
      count[0], stride_levels, proc, nb);
#endif

  CMX_ASSERT(proc >= 0);
  CMX_ASSERT(proc < p_config.size());
  CMX_ASSERT(NULL != src);
  CMX_ASSERT(NULL != dst);
  CMX_ASSERT(NULL != count);
  CMX_ASSERT(NULL != nb);
  CMX_ASSERT(count[0] > 0);
  CMX_ASSERT(stride_levels >= 0);
  CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);

  /* copy src info into structure */
  stride_src.ptr = static_cast<char*>(src);
  stride_src.stride_levels = stride_levels;
  stride_src.count[0] = count[0];
  for (i=0; i<stride_levels; ++i) {
    stride_src.stride[i] = src_stride[i];
    stride_src.count[i+1] = count[i+1];
  }
  for (/*no init*/; i<CMX_MAX_STRIDE_LEVEL; ++i) {
    stride_src.stride[i] = -1;
    stride_src.count[i+1] = -1;
  }

  CMX_ASSERT(stride_src.stride_levels >= 0);
  CMX_ASSERT(stride_src.stride_levels < CMX_MAX_STRIDE_LEVEL);

  /* copy dst info into structure */
  stride_dst = new stride_t;
  CMX_ASSERT(stride_dst);
  stride_dst->ptr = static_cast<char*>(dst);
  stride_dst->stride_levels = stride_levels;
  stride_dst->count[0] = count[0];
  for (i=0; i<stride_levels; ++i) {
    stride_dst->stride[i] = dst_stride[i];
    stride_dst->count[i+1] = count[i+1];
  }
  for (/*no init*/; i<CMX_MAX_STRIDE_LEVEL; ++i) {
    stride_dst->stride[i] = -1;
    stride_dst->count[i+1] = -1;
  }

  CMX_ASSERT(stride_dst->stride_levels >= 0);
  CMX_ASSERT(stride_dst->stride_levels < CMX_MAX_STRIDE_LEVEL);

  {
    char *message = NULL;
    size_t message_size = 0;
    size_t recv_size = 0;
    char *packed_buffer = NULL;
    header_t *header = NULL;
    int master_rank = -1;

    master_rank = p_config.master(proc);

    message_size = sizeof(header_t) + sizeof(stride_t);
    message = new char[message_size];
    header = reinterpret_cast<header_t*>(message);
    CMX_ASSERT(header);
    MAYBE_MEMSET(header, 0, sizeof(header_t));
    header->operation = OP_GET_PACKED;
    header->remote_address = static_cast<char*>(src);
    header->local_address = static_cast<char*>(dst);
    header->rank = proc;
    header->length = 0;

    recv_size = _packed_size(stride_dst->stride,
        stride_dst->count, stride_dst->stride_levels);
    CMX_ASSERT(recv_size > 0);
    packed_buffer = (char*)malloc(recv_size);
    CMX_ASSERT(packed_buffer);
    {
      /* prepost all receives backward */
      char *buf = (char*)packed_buffer + recv_size;
      size_t bytes_remaining = recv_size;
      do {
        size_t size = bytes_remaining>max_message_size ?
          max_message_size : bytes_remaining;
        buf -= size;
        if (size == bytes_remaining) {
          /* on the last recv, indicate a packed recv */
          nb_recv_packed(buf, size, master_rank, nb, stride_dst);
        }
        else {
          nb_recv(buf, size, master_rank, nb);
        }
        bytes_remaining -= size;
      } while (bytes_remaining > 0);
    }
    (void)memcpy(message+sizeof(header_t), &stride_src, sizeof(stride_t));
    nb_send_header(message, message_size, master_rank, nb);
  }
  nb->in_use = 1;
}


void p_Environment::nb_gets_datatype(
    void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels, int proc, _cmx_request *nb)
{
  MPI_Datatype dst_type;
  int64_t i;
  stride_t stride_src;

#if DEBUG
  fprintf(stderr, "[%d] nb_gets_datatype(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
      p_config.rank(), src, src_stride, dst, dst_stride,
      count[0], stride_levels, proc, nb);
#endif
#if DEBUG
  for (i=0; i<stride_levels; ++i) {
    fprintf(stderr, "\tsrc_stride[%d]=%d\n", i, src_stride[i]);
  }
  for (i=0; i<stride_levels; ++i) {
    fprintf(stderr, "\tdst_stride[%d]=%d\n", i, dst_stride[i]);
  }
  for (i=0; i<stride_levels+1; ++i) {
    fprintf(stderr, "\tcount[%d]=%d\n", i, count[i]);
  }
#endif

  CMX_ASSERT(proc >= 0);
  CMX_ASSERT(proc < p_config.size());
  CMX_ASSERT(NULL != src);
  CMX_ASSERT(NULL != dst);
  CMX_ASSERT(NULL != count);
  CMX_ASSERT(NULL != nb);
  CMX_ASSERT(count[0] > 0);
  CMX_ASSERT(stride_levels >= 0);
  CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);

  /* copy src info into structure */
  MAYBE_MEMSET(&stride_src, 0, sizeof(header_t));
  stride_src.ptr = static_cast<char*>(src);
  stride_src.stride_levels = stride_levels;
  stride_src.count[0] = count[0];
  for (i=0; i<stride_levels; ++i) {
    stride_src.stride[i] = src_stride[i];
    stride_src.count[i+1] = count[i+1];
  }
  for (/*no init*/; i<CMX_MAX_STRIDE_LEVEL; ++i) {
    stride_src.stride[i] = -1;
    stride_src.count[i+1] = -1;
  }

  CMX_ASSERT(stride_src.stride_levels >= 0);
  CMX_ASSERT(stride_src.stride_levels < CMX_MAX_STRIDE_LEVEL);

  {
    char *message = NULL;
    int64_t message_size = 0;
    header_t *header = NULL;
    int master_rank = -1;
    int ierr;

    master_rank = p_config.master(proc);

    message_size = sizeof(header_t) + sizeof(stride_t);
    message = new char[message_size];
    header = reinterpret_cast<header_t*>(message);
    CMX_ASSERT(header);
    MAYBE_MEMSET(header, 0, sizeof(header_t));
    header->operation = OP_GET_DATATYPE;
    header->remote_address = static_cast<char*>(src);
    header->local_address = static_cast<char*>(dst);
    header->rank = proc;
    header->length = 0;

    strided_to_subarray_dtype(dst_stride, count, stride_levels, MPI_BYTE, &dst_type);
    ierr = MPI_Type_commit(&dst_type);
    _translate_mpi_error(ierr,"nb_gets_datatype:MPI_Type_commit");

    nb_recv_datatype(dst, dst_type, master_rank, nb);
    (void)memcpy(message+sizeof(header_t), &stride_src, sizeof(stride_t));
    nb_send_header(message, message_size, master_rank, nb);
  }
  nb->in_use = 1;
}


void p_Environment::nb_accs(
    int datatype, void *scale,
    void *src, int64_t *src_stride,
    void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels,
    int proc, _cmx_request *nb)
{
  int64_t i, j;
  int64_t src_idx, dst_idx;  /* index offset of current block position to ptr */
  int64_t n1dim;  /* number of 1 dim block */
  int64_t src_bvalue[7], src_bunit[7];
  int64_t dst_bvalue[7], dst_bunit[7];

  /* if not actually a strided acc */
  if (0 == stride_levels) {
    nb_acc(datatype, scale, src, dst, count[0], proc, nb);
    return;
  }

  /* if not a strided acc to self or SMP, use packed algorithm */
  if (CMX_ENABLE_ACC_PACKED
      && (!CMX_ENABLE_ACC_SELF || p_config.rank() != proc)
      && (!CMX_ENABLE_ACC_SMP
        || p_config.hostid(proc) != p_config.hostid(p_config.rank()))) {
    nb_accs_packed(datatype, scale, src, src_stride, dst, dst_stride, count, stride_levels, proc, nb);
    return;
  }

  /* number of n-element of the first dimension */
  n1dim = 1;
  for(i=1; i<=stride_levels; i++) {
    n1dim *= count[i];
  }

  /* calculate the destination indices */
  src_bvalue[0] = 0; src_bvalue[1] = 0; src_bunit[0] = 1; src_bunit[1] = 1;
  dst_bvalue[0] = 0; dst_bvalue[1] = 0; dst_bunit[0] = 1; dst_bunit[1] = 1;

  for(i=2; i<=stride_levels; i++) {
    src_bvalue[i] = 0;
    dst_bvalue[i] = 0;
    src_bunit[i] = src_bunit[i-1] * count[i-1];
    dst_bunit[i] = dst_bunit[i-1] * count[i-1];
  }

  /* index mangling */
  for(i=0; i<n1dim; i++) {
    src_idx = 0;
    dst_idx = 0;
    for(j=1; j<=stride_levels; j++) {
      src_idx += (int64_t) src_bvalue[j] * (int64_t) src_stride[j-1];
      if((i+1) % src_bunit[j] == 0) {
        src_bvalue[j]++;
      }
      if(src_bvalue[j] > (count[j]-1)) {
        src_bvalue[j] = 0;
      }
    }

    for(j=1; j<=stride_levels; j++) {
      dst_idx += (int64_t) dst_bvalue[j] * (int64_t) dst_stride[j-1];
      if((i+1) % dst_bunit[j] == 0) {
        dst_bvalue[j]++;
      }
      if(dst_bvalue[j] > (count[j]-1)) {
        dst_bvalue[j] = 0;
      }
    }

    nb_acc(datatype, scale, (char *)src + src_idx, (char *)dst + dst_idx,
        count[0], proc, nb);
  }
}


void p_Environment::nb_accs_packed(
    int datatype, void *scale,
    void *src, int64_t *src_stride,
    void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels,
    int proc, _cmx_request *nb)
{
  int64_t i;
  int64_t packed_index = 0;
  char *packed_buffer = NULL;
  stride_t stride;

#if DEBUG
  fprintf(stderr, "[%d] nb_accs_packed(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
      p_config.rank(), src, src_stride, dst, dst_stride,
      count[0], stride_levels, proc, nb);
#endif

  CMX_ASSERT(proc >= 0);
  CMX_ASSERT(proc < p_config.size());
  CMX_ASSERT(NULL != scale);
  CMX_ASSERT(NULL != src);
  CMX_ASSERT(NULL != dst);
  CMX_ASSERT(NULL != count);
  CMX_ASSERT(NULL != nb);
  CMX_ASSERT(count[0] > 0);
  CMX_ASSERT(stride_levels >= 0);
  CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);

  /* copy dst info into structure */
  stride.ptr = static_cast<char*>(dst);
  stride.stride_levels = stride_levels;
  stride.count[0] = count[0];
  for (i=0; i<stride_levels; ++i) {
    stride.stride[i] = dst_stride[i];
    stride.count[i+1] = count[i+1];
  }
  /* assign remaining values to invalid */
  for (/*no init*/; i<CMX_MAX_STRIDE_LEVEL; ++i) {
    stride.stride[i] = -1;
    stride.count[i+1] = -1;
  }

  CMX_ASSERT(stride.stride_levels >= 0);
  CMX_ASSERT(stride.stride_levels < CMX_MAX_STRIDE_LEVEL);

#if DEBUG
  fprintf(stderr, "[%d] nb_accs_packed stride_levels=%d, count[0]=%d\n",
      p_config.rank(), stride_levels, count[0]);
  for (i=0; i<stride_levels; ++i) {
    printf("[%d] stride[%d]=%d count[%d+1]=%d\n",
        p_config.rank(), i, stride.stride[i], i, stride.count[i+1]);
  }
#endif

  packed_buffer = pack((char*)src, src_stride, count, stride_levels, &packed_index);

  CMX_ASSERT(NULL != packed_buffer);
  CMX_ASSERT(packed_index > 0);

  {
    header_t *header = NULL;
    char *message = NULL;
    size_t message_size = 0;
    int scale_size = 0;
    op_t operation = OP_NULL;
    int master_rank = -1;
    int use_eager = 0;

    switch (datatype) {
      case CMX_ACC_INT:
        operation = OP_ACC_INT_PACKED;
        scale_size = sizeof(int);
        break;
      case CMX_ACC_DBL:
        operation = OP_ACC_DBL_PACKED;
        scale_size = sizeof(double);
        break;
      case CMX_ACC_FLT:
        operation = OP_ACC_FLT_PACKED;
        scale_size = sizeof(float);
        break;
      case CMX_ACC_CPL:
        operation = OP_ACC_CPL_PACKED;
        scale_size = sizeof(SingleComplex);
        break;
      case CMX_ACC_DCP:
        operation = OP_ACC_DCP_PACKED;
        scale_size = sizeof(DoubleComplex);
        break;
      case CMX_ACC_LNG:
        operation = OP_ACC_LNG_PACKED;
        scale_size = sizeof(long);
        break;
      default: CMX_ASSERT(0);
    }
    use_eager = _eager_check(scale_size+sizeof(stride_t)+packed_index);

    master_rank = p_config.master(proc);

    /* only fence on the master */
    fence_array[master_rank] = 1;

    if (use_eager) {
      message_size = sizeof(header_t) + scale_size + sizeof(stride_t) + packed_index;
    }
    else {
      message_size = sizeof(header_t) + scale_size + sizeof(stride_t);
    }
    message = (char*)malloc(message_size);
    CMX_ASSERT(message);
    header = (header_t*)message;
    header->operation = operation;
    header->remote_address = static_cast<char*>(dst);
    header->local_address = NULL;
    header->rank = proc;
    header->length = packed_index;
    (void)memcpy(message+sizeof(header_t), scale, scale_size);
    (void)memcpy(message+sizeof(header_t)+scale_size, &stride, sizeof(stride_t));
    if (use_eager) {
      (void)memcpy(message+sizeof(header_t)+scale_size+sizeof(stride_t),
          packed_buffer, packed_index);
      nb_send_header(message, message_size, master_rank, nb);
      free(packed_buffer);
    }
    else {
      /* we send the buffer backwards */
      char *buf = packed_buffer + packed_index;
      size_t bytes_remaining = packed_index;
      nb_send_header(message, message_size, master_rank, nb);
      do {
        size_t size = bytes_remaining>max_message_size ?
          max_message_size : bytes_remaining;
        buf -= size;
        if (size == bytes_remaining) {
          nb_send_header(buf, size, master_rank, nb);
        }
        else {
          nb_send_buffer(buf, size, master_rank, nb);
        }
        bytes_remaining -= size;
      } while (bytes_remaining > 0);
    }
  }
  nb->in_use = 1;
}


void p_Environment::nb_putv(
    _cmx_giov_t *iov, int64_t iov_len,
    int proc, _cmx_request *nb)
{
  int i = 0;

  for (i=0; i<iov_len; ++i) {
    /* if not a vector put to self, use packed algorithm */
    if (CMX_ENABLE_PUT_IOV
        && (!CMX_ENABLE_PUT_SELF || p_config.rank() != proc)
        && (!CMX_ENABLE_PUT_SMP
          || p_config.hostid(proc) != p_config.hostid(p_config.rank()))) {
      nb_putv_packed(&iov[i], proc, nb);
    }
    else {
      int64_t j;
      void **src = iov[i].src;
      void **dst = iov[i].dst;
      int64_t bytes = iov[i].bytes;
      int64_t limit = iov[i].count;
      for (j=0; j<limit; ++j) {
        nb_put(src[j], dst[j], bytes, proc, nb);
      }
    }
  }
}


void p_Environment::nb_putv_packed(_cmx_giov_t *iov, int proc, _cmx_request *nb)
{
  int64_t i = 0;
  void **src = NULL;
  void **dst = NULL;
  int64_t bytes = 0;
  int64_t limit = 0;
  char *iov_buf = NULL;
  int64_t iov_off = 0;
  int64_t iov_size = 0;
  char *packed_buffer = NULL;
  int64_t packed_size = 0;
  int64_t packed_index = 0;

  src = iov->src;
  dst = iov->dst;
  bytes = iov->bytes;
  limit = iov->count;

#if DEBUG
  fprintf(stderr, "[%d] nb_putv_packed limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
      p_config.rank(), limit, bytes, src[0], dst[0]);
#endif

  /* allocate compressed iov */
  iov_size = 2*limit*sizeof(void*) + 2*sizeof(int64_t);
  iov_buf =  new char[iov_size];
  CMX_ASSERT(iov_buf);
  iov_off = 0;
  /* copy limit */
  (void)memcpy(&iov_buf[iov_off], &limit, sizeof(int64_t));
  iov_off += sizeof(int64_t);
  /* copy bytes */
  (void)memcpy(&iov_buf[iov_off], &bytes, sizeof(int64_t));
  iov_off += sizeof(int64_t);
  /* copy src pointers */
  (void)memcpy(&iov_buf[iov_off], src, limit*sizeof(void*));
  iov_off += limit*sizeof(void*);
  /* copy dst pointers */
  (void)memcpy(&iov_buf[iov_off], dst, limit*sizeof(void*));
  iov_off += limit*sizeof(void*);
  CMX_ASSERT(iov_off == iov_size);

  /* allocate send buffer */
  packed_size = bytes * limit;
  packed_buffer = new char[packed_size];
  CMX_ASSERT(packed_buffer);
  packed_index = 0;
  for (i=0; i<limit; ++i) {
    (void)memcpy(&packed_buffer[packed_index], src[i], bytes);
    packed_index += bytes;
  }
  CMX_ASSERT(packed_index == bytes*limit);

  {
    header_t *header = NULL;
    char *message;
    int master_rank = p_config.master(proc);

    /* only fence on the master */
    fence_array[master_rank] = 1;

    message = new char[sizeof(header_t)];
    header = reinterpret_cast<header_t*>(message);
    CMX_ASSERT(header);
    MAYBE_MEMSET(header, 0, sizeof(header_t));
    header->operation = OP_PUT_IOV;
    header->remote_address = NULL;
    header->local_address = NULL;
    header->rank = proc;
    header->length = iov_size;
    nb_send_header(message, sizeof(header_t), master_rank, nb);
    nb_send_header(iov_buf, iov_size, master_rank, nb);
    nb_send_header(packed_buffer, packed_size, master_rank, nb);
  }
}


void p_Environment::nb_getv(
    _cmx_giov_t *iov, int64_t iov_len,
    int proc, _cmx_request *nb)
{
  int64_t i = 0;

  for (i=0; i<iov_len; ++i) {
    /* if not a vector get from self, use packed algorithm */
    if (CMX_ENABLE_GET_IOV
        && (!CMX_ENABLE_GET_SELF || p_config.rank() != proc)
        && (!CMX_ENABLE_GET_SMP
          || p_config.hostid(proc) != p_config.hostid(p_config.rank()))) {
      nb_getv_packed(&iov[i], proc, nb);
    }
    else {
      int64_t j;
      void **src = iov[i].src;
      void **dst = iov[i].dst;
      int64_t bytes = iov[i].bytes;
      int64_t limit = iov[i].count;
      for (j=0; j<limit; ++j) {
        nb_get(src[j], dst[j], bytes, proc, nb);
      }
    }
  }
}


void p_Environment::nb_getv_packed(_cmx_giov_t *iov, int proc, _cmx_request *nb)
{
  void **src = NULL;
  void **dst = NULL;
  int64_t bytes = 0;
  int64_t limit = 0;
  char *iov_buf = NULL;
  int64_t iov_off = 0;
  int64_t iov_size = 0;
  _cmx_giov_t *iov_copy = NULL;
  char *packed_buffer = NULL;
  int64_t packed_size = 0;

  src = iov->src;
  dst = iov->dst;
  bytes = iov->bytes;
  limit = iov->count;

#if DEBUG
  fprintf(stderr, "[%d] nb_getv_packed limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
      p_config.rank(), limit, bytes, src[0], dst[0]);
#endif

  /* allocate compressed iov */
  iov_size = 2*limit*sizeof(void*) + 2*sizeof(int64_t);
  iov_buf = (char*)malloc(iov_size);
  iov_off = 0;
  CMX_ASSERT(iov_buf);
  /* copy limit */
  (void)memcpy(&iov_buf[iov_off], &limit, sizeof(int64_t));
  iov_off += sizeof(int64_t);
  /* copy bytes */
  (void)memcpy(&iov_buf[iov_off], &bytes, sizeof(int64_t));
  iov_off += sizeof(int64_t);
  /* copy src pointers */
  (void)memcpy(&iov_buf[iov_off], src, limit*sizeof(void*));
  iov_off += limit*sizeof(void*);
  /* copy dst pointers */
  (void)memcpy(&iov_buf[iov_off], dst, limit*sizeof(void*));
  iov_off += limit*sizeof(void*);
  CMX_ASSERT(iov_off == iov_size);

  /* copy given iov for later */
  iov_copy = (_cmx_giov_t*)malloc(sizeof(_cmx_giov_t));
  iov_copy->bytes = bytes;
  iov_copy->count = limit;
  iov_copy->src = (void**)malloc(sizeof(void*)*iov->count);
  CMX_ASSERT(iov_copy->src);
  (void)memcpy(iov_copy->src, iov->src, sizeof(void*)*iov->count);
  iov_copy->dst = (void**)malloc(sizeof(void*)*iov->count);
  CMX_ASSERT(iov_copy->dst);
  (void)memcpy(iov_copy->dst, iov->dst, sizeof(void*)*iov->count);

#if DEBUG
  fprintf(stderr, "[%d] nb_getv_packed limit=%d bytes=%d src[0]=%p dst[0]=%p copy\n",
      p_config.rank(), iov_copy->count, iov_copy->bytes,
      iov_copy->src[0], iov_copy->dst[0]);
#endif

  /* allocate recv buffer */
  packed_size = bytes * limit;
  packed_buffer = (char*)malloc(packed_size);
  CMX_ASSERT(packed_buffer);

  {
    header_t *header = NULL;
    int master_rank = p_config.master(proc);

    header = (header_t*)malloc(sizeof(header_t));
    CMX_ASSERT(header);
    MAYBE_MEMSET(header, 0, sizeof(header_t));
    header->operation = OP_GET_IOV;
    header->remote_address = NULL;
    header->local_address = NULL;
    header->rank = proc;
    header->length = iov_size;
    nb_recv_iov(packed_buffer, packed_size, master_rank, nb, iov_copy);
    nb_send_header(header, sizeof(header_t), master_rank, nb);
    nb_send_header(iov_buf, iov_size, master_rank, nb);
  }
}

void p_Environment::nb_accv(
    int datatype, void *scale,
    _cmx_giov_t *iov, int64_t iov_len,
    int proc, _cmx_request *nb)
{
  int64_t i = 0;

  for (i=0; i<iov_len; ++i) {
    /* if not a vector acc to self, use packed algorithm */
    if (CMX_ENABLE_ACC_IOV
        && (!CMX_ENABLE_ACC_SELF || p_config.rank() != proc)
        && (!CMX_ENABLE_ACC_SMP
          || p_config.hostid(proc) != p_config.hostid(p_config.rank()))) {
      nb_accv_packed(datatype, scale, &iov[i], proc, nb);
    }
    else {
      int64_t j;
      void **src = iov[i].src;
      void **dst = iov[i].dst;
      int64_t bytes = iov[i].bytes;
      int64_t limit = iov[i].count;
      for (j=0; j<limit; ++j) {
        nb_acc(datatype, scale, src[j], dst[j], bytes, proc, nb);
      }
    }
  }
}

void p_Environment::nb_accv_packed(
    int datatype, void *scale,
    _cmx_giov_t *iov,
    int proc, _cmx_request *nb)
{
  int64_t i = 0;
  void **src = NULL;
  void **dst = NULL;
  int64_t bytes = 0;
  int64_t limit = 0;
  char *iov_buf = NULL;
  int64_t iov_off = 0;
  int64_t iov_size = 0;
  char *packed_buffer = NULL;
  int64_t packed_size = 0;
  int64_t packed_index = 0;

  src = iov->src;
  dst = iov->dst;
  bytes = iov->bytes;
  limit = iov->count;

#if DEBUG
  fprintf(stderr, "[%d] nb_accv_packed limit=%d bytes=%d loc[0]=%p rem_offset[0]=%d\n",
      p_config.rank(), limit, bytes, loc[0], rem[0]);
#endif

  /* allocate compressed iov */
  iov_size = 2*limit*sizeof(void*) + 2*sizeof(int64_t);
  iov_buf = (char*)malloc(iov_size);
  iov_off = 0;
  CMX_ASSERT(iov_buf);
  /* copy limit */
  (void)memcpy(&iov_buf[iov_off], &limit, sizeof(int64_t));
  iov_off += sizeof(int64_t);
  /* copy bytes */
  (void)memcpy(&iov_buf[iov_off], &bytes, sizeof(int64_t));
  iov_off += sizeof(int64_t);
  /* copy src pointers */
  (void)memcpy(&iov_buf[iov_off], src, limit*sizeof(void*));
  iov_off += limit*sizeof(void*);
  /* copy dst pointers */
  (void)memcpy(&iov_buf[iov_off], dst, limit*sizeof(void*));
  iov_off += limit*sizeof(void*);
  CMX_ASSERT(iov_off == iov_size);


  /* allocate send buffer */
  packed_size = bytes * limit;
  packed_buffer = (char*)malloc(packed_size);
  CMX_ASSERT(packed_buffer);
  packed_index = 0;
  for (i=0; i<limit; ++i) {
    (void)memcpy(&packed_buffer[packed_index], src[i], bytes);
    packed_index += bytes;
  }
  CMX_ASSERT(packed_index == bytes*limit);


  {
    header_t *header = NULL;
    char *message = NULL;
    int64_t message_size = 0;
    int scale_size = 0;
    op_t operation = OP_NULL;
    int master_rank = p_config.master(proc);

    switch (datatype) {
      case CMX_ACC_INT:
        operation = OP_ACC_INT_IOV;
        scale_size = sizeof(int);
        break;
      case CMX_ACC_DBL:
        operation = OP_ACC_DBL_IOV;
        scale_size = sizeof(double);
        break;
      case CMX_ACC_FLT:
        operation = OP_ACC_FLT_IOV;
        scale_size = sizeof(float);
        break;
      case CMX_ACC_CPL:
        operation = OP_ACC_CPL_IOV;
        scale_size = sizeof(SingleComplex);
        break;
      case CMX_ACC_DCP:
        operation = OP_ACC_DCP_IOV;
        scale_size = sizeof(DoubleComplex);
        break;
      case CMX_ACC_LNG:
        operation = OP_ACC_LNG_IOV;
        scale_size = sizeof(long);
        break;
      default: CMX_ASSERT(0);
    }

    /* only fence on the master */
    fence_array[master_rank] = 1;

    message_size = sizeof(header_t) + scale_size;
    message = (char*)malloc(message_size);
    CMX_ASSERT(message);
    header = (header_t*)message;
    header->operation = operation;
    header->remote_address = NULL;
    header->local_address = NULL;
    header->rank = proc;
    header->length = iov_size;
    (void)memcpy(message+sizeof(header_t), scale, scale_size);
    nb_send_header(message, message_size, master_rank, nb);
    nb_send_header(iov_buf, iov_size, master_rank, nb);
    nb_send_header(packed_buffer, packed_size, master_rank, nb);
  }
}

/**
 * Utility function to catch and translate MPI errors. Returns silently if
 * no error detected.
 * @param ierr: Error code from MPI call
 * @param location: User specified string to indicate location of error
 */
void p_Environment::_translate_mpi_error(int ierr, const char* location)
{
  if (ierr == MPI_SUCCESS) return;
  char err_string[MPI_MAX_ERROR_STRING];
  int len;
  fprintf(stderr,"p[%d] Error in %s\n",p_config.rank(),location);
  MPI_Error_string(ierr,err_string,&len);
  fprintf(stderr,"p[%d] MPI_Error: %s\n",p_config.rank(),err_string);
}

/**
 * No checking for data consistency. Assume correctness has already been
 * established elsewhere. Individual elements are assumed to be one byte in size
 * stride_array: physical dimensions of array
 * count: number of elements along each array dimension
 * levels: number of stride levels (should be one less than array dimension)
 * type: MPI_Datatype returned to calling program
 */
void p_Environment::strided_to_subarray_dtype(int64_t *stride_array, int64_t *count,
    int levels, MPI_Datatype base_type, MPI_Datatype *type)
{
  int ndims = levels+1;
  int i = 0;
  int ierr = 0;
  int array_of_sizes[7];
  int array_of_starts[7];
  int array_of_subsizes[7];
  int stride = 0;

  ierr = MPI_Type_size(base_type,&stride);
  _translate_mpi_error(ierr,"strided_to_subarray_dtype:MPI_Type_size");

  /* the pointer to the local buffer points to the first data element
   * in data exchange, not the origin of the local array, so all starts
   * should be zero */
  for (i=0; i<levels; i++) {
    array_of_sizes[i] = static_cast<int>(stride_array[i]/stride);
    array_of_starts[i] = static_cast<int>(0);
    array_of_subsizes[i] = static_cast<int>(count[i]);
    if (array_of_sizes[i] < array_of_subsizes[i]) {
      fprintf(stderr, "p[%d] ERROR [strided_to_subarray_dtype]\n"
          "stride: %d\n"
          "stride_array[%d]: %ld\n"
          "array_of_sizes[%d]: %d\n"
          "array_of_subsizes[%d]: %d\n",
          p_config.rank(),
          stride,
          i,stride_array[i],
          i,array_of_sizes[i],
          i,array_of_subsizes[i]);
    }
    stride = stride_array[i];
  }
  array_of_sizes[levels] = static_cast<int>(count[levels]);
  array_of_starts[levels] = static_cast<int>(0);
  array_of_subsizes[levels] = static_cast<int>(count[levels]);
#if DEBUG
  for (i=0; i<ndims; i++) {
    fprintf(stderr, "p[%d] ndims: %d sizes[%d]: %d subsizes[%d]: %d starts[%d]: %d\n",
        p_config.rank(),
        ndims,
        i,array_of_sizes[i],
        i,array_of_subsizes[i],
        i,array_of_starts[i]);
  }
#endif

  ierr = MPI_Type_create_subarray(ndims, array_of_sizes,
      array_of_subsizes, array_of_starts, MPI_ORDER_FORTRAN,
      base_type, type);
  _translate_mpi_error(ierr,"strided_to_subarray_dtype:MPI_Type_create_subarray");
  if (MPI_SUCCESS != ierr) {
    fprintf(stderr, "p[%d] Error forming MPI_Datatype for one-sided strided operation."
        " Check that stride dimensions are compatible with local block"
        " dimensions\n",p_config.rank());
    for (i=0; i<levels; i++) {
      fprintf(stderr, "p[%d] count[%d]: %d stride[%d]: %d\n",
          p_config.rank(),
          i,count[i],
          i,stride_array[i]);
    }
    fprintf(stderr, "p[%d] count[%d]: %d\n",p_config.rank(),i,count[i]);
    _translate_mpi_error(ierr,"strided_to_subarray_dtype:MPI_Type_create_subarray");
  }
}

#if 0
void p_Environment::_group_init(void)
{
  int status = 0;
  int i = 0;
  int smallest_rank_with_same_hostid = 0;
  int largest_rank_with_same_hostid = 0;
  int size_node = 0;
  Group *group = NULL;
  long *sorted = NULL;
  int count = 0;
  MPI_Group igroup;

  /* need to figure out which proc is master on each node */
  g_state.hostid = (long*)malloc(sizeof(long)*g_state.size);
  g_state.hostid[p_config.rank()] = xgethostid();
  status = MPI_Allgather(MPI_IN_PLACE, 1, MPI_LONG,
      g_state.hostid, 1, MPI_LONG, g_state.comm);
  CMX_ASSERT(MPI_SUCCESS == status);
  /* First create a temporary node communicator and then
   * split further into number of gruoups within the node */
  MPI_Comm temp_node_comm;
  int temp_node_size;
  /* create node comm */
  /* MPI_Comm_split requires a non-negative color,
   * so sort and sanitize */
  sorted = (long*)malloc(sizeof(long) * g_state.size);
  (void)memcpy(sorted, g_state.hostid, sizeof(long)*g_state.size);
  qsort(sorted, g_state.size, sizeof(long), cmplong);
  for (i=0; i<g_state.size-1; ++i) {
    if (sorted[i] == g_state.hostid[p_config.rank()]) 
    {
      break;
    }
    if (sorted[i] != sorted[i+1]) {
      count += 1;
    }
  }
  free(sorted);
  status = MPI_Comm_split(MPI_COMM_WORLD, count,
      p_config.rank(), &temp_node_comm);
  int node_group_size, node_group_rank;
  MPI_Comm_size(temp_node_comm, &node_group_size);
  MPI_Comm_rank(temp_node_comm, &node_group_rank);
  int node_rank0, num_nodes;
  node_rank0 = (node_group_rank == 0) ? 1 : 0;
  MPI_Allreduce(&node_rank0, &num_nodes, 1, MPI_INT, MPI_SUM,
      g_state.comm);
  smallest_rank_with_same_hostid = p_config.rank();
  largest_rank_with_same_hostid = p_config.rank();
  for (i=0; i<g_state.size; ++i) {
    if (g_state.hostid[i] == g_state.hostid[p_config.rank()]) {
      ++size_node;
      if (i < smallest_rank_with_same_hostid) {
        smallest_rank_with_same_hostid = i;
      }
      if (i > largest_rank_with_same_hostid) {
        largest_rank_with_same_hostid = i;
      }
    }
  }
  /* Get number of Progress-Ranks per node from environment variable
   * equal to 1 by default */
  int num_progress_ranks_per_node = get_num_progress_ranks_per_node();
  /* Perform check on the number of Progress-Ranks */
  if (size_node < 2 * num_progress_ranks_per_node) {  
    p_error("ranks per node, must be at least", 
        2 * num_progress_ranks_per_node);
  }
  if (size_node % num_progress_ranks_per_node > 0) {  
    p_error("number of ranks per node must be multiple of number of process groups per node", -1);
  }
  int is_node_ranks_packed = get_progress_rank_distribution_on_node();
  int split_group_size;
  split_group_size = node_group_size / num_progress_ranks_per_node;
  MPI_Comm_free(&temp_node_comm);
  g_state.master = (int*)malloc(sizeof(int)*g_state.size);
  g_state.master[p_config.rank()] = get_my_master_rank_with_same_hostid(p_config.rank(), 
      split_group_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
      num_progress_ranks_per_node, is_node_ranks_packed);
  status = MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT,
      g_state.master, 1, MPI_INT, g_state.comm);
  CMX_ASSERT(MPI_SUCCESS == status);

  CMX_ASSERT(group_list == NULL);

  // put split group stamps
  int proc_split_group_stamp;
  int num_split_groups;
  num_split_groups = num_nodes * num_progress_ranks_per_node;
  int* split_group_list = (int*)malloc(sizeof(int)*num_split_groups);
  int split_group_index = 0;
  int j;
  for (i=0; i<g_state.size; i++) {
    for (j=0; j<i; j++) {
      if (g_state.master[i] == g_state.master[j])
        break; 
    }
    if(i == j) {
      split_group_list[split_group_index] = g_state.master[i];
      split_group_index++;
    }
  }
  // label each process
  for (j=0; j<num_split_groups; j++) {
    if (split_group_list[j] == g_state.master[p_config.rank()]) {
      proc_split_group_stamp = j;
    }
  }
  free(split_group_list);
  /* create a comm of only the workers */
  if (p_config.is_master()) {
    /* I'm a master */
    MPI_Comm delete_me;
    status = MPI_Comm_split(g_state.comm, 0, p_config.rank(), &delete_me);
    CMX_ASSERT(MPI_SUCCESS == status);
    /* masters don't need their own comm */
    if (MPI_COMM_NULL != delete_me) {
      MPI_Comm_free(&delete_me);
    }
    p_CMX_GROUP_WORLD = NULL;
  } else {
    /* I'm a worker */
    MPI_Comm comm;
    status = MPI_Comm_split(
        p_config.global_comm(), 1, p_config.rank(), &comm);
    CMX_ASSERT(MPI_SUCCESS == status);
    MPI_Group mpigroup;
    int size;
    status = MPI_Comm_size(comm, &size);
    std::vector<int> ranks(size);
    for (i=0; i<size; i++) ranks[i] = i;
    printf("p[%d]  (group_init) size of worker comm: %d\n",
        p_config.rank(),size);
    p_CMX_GROUP_WORLD = new Group(size, &ranks[0], comm);
  }
  status = MPI_Comm_split(MPI_COMM_WORLD, proc_split_group_stamp,
      p_config.rank(), &(g_state.node_comm));
  CMX_ASSERT(MPI_SUCCESS == status);
  /* node rank */
  status = MPI_Comm_rank(g_state.node_comm, &(g_state.node_rank));
  CMX_ASSERT(MPI_SUCCESS == status);
  /* node size */
  status = MPI_Comm_size(g_state.node_comm, &(g_state.node_size));
  CMX_ASSERT(MPI_SUCCESS == status);
}

/**
 * Creates and associates a cmx group with a cmx igroup.
 *
 * This does *not* initialize the members of the cmx igroup.
 */
static void _create_igroup(cmx_igroup_t **igroup)
{
    cmx_igroup_t *new_group_list_item = NULL;
    cmx_igroup_t *last_group_list_item = NULL;

#if DEBUG
    printf("[%d] _create_group(...)\n", p_config.rank());
#endif

    /* create, init, and insert the new node for the linked list */
    new_group_list_item = malloc(sizeof(cmx_igroup_t));
    new_group_list_item->next = NULL;
    new_group_list_item->comm = MPI_COMM_NULL;
    new_group_list_item->group = MPI_GROUP_NULL;
    new_group_list_item->size = -1;
    new_group_list_item->rank = -1;
    new_group_list_item->world_ranks = NULL;

    /* find the last group in the group linked list and insert */
    if (group_list) {
        last_group_list_item = group_list;
        while (last_group_list_item->next != NULL) {
            last_group_list_item = last_group_list_item->next;
        }
        last_group_list_item->next = new_group_list_item;
    }
    else {
        group_list = new_group_list_item;
    }

    /* return the group id and cmx igroup */
    *igroup = new_group_list_item;
}


int cmx_group_rank(cmx_igroup_t *igroup, int *rank)
{
    *rank = igroup->rank;

#if DEBUG
    printf("[%d] cmx_group_rank(group=%d, *rank=%d)\n",
            p_config.rank(), group, *rank);
#endif

    return CMX_SUCCESS;
}


int cmx_group_size(cmx_igroup_t *igroup, int *size)
{
    *size = igroup->size;

#if DEBUG
    printf("[%d] cmx_group_size(group=%d, *size=%d)\n",
            p_config.rank(), group, *size);
#endif

    return CMX_SUCCESS;
}


int cmx_group_comm(cmx_igroup_t *igroup, MPI_Comm *comm)
{
    *comm = igroup->comm;

#if DEBUG
    printf("[%d] cmx_group_comm(group=%d, comm)\n",
            p_config.rank(), group);
#endif

    return CMX_SUCCESS;
}

int cmx_group_translate_ranks(int n, cmx_group_t group_from,
    int *ranks_from, cmx_group_t group_to, int *ranks_to)
{
  int i;
  if (group_from == group_to) {
    for (i=0; i<n; i++) {
      ranks_to[i] = ranks_from[i];
    }
  } else {
    int status;
    status = MPI_Group_translate_ranks(group_from->group, n, ranks_from,
        group_to->group, ranks_to);
    if (status != MPI_SUCCESS) {
      p_error("MPI_Group_translate_ranks: Failed ", status);
    }
  }
  return CMX_SUCCESS;
}

int cmx_group_translate_world(
        cmx_igroup_t *igroup, int group_rank, int *world_rank)
{
#if DEBUG
    printf("[%d] cmx_group_translate_world("
            "group=%d, group_rank=%d, world_rank)\n",
            p_config.rank(), group, group_rank);
#endif

    if (CMX_GROUP_WORLD == igroup) {
        *world_rank = group_rank;
    }
    else {
        int status;

        CMX_ASSERT(group_list); /* first group is world worker group */
        status = MPI_Group_translate_ranks(igroup->group, 1, &group_rank,
                group_list->group, world_rank);
    }

    return CMX_SUCCESS;
}


/**
 * Destroys the given cmx igroup.
 */
static void _igroup_free(cmx_igroup_t *igroup)
{
    int status;

#if DEBUG
    printf("[%d] _igroup_free\n",
            p_config.rank());
#endif

    CMX_ASSERT(igroup);

    if (igroup->group != MPI_GROUP_NULL) {
        status = MPI_Group_free(&igroup->group);
        if (status != MPI_SUCCESS) {
            p_error("MPI_Group_free: Failed ", status);
        }
    }
#if DEBUG
    printf("[%d] free'd group\n", p_config.rank());
#endif

    if (igroup->comm != MPI_COMM_NULL) {
        status = MPI_Comm_free(&igroup->comm);
        if (status != MPI_SUCCESS) {
            p_error("MPI_Comm_free: Failed ", status);
        }
    }
#if DEBUG
    printf("[%d] free'd comm\n", p_config.rank());
#endif
    if (igroup->world_ranks != NULL) {
      free(igroup->world_ranks);
    }

    free(igroup);
}


int cmx_group_free(cmx_igroup_t *igroup)
{
    cmx_igroup_t *previous_group_list_item = NULL;
    cmx_igroup_t *current_group_list_item = group_list;

#if DEBUG
    printf("[%d] cmx_group_free(id=%d)\n", p_config.rank(), id);
#endif

    /* find the group to free */
    while (current_group_list_item != NULL) {
      if (current_group_list_item == igroup) {
        break;
      }
      previous_group_list_item = current_group_list_item;
      current_group_list_item = current_group_list_item->next;
    }

    /* make sure we found a group */
    CMX_ASSERT(current_group_list_item != NULL);
    /* remove the group from the linked list */
    if (previous_group_list_item != NULL) {
        previous_group_list_item->next = current_group_list_item->next;
    }
    /* free the igroup */
    _igroup_free(current_group_list_item);

    return CMX_SUCCESS;
}

void _igroup_set_world_ranks(cmx_igroup_t *igroup)
{
  int i = 0;
  int my_world_rank = p_config.rank();
  igroup->world_ranks = (int*)malloc(sizeof(int)*igroup->size);
  int status;

  for (i=0; i<igroup->size; ++i) {
    igroup->world_ranks[i] = MPI_PROC_NULL;
  }

  status = MPI_Allgather(&my_world_rank,1,MPI_INT,igroup->world_ranks,
      1,MPI_INT,igroup->comm);
  CMX_ASSERT(MPI_SUCCESS == status);

  for (i=0; i<igroup->size; ++i) {
    CMX_ASSERT(MPI_PROC_NULL != igroup->world_ranks[i]);
  }
}

int cmx_group_create(
        int n, int *pid_list, cmx_igroup_t *igroup_parent, cmx_igroup_t **igroup_child)
{
    int status = 0;
    int grp_me = 0;
    MPI_Group      *group_child = NULL;
    MPI_Comm       *comm_child = NULL;
    MPI_Group      *group_parent = NULL;
    MPI_Comm       *comm_parent = NULL;

#if DEBUG
    printf("[%d] cmx_group_create("
            "n=%d, pid_list=%p, id_parent=%d, id_child)\n",
            p_config.rank(), n, pid_list, id_parent);
    {
        int p;
        printf("[%d] pid_list={%d", p_config.rank(), pid_list[0]);
        for (p=1; p<n; ++p) {
            printf(",%d", pid_list[p]);
        }
        printf("}\n");
    }
#endif

    /* create the node in the linked list of groups and */
    /* get the child's MPI_Group and MPI_Comm, to be populated shortly */
    _create_igroup(igroup_child);
    group_child = &((*igroup_child)->group);
    comm_child  = &((*igroup_child)->comm);

    /* get the parent's MPI_Group and MPI_Comm */
    group_parent = &(igroup_parent->group);
    comm_parent  = &(igroup_parent->comm);

    status = MPI_Group_incl(*group_parent, n, pid_list, group_child);
    CMX_ASSERT(MPI_SUCCESS == status);

#if DEBUG
    printf("[%d] cmx_group_create before crazy logic\n", p_config.rank());
#endif
    {
        MPI_Comm comm, comm1, comm2;
        int lvl=1, local_ldr_pos;
        status = MPI_Group_rank(*group_child, &grp_me);
        CMX_ASSERT(MPI_SUCCESS == status);
        if (grp_me == MPI_UNDEFINED) {
            /* FIXME: keeping the group around for now */
#if DEBUG
    printf("[%d] cmx_group_create aborting -- not in group\n", p_config.rank());
#endif
            return CMX_SUCCESS;
        }
        /* SK: sanity check for the following bitwise operations */
        CMX_ASSERT(grp_me>=0);
        /* FIXME: can be optimized away */
        status = MPI_Comm_dup(MPI_COMM_SELF, &comm);
        CMX_ASSERT(MPI_SUCCESS == status);
        local_ldr_pos = grp_me;
        while(n>lvl) {
            int tag=0;
            int remote_ldr_pos = local_ldr_pos^lvl;
            if (remote_ldr_pos < n) {
                int remote_leader = pid_list[remote_ldr_pos];
                MPI_Comm peer_comm = *comm_parent;
                int high = (local_ldr_pos<remote_ldr_pos)?0:1;
                status = MPI_Intercomm_create(
                        comm, 0, peer_comm, remote_leader, tag, &comm1);
                CMX_ASSERT(MPI_SUCCESS == status);
                status = MPI_Comm_free(&comm);
                CMX_ASSERT(MPI_SUCCESS == status);
                status = MPI_Intercomm_merge(comm1, high, &comm2);
                CMX_ASSERT(MPI_SUCCESS == status);
                status = MPI_Comm_free(&comm1);
                CMX_ASSERT(MPI_SUCCESS == status);
                comm = comm2;
            }
            local_ldr_pos &= ((~0)^lvl);
            lvl<<=1;
        }
        *comm_child = comm;
        /* cleanup temporary group (from MPI_Group_incl above) */
        status = MPI_Group_free(group_child);
        CMX_ASSERT(MPI_SUCCESS == status);
        /* get the actual group associated with comm */
        status = MPI_Comm_group(*comm_child, group_child);
        CMX_ASSERT(MPI_SUCCESS == status);
        /* rank and size of new comm */
        status = MPI_Comm_size((*igroup_child)->comm, &((*igroup_child)->size));
        CMX_ASSERT(MPI_SUCCESS == status);
        status = MPI_Comm_rank((*igroup_child)->comm, &((*igroup_child)->rank));
        CMX_ASSERT(MPI_SUCCESS == status);
    }
#if DEBUG
    printf("[%d] cmx_group_create after crazy logic\n", p_config.rank());
#endif
    _igroup_set_world_ranks(*igroup_child);

    return CMX_SUCCESS;
}
#endif


int p_Environment::cmplong(const void *p1, const void *p2)
{
    return (int)(*((long*)p1) - *((long*)p2));
}

#if 0
/**
 * Initialize group linked list. Prepopulate with world group.
 */
void cmx_group_init() 
{
    int status = 0;
    int i = 0;
    int smallest_rank_with_same_hostid = 0;
    int largest_rank_with_same_hostid = 0;
    int size_node = 0;
    cmx_igroup_t *igroup = NULL;
    long *sorted = NULL;
    int count = 0;
    
    /* populate g_state */

    /* dup MPI_COMM_WORLD and get group, rank, and size */
    status = MPI_Comm_dup(MPI_COMM_WORLD, &(g_state.comm));
    CMX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Comm_group(g_state.comm, &(g_state.group));
    CMX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Comm_rank(g_state.comm, &(p_config.rank()));
    CMX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Comm_size(g_state.comm, &(g_state.size));
    CMX_ASSERT(MPI_SUCCESS == status);

#if DEBUG_TO_FILE
    {
        char pathname[80];
        sprintf(pathname, "trace.%d.log", p_config.rank());
        cmx_trace_file = fopen(pathname, "w");
        CMX_ASSERT(NULL != cmx_trace_file);

        printf("[%d] cmx_group_init()\n", p_config.rank());
    }
#endif

    /* need to figure out which proc is master on each node */
    g_state.hostid = (long*)malloc(sizeof(long)*g_state.size);
    g_state.hostid[p_config.rank()] = xgethostid();
    status = MPI_Allgather(MPI_IN_PLACE, 1, MPI_LONG,
            g_state.hostid, 1, MPI_LONG, g_state.comm);
    CMX_ASSERT(MPI_SUCCESS == status);
     /* First create a temporary node communicator and then
      * split further into number of gruoups within the node */
     MPI_Comm temp_node_comm;
     int temp_node_size;
    /* create node comm */
    /* MPI_Comm_split requires a non-negative color,
     * so sort and sanitize */
    sorted = (long*)malloc(sizeof(long) * g_state.size);
    (void)memcpy(sorted, g_state.hostid, sizeof(long)*g_state.size);
    qsort(sorted, g_state.size, sizeof(long), cmplong);
    for (i=0; i<g_state.size-1; ++i) {
        if (sorted[i] == g_state.hostid[p_config.rank()]) 
        {
            break;
        }
        if (sorted[i] != sorted[i+1]) {
            count += 1;
        }
    }
    free(sorted);
#if DEBUG
    printf("count: %d\n", count);
#endif
    status = MPI_Comm_split(MPI_COMM_WORLD, count,
            p_config.rank(), &temp_node_comm);
    int node_group_size, node_group_rank;
    MPI_Comm_size(temp_node_comm, &node_group_size);
    MPI_Comm_rank(temp_node_comm, &node_group_rank);
    int node_rank0, num_nodes;
    node_rank0 = (node_group_rank == 0) ? 1 : 0;
    MPI_Allreduce(&node_rank0, &num_nodes, 1, MPI_INT, MPI_SUM,
        g_state.comm);
    smallest_rank_with_same_hostid = p_config.rank();
    largest_rank_with_same_hostid = p_config.rank();
    for (i=0; i<g_state.size; ++i) {
        if (g_state.hostid[i] == g_state.hostid[p_config.rank()]) {
            ++size_node;
            if (i < smallest_rank_with_same_hostid) {
                smallest_rank_with_same_hostid = i;
            }
            if (i > largest_rank_with_same_hostid) {
                largest_rank_with_same_hostid = i;
            }
        }
    }
    /* Get number of Progress-Ranks per node from environment variable
     * equal to 1 by default */
    int num_progress_ranks_per_node = get_num_progress_ranks_per_node();
    /* Perform check on the number of Progress-Ranks */
    if (size_node < 2 * num_progress_ranks_per_node) {  
        p_error("ranks per node, must be at least", 
            2 * num_progress_ranks_per_node);
    }
    if (size_node % num_progress_ranks_per_node > 0) {  
        p_error("number of ranks per node must be multiple of number of process groups per node", -1);
    }
    int is_node_ranks_packed = get_progress_rank_distribution_on_node();
    int split_group_size;
    split_group_size = node_group_size / num_progress_ranks_per_node;
     MPI_Comm_free(&temp_node_comm);
    g_state.master = (int*)malloc(sizeof(int)*g_state.size);
    g_state.master[p_config.rank()] = get_my_master_rank_with_same_hostid(p_config.rank(), 
        split_group_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
        num_progress_ranks_per_node, is_node_ranks_packed);
#if DEBUG
    printf("[%d] rank; split_group_size: %d\n", p_config.rank(), split_group_size);
    printf("[%d] rank; largest_rank_with_same_hostid[%d]; my master is:[%d]\n",
        p_config.rank(), largest_rank_with_same_hostid, g_state.master[p_config.rank()]);
#endif
    status = MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT,
            g_state.master, 1, MPI_INT, g_state.comm);
    CMX_ASSERT(MPI_SUCCESS == status);

    CMX_ASSERT(group_list == NULL);

    // put split group stamps
    int proc_split_group_stamp;
    int num_split_groups;
    num_split_groups = num_nodes * num_progress_ranks_per_node;
    int* split_group_list = (int*)malloc(sizeof(int)*num_split_groups);
    int split_group_index = 0;
    int j;
    for (i=0; i<g_state.size; i++) {
      for (j=0; j<i; j++) {
        if (g_state.master[i] == g_state.master[j])
            break; 
      }
      if(i == j) {
        split_group_list[split_group_index] = g_state.master[i];
        split_group_index++;
      }
    }
    // label each process
    for (j=0; j<num_split_groups; j++) {
      if (split_group_list[j] == g_state.master[p_config.rank()]) {
        proc_split_group_stamp = j;
      }
    }
#if DEBUG
    printf("proc_split_group_stamp[%ld]: %ld\n", 
       p_config.rank(), proc_split_group_stamp);
#endif
    free(split_group_list);
    /* create a comm of only the workers */
    if (g_state.master[p_config.rank()] == p_config.rank()) {
        /* I'm a master */
        MPI_Comm delete_me;
        status = MPI_Comm_split(g_state.comm, 0, p_config.rank(), &delete_me);
        CMX_ASSERT(MPI_SUCCESS == status);
        /* masters don't need their own comm */
        if (MPI_COMM_NULL != delete_me) {
            MPI_Comm_free(&delete_me);
        }
        CMX_GROUP_WORLD = NULL;
#if DEBUG
        printf("Creating comm: I AM MASTER[%ld]\n", p_config.rank());
#endif
    } else {
        /* I'm a worker */
        /* create the head of the group linked list */
        igroup = (cmx_igroup_t*)malloc(sizeof(cmx_igroup_t));
        status = MPI_Comm_split(
                g_state.comm, 1, p_config.rank(), &(igroup->comm));
        CMX_ASSERT(MPI_SUCCESS == status);
        status = MPI_Comm_group(igroup->comm, &(igroup->group));
        CMX_ASSERT(MPI_SUCCESS == status);
        status = MPI_Comm_rank(igroup->comm, &(igroup->rank));
        CMX_ASSERT(MPI_SUCCESS == status);
        status = MPI_Comm_size(igroup->comm, &(igroup->size));
        igroup->next = NULL;
        CMX_ASSERT(MPI_SUCCESS == status);
        _igroup_set_world_ranks(igroup);
        CMX_ASSERT(igroup->world_ranks != NULL);
#if DEBUG
        printf("Creating comm: I AM WORKER[%ld]\n", p_config.rank());
#endif
        CMX_GROUP_WORLD = igroup;
        group_list = igroup;
    }
    status = MPI_Comm_split(MPI_COMM_WORLD, proc_split_group_stamp,
            p_config.rank(), &(g_state.node_comm));
    CMX_ASSERT(MPI_SUCCESS == status);
    /* node rank */
    status = MPI_Comm_rank(g_state.node_comm, &(g_state.node_rank));
    CMX_ASSERT(MPI_SUCCESS == status);
    /* node size */
    status = MPI_Comm_size(g_state.node_comm, &(g_state.node_size));
    CMX_ASSERT(MPI_SUCCESS == status);

#if DEBUG
    printf("node_rank[%d]/ size[%d]\n", g_state.node_rank, g_state.node_size);
    if (g_state.master[p_config.rank()] == p_config.rank()) {
        printf("[%d] world %d/%d\tI'm a master\n",
            p_config.rank(), p_config.rank(), g_state.size);
    }
    else {
        printf("[%d] world %d/%d\tI'm a worker\n",
            p_config.rank(), p_config.rank(), g_state.size);
    }
#endif
}


void cmx_group_finalize()
{
    int status;
    cmx_igroup_t *current_group_list_item = group_list;
    cmx_igroup_t *previous_group_list_item = NULL;

#if DEBUG
    printf("[%d] cmx_group_finalize()\n", p_config.rank());
#endif

    /* This loop will also clean up CMX_GROUP_WORLD */
    while (current_group_list_item != NULL) {
        previous_group_list_item = current_group_list_item;
        current_group_list_item = current_group_list_item->next;
        _igroup_free(previous_group_list_item);
    }

    free(g_state.master);
    free(g_state.hostid);
    status = MPI_Comm_free(&(g_state.node_comm));
    CMX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Group_free(&(g_state.group));
    CMX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Comm_free(&(g_state.comm));
    CMX_ASSERT(MPI_SUCCESS == status);
}
#endif


long p_Environment::xgethostid()
{
#if defined(__bgp__)
#warning BGP
    long nodeid;
    int matched,midplane,nodecard,computecard;
    char rack_row,rack_col;
    char location[128];
    char location_clean[128];
    (void) memset(location, '\0', 128);
    (void) memset(location_clean, '\0', 128);
    _BGP_Personality_t personality;
    Kernel_GetPersonality(&personality, sizeof(personality));
    BGP_Personality_getLocationString(&personality, location);
    matched = sscanf(location, "R%c%c-M%1d-N%2d-J%2d",
            &rack_row, &rack_col, &midplane, &nodecard, &computecard);
    assert(matched == 5);
    sprintf(location_clean, "%2d%02d%1d%02d%02d",
            (int)rack_row, (int)rack_col, midplane, nodecard, computecard);
    nodeid = atol(location_clean);
#elif defined(__bgq__)
#warning BGQ
    int nodeid;
    MPIX_Hardware_t hw;
    MPIX_Hardware(&hw);

    nodeid = hw.Coords[0] * hw.Size[1] * hw.Size[2] * hw.Size[3] * hw.Size[4]
        + hw.Coords[1] * hw.Size[2] * hw.Size[3] * hw.Size[4]
        + hw.Coords[2] * hw.Size[3] * hw.Size[4]
        + hw.Coords[3] * hw.Size[4]
        + hw.Coords[4];
#elif defined(__CRAYXT) || defined(__CRAYXE)
#warning CRAY
    int nodeid;
#  if defined(__CRAYXT)
    PMI_Portals_get_nid(p_config.rank(), &nodeid);
#  elif defined(__CRAYXE)
    PMI_Get_nid(p_config.rank(), &nodeid);
#  endif
#else
    long nodeid = gethostid();
#endif

    return nodeid;
}

int p_Environment::get_num_progress_ranks_per_node()
{
  int num_progress_ranks_per_node = 1;
  const char* num_progress_ranks_env_var = getenv("GA_NUM_PROGRESS_RANKS_PER_NODE");
  if (num_progress_ranks_env_var != NULL && num_progress_ranks_env_var[0] != '\0') {
    int env_number = atoi(getenv("GA_NUM_PROGRESS_RANKS_PER_NODE"));
    if ( env_number > 0 && env_number < 16)
      num_progress_ranks_per_node = env_number;
  }
  else {
    num_progress_ranks_per_node = 1;
  }
  return num_progress_ranks_per_node;
}

int p_Environment::get_progress_rank_distribution_on_node()
{
  const char* progress_ranks_packed_env_var = getenv("GA_PROGRESS_RANKS_DISTRIBUTION_PACKED");
  const char* progress_ranks_cyclic_env_var = getenv("GA_PROGRESS_RANKS_DISTRIBUTION_CYCLIC");
  int is_node_ranks_packed;
  int rank_packed =0;
  int rank_cyclic =0;

  if (progress_ranks_packed_env_var != NULL && progress_ranks_packed_env_var[0] != '\0') {
    if (strchr(progress_ranks_packed_env_var, 'y') != NULL ||
        strchr(progress_ranks_packed_env_var, 'Y') != NULL ||
        strchr(progress_ranks_packed_env_var, '1') != NULL ) {
      rank_packed = 1;
    }
  }
  if (progress_ranks_cyclic_env_var != NULL && progress_ranks_cyclic_env_var[0] != '\0') {
    if (strchr(progress_ranks_cyclic_env_var, 'y') != NULL ||
        strchr(progress_ranks_cyclic_env_var, 'Y') != NULL ||
        strchr(progress_ranks_cyclic_env_var, '1') != NULL ) {
      rank_cyclic = 1;
    }
  }
  if (rank_packed == 1 || rank_cyclic == 0) is_node_ranks_packed = 1;
  if (rank_packed == 0 && rank_cyclic == 1) is_node_ranks_packed = 0;
  return is_node_ranks_packed;
}

int p_Environment::get_my_master_rank_with_same_hostid(int rank, int split_group_size,
    int smallest_rank_with_same_hostid, int largest_rank_with_same_hostid,
    int num_progress_ranks_per_node, int is_node_ranks_packed)
{
  int my_master;

#if MASTER_IS_SMALLEST_SMP_RANK
  if(is_node_ranks_packed) {
    /* Contiguous packing of ranks on a node */
    my_master = smallest_rank_with_same_hostid
      + split_group_size *
      ((rank - smallest_rank_with_same_hostid)/split_group_size);
  }
  else {
    if(num_progress_ranks_per_node == 1) { 
      my_master = 2 * (split_group_size *
          ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
    } else {
      /* Cyclic packing of ranks on a node between two sockets
       * with even and odd numbering  */
      if(rank % 2 == 0) {
        my_master = 2 * (split_group_size *
            ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
      } else {
        my_master = 1 + 2 * (split_group_size *
            ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
      }
    }
  }
#else
  /* By default creates largest SMP rank as Master */
  if(is_node_ranks_packed) {
    /* Contiguous packing of ranks on a node */
    my_master = largest_rank_with_same_hostid
      - split_group_size *
      ((largest_rank_with_same_hostid - rank)/split_group_size);
  }
  else {
    if(num_progress_ranks_per_node == 1) { 
      my_master = largest_rank_with_same_hostid - 2 * (split_group_size *
          ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
    } else {
      /* Cyclic packing of ranks on a node between two sockets
       * with even and odd numbering  */
      if(rank % 2 == 0) {
        my_master = largest_rank_with_same_hostid - 1 - 2 * (split_group_size *
            ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
      } else {
        my_master = largest_rank_with_same_hostid - 2 * (split_group_size *
            ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
      }
    }
  }
#endif
  return my_master;
}

int p_Environment::get_my_rank_to_free(int rank, int split_group_size,
    int smallest_rank_with_same_hostid, int largest_rank_with_same_hostid,
    int num_progress_ranks_per_node, int is_node_ranks_packed)
{
  int my_rank_to_free;

#if MASTER_IS_SMALLEST_SMP_RANK
  /* By default creates largest SMP rank as Master */
  if(is_node_ranks_packed) {
    /* Contiguous packing of ranks on a node */
    my_rank_to_free = largest_rank_with_same_hostid
      - split_group_size *
      ((largest_rank_with_same_hostid - rank)/split_group_size);
  }
  else {
    if(num_progress_ranks_per_node == 1) { 
      my_rank_to_free = largest_rank_with_same_hostid - 2 * (split_group_size *
          ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
    } else {
      /* Cyclic packing of ranks on a node between two sockets
       * with even and odd numbering  */
      if(rank % 2 == 0) {
        my_rank_to_free = largest_rank_with_same_hostid - 1 - 2 * (split_group_size *
            ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
      } else {
        my_rank_to_free = largest_rank_with_same_hostid - 2 * (split_group_size *
            ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
      }
    }
  }
#else
  if(is_node_ranks_packed) {
    /* Contiguous packing of ranks on a node */
    my_rank_to_free = smallest_rank_with_same_hostid
      + split_group_size *
      ((rank - smallest_rank_with_same_hostid)/split_group_size);
  }
  else {
    if(num_progress_ranks_per_node == 1) { 
      my_rank_to_free = 2 * (split_group_size *
          ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
    } else {
      /* Cyclic packing of ranks on a node between two sockets
       * with even and odd numbering  */
      if(rank % 2 == 0) {
        my_rank_to_free = 2 * (split_group_size *
            ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
      } else {
        my_rank_to_free = 1 + 2 * (split_group_size *
            ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
      }
    }
  }
#endif
  return my_rank_to_free;
}

/**
 * Get group corresponding to world group
 * @return pointer to world group
 */
Group* p_Environment::getWorldGroup()
{
  return p_CMX_GROUP_WORLD;
}

} // namespace CMX

