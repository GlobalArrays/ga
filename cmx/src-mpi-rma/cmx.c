/* C and/or system headers */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

/* 3rd party headers */
#include <mpi.h>

/* Configuration options */
#define DEBUG 0

#define USE_MPI_REQUESTS
/*
#define USE_MPI_FLUSH_LOCAL
#define USE_MPI_WIN_ALLOC
*/

#ifdef USE_MPI_FLUSH_LOCAL
#define USE_MPI_REQUESTS
#endif

/* our headers */
#include "cmx.h"
#include "cmx_impl.h"
#include "groups.h"

/* exported state */
local_state l_state;

/* static state */
static int  initialized=0;  /* for cmx_initialized(), 0=false */
static char skip_lock=0;    /* don't acquire or release lock */

/* static function declarations */
static void acquire_remote_lock(int proc);
static void release_remote_lock(int proc);
static inline void acc(
    int datatype, int count, void *get_buf,
    void *src_ptr, long src_idx, void *scale);

/* definitions needed to implement mutexes */
MPI_Win *_mutex_list;
int **_mutex_buf;
int *_mutex_num;
int _mutex_total;

/* Maximum number of outstanding non-blocking requests */
/* static int nb_max_outstanding = CMX_MAX_NB_OUTSTANDING; */

typedef struct {
  MPI_Request request;
  MPI_Win win;
  int active;
#ifdef USE_MPI_FLUSH_LOCAL
  int remote_proc;
#endif
} nb_t;

static nb_t **nb_list = NULL;

/* needed for complex accumulate */
typedef struct {
  double real;
  double imag;
} DoubleComplex;

typedef struct {
  float real;
  float imag;
} SingleComplex;

/* Find first available non-blocking handle */
#ifdef ZUSE_MPI_REQUESTS
void get_nb_request(cmx_request_t *handle, nb_t **req)
{
  int i;
  for (i=0; i<nb_max_outstanding; i++) {
    if (nb_list[i]->active == 0) break;
  }
  if (i<nb_max_outstanding) {
    *handle = i;
    *req = nb_list[i];
  } else {
    i = -1;
    req = NULL;
  }
}
#endif

/**
 * Utility function to catch and translate MPI errors. Returns silently if
 * no error detected.
 * @param ierr: Error code from MPI call
 * @param location: User specified string to indicate location of error
 */
void translate_mpi_error(int ierr, const char* location)
{
  if (ierr == MPI_SUCCESS) return;
  char err_string[MPI_MAX_ERROR_STRING];
  int len;
  fprintf(stderr,"p[%d] Error in %s\n",l_state.rank,location);
  MPI_Error_string(ierr,err_string,&len);
  fprintf(stderr,"p[%d] MPI_Error: %s\n",l_state.rank,err_string);
}


/* Translate global process rank to local process rank */
int get_local_rank_from_win(MPI_Win win, int world_rank, int *local_rank)
{
  int status;
  cmx_igroup_t *world_igroup = CMX_GROUP_WORLD;
  MPI_Group group;
  status = MPI_Win_get_group(win, &group);
  translate_mpi_error(status,"get_local_rank_from_win:MPI_Win_get_group");
  status = MPI_Group_translate_ranks( world_igroup->group,
      1, &world_rank, group, local_rank);
  if (status != MPI_SUCCESS) {
    translate_mpi_error(status,"get_local_rank_from_win:MPI_Group_translate_ranks");
    cmx_error("MPI_Group_translate_ranks: Failed", status);
  }

  return CMX_SUCCESS;
}

int cmx_init()
{
  int i, status;

  if (initialized) {
    return 0;
  }
  initialized = 1;

  /* Assert MPI has been initialized */
  int init_flag;
  status = MPI_Initialized(&init_flag);
  assert(MPI_SUCCESS == status);
  assert(init_flag);

  /* Duplicate the World Communicator */
  status = MPI_Comm_dup(MPI_COMM_WORLD, &(l_state.world_comm));
  translate_mpi_error(status,"cmx_init:MPI_Comm_dup");
  assert(MPI_SUCCESS == status);
  assert(l_state.world_comm); 

  /* My Rank */
  status = MPI_Comm_rank(l_state.world_comm, &(l_state.rank));
  assert(MPI_SUCCESS == status);

  /* World Size */
  status = MPI_Comm_size(l_state.world_comm, &(l_state.size));
  assert(MPI_SUCCESS == status);

  /* groups */
  cmx_group_init();

  /* set mutex list equal to null */
  _mutex_list = NULL;
  _mutex_buf = NULL;
  _mutex_num = NULL;
  _mutex_total = 0;

  /* initialize non-blocking handles */
#ifdef ZUSE_MPI_REQUESTS
  nb_list = (nb_t**)malloc(sizeof(nb_t*) * nb_max_outstanding);
  CMX_ASSERT(nb_list);
  for (i=0; i<nb_max_outstanding; i++) {
    nb_list[i] = (nb_t*)malloc(sizeof(nb_t));
    nb_list[i]->active = 0;
  }
#endif

  /* sync - initialize first communication epoch */
  cmx_fence_all(CMX_GROUP_WORLD);
  /* Synch - Sanity Check */
  MPI_Barrier(l_state.world_comm);

  return CMX_SUCCESS;
}


int cmx_init_args(int *argc, char ***argv)
{
  int init_flag;

  MPI_Initialized(&init_flag);

  if(!init_flag) {
    MPI_Init(argc, argv);
  }

  return cmx_init();
}


int cmx_initialized()
{
  return initialized;
}


void cmx_error(char *msg, int code)
{
  fprintf(stderr,"[%d] Received an Error in Communication: (%d) %s\n",
      l_state.rank, code, msg);

  MPI_Abort(l_state.world_comm, code);
}


/**
 * blocking contiguous put
 * @param src pointer to source buffer
 * @param dst_offset offset from start of data allocation on remote process
 * @param bytes length of data to be transferred
 * @param proc rank of destination processor
 * @param cmx_hdl handle for data allocation
 */
int cmx_put(
    void *src, cmxInt dst_offset, cmxInt bytes,
    int proc, cmx_handle_t cmx_hdl)
{
  MPI_Aint displ;
  int lproc, ierr;
  MPI_Win win = cmx_hdl.win;
#ifdef USE_MPI_REQUESTS
  MPI_Request request;
  MPI_Status status;
#endif
  displ = (MPI_Aint)(dst_offset);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
/**
 * All processes called MPI_WIN_LOCK_ALL on the window (win) used for
 * these calls. This call:
 *
 * Starts an RMA access epoch to all processors in win with a lock of
 * type MPI_LOCK_SHARED. During the epoch, the calling process can access
 * the window memory on all processes in win using RMA operations.
 * A window locked with MPI_WIN_LOCK_ALL must be unlocked with
 * MPI_WIN_UNLOCK_ALL. This routine is not collective. The ALL refers to a
 * lock on all members of the group of the window.
 *
 * Locks are used to protect accesses to the locked target window effected
 * by RMA calls issued between the lock and unlock calls, and to protect
 * load/store accesses to a locked local or shared memory window executed
 * between the lock and unlock calls. Accesses that are protected by an
 * exclusive lock will not be concurrent at the window site with other
 * accesses to the same window that are lock protected. Accesses that are
 * protected by a shared lock will not be concurrent at the window site with
 * accesses protected by an exclusive lock to the same window.
 *
 * A passive synchronization epoch is created by calling MPI_WIN_LOCK_ALL on
 * win
 */
/**
 * MPI_WIN_FLUSH completes all outstanding RMA operations initiated by the
 * calling process to the target rank on the specified window. The
 * operations are completed both at the origin and at the target
 */
#ifdef USE_PRIOR_MPI_WIN_FLUSH
  ierr = MPI_Win_flush(lproc, reg_win->win);
  translate_mpi_error(ierr,"cmx_put:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
/**
 * The local communication buffer of an RMA call should not be updated, and
 * the local communication buffer of a get call should not be accessed after
 * the RMA call until the operation completes at the origin.
 */
  ierr = MPI_Put(src, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
      win);
  translate_mpi_error(ierr,"cmx_put:MPI_Put");
/**
 * MPI_WIN_FLUSH_LOCAL locally completes at the origin all outstanding RMA
 * operations initiated by the calling process to the target process
 * specified by the rank on the specified window. For example, after the
 * routine completes, the user may reused any buffers provided to put, get,
 * or accumulate operations
 */
  ierr = MPI_Win_flush_local(lproc, win);
  translate_mpi_error(ierr,"cmx_put:MPI_Win_flush_local");
#else
/**
 * MPI_RPUT is similar to MPI_PUT, except that it allocates a communication
 * request object and associates it with the request handle. The completion
 * of an MPI_RPUT operation (i.e. after the corresponding test or wait)
 * indicates that the sender is now free to update the locations in the
 * origin buffer. It does not indicate that the data is available at the
 * target window. If remote completion is required, MPI_WIN_FLUSH,
 * MPI_WIN_FLUSH_ALL, MPI_WIN_UNLOCK or MPI_WIN_UNLOCK_ALL can be used.
 */
  ierr = MPI_Rput(src, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
      win, &request);
  translate_mpi_error(ierr,"cmx_put:MPI_Rput");
/**
 * A call to MPI_WAIT returns when the operation identified by the request
 * is complete. If the request is an active persistent request, it is marked
 * inactive. Any other type of request is and the request handle is set to
 * MPI_REQUEST_NULL. MPI_WAIT is a non-local operation
 */
  ierr = MPI_Wait(&request, &status);
  translate_mpi_error(ierr,"cmx_put:MPI_Wait");
#endif
#else
  ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,win);
  translate_mpi_error(ierr,"cmx_put:MPI_Win_lock");
  ierr = MPI_Put(src, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
      win);
  translate_mpi_error(ierr,"cmx_put:MPI_Put");
  ierr = MPI_Win_unlock(lproc,win);
  translate_mpi_error(ierr,"cmx_put:MPI_Win_unlock");
#endif
  return CMX_SUCCESS;
}


/**
 * blocking contiguous get
 * @param dst pointer to source local buffer
 * @param src_offset offset from start of data allocation on remote process
 * @param bytes length of data to be transferred
 * @param proc rank of destination processor
 * @param cmx_hdl handle for data allocation
 */
int cmx_get(
    void *dst, cmxInt src_offset, cmxInt bytes,
    int proc, cmx_handle_t cmx_hdl)
{
  MPI_Aint displ;
  int lproc, ierr;
  MPI_Win win = cmx_hdl.win;
#ifdef USE_MPI_REQUESTS
  MPI_Request request;
  MPI_Status status;
#endif
  displ = (MPI_Aint)(src_offset);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
#ifdef USE_PRIOR_MPI_WIN_FLUSH
  ierr = MPI_Win_flush(lproc, reg_win->win);
  translate_mpi_error(ierr,"cmx_get:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Get(dst, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
      win);
  translate_mpi_error(ierr,"cmx_get:MPI_Get");
  ierr = MPI_Win_flush_local(lproc, win);
  translate_mpi_error(ierr,"cmx_get:MPI_Win_flush_local");
#else
  ierr = MPI_Rget(dst, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
      win, &request);
  translate_mpi_error(ierr,"cmx_get:MPI_Rget");
  ierr = MPI_Wait(&request, &status);
  translate_mpi_error(ierr,"cmx_get:MPI_Wait");
#endif
#else
  ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,win);
  translate_mpi_error(ierr,"cmx_get:MPI_Win_lock");
  ierr = MPI_Get(dst, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
      win);
  translate_mpi_error(ierr,"cmx_get:MPI_Get");
  ierr = MPI_Win_unlock(lproc,win);
  translate_mpi_error(ierr,"cmx_get:MPI_Win_unlock");
#endif
  return CMX_SUCCESS;
}

/**
 * blocking contiguous accumulate
 * @param op CMX op for accumulate operation
 * @param scale value to scale data by before accumulating it
 * @param src pointer to source buffer
 * @param dst_offset offset from start of data allocation on remote processor
 * @param bytes length of data to be transferred
 * @param proc rank of destination processors
 * @param cmx_hdl handle for data allocation
 */
int cmx_acc(
    int op, void *scale,
    void *src, cmxInt dst_offset, cmxInt bytes,
    int proc, cmx_handle_t cmx_hdl)
{
  MPI_Aint displ, count;
  void *buf_ptr;
  int i, lproc, ierr;
  MPI_Win win = cmx_hdl.win;
#ifdef USE_MPI_REQUESTS
  MPI_Request request;
  MPI_Status status;
#endif
  MPI_Datatype mpi_type;
  displ = (MPI_Aint)(dst_offset);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
  if (op == CMX_ACC_INT) {
    int *buf;
    int *isrc = (int*)src;
    int iscale = *((int*)scale);
    count = (MPI_Aint)(bytes/sizeof(int));
    buf = (int*)malloc(bytes);
    buf_ptr = buf;
    for (i=0; i<count; i++) {
      buf[i] = isrc[i]*iscale;
    }
    mpi_type = MPI_INT;
  } else if (op == CMX_ACC_LNG) {
    long *buf;
    long *lsrc = (long*)src;
    long lscale = *((long*)scale);
    count = (MPI_Aint)(bytes/sizeof(long));
    buf = (long*)malloc(bytes);
    buf_ptr = buf;
    for (i=0; i<count; i++) {
      buf[i] = lsrc[i]*lscale;
    }
    mpi_type = MPI_LONG;
  } else if (op == CMX_ACC_FLT) {
    float *buf;
    float *fsrc = (float*)src;
    float fscale = *((float*)scale);
    count = (MPI_Aint)(bytes/sizeof(float));
    buf = (float*)malloc(bytes);
    buf_ptr = buf;
    for (i=0; i<count; i++) {
      buf[i] = fsrc[i]*fscale;
    }
    mpi_type = MPI_FLOAT;
  } else if (op == CMX_ACC_DBL) {
    double *buf;
    double *dsrc = (double*)src;
    double dscale = *((double*)scale);
    count = (MPI_Aint)(bytes/sizeof(double));
    buf = (double*)malloc(bytes);
    buf_ptr = buf;
    for (i=0; i<count; i++) {
      buf[i] = dsrc[i]*dscale;
    }
    mpi_type = MPI_DOUBLE;
  } else if (op == CMX_ACC_CPL) {
    int cnum;
    float *buf;
    float *csrc = (float*)src;
    float crscale = *((float*)scale);
    float ciscale = *((float*)scale+1);
    count = (MPI_Aint)(bytes/sizeof(float));
    cnum = count/2;
    buf = (float*)malloc(bytes);
    buf_ptr = buf;
    for (i=0; i<cnum; i++) {
      buf[2*i] = csrc[2*i]*crscale-csrc[2*i+1]*ciscale;
      buf[2*i+1] = csrc[2*i]*ciscale+csrc[2*i+1]*crscale;
    }
    mpi_type = MPI_FLOAT;
  } else if (op == CMX_ACC_DCP) {
    int cnum;
    double *buf;
    double *csrc = (double*)src;
    double crscale = *((double*)scale);
    double ciscale = *((double*)scale+1);
    count = (MPI_Aint)(bytes/sizeof(double));
    cnum = count/2;
    buf = (double*)malloc(bytes);
    buf_ptr = buf;
    for (i=0; i<cnum; i++) {
      buf[2*i] = csrc[2*i]*crscale-csrc[2*i+1]*ciscale;
      buf[2*i+1] = csrc[2*i]*ciscale+csrc[2*i+1]*crscale;
    }
    mpi_type = MPI_DOUBLE;
  } else {
    assert(0);
  }
#ifdef USE_PRIOR_MPI_WIN_FLUSH
  ierr = MPI_Win_flush(lproc, reg_win->win);
  translate_mpi_error(ierr,"cmx_acc:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Accumulate(buf_ptr,count,mpi_type,lproc,displ,count,
      mpi_type,MPI_SUM,win);
  translate_mpi_error(ierr,"cmx_acc:MPI_Accumulate");
  ierr = MPI_Win_flush_local(lproc, win);
  translate_mpi_error(ierr,"cmx_acc:MPI_Win_flush_local");
#else
  ierr = MPI_Raccumulate(buf_ptr,count,mpi_type,lproc,displ,count,
      mpi_type,MPI_SUM,win,&request);
  translate_mpi_error(ierr,"cmx_acc:MPI_Raccumulate");
  ierr = MPI_Wait(&request, &status);
  translate_mpi_error(ierr,"cmx_acc:MPI_Wait");
#endif
#else
  ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,win);
  translate_mpi_error(ierr,"cmx_acc:MPI_Win_lock");
  ierr = MPI_Accumulate(buf_ptr,count,mpi_type,lproc,displ,count,
      mpi_type,MPI_SUM,win);
  translate_mpi_error(ierr,"cmx_acc:MPI_Accumulate");
  ierr = MPI_Win_unlock(lproc,win);
  translate_mpi_error(ierr,"cmx_acc:MPI_Win_unlock");
#endif
  free(buf_ptr);
  return CMX_SUCCESS;
}

/**
 * No checking for data consistency. Assume correctness has already been
 * established elsewhere. Individual elements are assumed to be one byte in size
 * @param stride_array: physical dimensions of array
 * @param count: number of elements along each array dimension
 * @param levels: number of stride levels (should be one less than array dimension)
 * @param type: MPI_Datatype returned to calling program
 */
void strided_to_subarray_dtype(int *stride_array, int *count, int levels,
    MPI_Datatype base_type, MPI_Datatype *type)
{
  int ndims, i, ierr;
  int array_of_sizes[7];
  int array_of_starts[7];
  int array_of_subsizes[7];
  int stride;
  ierr = MPI_Type_size(base_type,&stride);
  translate_mpi_error(ierr,"strided_to_subarray_dtype:MPI_Type_size");
  ndims = levels+1;
  /* the pointer to the local buffer points to the first data element in
     data exchange, not the origin of the local array, so all starts should
     be zero */
  for (i=0; i<levels; i++) {
    array_of_sizes[i] = stride_array[i]/stride;
    array_of_starts[i] = 0;
    array_of_subsizes[i] = count[i];
    stride = stride_array[i];
  }
  array_of_sizes[levels] = count[levels];
  array_of_starts[levels] = 0;
  array_of_subsizes[levels] = count[levels];
#if 0
  for (i=0; i<ndims; i++) {
    printf("p[%d] ndims: %d sizes[%d]: %d subsizes[%d]: %d starts[%d]: %d\n",
        l_state.rank,ndims,i,array_of_sizes[i],i,array_of_subsizes[i],
        i,array_of_starts[i]);
  }
#endif

  ierr = MPI_Type_create_subarray(ndims, array_of_sizes,array_of_subsizes,
      array_of_starts,MPI_ORDER_FORTRAN,base_type,type);
  if (ierr != 0) {
    printf("p[%d] Error forming MPI_Datatype for one-sided strided operation."
        " Check that stride dimensions are compatible with local block"
        " dimensions\n",l_state.rank);
    for (i=0; i<levels; i++) {
      printf("p[%d] count[%d]: %d stride[%d]: %d\n",l_state.rank,i,count[i],
          i,stride_array[i]);
    }
    printf("p[%d] count[%d]: %d\n",l_state.rank,i,count[i]);
    translate_mpi_error(ierr,"strided_to_subarray_dtype:MPI_Type_create_subarray");
  }
}

/**
 * blocking strided put
 * @param src_ptr: pointer to origin of data on source processor
 * @param src_stride_ar: physical dimensions of array containing source data
 * @param dst_offset: offset from start of data allocation on remote process
 * @param dst_stride_ar: physical dimensions of destination array
 * @param count: array containing number of data points along each dimension. The 0
 * dimension contains the actual length of contiguous segments in bytes (number
 * of elements times the size of each element).
 * @param stride_levels: this should be one less than the dimension of the array
 * @param proc: global rank of destination array
 * @param cmx_hdl: handle for data allocation
 */
int cmx_puts(
    void *src_ptr, cmxInt *src_stride_ar,
    cmxInt dst_offset, cmxInt *dst_stride_ar,
    cmxInt *count, int stride_levels,
    int proc, cmx_handle_t cmx_hdl)
{
  MPI_Datatype src_type, dst_type;
  MPI_Aint displ;
  int lproc, ierr;
  MPI_Win win = cmx_hdl.win;
#ifdef USE_MPI_REQUESTS
  MPI_Request request;
  MPI_Status status;
#endif
  /* If data is contiguous, use cmx_put */
  if (stride_levels == 0) {
    return cmx_put(src_ptr, dst_offset, count[0], proc, cmx_hdl);
  }
  displ = (MPI_Aint)(dst_offset);

  strided_to_subarray_dtype(src_stride_ar, count, stride_levels,
      MPI_BYTE, &src_type);
  strided_to_subarray_dtype(dst_stride_ar, count, stride_levels,
      MPI_BYTE, &dst_type);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
  ierr = MPI_Type_commit(&src_type);
  translate_mpi_error(ierr,"cmx_puts:MPI_Type_commit");
  ierr = MPI_Type_commit(&dst_type);
  translate_mpi_error(ierr,"cmx_puts:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
  ierr = MPI_Win_flush(lproc, reg_win->win);
  translate_mpi_error(ierr,"cmx_puts:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Put(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
      win);
  translate_mpi_error(ierr,"cmx_puts:MPI_Put");
  ierr = MPI_Win_flush_local(lproc, win);
  translate_mpi_error(ierr,"cmx_puts:MPI_Win_flush_local");
#else
  ierr = MPI_Rput(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
      win, &request);
  translate_mpi_error(ierr,"cmx_puts:MPI_Rput");
  ierr = MPI_Wait(&request, &status);
  translate_mpi_error(ierr,"cmx_puts:MPI_Wait");
#endif
#else
  ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,win);
  translate_mpi_error(ierr,"cmx_puts:MPI_Win_lock");
  ierr = MPI_Put(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
      win);
  translate_mpi_error(ierr,"cmx_puts:MPI_Put");
  ierr = MPI_Win_unlock(lproc,win);
  translate_mpi_error(ierr,"cmx_puts:MPI_Win_unlock");
#endif
  ierr = MPI_Type_free(&src_type);
  translate_mpi_error(ierr,"cmx_puts:MPI_Type_free");
  ierr = MPI_Type_free(&dst_type);
  translate_mpi_error(ierr,"cmx_puts:MPI_Type_free");
  return CMX_SUCCESS;
}


/**
 * blocking strided get
 * @param dst_ptr: pointer to origin of data on destination processor
 * @param dst_stride_ar: physical dimensions of destination array
 * @param src_offset: offset from start of data allocation on remote process
 * @param src_stride_ar: physical dimensions of array containing source data
 * @param count: array containing number of data points along each dimension. The 0
 * dimension contains the actual length of contiguous segments in bytes (number
 * of elements times the size of each element).
 * @param stride_levels: this should be one less than the dimension of the array
 * @param proc: global rank of source array
 * @param cmx_hdl: handle for data allocation
 */
int cmx_gets(
    void *dst_ptr, cmxInt *dst_stride_ar,
    cmxInt src_offset, cmxInt *src_stride_ar,
    cmxInt *count, int stride_levels,
    int proc, cmx_handle_t cmx_hdl)
{
  MPI_Datatype src_type, dst_type;
  MPI_Aint displ;
  int lproc,ierr;
  MPI_Win win = cmx_hdl.win;
#ifdef USE_MPI_REQUESTS
  MPI_Request request;
  MPI_Status status;
#endif
  /* If data is contiguous, use cmx_get */
  if (stride_levels == 0) {
    return cmx_get(dst_ptr, src_offset, count[0], proc, cmx_hdl);
  }
  displ = (MPI_Aint)(src_offset);

  strided_to_subarray_dtype(src_stride_ar, count, stride_levels,
      MPI_BYTE, &src_type);
  strided_to_subarray_dtype(dst_stride_ar, count, stride_levels,
      MPI_BYTE, &dst_type);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
  ierr = MPI_Type_commit(&src_type);
  translate_mpi_error(ierr,"cmx_gets:MPI_Type_commit");
  ierr = MPI_Type_commit(&dst_type);
  translate_mpi_error(ierr,"cmx_gets:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
  ierr = MPI_Win_flush(lproc, reg_win->win);
  translate_mpi_error(ierr,"cmx_gets:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Get(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
      win);
  translate_mpi_error(ierr,"cmx_gets:MPI_Get");
  ierr = MPI_Win_flush_local(lproc, win);
  translate_mpi_error(ierr,"cmx_gets:MPI_Win_flush_local");
#else
  ierr = MPI_Rget(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
      win, &request);
  translate_mpi_error(ierr,"cmx_gets:MPI_Rget");
  ierr = MPI_Wait(&request, &status);
  translate_mpi_error(ierr,"cmx_gets:MPI_Wait");
#endif
#else
  ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,win);
  translate_mpi_error(ierr,"cmx_gets:MPI_Win_lock");
  ierr = MPI_Get(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
      win);
  translate_mpi_error(ierr,"cmx_gets:MPI_Get");
  ierr = MPI_Win_unlock(lproc,win);
  translate_mpi_error(ierr,"cmx_gets:MPI_Win_lock");
#endif
  ierr = MPI_Type_free(&src_type);
  translate_mpi_error(ierr,"cmx_gets:MPI_Type_free");
  ierr = MPI_Type_free(&dst_type);
  translate_mpi_error(ierr,"cmx_gets:MPI_Type_free");
  return CMX_SUCCESS;
}

/**
 * Utility function for allocating a buffer that can be used to scale strided
 * data before sending it to remote processor for accumulate operation. Data is
 * copied into buffer after allocation. 
 * @param ptr: pointer to first member of data array
 * @param strides: physical dimensions of array containing data
 * @param count: array containing the number of data points along each dimension.
 * The 0 element contains the actual length of the contiguous segments in bytes
 * @param levels: this should be one less than the dimension of the array
 * @param size: size (in bytes) of allocated buffer
 * @param new_strides: stride array for newly allocated buffer
 */
void* malloc_strided_acc_buffer(void* ptr, cmxInt *strides, cmxInt *count,
    int levels, cmxInt *size, cmxInt *new_strides)
{
  cmxInt index[7];
  cmxInt lcnt[7];
  cmxInt i, j, idx, stride, src_idx;
  void *new_ptr;
  char *cursor, *src_cursor;
  cmxInt n1dim;
  *size = 1;
  /* evaluate size of new array */
  for (i=0; i<=levels; i++) {
    *size *= count[i];
  }
  new_ptr = malloc(*size);
  cursor = (char*)new_ptr;
  stride = 1;
  /* calculate strides in new array */
  n1dim = 1;
  for (i=0; i<levels; i++) {
    new_strides[i] = stride*count[i];
    stride = new_strides[i];
    n1dim *= count[i+1];
  }

  /* copy data from original buffer to new buffer */
  for (i=0; i<n1dim; i++) {
    idx = i;
    /* calculate indices corresponding to i except for first index */
    for (j=0; j<levels; j++) {
      index[j] = idx%count[j+1];
      if (j<levels-1) idx = (idx-index[j])/count[j+1];
    }
    if (levels > 0) index[levels-1] = idx;
    /* evaluate offset in source ptr */
    src_idx = 0;
    if (levels > 0) stride = strides[0];
    for (j=0; j<levels; j++) {
      src_idx += index[j]*stride;
      if (j+1<levels) stride = strides[j+1];
    }
    src_cursor = (char*)ptr+src_idx;
    memcpy(cursor, src_cursor, count[0]);
    cursor += count[0];
  }
  return new_ptr;
}

/**
 * blocking strided accumulate
 * @param op: operation
 * @param scale: factor to scale data before adding to contents in destination array
 * @param src_ptr: pointer to origin of data on source processor
 * @param src_stride_ar: physical dimensions of array containing source data
 * @param dst_ptr: pointer to origin of data on destination processor
 * @param dst_stride_ar: physical dimensions of destination array
 * @param count: array containing number of data points along each dimension. The 0
 * dimension contains the actual length of contiguous segments in bytes (number
 * of elements times the size of each element).
 * @param stride_levels: this should be one less than the dimension of the array
 * @param proc: global rank of destination array
 * @param cmx_hdl: handle of data allocation
 */
int cmx_accs(
    int op, void *scale,
    void *src_ptr, cmxInt *src_stride_ar,
    cmxInt dst_offset, cmxInt *dst_stride_ar,
    cmxInt *count, int stride_levels,
    int proc, cmx_handle_t cmx_hdl)
{
  MPI_Datatype src_type, dst_type;
  MPI_Aint displ;
  int lproc, i, ierr;
  void *packbuf;
  cmxInt bufsize;
  cmxInt new_strides[7], new_count[7];
  MPI_Win win = cmx_hdl.win;
#ifdef USE_MPI_REQUESTS
  MPI_Request request;
  MPI_Status status;
#endif

  /* If data is contiguous, use cmx_acc */
  if (stride_levels == 0) {
    return cmx_acc(op, scale, src_ptr,
        dst_offset, count[0], proc, cmx_hdl);
  }
  displ = (MPI_Aint)(dst_offset);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
  packbuf = malloc_strided_acc_buffer(src_ptr, src_stride_ar, count,
      stride_levels, &bufsize, new_strides);

  for (i=0; i<stride_levels+1; i++) new_count[i] = count[i];
  if (op == CMX_ACC_INT) {
    int *buf;
    int iscale = *((int*)scale);
    cmxInt nvar = bufsize/sizeof(int);
    buf = (int*)packbuf;
    for (i=0; i<nvar; i++) buf[i] = iscale*buf[i];
    new_count[0] = new_count[0]/sizeof(int);
    strided_to_subarray_dtype(new_strides, new_count, stride_levels,
        MPI_INT, &src_type);
    strided_to_subarray_dtype(dst_stride_ar, new_count, stride_levels,
        MPI_INT, &dst_type);
  } else if (op == CMX_ACC_LNG) {
    long *buf;
    long lscale = *((long*)scale);
    cmxInt nvar = bufsize/sizeof(long);
    buf = (long*)packbuf;
    for (i=0; i<nvar; i++) buf[i] = lscale*buf[i];
    new_count[0] = new_count[0]/sizeof(long);
    strided_to_subarray_dtype(new_strides, new_count, stride_levels,
        MPI_LONG, &src_type);
    strided_to_subarray_dtype(dst_stride_ar, new_count, stride_levels,
        MPI_LONG, &dst_type);
  } else if (op == CMX_ACC_FLT) {
    float *buf;
    float fscale = *((float*)scale);
    cmxInt nvar = bufsize/sizeof(float);
    buf = (float*)packbuf;
    for (i=0; i<nvar; i++) buf[i] = fscale*buf[i];
    new_count[0] = new_count[0]/sizeof(float);
    strided_to_subarray_dtype(new_strides, new_count, stride_levels,
        MPI_FLOAT, &src_type);
    strided_to_subarray_dtype(dst_stride_ar, new_count, stride_levels,
        MPI_FLOAT, &dst_type);
  } else if (op == CMX_ACC_DBL) {
    double *buf;
    double dscale = *((double*)scale);
    cmxInt nvar = bufsize/sizeof(double);
    buf = (double*)packbuf;
    for (i=0; i<nvar; i++) buf[i] = dscale*buf[i];
    new_count[0] = new_count[0]/sizeof(double);
    strided_to_subarray_dtype(new_strides, new_count, stride_levels,
        MPI_DOUBLE, &src_type);
    strided_to_subarray_dtype(dst_stride_ar, new_count, stride_levels,
        MPI_DOUBLE, &dst_type);
  } else if (op == CMX_ACC_CPL) {
    float *buf;
    float crscale = *((float*)scale);
    float ciscale = *((float*)scale+1);
    cmxInt nvar = bufsize/(2*sizeof(float));
    buf = (float*)packbuf;
    float tmp;
    for (i=0; i<nvar; i++) {
      tmp = buf[2*i];
      buf[2*i] = crscale*buf[2*i]-ciscale*buf[2*i+1];
      buf[2*i+1] = ciscale*tmp+crscale*buf[2*i+1];
    }
    new_count[0] = new_count[0]/sizeof(float);
    strided_to_subarray_dtype(new_strides, new_count, stride_levels,
        MPI_FLOAT, &src_type);
    strided_to_subarray_dtype(dst_stride_ar, new_count, stride_levels,
        MPI_FLOAT, &dst_type);
  } else if (op == CMX_ACC_DCP) {
    double *buf;
    double crscale = *((double*)scale);
    double ciscale = *((double*)scale+1);
    cmxInt nvar = bufsize/(2*sizeof(double));
    buf = (double*)packbuf;
    double tmp;
    for (i=0; i<nvar; i++) {
      tmp = buf[2*i];
      buf[2*i] = crscale*buf[2*i]-ciscale*buf[2*i+1];
      buf[2*i+1] = ciscale*tmp+crscale*buf[2*i+1];
    }
    new_count[0] = new_count[0]/sizeof(double);
    strided_to_subarray_dtype(new_strides, new_count, stride_levels,
        MPI_DOUBLE, &src_type);
    strided_to_subarray_dtype(dst_stride_ar, new_count, stride_levels,
        MPI_DOUBLE, &dst_type);
  } else {
    assert(0);
  }
  ierr = MPI_Type_commit(&src_type);
  translate_mpi_error(ierr,"cmx_accs:MPI_Type_commit");
  ierr = MPI_Type_commit(&dst_type);
  translate_mpi_error(ierr,"cmx_accs:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
  ierr = MPI_Win_flush(lproc, reg_win->win);
  translate_mpi_error(ierr,"cmx_accs:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Accumulate(packbuf,1,src_type,lproc,displ,1,dst_type,
      MPI_SUM,win);
  translate_mpi_error(ierr,"cmx_accs:MPI_Accumulate");
  ierr = MPI_Win_flush_local(lproc, win);
  translate_mpi_error(ierr,"cmx_accs:MPI_Win_flush_local");
#else
  ierr = MPI_Raccumulate(packbuf,1,src_type,lproc,displ,1,dst_type,
      MPI_SUM,win,&request);
  translate_mpi_error(ierr,"cmx_accs:MPI_Raccumulate");
  ierr = MPI_Wait(&request, &status);
  translate_mpi_error(ierr,"cmx_accs:MPI_Wait");
#endif
#else
  ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,win);
  translate_mpi_error(ierr,"cmx_accs:MPI_Win_lock");
  ierr = MPI_Accumulate(packbuf,1,src_type,lproc,displ,1,dst_type,
      MPI_SUM,win);
  translate_mpi_error(ierr,"cmx_accs:MPI_Accumulate");
  ierr = MPI_Win_unlock(lproc,win);
  translate_mpi_error(ierr,"cmx_accs:MPI_Win_unlock");
#endif
  ierr = MPI_Type_free(&src_type);
  translate_mpi_error(ierr,"cmx_accs:MPI_Type_free");
  ierr = MPI_Type_free(&dst_type);
  translate_mpi_error(ierr,"cmx_accs:MPI_Type_free");
  free(packbuf);

  return CMX_SUCCESS;
}


/**
 * Utility function to create MPI data type for vector data object
 * @param src_ptr: location of first data point in vector data object for source
 * @param iov: cmx vector struct containing data information
 * @param iov_len: number of elements in vector data object
 * @param base_type: MPI_Datatype of elements
 * @param type: returned MPI_Datatype corresponding to iov for source data
 * @param type: returned MPI_Datatype corresponding to iov for destination data
 */
void vector_to_struct_dtype(void* loc_ptr, cmx_giov_t *iov,
    int iov_len, MPI_Datatype base_type, MPI_Datatype *loc_type,
    MPI_Datatype *rem_type)
{
  cmxInt i, j, size, ierr;
  cmxInt nelems, icnt;
  cmxInt *blocklengths;
  MPI_Aint *displacements;
  MPI_Datatype *types;
  MPI_Aint displ;
  MPI_Type_size(base_type,&size);
  /* find total number of elements */
  nelems = 0;
  for (i=0; i<iov_len; i++) {
    nelems += iov[i].count;
  }
  /* allocate buffers to create data types */
  blocklengths = (cmxInt*)malloc(nelems*sizeof(cmxInt));
  displacements = (MPI_Aint*)malloc(nelems*sizeof(MPI_Aint));
  types = (MPI_Datatype*)malloc(nelems*sizeof(MPI_Datatype));
  icnt = 0;
  for (i=0; i<iov_len; i++) {
    for (j=0; j<iov[i].count; j++) {
      displ = (MPI_Aint)iov[i].loc[j]-(MPI_Aint)loc_ptr;
      blocklengths[icnt] = iov[i].bytes/size;
      displacements[icnt] = displ;
      types[icnt] = base_type;
      icnt++;
    }
  }
  ierr = MPI_Type_create_struct(nelems, blocklengths, displacements, types,
      loc_type);
  translate_mpi_error(ierr,"vector_to_struct_dtype:MPI_Type_create_struct");
  icnt = 0;
  for (i=0; i<iov_len; i++) {
    for (j=0; j<iov[i].count; j++) {
      displ = size*(MPI_Aint)iov[i].rem[j];
      displacements[icnt] = displ;
      icnt++;
    }
  }
  ierr = MPI_Type_create_struct(nelems, blocklengths, displacements, types,
      rem_type);
  translate_mpi_error(ierr,"vector_to_struct_dtype:MPI_Type_create_struct");
}

/**
 * vector put operation
 * @param iov: descriptor array
 * @param iov_len: length of descriptor array
 * @param proc: remote processor id
 * @param cmx_hdl: handle for data allocation
 */
int cmx_putv(
    cmx_giov_t *iov, cmxInt iov_len,
    int proc, cmx_handle_t cmx_hdl)
{
  MPI_Datatype src_type, dst_type;
  MPI_Aint displ;
  void *src_ptr;
  int lproc, ierr;
  MPI_Win win = cmx_hdl.win;
#ifdef USE_MPI_REQUESTS
  MPI_Request request;
  MPI_Status status;
#endif
  src_ptr = iov[0].loc[0];
  displ = 0;
  vector_to_struct_dtype(src_ptr, iov, iov_len,
      MPI_BYTE, &src_type, &dst_type);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
  ierr = MPI_Type_commit(&src_type);
  translate_mpi_error(ierr,"cmx_putv:MPI_Type_commit");
  ierr = MPI_Type_commit(&dst_type);
  translate_mpi_error(ierr,"cmx_putv:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
  ierr = MPI_Win_flush(lproc, reg_win->win);
  translate_mpi_error(ierr,"cmx_putv:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Put(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
      win);
  translate_mpi_error(ierr,"cmx_putv:MPI_Put");
  ierr = MPI_Win_flush_local(lproc, win);
  translate_mpi_error(ierr,"cmx_putv:MPI_Win_flush_local");
#else
  ierr = MPI_Rput(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
      win, &request);
  translate_mpi_error(ierr,"cmx_putv:MPI_Rput");
  ierr = MPI_Wait(&request, &status);
  translate_mpi_error(ierr,"cmx_putv:MPI_Wait");
#endif
#else
  ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,win);
  translate_mpi_error(ierr,"cmx_putv:MPI_Win_lock");
  ierr = MPI_Put(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
      win);
  translate_mpi_error(ierr,"cmx_putv:MPI_Put");
  ierr = MPI_Win_unlock(lproc,win);
  translate_mpi_error(ierr,"cmx_putv:MPI_Win_unlock");
#endif
  ierr = MPI_Type_free(&src_type);
  translate_mpi_error(ierr,"cmx_putv:MPI_Type_free");
  ierr = MPI_Type_free(&dst_type);
  translate_mpi_error(ierr,"cmx_putv:MPI_Type_free");
  return CMX_SUCCESS;
}

/**
 * Vector get operation
 * @param iov: descriptor array
 * @param iov_len: length of descriptor array
 * @param proc: remote processor id
 * @param cmx_hdl: handle for data allocation
 */
int cmx_getv(
    cmx_giov_t *iov, int iov_len,
    int proc, cmx_handle_t cmx_hdl)
{
  MPI_Datatype src_type, dst_type;
  MPI_Aint displ;
  void *dst_ptr;
  int lproc, ierr;
  MPI_Win win = cmx_hdl.win;
#ifdef USE_MPI_REQUESTS
  MPI_Request request;
  MPI_Status status;
#endif
  dst_ptr = iov[0].loc[0];
  displ = 0;
  vector_to_struct_dtype(dst_ptr, iov, iov_len,
      MPI_BYTE, &dst_type, &src_type);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
  ierr = MPI_Type_commit(&src_type);
  translate_mpi_error(ierr,"cmx_getv:MPI_Type_commit");
  ierr = MPI_Type_commit(&dst_type);
  translate_mpi_error(ierr,"cmx_getv:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
  ierr = MPI_Win_flush(lproc, reg_win->win);
  translate_mpi_error(ierr,"cmx_getv:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Get(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
      win);
  translate_mpi_error(ierr,"cmx_getv:MPI_Get");
  ierr = MPI_Win_flush_local(lproc, win);
  translate_mpi_error(ierr,"cmx_getv:MPI_Win_flush_local");
#else
  ierr = MPI_Rget(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
      win, &request);
  translate_mpi_error(ierr,"cmx_getv:MPI_Rget");
  ierr = MPI_Wait(&request, &status);
  translate_mpi_error(ierr,"cmx_getv:MPI_Wait");
#endif
#else
  ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,win);
  translate_mpi_error(ierr,"cmx_getv:MPI_Win_lock");
  ierr = MPI_Get(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
      win);
  translate_mpi_error(ierr,"cmx_getv:MPI_Get");
  ierr = MPI_Win_unlock(lproc,win);
  translate_mpi_error(ierr,"cmx_getv:MPI_Win_unlock");
#endif
  ierr = MPI_Type_free(&src_type);
  translate_mpi_error(ierr,"cmx_getv:MPI_Type_free");
  ierr = MPI_Type_free(&dst_type);
  translate_mpi_error(ierr,"cmx_getv:MPI_Type_free");
  return CMX_SUCCESS;
}

/**
 * Utility function to create MPI data type for vector data object used in
 * vector accumulate operation. This also handles creation of a temporary buffer
 * to handle scaling of the accumulated values. The returned buffer must be
 * deallocated using free
 * @param loc_ptr: location of first data point in vector data object for
 *        local data
 * @param iov: cmx vector struct containing data information
 * @param iov_len: number of elements in vector data object
 * @param base_type: MPI_Datatype of elements
 * @param src_type: returned MPI_Datatype corresponding to iov for source data
 * @param dst_type: returned MPI_Datatype corresponding to iov for destination data
 */
void* create_vector_buf_and_dtypes(void *loc_ptr,
    cmx_giov_t *iov, int iov_len, int cmx_size,
    void *scale, MPI_Datatype base_type,
    MPI_Datatype *src_type, MPI_Datatype *dst_type)
{
  int i, j, size, ierr;
  int nelems, icnt, ratio;
  void* src_buf;
  int *blocklengths;
  MPI_Aint *displacements;
  MPI_Datatype *types;
  MPI_Aint displ;
  MPI_Type_size(base_type,&size);
  /* find total number of elements */
  nelems = 0;
  for (i=0; i<iov_len; i++) {
    nelems += iov[i].count;
  }
  /* create temporary buffers for scaled accumulate values */
  MPI_Type_size(base_type, &size);
  ratio = cmx_size/size;
  /*
  printf("p[%d] Ratio: %d cmx_size: %d size: %d\n",l_state.rank,
      ratio,cmx_size,size);
  */
  src_buf=malloc(nelems*cmx_size);
  if (base_type == MPI_INT) {
    int *buf = (int*)src_buf;
    int iscale = *((int*)scale);
    icnt = 0;
    for (i=0; i<iov_len; i++) {
      for (j=0; j<iov[i].count; j++) {
        buf[icnt] = *((int*)(iov[i].loc[j]));
        buf[icnt] *= iscale;
        icnt++;
      }
    }
  } else if (base_type == MPI_LONG) {
    long *buf = (long*)src_buf;
    long lscale = *((long*)scale);
    icnt = 0;
    for (i=0; i<iov_len; i++) {
      for (j=0; j<iov[i].count; j++) {
        buf[icnt] = *((long*)(iov[i].loc[j]));
        buf[icnt] *= lscale;
        icnt++;
      }
    }
  } else if (base_type == MPI_FLOAT && ratio == 1) {
    float *buf = (float*)src_buf;
    float fscale = *((float*)scale);
    icnt = 0;
    for (i=0; i<iov_len; i++) {
      for (j=0; j<iov[i].count; j++) {
        buf[icnt] = *((float*)(iov[i].loc[j]));
        buf[icnt] *= fscale;
        icnt++;
      }
    }
  } else if (base_type == MPI_DOUBLE && ratio == 1) {
    double *buf = (double*)src_buf;
    double dscale = *((double*)scale);
    icnt = 0;
    for (i=0; i<iov_len; i++) {
      for (j=0; j<iov[i].count; j++) {
        buf[icnt] = *((double*)(iov[i].loc[j]));
        buf[icnt] *= dscale;
        icnt++;
      }
    }
  } else if (base_type == MPI_FLOAT && ratio == 2) {
    float *buf = (float*)src_buf;
    float crscale = *((float*)scale);
    float ciscale = *((float*)scale+1);
    float rval, ival;
    icnt = 0;
    for (i=0; i<iov_len; i++) {
      for (j=0; j<iov[i].count; j++) {
        rval = *((float*)(iov[i].loc[j]));
        ival = *((float*)(iov[i].loc[j])+1);
        buf[2*icnt] = rval*crscale-ival*ciscale;
        buf[2*icnt+1] = rval*ciscale+ival*crscale;
        icnt++;
      }
    }
  } else if (base_type == MPI_DOUBLE && ratio == 2) {
    double *buf = (double*)src_buf;
    double crscale = *((double*)scale);
    double ciscale = *((double*)scale+1);
    double rval, ival;
    icnt = 0;
    for (i=0; i<iov_len; i++) {
      for (j=0; j<iov[i].count; j++) {
        rval = *((double*)(iov[i].loc[j]));
        ival = *((double*)(iov[i].loc[j])+1);
        buf[2*icnt] = rval*crscale-ival*ciscale;
        buf[2*icnt+1] = rval*ciscale+ival*crscale;
        icnt++;
      }
    }
  }

  /* allocate buffers to create data types */
  blocklengths = (int*)malloc(ratio*nelems*sizeof(int));
  displacements = (MPI_Aint*)malloc(ratio*nelems*sizeof(MPI_Aint));
  types = (MPI_Datatype*)malloc(ratio*nelems*sizeof(MPI_Datatype));
  if (base_type == MPI_INT) {
    int *buf;
    buf = (int*)src_buf;
    icnt = 0;
    for (i=0; i<iov_len; i++) {
      for (j=0; j<iov[i].count; j++) {
        displ = (MPI_Aint)(buf+icnt)-(MPI_Aint)buf;
        blocklengths[icnt] = 1;
        displacements[icnt] = displ;
        types[icnt] = base_type;
        icnt++;
      }
    }
  } else if (base_type == MPI_LONG) {
    long *buf;
    buf = (long*)src_buf;
    icnt = 0;
    for (i=0; i<iov_len; i++) {
      for (j=0; j<iov[i].count; j++) {
        displ = (MPI_Aint)(buf+icnt)-(MPI_Aint)buf;
        blocklengths[icnt] = 1;
        displacements[icnt] = displ;
        types[icnt] = base_type;
        icnt++;
      }
    }
  } else if (base_type == MPI_FLOAT && ratio == 1) {
    float *buf;
    buf = (float*)src_buf;
    icnt = 0;
    for (i=0; i<iov_len; i++) {
      for (j=0; j<iov[i].count; j++) {
        displ = (MPI_Aint)(buf+icnt)-(MPI_Aint)buf;
        blocklengths[icnt] = 1;
        displacements[icnt] = displ;
        types[icnt] = base_type;
        icnt++;
      }
    }
  } else if (base_type == MPI_DOUBLE && ratio == 1) {
    double *buf;
    buf = (double*)src_buf;
    icnt = 0;
    for (i=0; i<iov_len; i++) {
      for (j=0; j<iov[i].count; j++) {
        displ = (MPI_Aint)(buf+icnt)-(MPI_Aint)buf;
        blocklengths[icnt] = 1;
        displacements[icnt] = displ;
        types[icnt] = base_type;
        icnt++;
      }
    }
  } else if (base_type == MPI_FLOAT && ratio == 2) {
    float *buf;
    buf = (float*)src_buf;
    icnt = 0;
    for (i=0; i<iov_len; i++) {
      for (j=0; j<iov[i].count; j++) {
        displ = (MPI_Aint)(buf+2*icnt)-(MPI_Aint)buf;
        blocklengths[icnt] = 2;
        displacements[icnt] = displ;
        types[icnt] = base_type;
        icnt++;
      }
    }
  } else if (base_type == MPI_DOUBLE && ratio == 2) {
    double *buf;
    buf = (double*)src_buf;
    icnt = 0;
    for (i=0; i<iov_len; i++) {
      for (j=0; j<iov[i].count; j++) {
        displ = (MPI_Aint)(buf+2*icnt)-(MPI_Aint)buf;
        blocklengths[icnt] = 2;
        displacements[icnt] = displ;
        types[icnt] = base_type;
        icnt++;
      }
    }
  }
  ierr = MPI_Type_create_struct(nelems, blocklengths, displacements, types,
      src_type);
  translate_mpi_error(ierr,"create_vector_buf_and_dtypes:MPI_Type_create_struct");
  icnt = 0;
  for (i=0; i<iov_len; i++) {
    for (j=0; j<iov[i].count; j++) {
      displ = (MPI_Aint)iov[i].rem[j];
      displacements[icnt] = displ;
      icnt++;
    }
  }
  ierr = MPI_Type_create_struct(nelems, blocklengths, displacements, types,
      dst_type);
  translate_mpi_error(ierr,"create_vector_buf_and_dtypes:MPI_Type_create_struct");
  return src_buf;
}

/**
 * Vector accumulate operation
 * @param op: operation
 * @param scale: scale factor in accumulate operation
 * @param iov: descriptor array
 * @param iov_len: length of descriptor array
 * @param proc: remote processor id
 * @param cmx_hdl: handle for data allocation
 */
int cmx_accv(
    int op, void *scale,
    cmx_giov_t *iov, int iov_len,
    int proc, cmx_handle_t cmx_hdl)
{
  MPI_Datatype src_type, dst_type;
  MPI_Aint displ;
  MPI_Datatype base_type;
  void *ptr, *src_ptr;
  int ierr;
#ifdef USE_MPI_REQUESTS
  MPI_Request request;
  MPI_Status status;
#endif
  MPI_Win win = cmx_hdl.win;
  int lproc, size;
  if (op == CMX_ACC_INT) {
    size = sizeof(int);
    base_type = MPI_INT;
  } else if (op == CMX_ACC_LNG) {
    size = sizeof(long);
    base_type = MPI_LONG;
  } else if (op == CMX_ACC_FLT) {
    size = sizeof(float);
    base_type = MPI_FLOAT;
  } else if (op == CMX_ACC_DBL) {
    size = sizeof(double);
    base_type = MPI_DOUBLE;
  } else if (op == CMX_ACC_CPL) {
    size = 2*sizeof(float);
    base_type = MPI_FLOAT;
  } else if (op == CMX_ACC_DCP) {
    size = 2*sizeof(double);
    base_type = MPI_DOUBLE;
  }
  ptr = cmx_hdl.buf;
  displ = 0;
  src_ptr = create_vector_buf_and_dtypes(ptr, iov,
      iov_len, size, scale, base_type, &src_type, &dst_type);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
  ierr = MPI_Type_commit(&src_type);
  translate_mpi_error(ierr,"cmx_accv:MPI_Type_commit");
  ierr = MPI_Type_commit(&dst_type);
  translate_mpi_error(ierr,"cmx_accv:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
  ierr = MPI_Win_flush(lproc, reg_win->win);
  translate_mpi_error(ierr,"cmx_getv:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Accumulate(src_ptr,1,src_type,lproc,displ,1,dst_type,
      MPI_SUM,win);
  translate_mpi_error(ierr,"cmx_accv:MPI_Accumulate");
  ierr = MPI_Win_flush_local(lproc, win);
  translate_mpi_error(ierr,"cmx_accv:MPI_Win_flush_local");
#else
  ierr = MPI_Raccumulate(src_ptr,1,src_type,lproc,displ,1,dst_type,
      MPI_SUM,win,&request);
  translate_mpi_error(ierr,"cmx_accv:MPI_Raccumulate");
  ierr = MPI_Wait(&request, &status);
  translate_mpi_error(ierr,"cmx_accv:MPI_Wait");
#endif
#else
  ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,win);
  translate_mpi_error(ierr,"cmx_accv:MPI_Win_lock");
  ierr = MPI_Accumulate(src_ptr,1,src_type,lproc,displ,1,dst_type,
      MPI_SUM,win);
  translate_mpi_error(ierr,"cmx_accv:MPI_Accumulate");
  ierr = MPI_Win_unlock(lproc,win);
  translate_mpi_error(ierr,"cmx_accv:MPI_Win_unlock");
#endif
  ierr = MPI_Type_free(&src_type);
  translate_mpi_error(ierr,"cmx_accv:MPI_Type_free");
  ierr = MPI_Type_free(&dst_type);
  translate_mpi_error(ierr,"cmx_accv:MPI_Type_free");
  free(src_ptr);
  return CMX_SUCCESS;
}


/**
 * Flush all outgoing messages to all procs in group
 * @param group: group containing processors
 */
int cmx_fence_all(cmx_group_t group)
{
  return cmx_wait_all(group);
}


/**
 * Flush all messages from me going to processor proc
 * @param proc: target processor
 * @param group: processor group
 */
int cmx_fence_proc(int proc, cmx_group_t group)
{
  cmx_igroup_t *igroup = group;
  win_link_t *curr_win;
  int ierr;
  if (igroup != NULL) {
    curr_win = igroup->win_list;
    while (curr_win != NULL) {
      ierr = MPI_Win_flush(proc, curr_win->win);
      translate_mpi_error(ierr,"cmx_fence_proc:MPI_Win_flush");
      curr_win = curr_win->next;
    }
  }
}


/**
 * cmx_barrier is cmx_fence_all + MPI_Barrier
 * @param group: processor group
 */
int cmx_barrier(cmx_group_t group)
{
  MPI_Comm comm;

  cmx_fence_all(group);
  MPI_Barrier(group->comm);

  return CMX_SUCCESS;
}

/**
 * local allocation of registered memory
 * @param size: size in bytes of requested segment
 */
void *cmx_malloc_local(size_t size)
{
  void *ptr;
  if (size == 0) {
    ptr = NULL;
  } else {
    MPI_Aint tsize;
    int ierr;
    tsize = size;
    ierr = MPI_Alloc_mem(tsize,MPI_INFO_NULL,&ptr);
    translate_mpi_error(ierr,"cmx_malloc_local:MPI_Alloc_mem");
  }
  return ptr;
}


/**
 * free segment of registered memory
 * @param ptr: pointer to start of registered memory segment
 */
int cmx_free_local(void *ptr)
{
  if (ptr != NULL) {
    int ierr;
    ierr = MPI_Free_mem(ptr);
    translate_mpi_error(ierr,"cmx_free_local:MPI_Free_mem");
  }

  return CMX_SUCCESS;
}

/**
 * Terminate cmx and clean up resources
 */
int cmx_finalize()
{
  int i, ierr;
  /* it's okay to call multiple times -- extra calls are no-ops */
  if (!initialized) {
    return CMX_SUCCESS;
  }

  initialized = 0;

  /* Make sure that all outstanding operations are done */
  cmx_wait_all(CMX_GROUP_WORLD);

  /* groups */
  cmx_group_finalize();

  MPI_Barrier(l_state.world_comm);

  /* destroy the communicators */
#if 1
  ierr = MPI_Comm_free(&l_state.world_comm);
  translate_mpi_error(ierr,"cmx_finalize:MPI_Comm_free");
#endif

  /* Clean up request list */
#ifdef ZUSE_MPI_REQUESTS
  for (i=0; i<nb_max_outstanding; i++) {
    free(nb_list[i]);
  }
  free(nb_list);
#endif

  return CMX_SUCCESS;
}


/**
 * Wait for all non-blocking calls on a particular proc to a particular proc
 * to finish locally (buffers are available for reuse).
 * @param proc: target processor
 * @param group: processor group
 */
int cmx_wait_proc(int proc, cmx_group_t group)
{
  assert(0);

  return CMX_SUCCESS;
}


/**
 * Wait for completion of non-blocking operation associated with handle req
 * @param req: non-blocking request handle
 */
int cmx_wait(cmx_request_t* req)
{
  int ierr;
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Win_flush_local(req->remote_proc,req->win);
  translate_mpi_error(ierr,"cmx_wait:MPI_Win_flush_local");
#else
  MPI_Status status;
  ierr = MPI_Wait(&(req->request),&status);
  translate_mpi_error(ierr,"cmx_wait:MPI_Wait");
#endif
  req->active = 0;
#else
  /* Non-blocking functions not implemented */
#endif
  return CMX_SUCCESS;
}


/**
 * Check compoletion status of non-blocking request. Returns true if request
 * has completed
 * @param req: request handle
 * @param status: returns 0 if complete, 1 otherwise
 */
int cmx_test(cmx_request_t *req, int *status)
{
#ifdef USE_MPI_REQUESTS
  int flag;
  MPI_Status stat;
  MPI_Test(&req->request,&flag,&stat);
  if (flag) {
    *status = 0;
  } else {
    *status = 1;
  }
  return CMX_SUCCESS;
#else
  *status = 0;
  return CMX_SUCCESS;
#endif
}


/**
 * Wait for all non-blocking operations on the group to finish locally
 * @param proc: processor group
 */
int cmx_wait_all(cmx_group_t group)
{
  cmx_igroup_t *igroup = group;
  win_link_t *curr_win;
  if (igroup != NULL) {
    curr_win = igroup->win_list;
    while (curr_win != NULL) {
#ifdef USE_MPI_REQUESTS
      MPI_Win_flush_all(curr_win->win);
#else
      MPI_Win_fence(0,curr_win->win);
#endif
      curr_win = curr_win->next;
    }
    return CMX_SUCCESS;
  }
  return CMX_FAILURE;
}

/**
 * Non-blocking contiguous put
 *
 * @param src: pointer to source buffer
 * @param dst_offset: offset from start of data allocation on remote processor
 * @param bytes: number of bytes in data transfer
 * @param proc: remote processor id
 * @param cmx_hdl: handle for data allocation
 * @param req: non-blocking request handle
 */
int cmx_nbput(
    void *src, cmxInt dst_offset, cmxInt bytes,
    int proc, cmx_handle_t cmx_hdl,
    cmx_request_t *req)
{
#ifdef USE_MPI_REQUESTS
  if (req == NULL) {
    return cmx_put(src, dst_offset, bytes, proc, cmx_hdl);
  }
  MPI_Aint displ;
  void *ptr;
  int lproc, ierr;
  MPI_Win win = cmx_hdl.win;
  MPI_Request request;
  ptr = cmx_hdl.buf;
  displ = (MPI_Aint)(dst_offset);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Put(src, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
      win);
  translate_mpi_error(ierr,"cmx_nbput:MPI_Put");
  req->remote_proc = lproc;
  req->win = win;
#else
  ierr = MPI_Rput(src, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
         win, &request);
  translate_mpi_error(ierr,"cmx_nbput:MPI_Rput");
#endif
  req->request = request;
  req->active = 1;
  return CMX_SUCCESS;
#else
  return cmx_put(src, dst_offset, bytes, proc, cmx_hdl);
#endif
}


/**
 * Non-blocking contiguous put
 *
 * @param dst: pointer to local buffer
 * @param src_offset: offset from start of data allocation on remote processor
 * @param bytes: number of bytes in data transfer
 * @param proc: remote processor id
 * @param cmx_hdl: handle for data allocation
 * @param req: non-blocking request handle
 */
int cmx_nbget(
    void *dst, cmxInt src_offset, int bytes,
    int proc, cmx_handle_t cmx_hdl,
    cmx_request_t *req)
{
#ifdef USE_MPI_REQUESTS
  if (req == NULL) {
    return cmx_get(dst, src_offset, bytes, proc, cmx_hdl);
  }
  MPI_Aint displ;
  void *ptr;
  int lproc, ierr;
  MPI_Win win = cmx_hdl.win;
  MPI_Request request;
  ptr = cmx_hdl.buf;
  displ = (MPI_Aint)(src_offset);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Get(dst, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
      win);
  translate_mpi_error(ierr,"cmx_nbget:MPI_Get");
  req->remote_proc = lproc;
  req->win = win;
#else
  ierr = MPI_Rget(dst, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
      win, &request);
  translate_mpi_error(ierr,"cmx_nbget:MPI_Rget");
#endif
  req->request = request;
  req->active = 1;
  return CMX_SUCCESS;
#else
  return cmx_get(dst, src_offset, bytes, proc, cmx_hdl);
#endif
}


/**
 * Non-blocking contiguous accumulate
 *
 * @param op: operation
 * @param scale: scale factor x += scale*y
 * @param src_ptr: pointer to source buffer
 * @param dst_offset: offset from start of data allocation on remote processor
 * @param bytes: number of bytes in data transfer
 * @param proc: remote processor id
 * @param cmx_hdl: handle for data allocation
 * @param req: non-blocking request handle
 */
int cmx_nbacc(
    int op, void *scale,
    void *src_ptr, cmxInt dst_offset, cmxInt bytes,
    int proc, cmx_handle_t cmx_hdl,
    cmx_request_t *req)
{
#ifdef USE_MPI_REQUESTS
  if (req == NULL) {
    return cmx_acc( op, scale,
        src_ptr, dst_offset, bytes, proc, cmx_hdl);
  }
  MPI_Aint displ;
  void *ptr;
  int count, i, lproc, ierr;
  MPI_Win win = cmx_hdl.win;
  MPI_Request request;
  MPI_Status status;
  ptr = cmx_hdl.buf;
  displ = (MPI_Aint)(dst_offset);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
  if (op == CMX_ACC_INT) {
    int *buf;
    int *isrc = (int*)src_ptr;
    int iscale = *((int*)scale);
    count = bytes/sizeof(int);
    buf = (int*)malloc(bytes);
    for (i=0; i<count; i++) {
      buf[i] = isrc[i]*iscale;
    }
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Accumulate(buf,count,MPI_INT,lproc,displ,count,
        MPI_INT,MPI_SUM,win);
    translate_mpi_error(ierr,"cmx_nbacc:MPI_Accumulate");
    req->remote_proc = lproc;
    req->win = win;
#else
    ierr = MPI_Raccumulate(buf,count,MPI_INT,lproc,displ,count,
        MPI_INT,MPI_SUM,win,&request);
    translate_mpi_error(ierr,"cmx_nbacc:MPI_Raccumulate");
#endif
    req->request = request;
    req->active = 1;
    free(buf);
  } else if (op == CMX_ACC_LNG) {
    long *buf;
    long *lsrc = (long*)src_ptr;
    long lscale = *((long*)scale);
    count = bytes/sizeof(long);
    buf = (long*)malloc(bytes);
    for (i=0; i<count; i++) {
      buf[i] = lsrc[i]*lscale;
    }
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Accumulate(buf,count,MPI_LONG,lproc,displ,count,
        MPI_LONG,MPI_SUM,win);
    translate_mpi_error(ierr,"cmx_nbacc:MPI_Accumulate");
    req->remote_proc = lproc;
    req->win = win;
#else
    ierr = MPI_Raccumulate(buf,count,MPI_LONG,lproc,displ,count,
        MPI_LONG,MPI_SUM,win,&request);
    translate_mpi_error(ierr,"cmx_nbacc:MPI_Raccumulate");
#endif
    req->request = request;
    req->active = 1;
    free(buf);
  } else if (op == CMX_ACC_FLT) {
    float *buf;
    float *fsrc = (float*)src_ptr;
    float fscale = *((float*)scale);
    count = bytes/sizeof(float);
    buf = (float*)malloc(bytes);
    for (i=0; i<count; i++) {
      buf[i] = fsrc[i]*fscale;
    }
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Accumulate(buf,count,MPI_FLOAT,lproc,displ,count,
        MPI_FLOAT,MPI_SUM,win);
    translate_mpi_error(ierr,"cmx_nbacc:MPI_Accumulate");
    req->remote_proc = lproc;
    req->win = win;
#else
    ierr = MPI_Raccumulate(buf,count,MPI_FLOAT,lproc,displ,count,
        MPI_FLOAT,MPI_SUM,win,&request);
    translate_mpi_error(ierr,"cmx_nbacc:MPI_Raccumulate");
    req->request = request;
#endif
    req->active = 1;
    free(buf);
  } else if (op == CMX_ACC_DBL) {
    double *buf;
    double *dsrc = (double*)src_ptr;
    double dscale = *((double*)scale);
    count = bytes/sizeof(double);
    buf = (double*)malloc(bytes);
    for (i=0; i<count; i++) {
      buf[i] = dsrc[i]*dscale;
    }
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Accumulate(buf,count,MPI_DOUBLE,lproc,displ,count,
        MPI_DOUBLE,MPI_SUM,win);
    translate_mpi_error(ierr,"cmx_nbacc:MPI_Accumulate");
    req->remote_proc = lproc;
    req->win = win;
#else
    ierr = MPI_Raccumulate(buf,count,MPI_DOUBLE,lproc,displ,count,
        MPI_DOUBLE,MPI_SUM,win,&request);
    translate_mpi_error(ierr,"cmx_nbacc:MPI_Raccumulate");
    req->request = request;
#endif
    req->active = 1;
    free(buf);
  } else if (op == CMX_ACC_CPL) {
    int cnum;
    float *buf;
    float *csrc = (float*)src_ptr;
    float crscale = *((float*)scale);
    float ciscale = *((float*)scale+1);
    count = bytes/sizeof(float);
    cnum = count/2;
    buf = (float*)malloc(bytes);
    for (i=0; i<cnum; i++) {
      buf[2*i] = csrc[2*i]*crscale-csrc[2*i+1]*ciscale;
      buf[2*i+1] = csrc[2*i]*ciscale+csrc[2*i+1]*crscale;
    }
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Accumulate(buf,count,MPI_FLOAT,lproc,displ,count,
        MPI_FLOAT,MPI_SUM,win);
    translate_mpi_error(ierr,"cmx_nbacc:MPI_Accumulate");
    req->remote_proc = lproc;
    req->win = win;
#else
    ierr = MPI_Raccumulate(buf,count,MPI_FLOAT,lproc,displ,count,
        MPI_FLOAT,MPI_SUM,win,&request);
    translate_mpi_error(ierr,"cmx_nbacc:MPI_Raccumulate");
#endif
    req->request = request;
    req->active = 1;
    free(buf);
  } else if (op == CMX_ACC_DCP) {
    int cnum;
    double *buf;
    double *csrc = (double*)src_ptr;
    double crscale = *((double*)scale);
    double ciscale = *((double*)scale+1);
    count = bytes/sizeof(double);
    cnum = count/2;
    buf = (double*)malloc(bytes);
    for (i=0; i<cnum; i++) {
      buf[2*i] = csrc[2*i]*crscale-csrc[2*i+1]*ciscale;
      buf[2*i+1] = csrc[2*i]*ciscale+csrc[2*i+1]*crscale;
    }
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Accumulate(buf,count,MPI_DOUBLE,lproc,displ,count,
        MPI_DOUBLE,MPI_SUM,win);
    translate_mpi_error(ierr,"cmx_nbacc:MPI_Accumulate");
    req->remote_proc = lproc;
    req->win = win;
#else
    ierr = MPI_Raccumulate(buf,count,MPI_DOUBLE,lproc,displ,count,
        MPI_DOUBLE,MPI_SUM,win,&request);
    translate_mpi_error(ierr,"cmx_nbacc:MPI_Raccumulate");
#endif
    req->request = request;
    req->active = 1;
    free(buf);
  } else {
    assert(0);
  }
  return CMX_SUCCESS;
#else
  return cmx_acc( op, scale,
      src_ptr, dst_offset, bytes, proc, cmx_hdl);
#endif
}


/**
 * non-blocking strided put
 * @param src: pointer to origin of data on source processor
 * @param src_stride: physical dimensions of array containing source data
 * @param dst_offset: offset from start of data allocation on remote process
 * @param dst_stride: physical dimensions of destination array
 * @param count: array containing number of data points along each dimension. The 0
 * dimension contains the actual length of contiguous segments in bytes (number
 * of elements times the size of each element).
 * @param stride_levels: this should be one less than the dimension of the array
 * @param proc: global rank of destination array
 * @param cmx_hdl: handle for data allocation
 * @param req: non-blocking request handle
 */
int cmx_nbputs(
    void *src, cmxInt *src_stride,
    cmxInt dst_offset, cmxInt *dst_stride,
    cmxInt *count, int stride_levels, 
    int proc, cmx_handle_t cmx_hdl,
    cmx_request_t *req)
{
#ifdef USE_MPI_REQUESTS
  if (req == NULL) {
    return cmx_puts(src, src_stride, dst_offset, dst_stride,
        count, stride_levels, proc, cmx_hdl);
  }
  MPI_Datatype src_type, dst_type;
  MPI_Aint displ;
  void *ptr;
  int lproc, ierr;
  MPI_Win win = cmx_hdl.win;
  MPI_Request request;
  MPI_Status status;
  ptr = cmx_hdl.buf;
  displ = (MPI_Aint)(dst_offset);

  strided_to_subarray_dtype(src_stride, count, stride_levels,
      MPI_BYTE, &src_type);
  strided_to_subarray_dtype(dst_stride, count, stride_levels,
      MPI_BYTE, &dst_type);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
  MPI_Type_commit(&src_type);
  MPI_Type_commit(&dst_type);
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Put(src, 1, src_type, lproc, displ, 1, dst_type,
      win);
  translate_mpi_error(ierr,"cmx_nbputs:MPI_Put");
  req->remote_proc = lproc;
  req->win = win;
#else
  ierr = MPI_Rput(src, 1, src_type, lproc, displ, 1, dst_type,
      win, &request);
  translate_mpi_error(ierr,"cmx_nbputs:MPI_Rput");
#endif
  req->request = request;
  req->active = 1;
  MPI_Type_free(&src_type);
  MPI_Type_free(&dst_type);
  return CMX_SUCCESS;
#else
  return cmx_puts(src, src_stride, dst_offset, dst_stride,
      count, stride_levels, proc, cmx_hdl);
#endif
}


/**
 * non-blocking strided get
 * @param dst_ptr: pointer to origin of data on destination processor
 * @param dst_stride: physical dimensions of destination array
 * @param src_offset: offset from start of data allocation on remote process
 * @param src_stride: physical dimensions of array containing source data
 * @param count: array containing number of data points along each dimension. The 0
 * dimension contains the actual length of contiguous segments in bytes (number
 * of elements times the size of each element).
 * @param stride_levels: this should be one less than the dimension of the array
 * @param proc: global rank of source array
 * @param cmx_hdl: handle for data allocation
 * @param req: non-blocking request handle
 */
int cmx_nbgets(
    void *dst, cmxInt *dst_stride,
    cmxInt src_offset, cmxInt *src_stride,
    cmxInt *count, int stride_levels, 
    int proc, cmx_handle_t cmx_hdl,
    cmx_request_t *req) 
{
#ifdef USE_MPI_REQUESTS
  if (req == NULL) {
    return cmx_gets(dst, dst_stride, src_offset, src_stride,
        count, stride_levels, proc, cmx_hdl);
  }
  MPI_Datatype src_type, dst_type;
  MPI_Aint displ;
  void *ptr;
  int lproc, ierr;
  MPI_Win win = cmx_hdl.win;
  MPI_Request request;
  MPI_Status status;
  ptr = cmx_hdl.buf;
  displ = (MPI_Aint)(src_offset);

  strided_to_subarray_dtype(src_stride, count, stride_levels,
      MPI_BYTE, &src_type);
  strided_to_subarray_dtype(dst_stride, count, stride_levels,
      MPI_BYTE, &dst_type);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
  MPI_Type_commit(&src_type);
  MPI_Type_commit(&dst_type);
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Get(dst, 1, dst_type, lproc, displ, 1, src_type,
      win);
  translate_mpi_error(ierr,"cmx_nbgets:MPI_Get");
  req->remote_proc = lproc;
  req->win = win;
#else
  ierr = MPI_Rget(dst, 1, dst_type, lproc, displ, 1, src_type,
      win, &request);
  translate_mpi_error(ierr,"cmx_nbgets:MPI_Rget");
#endif
  req->request = request;
  req->active = 1;
  MPI_Type_free(&src_type);
  MPI_Type_free(&dst_type);
  return CMX_SUCCESS;
#else
  return cmx_gets(dst, dst_stride, src_offset, src_stride,
      count, stride_levels, proc, cmx_hdl);
#endif
}


/**
 * non-blocking strided accumulate
 * @param op: operation
 * @param scale: scale factor x += scale*y
 * @param src: pointer to origin of data on source processor
 * @param src_stride: physical dimensions of array containing source data
 * @param dst_offset: offset from start of data allocation on remote process
 * @param dst_stride: physical dimensions of destination array
 * @param count: array containing number of data points along each dimension. The 0
 * dimension contains the actual length of contiguous segments in bytes (number
 * of elements times the size of each element).
 * @param stride_levels: this should be one less than the dimension of the array
 * @param proc: global rank of destination array
 * @param cmx_hdl: handle for data allocation
 * @param req: non-blocking request handle
 */
int cmx_nbaccs(
    int op, void *scale,
    void *src, cmxInt *src_stride,
    cmxInt dst_offset, cmxInt *dst_stride,
    cmxInt *count, int stride_levels,
    int proc, cmx_handle_t cmx_hdl,
    cmx_request_t *req)
{
#ifdef USE_MPI_REQUESTS
  if (req == NULL) {
    return cmx_accs(op, scale,
        src, src_stride, dst_offset, dst_stride,
        count, stride_levels, proc, cmx_hdl);
  }
  MPI_Datatype src_type, dst_type;
  MPI_Aint displ;
  void *ptr;
  int lproc, i, ierr;
  void *packbuf;
  int bufsize;
  int new_strides[7];
  MPI_Win win = cmx_hdl.win;
  MPI_Request request;
  MPI_Status status;
  ptr = cmx_hdl.buf;
  displ = (MPI_Aint)(dst_offset);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
  packbuf = malloc_strided_acc_buffer(src, src_stride, count,
      stride_levels, &bufsize, new_strides);

  if (op == CMX_ACC_INT) {
    int *buf;
    int iscale = *((int*)scale);
    int nvar = bufsize/sizeof(int);
    buf = (int*)packbuf;
    for (i=0; i<nvar; i++) buf[i] = iscale*buf[i];
    strided_to_subarray_dtype(new_strides, count, stride_levels,
        MPI_INT, &src_type);
    strided_to_subarray_dtype(dst_stride, count, stride_levels,
        MPI_INT, &dst_type);
  } else if (op == CMX_ACC_LNG) {
    long *buf;
    long lscale = *((long*)scale);
    int nvar = bufsize/sizeof(long);
    buf = (long*)packbuf;
    for (i=0; i<nvar; i++) buf[i] = lscale*buf[i];
    strided_to_subarray_dtype(new_strides, count, stride_levels,
        MPI_LONG, &src_type);
    strided_to_subarray_dtype(dst_stride, count, stride_levels,
        MPI_LONG, &dst_type);
  } else if (op == CMX_ACC_FLT) {
    float *buf;
    float fscale = *((float*)scale);
    int nvar = bufsize/sizeof(float);
    buf = (float*)packbuf;
    for (i=0; i<nvar; i++) buf[i] = fscale*buf[i];
    strided_to_subarray_dtype(new_strides, count, stride_levels,
        MPI_FLOAT, &src_type);
    strided_to_subarray_dtype(dst_stride, count, stride_levels,
        MPI_FLOAT, &dst_type);
  } else if (op == CMX_ACC_DBL) {
    double *buf;
    double dscale = *((double*)scale);
    int nvar = bufsize/sizeof(double);
    buf = (double*)packbuf;
    for (i=0; i<nvar; i++) buf[i] = dscale*buf[i];
    strided_to_subarray_dtype(new_strides, count, stride_levels,
        MPI_DOUBLE, &src_type);
    strided_to_subarray_dtype(dst_stride, count, stride_levels,
        MPI_DOUBLE, &dst_type);
  } else if (op == CMX_ACC_CPL) {
    float *buf;
    float crscale = *((float*)scale);
    float ciscale = *((float*)scale+1);
    int nvar = bufsize/(2*sizeof(float));
    buf = (float*)packbuf;
    for (i=0; i<nvar; i++) {
      buf[2*i] = crscale*buf[2*i]-ciscale*buf[2*i+1];
      buf[2*i+1] = ciscale*buf[2*i]+crscale*buf[2*i+1];
    }
    strided_to_subarray_dtype(new_strides, count, stride_levels,
        MPI_FLOAT, &src_type);
    strided_to_subarray_dtype(dst_stride, count, stride_levels,
        MPI_FLOAT, &dst_type);
  } else if (op == CMX_ACC_DCP) {
    double *buf;
    double crscale = *((double*)scale);
    double ciscale = *((double*)scale+1);
    int nvar = bufsize/(2*sizeof(double));
    buf = (double*)packbuf;
    for (i=0; i<nvar; i++) {
      buf[2*i] = crscale*buf[2*i]-ciscale*buf[2*i+1];
      buf[2*i+1] = ciscale*buf[2*i]+crscale*buf[2*i+1];
    }
    strided_to_subarray_dtype(new_strides, count, stride_levels,
        MPI_DOUBLE, &src_type);
    strided_to_subarray_dtype(dst_stride, count, stride_levels,
        MPI_DOUBLE, &dst_type);
  } else {
    assert(0);
  }
  MPI_Type_commit(&src_type);
  MPI_Type_commit(&dst_type);
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Accumulate(packbuf,1,src_type,lproc,displ,1,dst_type,
      MPI_SUM,win);
  translate_mpi_error(ierr,"cmx_nbaccs:MPI_Accumulate");
  req->remote_proc = lproc;
  req->win = win;
#else
  ierr = MPI_Raccumulate(packbuf,1,src_type,lproc,displ,1,dst_type,
      MPI_SUM,win,&request);
  translate_mpi_error(ierr,"cmx_nbaccs:MPI_Rget");
#endif
  req->request = request;
  req->active = 1;
  MPI_Type_free(&src_type);
  MPI_Type_free(&dst_type);
  free(packbuf);

  return CMX_SUCCESS;
#else
  return cmx_accs(op, scale,
      src, src_stride, dst_offset, dst_stride,
      count, stride_levels, proc, cmx_hdl);
#endif
}


/**
 * non-blocking vector put operation
 * @param iov: descriptor array
 * @param iov_len: length of descriptor array
 * @param proc: remote processor id
 * @param cmx_hdl: handle for data allocation
 * @param req: non-blocking request handle
 */
int cmx_nbputv(
    cmx_giov_t *iov, cmxInt iov_len,
    int proc, cmx_handle_t cmx_hdl,
    cmx_request_t* req)
{
#ifdef USE_MPI_REQUESTS
  MPI_Datatype src_type, dst_type;
  MPI_Aint displ;
  void *ptr, *src_ptr;
  int lproc, ierr;
  MPI_Win win = cmx_hdl.win;
  MPI_Request request;
  MPI_Status status;
  if (req == NULL) {
    return cmx_putv(iov, iov_len, proc, cmx_hdl);
  }
  src_ptr = iov[0].loc[0];
  ptr = cmx_hdl.buf;
  displ = 0;
  vector_to_struct_dtype(src_ptr, iov, iov_len,
      MPI_BYTE, &src_type, &dst_type);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
  MPI_Type_commit(&src_type);
  MPI_Type_commit(&dst_type);
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Put(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
      win);
  translate_mpi_error(ierr,"cmx_nbputv:MPI_Put");
  req->remote_proc = lproc;
  req->win = win;
#else
  ierr = MPI_Rput(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
      win, &request);
  translate_mpi_error(ierr,"cmx_nbputv:MPI_Rput");
#endif
  req->request = request;
  req->active = 1;
  MPI_Type_free(&src_type);
  MPI_Type_free(&dst_type);
  return CMX_SUCCESS;
#else
  return cmx_putv(iov, iov_len, proc, cmx_hdl);
#endif
}

/**
 * non-blocking vector get operation
 * @param iov: descriptor array
 * @param iov_len: length of descriptor array
 * @param proc: remote processor id
 * @param cmx_hdl: handle for data allocation
 * @param req: non-blocking request handle
 */
int cmx_nbgetv(
    cmx_giov_t *iov, cmxInt iov_len,
    int proc, cmx_handle_t cmx_hdl,
    cmx_request_t* req)
{
#ifdef USE_MPI_REQUESTS
  MPI_Datatype src_type, dst_type;
  MPI_Aint displ;
  void *ptr, *dst_ptr;
  int lproc, ierr;
  MPI_Win win = cmx_hdl.win;
  MPI_Request request;
  MPI_Status status;
  /**
   * ALERT: This is a temporary hack to get the ARMCI interface to work.
   * The CMX tests work but the corresponding ARMCI interface tests don't
   * which may reflect issues with the underlying MPI implementation or a
   * problem with CMX. This issue needs to be revisited.
   */
  return cmx_getv(iov, iov_len, proc, cmx_hdl);
  if (req == NULL) {
    return cmx_getv(iov, iov_len, proc, cmx_hdl);
  }
  dst_ptr = iov[0].loc[0];
  ptr = cmx_hdl.buf;
  displ = 0;
  vector_to_struct_dtype(ptr, iov, iov_len,
      MPI_BYTE, &src_type, &dst_type);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
  MPI_Type_commit(&src_type);
  MPI_Type_commit(&dst_type);
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Get(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
      win);
  translate_mpi_error(ierr,"cmx_nbgetv:MPI_Get");
  req->remote_proc = lproc;
  req->win = win;
#else
  ierr = MPI_Rget(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
      win, &request);
  translate_mpi_error(ierr,"cmx_nbgetv:MPI_Rget");
#endif
  req->request = request;
  req->active = 1;
  MPI_Type_free(&src_type);
  MPI_Type_free(&dst_type);
  return CMX_SUCCESS;
#else
  return cmx_getv(iov, iov_len, proc, cmx_hdl);
#endif
}


/**
 * non-blocking vector accumulate operation
 * @param op: operation
 * @param scale: scale factor x += scale*y
 * @param iov: descriptor array
 * @param iov_len: length of descriptor array
 * @param proc: remote processor id
 * @param cmx_hdl: handle for data allocation
 * @param req: non-blocking request handle
 */
int cmx_nbaccv(
    int op, void *scale,
    cmx_giov_t *iov, cmxInt iov_len,
    int proc, cmx_handle_t cmx_hdl,
    cmx_request_t* req)
{
#ifdef USE_MPI_REQUESTS
  MPI_Datatype src_type, dst_type;
  MPI_Aint displ;
  MPI_Datatype base_type;
  void *ptr, *src_ptr;
  MPI_Request request;
  MPI_Status status;
  MPI_Win win = cmx_hdl.win;
  if (req == NULL) {
    return cmx_accv(op, scale, iov, iov_len, proc, cmx_hdl);
  }
  int lproc, size, ierr;
  if (op == CMX_ACC_INT) {
    size = sizeof(int);
    base_type = MPI_INT;
  } else if (op == CMX_ACC_LNG) {
    size = sizeof(long);
    base_type = MPI_LONG;
  } else if (op == CMX_ACC_FLT) {
    size = sizeof(float);
    base_type = MPI_FLOAT;
  } else if (op == CMX_ACC_DBL) {
    size = sizeof(double);
    base_type = MPI_DOUBLE;
  } else if (op == CMX_ACC_CPL) {
    size = 2*sizeof(float);
    base_type = MPI_FLOAT;
  } else if (op == CMX_ACC_DCP) {
    size = 2*sizeof(double);
    base_type = MPI_DOUBLE;
  }
  ptr = cmx_hdl.buf;
  displ = 0;
  src_ptr = create_vector_buf_and_dtypes(ptr, iov,
      iov_len, size, scale, base_type, &src_type, &dst_type);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
  MPI_Type_commit(&src_type);
  MPI_Type_commit(&dst_type);
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Accumulate(src_ptr,1,src_type,lproc,displ,1,dst_type,
      MPI_SUM,win);
  translate_mpi_error(ierr,"cmx_nbaccv:MPI_Accumulate");
  req->remote_proc = lproc;
  req->win = win;
#else
  ierr = MPI_Raccumulate(src_ptr,1,src_type,lproc,displ,1,dst_type,
      MPI_SUM,win,&request);
  translate_mpi_error(ierr,"cmx_nbaccv:MPI_Raccumulate");
#endif
  req->request = request;
  req->active = 1;
  MPI_Type_free(&src_type);
  MPI_Type_free(&dst_type);
  free(src_ptr);
  return CMX_SUCCESS;
#else
  return cmx_accv(op, scale, iov, iov_len, proc, cmx_hdl);
#endif
}


/**
 * Read-modify_write atomic operation
 * @param op: operation
 * @param ploc: the value to update locally
 * @param rem_offset: offset to remote value
 * @param extra: amount to increment remote value
 * @param proc: rank of remote processor
 * @param cmx_hdl: handle for data allocation
 */
int cmx_rmw(
    int op, void *ploc, cmxInt rem_offset, int extra,
    int proc, cmx_handle_t cmx_hdl)
{
  MPI_Aint displ;
  void *ptr;
  int lproc, ierr;
  MPI_Win win = cmx_hdl.win;
  ptr = cmx_hdl.buf;
  if (ptr == NULL) return CMX_FAILURE;
  displ = (MPI_Aint)(rem_offset);
  if (!(get_local_rank_from_win(win, proc, &lproc)
        == CMX_SUCCESS)) {
    assert(0);
  }
  if (op == CMX_FETCH_AND_ADD) {
    int incr = extra;
#ifdef USE_MPI_REQUESTS
    ierr = MPI_Fetch_and_op(&incr,ploc,MPI_INT,lproc,displ,
        MPI_SUM,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Fetch_and_op");
    ierr = MPI_Win_flush(lproc,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Win_flush");
#else
    ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Win_lock");
    ierr = MPI_Fetch_and_op(&incr,ploc,MPI_INT,lproc,displ,
        MPI_SUM,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Fetch_and_op");
    ierr = MPI_Win_unlock(lproc,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Win_unlock");
#endif
  } else if (op == CMX_FETCH_AND_ADD_LONG) {
    long incr = extra;
#ifdef USE_MPI_REQUESTS
    ierr = MPI_Fetch_and_op(&incr,ploc,MPI_LONG,lproc,displ,
        MPI_SUM,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Fetch_and_op");
    ierr = MPI_Win_flush(lproc,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Win_flush");
#else
    ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Win_lock");
    ierr = MPI_Fetch_and_op(&incr,ploc,MPI_LONG,lproc,displ,
        MPI_SUM,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Fetch_and_op");
    ierr = MPI_Win_unlock(lproc,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Win_unlock");
#endif
  } else if (op == CMX_SWAP) {
    int incr = *((int*)ploc);
#ifdef USE_MPI_REQUESTS
    ierr = MPI_Fetch_and_op(&incr,ploc,MPI_INT,lproc,displ,
        MPI_REPLACE,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Fetch_and_op");
    ierr = MPI_Win_flush(lproc,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Win_flush");
#else
    ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Win_lock");
    ierr = MPI_Fetch_and_op(&incr,ploc,MPI_INT,lproc,displ,
        MPI_REPLACE,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Fetch_and_op");
    ierr = MPI_Win_unlock(lproc,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Win_unlock");
#endif
  } else if (op == CMX_SWAP_LONG) {
    long incr = *((long*)ploc);
#ifdef USE_MPI_REQUESTS
    ierr = MPI_Fetch_and_op(&incr,ploc,MPI_LONG,lproc,displ,
        MPI_REPLACE,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Fetch_and_op");
    ierr = MPI_Win_flush(lproc,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Win_flush");
#else
    ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Win_lock");
    ierr = MPI_Fetch_and_op(&incr,ploc,MPI_LONG,lproc,displ,
        MPI_REPLACE,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Fetch_and_op");
    ierr = MPI_Win_unlock(lproc,win);
    translate_mpi_error(ierr,"cmx_rmw:MPI_Win_unlock");
#endif
  } else  {
    assert(0);
  }

  return CMX_SUCCESS;
}


/* Mutex Operations. These are implemented using an MPI-based algorithm
   described in R. Latham, R. Ross, R. Thakur, The International Journal of
   High Performance Computing Applications, vol. 21, pp. 132-143, (2007).
   The actual algorithm appears to have some errors, the code below is a
   combination of this paper and some examples from the web.
   */

/**
 * create some mutexes on this processor. This is a collective operation
 * on the world group but different processors may create different numbers
 * of mutexes
 * num: number of mutexes to create to create on this processor
 */
int cmx_create_mutexes(int num)
{
  int i, j, k, idx, isize, nsize;
  cmx_igroup_t *igroup = NULL;
  MPI_Comm comm;
  int *sbuf;
  int me = l_state.rank;
  int nproc = l_state.size;

  igroup = CMX_GROUP_WORLD;
  comm = igroup->comm;

  sbuf = (int*)malloc(nproc*sizeof(int));
  _mutex_num = (int*)malloc(nproc*sizeof(int));
  for (i=0; i<nproc; i++) sbuf[i] = 0;
  sbuf[me] = num;

  MPI_Allreduce(sbuf, _mutex_num, nproc, MPI_INT, MPI_SUM, comm);
  free(sbuf);

  _mutex_total = 0;
  for (i=0; i<nproc; i++) _mutex_total += _mutex_num[i];

  if (_mutex_list != NULL) cmx_destroy_mutexes();
  _mutex_list = (MPI_Win*)malloc(_mutex_total*sizeof(MPI_Win));
  _mutex_buf = (int**)malloc(_mutex_total*sizeof(int*));
  isize = sizeof(int);

  /* create windows for mutexes */
  idx = 0;
  for (i=0; i<nproc; i++) {
    for (j=0; j<_mutex_num[i]; j++) {
      if (i == me) {
        nsize = isize*nproc;
        MPI_Alloc_mem(nsize,MPI_INFO_NULL,&_mutex_buf[idx]);
        sbuf = _mutex_buf[idx];
        for (k=0; k<nproc; k++) sbuf[k] = 0;
      } else {
        nsize = 0;
        _mutex_buf[idx] = NULL;
      }
      MPI_Win_create(_mutex_buf[idx],nsize,1,MPI_INFO_NULL,comm,
          &_mutex_list[idx]);
      idx++;
    } 
  }

  return CMX_SUCCESS;
}


/**
 * destroy all mutexes
 */
int cmx_destroy_mutexes()
{
  int i;
  if (_mutex_list == NULL) return CMX_SUCCESS;
  for (i=0; i<_mutex_total; i++) {
    MPI_Win_free(&_mutex_list[i]);
    if (_mutex_buf[i] != NULL) MPI_Free_mem(_mutex_buf[i]);
  }
  free(_mutex_list);
  free(_mutex_buf);
  free(_mutex_num);
  _mutex_list = NULL;
  _mutex_buf = NULL;
  _mutex_num = NULL;
  _mutex_total = 0;

  return CMX_SUCCESS;
}


/**
 * lock a mutex on some processor
 * mutex: index of mutex on processor proc
 * proc: rank of process containing mutex
 */
int cmx_lock(int mutex, int proc)
{
  int i;
  int idx = 0;
  int lock = 1;
  int *waitlistcopy;
  int nproc = l_state.size;
  int me = l_state.rank;
  int ok;

  /* evaluate index of window corresponding to mutex */ 
  for (i=0; i<proc; i++) idx += _mutex_num[i];
  idx += mutex;
  if (idx < 0 || idx >= _mutex_total) assert(0);

  waitlistcopy = (int*)malloc(nproc*sizeof(int));

  /* Set value in the wait list */
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE,proc,0,_mutex_list[idx]);
  MPI_Get(waitlistcopy,nproc,MPI_INT,proc,0,nproc,MPI_INT,_mutex_list[idx]);
  MPI_Put(&lock,1,MPI_INT,proc,me*sizeof(int),1,MPI_INT,_mutex_list[idx]);
  MPI_Win_unlock(proc,_mutex_list[idx]);

  /* Check to see if lock is already held */
  ok = 1;
  for (i=0; i<nproc && ok == 1; i++) {
    if (waitlistcopy[i] != 0 && i != me) ok = 0;
  }

  /* Wait until some one else frees the lock */
  if (!ok) {
    MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, idx, MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
  }

  free(waitlistcopy);
  return CMX_SUCCESS;
}


/**
 * unlock a mutex on some processor
 * mutex: index of mutex on processor proc
 * proc: rank of process containing mutex
 */
int cmx_unlock(int mutex, int proc)
{
  int i;
  int idx = 0;
  int lock = 0;
  int *waitlistcopy;
  int nproc = l_state.size;
  int me = l_state.rank;
  int ok, nextlock;

  /* evaluate index of window corresponding to mutex */ 
  for (i=0; i<proc; i++) idx += _mutex_num[i];
  idx += mutex;
  if (idx < 0 || idx >= _mutex_total) assert(0);

  waitlistcopy = (int*)malloc(nproc*sizeof(int));

  /* Set value in the wait list */
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE,proc,0,_mutex_list[idx]);
  MPI_Get(waitlistcopy,nproc,MPI_INT,proc,0,nproc,MPI_INT,_mutex_list[idx]);
  MPI_Put(&lock,1,MPI_INT,proc,me*sizeof(int),1,MPI_INT,_mutex_list[idx]);
  MPI_Win_unlock(proc,_mutex_list[idx]);

  /* Check to see if someone is waiting for lock */
  ok = 1;
  for (i=0; i<nproc && ok == 1; i++) {
    if (waitlistcopy[i] != 0 && i != me) {
      ok = 0;
      nextlock = i;
    }
  }

  if (!ok) {
    MPI_Send(&lock, 0, MPI_INT, nextlock, idx, MPI_COMM_WORLD);
  }

  free(waitlistcopy);
  return CMX_SUCCESS;
}


int cmx_malloc(cmx_handle_t *handle, cmxInt bytes, cmx_group_t group)
{
  cmx_igroup_t *igroup = group;
  MPI_Comm comm = MPI_COMM_NULL;
  int i;
  int comm_rank = -1;
  int comm_size = -1;
  int tsize;

  comm = igroup->comm;
  assert(comm != MPI_COMM_NULL);
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

#if DEBUG
  printf("[%d] cmx_malloc(ptrs=%p, size=%lu, group=%d)\n",
      comm_rank, ptrs, (long unsigned)bytes, group);
#endif

  /* is this needed? */
  /* cmx_barrier(group); */

  /* allocate ret_entry_t object for this process */
  handle->rank = comm_rank;
  handle->bytes = bytes;
  handle->comm = comm;
  handle->group = group;

  /* allocate and register segment. We need to allocate something even if size
     is zero so allocate a nominal amount so that routines don't break. Count
     on the fact that the length is zero to keep things from breaking down */
  if (bytes > 0) {
    tsize = bytes;
  } else {
    tsize = 8;
  }
#ifdef USE_MPI_WIN_ALLOC
  MPI_Win_allocate(sizeof(char)*tsize,1,MPI_INFO_NULL,comm,&(handle->buf),
      &(handle->win));
#else
  MPI_Alloc_mem(tsize,MPI_INFO_NULL,&(handle->buf));
  MPI_Win_create(handle->buf,tsize,1,MPI_INFO_NULL,comm,
      &(handle->win));
#endif
#ifdef USE_MPI_REQUESTS
  MPI_Win_lock_all(0,(handle->win));
#endif

  cmx_igroup_add_win(igroup,handle->win);
  cmx_wait_all(igroup);
  /* MPI_Win_fence(0,reg_entries[comm_rank].win); */
  MPI_Barrier(comm);

  return CMX_SUCCESS;
}

/**
 * Access local buffer from CMX handle
 * @param handle CMX handle for data allocation
 * @param buf pointer to local buffer
 * @return CMX_SUCCESS on success
 */
int cmx_access(cmx_handle_t cmx_hdl, void **buf)
{
  *buf = cmx_hdl.buf;
  return CMX_SUCCESS;
}

/**
 * Extact group object from CMX allocation handle
 *
 * @param handle CMX handle for data allocation
 * @param group CMX group associated with CMX data allocation
 * @return CMX_SUCCESS on success
 */
int cmx_get_group_from_handle(cmx_handle_t handle, cmx_group_t *group)
{
  *group = handle.group;
  return CMX_SUCCESS;
}



int cmx_free(cmx_handle_t handle)
{
  cmx_igroup_t *group = handle.group;
  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Win window;
  int comm_rank;
  int comm_size;
  int i;

  comm = handle.comm;
  assert(comm != MPI_COMM_NULL);
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  window = handle.win;

  /* Remove window from group list */
  cmx_igroup_delete_win(group, window);

  /* free up window */
#ifdef USE_MPI_REQUESTS
  MPI_Win_unlock_all(window);
#endif
  MPI_Win_free(&window);
#ifndef USE_MPI_WIN_ALLOC
  /* Clear memory for this window */
  MPI_Free_mem(handle.buf);
#endif

  /* Is this needed? */
  MPI_Barrier(comm);

  return CMX_SUCCESS;
}


static void acquire_remote_lock(int proc)
{
  assert(0);
}


static void release_remote_lock(int proc)
{
  assert(0);
}


static inline void acc(
    int datatype, int count, void *get_buf,
    void *src_ptr, long src_idx, void *scale)
{
#define EQ_ONE_REG(A) ((A) == 1.0)
#define EQ_ONE_CPL(A) ((A).real == 1.0 && (A).imag == 0.0)
#define IADD_REG(A,B) (A) += (B)
#define IADD_CPL(A,B) (A).real += (B).real; (A).imag += (B).imag
#define IADD_SCALE_REG(A,B,C) (A) += (B) * (C)
#define IADD_SCALE_CPL(A,B,C) (A).real += ((B).real*(C).real) - ((B).imag*(C).imag);\
  (A).imag += ((B).real*(C).imag) + ((B).imag*(C).real);
#define ACC(WHICH, CMX_TYPE, C_TYPE)                                  \
  if (datatype == CMX_TYPE) {                                       \
    int m;                                                          \
    int m_lim = count/sizeof(C_TYPE);                               \
    C_TYPE *iterator = (C_TYPE *)get_buf;                           \
    C_TYPE *value = (C_TYPE *)((char *)src_ptr + src_idx);          \
    C_TYPE calc_scale = *(C_TYPE *)scale;                           \
    if (EQ_ONE_##WHICH(calc_scale)) {                               \
      for (m = 0 ; m < m_lim; ++m) {                              \
        IADD_##WHICH(iterator[m], value[m]);                    \
      }                                                           \
    }                                                               \
    else {                                                          \
      for (m = 0 ; m < m_lim; ++m) {                              \
        IADD_SCALE_##WHICH(iterator[m], value[m], calc_scale);  \
      }                                                           \
    }                                                               \
  } else
  ACC(REG, CMX_ACC_DBL, double)
    ACC(REG, CMX_ACC_FLT, float)
    ACC(REG, CMX_ACC_INT, int)
    ACC(REG, CMX_ACC_LNG, long)
    ACC(CPL, CMX_ACC_DCP, DoubleComplex)
    ACC(CPL, CMX_ACC_CPL, SingleComplex)
    {
      assert(0);
    }
#undef ACC
#undef EQ_ONE_REG
#undef EQ_ONE_CPL
#undef IADD_REG
#undef IADD_CPL
#undef IADD_SCALE_REG
#undef IADD_SCALE_CPL
}

