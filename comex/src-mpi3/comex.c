#if HAVE_CONFIG_H
#   include "config.h"
#endif

/* C and/or system headers */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

/* 3rd party headers */
#include <mpi.h>
#if USE_SICM
#include <sicm_low.h>
//#include <sicm_impl.h>
#endif

/* our headers */
#include "comex.h"
#include "comex_impl.h"
#include "groups.h"
#include "reg_win.h"

#define DEBUG 0

#define XSTR(x) #x
#define STR(x) XSTR(x)

/*
#define USE_PRIOR_MPI_WIN_FLUSH
#define USE_POST_MPI_WIN_FLUSH
*/

/*
*/
#define USE_MPI_DATATYPES


#define USE_MPI_REQUESTS
/*
#define USE_MPI_FLUSH_LOCAL
#define USE_MPI_WIN_ALLOC
*/

#ifdef USE_MPI_FLUSH_LOCAL
#define USE_MPI_REQUESTS
#endif

#define XSTR(x) #x
#define STR(x) XSTR(x)

#if USE_SICM
static sicm_device_list devices    = {0};
#if SICM_OLD
static sicm_device *device_dram    = NULL;
static sicm_device *device_knl_hbm = NULL;
static sicm_device *device_ppc_hbm = NULL;
#else
static sicm_device_list device_dram     = {0};
static sicm_device_list device_knl_hbm  = {0};
static sicm_device_list device_ppc_hbm  = {0};
#endif
#endif

/* exported state */
local_state l_state;

/* static state */
static int  initialized=0;  /* for comex_initialized(), 0=false */
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
static int nb_max_outstanding = COMEX_MAX_NB_OUTSTANDING;

typedef struct {
  MPI_Request request;
  MPI_Win win;
  int use_type;
  MPI_Datatype src_type;
  MPI_Datatype dst_type;
  int active;
#ifdef USE_MPI_FLUSH_LOCAL
  int remote_proc;
#else
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

static int nb_full_count = 0;

/* Find first available non-blocking handle */
#ifdef USE_MPI_REQUESTS
void get_nb_request(comex_request_t *handle, nb_t **req)
{
  int i;
  for (i=0; i<nb_max_outstanding; i++) {
    if (nb_list[i]->active == 0) break;
  }
  if (i<nb_max_outstanding) {
    *handle = i;
    *req = nb_list[i];
  } else {
    /* call wait on an existing handle and then return that
     * handle as available. This is NOT thread safe */
    *handle = nb_full_count;
    comex_wait(handle);
    *req = nb_list[nb_full_count];
    nb_full_count++;
    nb_full_count = nb_full_count%nb_max_outstanding;
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
  comex_igroup_t *world_igroup =
    comex_get_igroup_from_group(COMEX_GROUP_WORLD);
  MPI_Group group;
  status = MPI_Win_get_group(win, &group);
  translate_mpi_error(status,"get_local_rank_from_win:MPI_Win_get_group");
  status = MPI_Group_translate_ranks( world_igroup->group,
      1, &world_rank, group, local_rank);
  if (status != MPI_SUCCESS) {
    translate_mpi_error(status,"get_local_rank_from_win:MPI_Group_translate_ranks");
    comex_error("MPI_Group_translate_ranks: Failed", status);
  }

  return COMEX_SUCCESS;
}

int comex_init()
{
    int i, status;
    int init_flag;

    if (initialized) {
        return 0;
    }
    initialized = 1;

    /* Assert MPI has been initialized */
    status = MPI_Initialized(&init_flag);
    assert(MPI_SUCCESS == status);
    assert(init_flag);
    status = MPI_Comm_size(MPI_COMM_WORLD, &status);
    
    /* Duplicate the World Communicator */
    status = MPI_Comm_dup(MPI_COMM_WORLD, &(l_state.world_comm));
    translate_mpi_error(status,"comex_init:MPI_Comm_dup");
    assert(MPI_SUCCESS == status);
    assert(l_state.world_comm); 

    /* My Rank */
    status = MPI_Comm_rank(l_state.world_comm, &(l_state.rank));
    assert(MPI_SUCCESS == status);

    /* World Size */
    status = MPI_Comm_size(l_state.world_comm, &(l_state.size));
    assert(MPI_SUCCESS == status);

    /* Pick up environment variables */
    {
      char *value = NULL;
      nb_max_outstanding = COMEX_MAX_NB_OUTSTANDING; /* default */
      value = getenv("COMEX_MAX_NB_OUTSTANDING");
      if (NULL != value) {
        nb_max_outstanding = atoi(value);
      }
      COMEX_ASSERT(nb_max_outstanding > 0);
    }

    
    /* groups */
    comex_group_init();

    /* register windows initialization */
    reg_win_init(l_state.size);

    /* set mutex list equal to null */
    _mutex_list = NULL;
    _mutex_buf = NULL;
    _mutex_num = NULL;
    _mutex_total = 0;

    /* initialize non-blocking handles */
#ifdef USE_MPI_REQUESTS
    nb_list = (nb_t**)malloc(sizeof(nb_t*) * nb_max_outstanding);
    COMEX_ASSERT(nb_list);
    for (i=0; i<nb_max_outstanding; i++) {
      nb_list[i] = (nb_t*)malloc(sizeof(nb_t));
      nb_list[i]->active = 0;
    }
#endif

#if USE_SICM
    devices = sicm_init();
#if SICM_OLD
    for(i = 0; i < devices.count; i++){
      if(devices.devices[i].tag == SICM_DRAM){
        device_dram = &(devices.devices[i]);
      }
      if(devices.devices[i].tag == SICM_KNL_HBM){
        device_knl_hbm = &(devices.devices[i]);
      }
      if(devices.devices[i].tag == SICM_POWERPC_HBM){
        device_ppc_hbm = &(devices.devices[i]);
      }
    }
    if(!device_dram){
      printf("Device DRAM not found\n");
      exit(18);
    }
#else
   for(i =0; i < devices.count; ++i){
      sicm_device *curr = devices.devices[i];
      if(curr->tag == SICM_DRAM){
         device_dram.count = 1;
         device_dram.devices = &devices.devices[i];
      }
      if (curr->tag == SICM_KNL_HBM) {
         device_knl_hbm.count = 1;
         device_knl_hbm.devices = &devices.devices[i];
      }
      if (curr->tag == SICM_POWERPC_HBM) {
         device_ppc_hbm.count = 1;
         device_ppc_hbm.devices = &devices.devices[i];
      }
   }
   if(device_dram.devices == NULL){
      printf("Device DRAM not found\n");
      exit(18);
   }
#endif
#endif


    /* sync - initialize first communication epoch */
    comex_fence_all(COMEX_GROUP_WORLD);
    /* Synch - Sanity Check */
    MPI_Barrier(l_state.world_comm);

    return COMEX_SUCCESS;
}


int comex_init_args(int *argc, char ***argv)
{
    int init_flag;
    
    MPI_Initialized(&init_flag);
    
    if(!init_flag) {
        MPI_Init(argc, argv);
    }
    
    return comex_init();
}


int comex_initialized()
{
    return initialized;
}


void comex_error(char *msg, int code)
{
    fprintf(stderr,"[%d] Received an Error in Communication: (%d) %s\n",
            l_state.rank, code, msg);
    
    MPI_Abort(l_state.world_comm, code);
}


/**
 * @param src pointer to source buffer
 * @param dst pointer to destination (remote) buffer
 * @param bytes length of data to be transferred
 * @param proc rank of destination processors
 * @param group handle for group that transfer is defined on
 */
int comex_put(
        void *src, void *dst, int bytes,
        int proc, comex_group_t group)
{
    MPI_Aint displ;
    void *ptr;
    int lproc, ierr;
    reg_entry_t *reg_win;
#ifdef USE_MPI_REQUESTS
    MPI_Request request;
    MPI_Status status;
#endif
    reg_win = reg_win_find(proc, dst, 0);
    ptr = reg_win->buf;
    displ = (MPI_Aint)(dst) - (MPI_Aint)(ptr);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    /**
     * All processes called MPI_WIN_LOCK_ALL on the reg_win->win window used for
     * these calls. This call:
     *
     * Starts an RMA access epoch to all processors in reg_win->win with a lock
     * type of MPI_LOCK_SHARED. During the epoch, the calling process can access
     * the window memory on all processes in reg_win->win using RMA operations.
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
     * reg_win->win.
     */
    /**
     * MPI_WIN_FLUSH completes all outstanding RMA operations initiated by the
     * calling process to the target rank on the specified window. The
     * operations are completed both at the origin and at the target
     */
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_put:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
    /**
     * The local communication buffer of an RMA call should not be updated, and
     * the local communication buffer of a get call should not be accessed after
     * the RMA call until the operation completes at the origin.
     */
    ierr = MPI_Put(src, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
        reg_win->win);
    translate_mpi_error(ierr,"comex_put:MPI_Put");
    /**
     * MPI_WIN_FLUSH_LOCAL locally completes at the origin all outstanding RMA
     * operations initiated by the calling process to the target process
     * specified by the rank on the specified window. For example, after the
     * routine completes, the user may reused any buffers provided to put, get,
     * or accumulate operations
     */
    ierr = MPI_Win_flush_local(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_put:MPI_Win_flush_local");
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
        reg_win->win, &request);
    translate_mpi_error(ierr,"comex_put:MPI_Rput");
    /**
     * A call to MPI_WAIT returns when the operation identified by the request
     * is complete. If the request is an active persistent request, it is marked
     * inactive. Any other type of request is and the request handle is set to
     * MPI_REQUEST_NULL. MPI_WAIT is a non-local operation
     */
    ierr = MPI_Wait(&request, &status);
    translate_mpi_error(ierr,"comex_put:MPI_Wait");
#ifdef USE_POST_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_put:MPI_Win_flush");
#endif
#endif
#else
    ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,reg_win->win);
    translate_mpi_error(ierr,"comex_put:MPI_Win_lock");
    ierr = MPI_Put(src, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
        reg_win->win);
    translate_mpi_error(ierr,"comex_put:MPI_Put");
    ierr = MPI_Win_unlock(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_put:MPI_Win_unlock");
#endif
    return COMEX_SUCCESS;
}


/**
 * @param src pointer to source (remote) buffer
 * @param dst pointer to destination buffer
 * @param bytes length of data to be transferred
 * @param proc rank of destination processors
 * @param group handle for group that transfer is defined on
 */
int comex_get(
        void *src, void *dst, int bytes,
        int proc, comex_group_t group)
{
    MPI_Aint displ;
    void *ptr;
    int lproc, ierr;
    reg_entry_t *reg_win;
#ifdef USE_MPI_REQUESTS
    MPI_Request request;
    MPI_Status status;
#endif
    reg_win = reg_win_find(proc, src, 0);
    ptr = reg_win->buf;
    displ = (MPI_Aint)(src) - (MPI_Aint)(ptr);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_get:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Get(dst, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
        reg_win->win);
    translate_mpi_error(ierr,"comex_get:MPI_Get");
    ierr = MPI_Win_flush_local(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_get:MPI_Win_flush_local");
#else
    ierr = MPI_Rget(dst, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
        reg_win->win, &request);
    translate_mpi_error(ierr,"comex_get:MPI_Rget");
    ierr = MPI_Wait(&request, &status);
    translate_mpi_error(ierr,"comex_get:MPI_Wait");
#ifdef USE_POST_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_get:MPI_Win_flush");
#endif
#endif
#else
    ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,reg_win->win);
    translate_mpi_error(ierr,"comex_get:MPI_Win_lock");
    ierr = MPI_Get(dst, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
        reg_win->win);
    translate_mpi_error(ierr,"comex_get:MPI_Get");
    ierr = MPI_Win_unlock(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_get:MPI_Win_unlock");
#endif
    return COMEX_SUCCESS;
}

/**
 * @param datatype COMEX datatype for accumulate operation
 * @param scale value to scale data by before accumulating it
 * @param src pointer to source buffer
 * @param dst pointer to destination (remote) buffer
 * @param bytes length of data to be transferred
 * @param proc rank of destination processors
 * @param group handle for group that transfer is defined on
 */
int comex_acc(
        int datatype, void *scale,
        void *src, void *dst, int bytes,
        int proc, comex_group_t group)
{
    MPI_Aint displ;
    void *ptr, *tbuf;
    int count, i, lproc, ierr;
    reg_entry_t *reg_win;
#ifdef USE_MPI_REQUESTS
    MPI_Request request;
    MPI_Status status;
#endif
    MPI_Datatype mpi_type;
    reg_win = reg_win_find(proc, dst, 0);
    ptr = reg_win->buf;
    displ = (MPI_Aint)(dst) - (MPI_Aint)(ptr);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    if (datatype == COMEX_ACC_INT) {
      int *buf;
      int *isrc = (int*)src;
      int iscale = *((int*)scale);
      count = bytes/sizeof(int);
      buf = (int*)malloc(bytes);
      tbuf = buf;
      for (i=0; i<count; i++) {
        buf[i] = isrc[i]*iscale;
      }
      mpi_type = MPI_INT;
    } else if (datatype == COMEX_ACC_LNG) {
      long *buf;
      long *lsrc = (long*)src;
      long lscale = *((long*)scale);
      count = bytes/sizeof(long);
      buf = (long*)malloc(bytes);
      tbuf = buf;
      for (i=0; i<count; i++) {
        buf[i] = lsrc[i]*lscale;
      }
      mpi_type = MPI_LONG;
    } else if (datatype == COMEX_ACC_FLT) {
      float *buf;
      float *fsrc = (float*)src;
      float fscale = *((float*)scale);
      count = bytes/sizeof(float);
      buf = (float*)malloc(bytes);
      tbuf = buf;
      for (i=0; i<count; i++) {
        buf[i] = fsrc[i]*fscale;
      }
      mpi_type = MPI_FLOAT;
    } else if (datatype == COMEX_ACC_DBL) {
      double *buf;
      double *dsrc = (double*)src;
      double dscale = *((double*)scale);
      count = bytes/sizeof(double);
      buf = (double*)malloc(bytes);
      tbuf = buf;
      for (i=0; i<count; i++) {
        buf[i] = dsrc[i]*dscale;
      }
      mpi_type = MPI_DOUBLE;
    } else if (datatype == COMEX_ACC_CPL) {
      float *buf;
      int cnum;
      float *csrc = (float*)src;
      float crscale = *((float*)scale);
      float ciscale = *((float*)scale+1);
      count = bytes/sizeof(float);
      cnum = count/2;
      buf = (float*)malloc(bytes);
      tbuf = buf;
      for (i=0; i<cnum; i++) {
        buf[2*i] = csrc[2*i]*crscale-csrc[2*i+1]*ciscale;
        buf[2*i+1] = csrc[2*i]*ciscale+csrc[2*i+1]*crscale;
      }
      mpi_type = MPI_FLOAT;
    } else if (datatype == COMEX_ACC_DCP) {
      double *buf;
      int cnum;
      double *csrc = (double*)src;
      double crscale = *((double*)scale);
      double ciscale = *((double*)scale+1);
      count = bytes/sizeof(double);
      cnum = count/2;
      buf = (double*)malloc(bytes);
      tbuf = buf;
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
    translate_mpi_error(ierr,"comex_acc:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Accumulate(tbuf,count,mpi_type,lproc,displ,count,
        mpi_type,MPI_SUM,reg_win->win);
    translate_mpi_error(ierr,"comex_acc:MPI_Accumulate");
    ierr = MPI_Win_flush_local(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_acc:MPI_Win_flush_local");
#else
    ierr = MPI_Raccumulate(tbuf,count,mpi_type,lproc,displ,count,
        mpi_type,MPI_SUM,reg_win->win,&request);
    translate_mpi_error(ierr,"comex_acc:MPI_Raccumulate");
    ierr = MPI_Wait(&request, &status);
    translate_mpi_error(ierr,"comex_acc:MPI_Wait");
#ifdef USE_POST_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_acc:MPI_Win_flush");
#endif
#endif
#else
    ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,reg_win->win);
    translate_mpi_error(ierr,"comex_acc:MPI_Win_lock");
    ierr = MPI_Accumulate(tbuf,count,mpi_type,lproc,displ,count,
        mpi_type,MPI_SUM,reg_win->win);
    translate_mpi_error(ierr,"comex_acc:MPI_Accumulate");
    ierr = MPI_Win_unlock(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_acc:MPI_Win_unlock");
#endif
    free(tbuf);
    return COMEX_SUCCESS;
}

/**
 * No checking for data consistency. Assume correctness has already been
 * established elsewhere. Individual elements are assumed to be one byte in size
 * stride_array: physical dimensions of array
 * count: number of elements along each array dimension
 * levels: number of stride levels (should be one less than array dimension)
 * type: MPI_Datatype returned to calling program
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
     if (array_of_sizes[i] < array_of_subsizes[i]) {
       printf("p[%d] ERROR [strided_to_subarray_dtype]\n"
              "array_of_sizes[%d]: %d\n"
              "array_of_subsizes[%d]: %d\n",l_state.rank,
              i,array_of_sizes[i],i,array_of_subsizes[i]);
     }
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
 * src_ptr: pointer to origin of data on source processor
 * dst_ptr: pointer to origin of data on destination processor
 * src_stride_ar: physical dimensions of array containing source data
 * dst_strice_ar: physical dimensions of destination array
 * count: array containing number of data points along each dimension. The 0
 * dimension contains the actual length of contiguous segments in bytes (number
 * of elements times the size of each element).
 * stride_levels: this should be one less than the dimension of the array
 * proc: global rank of destination array
 * group: ID of group over which transfer takes place
 */
int comex_puts(
        void *src_ptr, int *src_stride_ar,
        void *dst_ptr, int *dst_stride_ar,
        int *count, int stride_levels,
        int proc, comex_group_t group)
{
#ifndef USE_MPI_DATATYPES
    int i, j;
    long src_idx, dst_idx;  /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    int src_bvalue[7], src_bunit[7];
    int dst_bvalue[7], dst_bunit[7];
    int status;

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
            src_idx += src_bvalue[j] * src_stride_ar[j-1];
            if((i+1) % src_bunit[j] == 0) {
                src_bvalue[j]++;
            }
            if(src_bvalue[j] > (count[j]-1)) {
                src_bvalue[j] = 0;
            }
        }

        for(j=1; j<=stride_levels; j++) {
            dst_idx += dst_bvalue[j] * dst_stride_ar[j-1];
            if((i+1) % dst_bunit[j] == 0) {
                dst_bvalue[j]++;
            }
            if(dst_bvalue[j] > (count[j]-1)) {
                dst_bvalue[j] = 0;
            }
        }
        
        status = comex_put((char *)src_ptr + src_idx, 
                (char *)dst_ptr + dst_idx, count[0], proc, group);
        assert(status == COMEX_SUCCESS);
    }

    return COMEX_SUCCESS;
#else
    MPI_Datatype src_type, dst_type;
    MPI_Aint displ;
    void *ptr;
    int lproc, ierr;
    reg_entry_t *reg_win;
#ifdef USE_MPI_REQUESTS
    MPI_Request request;
    MPI_Status status;
#endif
    /* If data is contiguous, use comex_put */
    if (stride_levels == 0) {
      return comex_put(src_ptr, dst_ptr, count[0], proc, group);
    }
    reg_win = reg_win_find(proc, dst_ptr, 0);
    ptr = reg_win->buf;
    displ = (MPI_Aint)(dst_ptr) - (MPI_Aint)(ptr);

    strided_to_subarray_dtype(src_stride_ar, count, stride_levels,
      MPI_BYTE, &src_type);
    strided_to_subarray_dtype(dst_stride_ar, count, stride_levels,
      MPI_BYTE, &dst_type);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    ierr = MPI_Type_commit(&src_type);
    translate_mpi_error(ierr,"comex_puts:MPI_Type_commit");
    ierr = MPI_Type_commit(&dst_type);
    translate_mpi_error(ierr,"comex_puts:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_puts:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Put(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
        reg_win->win);
    translate_mpi_error(ierr,"comex_puts:MPI_Put");
    ierr = MPI_Win_flush_local(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_puts:MPI_Win_flush_local");
#else
    ierr = MPI_Rput(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
        reg_win->win, &request);
    translate_mpi_error(ierr,"comex_puts:MPI_Rput");
    ierr = MPI_Wait(&request, &status);
    translate_mpi_error(ierr,"comex_puts:MPI_Wait");
#ifdef USE_POST_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_puts:MPI_Win_flush");
#endif
#endif
#else
    ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,reg_win->win);
    translate_mpi_error(ierr,"comex_puts:MPI_Win_lock");
    ierr = MPI_Put(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
        reg_win->win);
    translate_mpi_error(ierr,"comex_puts:MPI_Put");
    ierr = MPI_Win_unlock(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_puts:MPI_Win_unlock");
#endif
    ierr = MPI_Type_free(&src_type);
    translate_mpi_error(ierr,"comex_puts:MPI_Type_free");
    ierr = MPI_Type_free(&dst_type);
    translate_mpi_error(ierr,"comex_puts:MPI_Type_free");
    return COMEX_SUCCESS;
#endif
}


/**
 * src_ptr: pointer to origin of data on source processor
 * dst_ptr: pointer to origin of data on destination processor
 * src_stride_ar: physical dimensions of array containing source data
 * dst_strice_ar: physical dimensions of destination array
 * count: array containing number of data points along each dimension. The 0
 * dimension contains the actual length of contiguous segments in bytes (number
 * of elements times the size of each element).
 * stride_levels: this should be one less than the dimension of the array
 * proc: global rank of source array
 * group: ID of group over which transfer takes place
 */
int comex_gets(
        void *src_ptr, int *src_stride_ar,
        void *dst_ptr, int *dst_stride_ar,
        int *count, int stride_levels,
        int proc, comex_group_t group)
{
#ifndef USE_MPI_DATATYPES
    int i, j;
    long src_idx, dst_idx;  /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    int src_bvalue[7], src_bunit[7];
    int dst_bvalue[7], dst_bunit[7];
    int status;

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
            src_idx += src_bvalue[j] * src_stride_ar[j-1];
            if((i+1) % src_bunit[j] == 0) {
                src_bvalue[j]++;
            }
            if(src_bvalue[j] > (count[j]-1)) {
                src_bvalue[j] = 0;
            }
        }

        dst_idx = 0;
        
        for(j=1; j<=stride_levels; j++) {
            dst_idx += dst_bvalue[j] * dst_stride_ar[j-1];
            if((i+1) % dst_bunit[j] == 0) {
                dst_bvalue[j]++;
            }
            if(dst_bvalue[j] > (count[j]-1)) {
                dst_bvalue[j] = 0;
            }
        }
        
        status = comex_get((char *)src_ptr + src_idx, 
                (char *)dst_ptr + dst_idx, count[0], proc, group);
        assert(status == COMEX_SUCCESS);
    }
    
    return COMEX_SUCCESS;
#else
    MPI_Datatype src_type, dst_type;
    MPI_Aint displ;
    void *ptr;
    int lproc,ierr;
    reg_entry_t *reg_win;
#ifdef USE_MPI_REQUESTS
    MPI_Request request;
    MPI_Status status;
#endif
    /* If data is contiguous, use comex_get */
    if (stride_levels == 0) {
      return comex_get(src_ptr, dst_ptr, count[0], proc, group);
    }
    reg_win = reg_win_find(proc, src_ptr, 0);
    ptr = reg_win->buf;
    displ = (MPI_Aint)(src_ptr) - (MPI_Aint)(ptr);

    strided_to_subarray_dtype(src_stride_ar, count, stride_levels,
      MPI_BYTE, &src_type);
    strided_to_subarray_dtype(dst_stride_ar, count, stride_levels,
      MPI_BYTE, &dst_type);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    ierr = MPI_Type_commit(&src_type);
    translate_mpi_error(ierr,"comex_gets:MPI_Type_commit");
    ierr = MPI_Type_commit(&dst_type);
    translate_mpi_error(ierr,"comex_gets:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_gets:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Get(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
        reg_win->win);
    translate_mpi_error(ierr,"comex_gets:MPI_Get");
    ierr = MPI_Win_flush_local(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_gets:MPI_Win_flush_local");
#else
    ierr = MPI_Rget(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
        reg_win->win, &request);
    translate_mpi_error(ierr,"comex_gets:MPI_Rget");
    ierr = MPI_Wait(&request, &status);
    translate_mpi_error(ierr,"comex_gets:MPI_Wait");
#ifdef USE_POST_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_gets:MPI_Win_flush");
#endif
#endif
#else
    ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,reg_win->win);
    translate_mpi_error(ierr,"comex_gets:MPI_Win_lock");
    ierr = MPI_Get(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
        reg_win->win);
    translate_mpi_error(ierr,"comex_gets:MPI_Get");
    ierr = MPI_Win_unlock(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_gets:MPI_Win_lock");
#endif
    ierr = MPI_Type_free(&src_type);
    translate_mpi_error(ierr,"comex_gets:MPI_Type_free");
    ierr = MPI_Type_free(&dst_type);
    translate_mpi_error(ierr,"comex_gets:MPI_Type_free");
    return COMEX_SUCCESS;
#endif
}

/**
 * Utility function for allocating a buffer that can be used to scale strided
 * data before sending it to remote processor for accumulate operation. Data is
 * copied into buffer after allocation. 
 * ptr: pointer to first member of data array
 * strides: physical dimensions of array containing data
 * count: array containing the number of data points along each dimension. The
 * 0 element contains the actual length of the contiguous segments in bytes
 * levels: this should be one less than the dimension of the array
 * size: size (in bytes) of allocated buffer
 * new_strides: stride array for newly allocated buffer
 */
void* malloc_strided_acc_buffer(void* ptr, int *strides, int *count,
    int levels, int *size, int *new_strides)
{
#if 0
  int i, j, stride, src_idx;
  void *new_ptr;
  char *cursor, *src_cursor;
  int n1dim, src_bvalue[7], src_bunit[7];
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
  src_bvalue[0] = 0;
  src_bvalue[1] = 0;
  src_bunit[0] = 1;
  src_bunit[1] = 1;

  /* set up evaluation of source indices */
  for (i=2; i<=levels; i++) {
    src_bvalue[i] = 0;
    src_bunit[i] = src_bunit[i-1]*count[i-1];
  }
  for (i=0; i<n1dim; i++) {
    src_idx = 0;
    for (j=1; j<=levels; j++) {
      src_idx += src_bvalue[j]*strides[j-1];
      if ((i+1) % src_bunit[j] == 0) {
        src_bvalue[j]++;
      }
      if (src_bvalue[j] > (count[j]-1)) {
        src_bvalue[j] = 0;
      }
    }
    src_cursor = (char*)ptr+src_idx;
    memcpy(cursor, src_cursor, count[0]);
    cursor += count[0];
  }

  return new_ptr;
#else
  int index[7];
  int lcnt[7];
  int idx, i, j, stride, src_idx;
  void *new_ptr;
  char *cursor, *src_cursor;
  int n1dim;
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
#endif
}

/**
 * datatype: array data type
 * scale: factor to scale data before adding to contents in destination array
 * src_ptr: pointer to origin of data on source processor
 * dst_ptr: pointer to origin of data on destination processor
 * src_stride_ar: physical dimensions of array containing source data
 * dst_stride_ar: physical dimensions of destination array
 * count: array containing number of data points along each dimension. The 0
 * dimension contains the actual length of contiguous segments in bytes (number
 * of elements times the size of each element).
 * stride_levels: this should be one less than the dimension of the array
 * proc: global rank of destination array
 * group: ID of group over which transfer takes place
 */
int comex_accs(
        int datatype, void *scale,
        void *src_ptr, int *src_stride_ar,
        void *dst_ptr, int *dst_stride_ar,
        int *count, int stride_levels,
        int proc, comex_group_t group)
{
#ifndef USE_MPI_DATATYPES
    int i, j;
    long src_idx, dst_idx;  /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    int src_bvalue[7], src_bunit[7];
    int dst_bvalue[7], dst_bunit[7];
    void *get_buf;

    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++)
        n1dim *= count[i];

    /* calculate the destination indices */
    src_bvalue[0] = 0; src_bvalue[1] = 0; src_bunit[0] = 1; src_bunit[1] = 1;
    dst_bvalue[0] = 0; dst_bvalue[1] = 0; dst_bunit[0] = 1; dst_bunit[1] = 1;

    for(i=2; i<=stride_levels; i++)
    {
        src_bvalue[i] = 0;
        dst_bvalue[i] = 0;
        src_bunit[i] = src_bunit[i-1] * count[i-1];
        dst_bunit[i] = dst_bunit[i-1] * count[i-1];
    }

    for(i=0; i<n1dim; i++) {
        src_idx = 0;
        for(j=1; j<=stride_levels; j++) {
            src_idx += src_bvalue[j] * src_stride_ar[j-1];
            if((i+1) % src_bunit[j] == 0) {
                src_bvalue[j]++;
            }
            if(src_bvalue[j] > (count[j]-1)) {
                src_bvalue[j] = 0;
            }
        }

        dst_idx = 0;

        for(j=1; j<=stride_levels; j++) {
            dst_idx += dst_bvalue[j] * dst_stride_ar[j-1];
            if((i+1) % dst_bunit[j] == 0) {
                dst_bvalue[j]++;
            }
            if(dst_bvalue[j] > (count[j]-1)) {
                dst_bvalue[j] = 0;
            }
        }
        comex_acc(datatype, scale, (char*)src_ptr+src_idx, (char*)dst_ptr+dst_idx,
            count[0], proc, group);
    }

    return COMEX_SUCCESS;
#else
    MPI_Datatype src_type, dst_type;
    MPI_Aint displ;
    void *ptr;
    int lproc, i, ierr;
    void *packbuf;
    int bufsize;
    int new_strides[7], new_count[7];
    reg_entry_t *reg_win;
#ifdef USE_MPI_REQUESTS
    MPI_Request request;
    MPI_Status status;
#endif

    /* If data is contiguous, use comex_acc */
    if (stride_levels == 0) {
      return comex_acc(datatype, scale, src_ptr,
          dst_ptr, count[0], proc, group);
    }
    reg_win = reg_win_find(proc, dst_ptr, 0);
    ptr = reg_win->buf;
    displ = (MPI_Aint)(dst_ptr) - (MPI_Aint)(ptr);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    packbuf = malloc_strided_acc_buffer(src_ptr, src_stride_ar, count,
        stride_levels, &bufsize, new_strides);

    for (i=0; i<stride_levels+1; i++) new_count[i] = count[i];
    if (datatype == COMEX_ACC_INT) {
      int *buf;
      int iscale = *((int*)scale);
      int nvar = bufsize/sizeof(int);
      buf = (int*)packbuf;
      for (i=0; i<nvar; i++) buf[i] = iscale*buf[i];
      new_count[0] = new_count[0]/sizeof(int);
      strided_to_subarray_dtype(new_strides, new_count, stride_levels,
          MPI_INT, &src_type);
      strided_to_subarray_dtype(dst_stride_ar, new_count, stride_levels,
          MPI_INT, &dst_type);
    } else if (datatype == COMEX_ACC_LNG) {
      long *buf;
      long lscale = *((long*)scale);
      int nvar = bufsize/sizeof(long);
      buf = (long*)packbuf;
      for (i=0; i<nvar; i++) buf[i] = lscale*buf[i];
      new_count[0] = new_count[0]/sizeof(long);
      strided_to_subarray_dtype(new_strides, new_count, stride_levels,
          MPI_LONG, &src_type);
      strided_to_subarray_dtype(dst_stride_ar, new_count, stride_levels,
          MPI_LONG, &dst_type);
    } else if (datatype == COMEX_ACC_FLT) {
      float *buf;
      float fscale = *((float*)scale);
      int nvar = bufsize/sizeof(float);
      buf = (float*)packbuf;
      for (i=0; i<nvar; i++) buf[i] = fscale*buf[i];
      new_count[0] = new_count[0]/sizeof(float);
      strided_to_subarray_dtype(new_strides, new_count, stride_levels,
          MPI_FLOAT, &src_type);
      strided_to_subarray_dtype(dst_stride_ar, new_count, stride_levels,
          MPI_FLOAT, &dst_type);
    } else if (datatype == COMEX_ACC_DBL) {
      double *buf;
      double dscale = *((double*)scale);
      int nvar = bufsize/sizeof(double);
      buf = (double*)packbuf;
      for (i=0; i<nvar; i++) buf[i] = dscale*buf[i];
      new_count[0] = new_count[0]/sizeof(double);
      strided_to_subarray_dtype(new_strides, new_count, stride_levels,
          MPI_DOUBLE, &src_type);
      strided_to_subarray_dtype(dst_stride_ar, new_count, stride_levels,
          MPI_DOUBLE, &dst_type);
    } else if (datatype == COMEX_ACC_CPL) {
      float *buf;
      float crscale = *((float*)scale);
      float ciscale = *((float*)scale+1);
      int nvar = bufsize/(2*sizeof(float));
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
    } else if (datatype == COMEX_ACC_DCP) {
      double *buf;
      double crscale = *((double*)scale);
      double ciscale = *((double*)scale+1);
      int nvar = bufsize/(2*sizeof(double));
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
    translate_mpi_error(ierr,"comex_accs:MPI_Type_commit");
    ierr = MPI_Type_commit(&dst_type);
    translate_mpi_error(ierr,"comex_accs:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_accs:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Accumulate(packbuf,1,src_type,lproc,displ,1,dst_type,
        MPI_SUM,reg_win->win);
    translate_mpi_error(ierr,"comex_accs:MPI_Accumulate");
    ierr = MPI_Win_flush_local(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_accs:MPI_Win_flush_local");
#else
    ierr = MPI_Raccumulate(packbuf,1,src_type,lproc,displ,1,dst_type,
        MPI_SUM,reg_win->win,&request);
    translate_mpi_error(ierr,"comex_accs:MPI_Raccumulate");
    ierr = MPI_Wait(&request, &status);
    translate_mpi_error(ierr,"comex_accs:MPI_Wait");
#ifdef USE_POST_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_accs:MPI_Win_flush");
#endif
#endif
#else
    ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,reg_win->win);
    translate_mpi_error(ierr,"comex_accs:MPI_Win_lock");
    ierr = MPI_Accumulate(packbuf,1,src_type,lproc,displ,1,dst_type,
        MPI_SUM,reg_win->win);
    translate_mpi_error(ierr,"comex_accs:MPI_Accumulate");
    ierr = MPI_Win_unlock(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_accs:MPI_Win_unlock");
#endif
    ierr = MPI_Type_free(&src_type);
    translate_mpi_error(ierr,"comex_accs:MPI_Type_free");
    ierr = MPI_Type_free(&dst_type);
    translate_mpi_error(ierr,"comex_accs:MPI_Type_free");
    free(packbuf);

    return COMEX_SUCCESS;
#endif
}


/**
 * Utility function to create MPI data type for vector data object
 * src_ptr: location of first data point in vector data object for source
 * dst_ptr: location of first data point in vector data object for destination
 * iov: comex vector struct containing data information
 * iov_len: number of elements in vector data object
 * base_type: MPI_Datatype of elements
 * type: returned MPI_Datatype corresponding to iov for source data
 * type: returned MPI_Datatype corresponding to iov for destination data
 */
void vector_to_struct_dtype(void* src_ptr, void *dst_ptr, comex_giov_t *iov,
    int iov_len, MPI_Datatype base_type, MPI_Datatype *src_type,
    MPI_Datatype *dst_type)
{
#if 0
  int i, size;
  int *blocklengths;
  MPI_Aint *displacements;
  MPI_Datatype *types;
  MPI_Aint displ;
  /* allocate buffers to create data types */
  blocklengths = (int*)malloc(iov_len*sizeof(int));
  displacements = (MPI_Aint*)malloc(iov_len*sizeof(MPI_Aint));
  types = (MPI_Datatype*)malloc(iov_len*sizeof(MPI_Datatype));
  for (i=0; i<iov_len; i++) {
    displ = (MPI_Aint)iov[i].src[0]-(MPI_Aint)src_ptr;
    blocklengths[i] = iov[i].count;
    displacements[i] = displ;
    types[i] = base_type;
  }
  MPI_Type_create_struct(iov_len, blocklengths, displacements, types,
      src_type);
  for (i=0; i<iov_len; i++) {
    displ = (MPI_Aint)iov[i].dst[0]-(MPI_Aint)dst_ptr;
    displacements[i] = displ;
  }
  MPI_Type_create_struct(iov_len, blocklengths, displacements, types,
      dst_type);
#else
  int i, j, size, ierr;
  int nelems, icnt;
  /* find total number of elements */
  nelems = 0;
  for (i=0; i<iov_len; i++) {
    nelems += iov[i].count;
  }
  int *blocklengths;
  MPI_Aint *displacements;
  MPI_Datatype *types;
  MPI_Aint displ;
  MPI_Type_size(base_type,&size);
  /* allocate buffers to create data types */
  blocklengths = (int*)malloc(nelems*sizeof(int));
  displacements = (MPI_Aint*)malloc(nelems*sizeof(MPI_Aint));
  types = (MPI_Datatype*)malloc(nelems*sizeof(MPI_Datatype));
  icnt = 0;
  for (i=0; i<iov_len; i++) {
    for (j=0; j<iov[i].count; j++) {
      displ = (MPI_Aint)iov[i].src[j]-(MPI_Aint)src_ptr;
      blocklengths[icnt] = iov[i].bytes/size;
      displacements[icnt] = displ;
      types[icnt] = base_type;
      icnt++;
    }
  }
  ierr = MPI_Type_create_struct(nelems, blocklengths, displacements, types,
      src_type);
  translate_mpi_error(ierr,"vector_to_struct_dtype:MPI_Type_create_struct");
  icnt = 0;
  for (i=0; i<iov_len; i++) {
    for (j=0; j<iov[i].count; j++) {
      displ = (MPI_Aint)iov[i].dst[j]-(MPI_Aint)dst_ptr;
      displacements[icnt] = displ;
      icnt++;
    }
  }
  ierr = MPI_Type_create_struct(nelems, blocklengths, displacements, types,
      dst_type);
  translate_mpi_error(ierr,"vector_to_struct_dtype:MPI_Type_create_struct");
#endif
}

int comex_putv(
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group)
{
#ifndef USE_MPI_DATATYPES
    int status;
    int i;

    for (i=0; i<iov_len; ++i) {
        int j;
        void **src = iov[i].src;
        void **dst = iov[i].dst;
        int bytes = iov[i].bytes;
        int limit = iov[i].count;
        for (j=0; j<limit; ++j) {
            status = comex_put(src[j], dst[j], bytes, proc, group);
            assert(status == COMEX_SUCCESS);
        }
    }

    return COMEX_SUCCESS;
#else
    MPI_Datatype src_type, dst_type;
    MPI_Aint displ;
    void *ptr, *src_ptr, *dst_ptr;
    int lproc, ierr;
    reg_entry_t *reg_win;
#ifdef USE_MPI_REQUESTS
    MPI_Request request;
    MPI_Status status;
#endif
    src_ptr = iov[0].src[0];
    dst_ptr = iov[0].dst[0];
    reg_win = reg_win_find(proc, dst_ptr, 0);
    ptr = reg_win->buf;
    displ = 0;
    vector_to_struct_dtype(src_ptr, ptr, iov, iov_len,
        MPI_BYTE, &src_type, &dst_type);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    ierr = MPI_Type_commit(&src_type);
    translate_mpi_error(ierr,"comex_putv:MPI_Type_commit");
    ierr = MPI_Type_commit(&dst_type);
    translate_mpi_error(ierr,"comex_putv:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_putv:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Put(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
        reg_win->win);
    translate_mpi_error(ierr,"comex_putv:MPI_Put");
    ierr = MPI_Win_flush_local(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_putv:MPI_Win_flush_local");
#else
    ierr = MPI_Rput(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
        reg_win->win, &request);
    translate_mpi_error(ierr,"comex_putv:MPI_Rput");
    ierr = MPI_Wait(&request, &status);
    translate_mpi_error(ierr,"comex_putv:MPI_Wait");
#ifdef USE_POST_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_putv:MPI_Win_flush");
#endif
#endif
#else
    ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,reg_win->win);
    translate_mpi_error(ierr,"comex_putv:MPI_Win_lock");
    ierr = MPI_Put(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
        reg_win->win);
    translate_mpi_error(ierr,"comex_putv:MPI_Put");
    ierr = MPI_Win_unlock(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_putv:MPI_Win_unlock");
#endif
    ierr = MPI_Type_free(&src_type);
    translate_mpi_error(ierr,"comex_putv:MPI_Type_free");
    ierr = MPI_Type_free(&dst_type);
    translate_mpi_error(ierr,"comex_putv:MPI_Type_free");
    return COMEX_SUCCESS;
#endif
}

int comex_getv(
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group)
{
#ifndef USE_MPI_DATATYPES
    int status;
    int i;

    for (i=0; i<iov_len; ++i) {
        int j;
        void **src = iov[i].src;
        void **dst = iov[i].dst;
        int bytes = iov[i].bytes;
        int limit = iov[i].count;
        for (j=0; j<limit; ++j) {
            status = comex_get(src[j], dst[j], bytes, proc, group);
            assert(status == COMEX_SUCCESS);
        }
    }

    return COMEX_SUCCESS;
#else
    MPI_Datatype src_type, dst_type;
    MPI_Aint displ;
    void *ptr, *src_ptr, *dst_ptr;
    int lproc, ierr;
    reg_entry_t *reg_win;
#ifdef USE_MPI_REQUESTS
    MPI_Request request;
    MPI_Status status;
#endif
    src_ptr = iov[0].src[0];
    dst_ptr = iov[0].dst[0];
    reg_win = reg_win_find(proc, src_ptr, 0);
    ptr = reg_win->buf;
    displ = 0;
    vector_to_struct_dtype(ptr, dst_ptr, iov, iov_len,
        MPI_BYTE, &src_type, &dst_type);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    ierr = MPI_Type_commit(&src_type);
    translate_mpi_error(ierr,"comex_getv:MPI_Type_commit");
    ierr = MPI_Type_commit(&dst_type);
    translate_mpi_error(ierr,"comex_getv:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_getv:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Get(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
        reg_win->win);
    translate_mpi_error(ierr,"comex_getv:MPI_Get");
    ierr = MPI_Win_flush_local(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_getv:MPI_Win_flush_local");
#else
    ierr = MPI_Rget(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
        reg_win->win, &request);
    translate_mpi_error(ierr,"comex_getv:MPI_Rget");
    ierr = MPI_Wait(&request, &status);
    translate_mpi_error(ierr,"comex_getv:MPI_Wait");
#ifdef USE_POST_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_getv:MPI_Win_flush");
#endif
#endif
#else
    ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,reg_win->win);
    translate_mpi_error(ierr,"comex_getv:MPI_Win_lock");
    ierr = MPI_Get(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
        reg_win->win);
    translate_mpi_error(ierr,"comex_getv:MPI_Get");
    ierr = MPI_Win_unlock(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_getv:MPI_Win_unlock");
#endif
    ierr = MPI_Type_free(&src_type);
    translate_mpi_error(ierr,"comex_getv:MPI_Type_free");
    ierr = MPI_Type_free(&dst_type);
    translate_mpi_error(ierr,"comex_getv:MPI_Type_free");
    return COMEX_SUCCESS;
#endif
}

/**
 * Utility function to create MPI data type for vector data object used in
 * vector accumulate operation. This also handles creation of a temporary buffer
 * to handle scaling of the accumulated values. The returned buffer must be
 * deallocated using free
 * dst_ptr: location of first data point in vector data object for destination
 * iov: comex vector struct containing data information
 * iov_len: number of elements in vector data object
 * base_type: MPI_Datatype of elements
 * type: returned MPI_Datatype corresponding to iov for source data
 * type: returned MPI_Datatype corresponding to iov for destination data
 */
void* create_vector_buf_and_dtypes(void *dst_ptr,
    comex_giov_t *iov, int iov_len, int comex_size,
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
  ratio = comex_size/size;
#if DEBUG
  printf("p[%d] Ratio: %d comex_size: %d size: %d\n",l_state.rank,
      ratio,comex_size,size);
#endif
  src_buf=malloc(nelems*comex_size);
  if (base_type == MPI_INT) {
    int *buf = (int*)src_buf;
    int iscale = *((int*)scale);
    icnt = 0;
    for (i=0; i<iov_len; i++) {
      for (j=0; j<iov[i].count; j++) {
        buf[icnt] = *((int*)(iov[i].src[j]));
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
        buf[icnt] = *((long*)(iov[i].src[j]));
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
        buf[icnt] = *((float*)(iov[i].src[j]));
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
        buf[icnt] = *((double*)(iov[i].src[j]));
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
        rval = *((float*)(iov[i].src[j]));
        ival = *((float*)(iov[i].src[j])+1);
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
        rval = *((double*)(iov[i].src[j]));
        ival = *((double*)(iov[i].src[j])+1);
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
  ierr =MPI_Type_create_struct(nelems, blocklengths, displacements, types,
      src_type);
  translate_mpi_error(ierr,"create_vector_buf_and_dtypes:MPI_Type_create_struct");
  icnt = 0;
  for (i=0; i<iov_len; i++) {
    for (j=0; j<iov[i].count; j++) {
      displ = (MPI_Aint)iov[i].dst[j]-(MPI_Aint)dst_ptr;
      displacements[icnt] = displ;
      icnt++;
    }
  }
  ierr = MPI_Type_create_struct(nelems, blocklengths, displacements, types,
      dst_type);
  translate_mpi_error(ierr,"create_vector_buf_and_dtypes:MPI_Type_create_struct");
  return src_buf;
}

int comex_accv(
        int datatype, void *scale,
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group)
{
#ifndef USE_MPI_DATATYPES
    int i;
    
    for (i=0; i<iov_len; ++i) {
        int j;
        void **src = iov[i].src;
        void **dst = iov[i].dst;
        int bytes = iov[i].bytes;
        int limit = iov[i].count;
        for (j=0; j<limit; ++j) {
            comex_acc(datatype, scale, src[j], dst[j], bytes, proc, group);
        }
    }

    return COMEX_SUCCESS;
#else
    MPI_Datatype src_type, dst_type;
    MPI_Aint displ;
    MPI_Datatype base_type;
    void *ptr, *src_ptr, *dst_ptr;
    int ierr;
#ifdef USE_MPI_REQUESTS
    MPI_Request request;
    MPI_Status status;
#endif
    int lproc, size;
    if (datatype == COMEX_ACC_INT) {
      size = sizeof(int);
      base_type = MPI_INT;
    } else if (datatype == COMEX_ACC_LNG) {
      size = sizeof(long);
      base_type = MPI_LONG;
    } else if (datatype == COMEX_ACC_FLT) {
      size = sizeof(float);
      base_type = MPI_FLOAT;
    } else if (datatype == COMEX_ACC_DBL) {
      size = sizeof(double);
      base_type = MPI_DOUBLE;
    } else if (datatype == COMEX_ACC_CPL) {
      size = 2*sizeof(float);
      base_type = MPI_FLOAT;
    } else if (datatype == COMEX_ACC_DCP) {
      size = 2*sizeof(double);
      base_type = MPI_DOUBLE;
    }
    reg_entry_t *reg_win;
    dst_ptr = iov[0].dst[0];
    reg_win = reg_win_find(proc, dst_ptr, 0);
    ptr = reg_win->buf;
    displ = 0;
    src_ptr = create_vector_buf_and_dtypes(ptr, iov,
        iov_len, size, scale, base_type, &src_type, &dst_type);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    ierr = MPI_Type_commit(&src_type);
    translate_mpi_error(ierr,"comex_accv:MPI_Type_commit");
    ierr = MPI_Type_commit(&dst_type);
    translate_mpi_error(ierr,"comex_accv:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_accv:MPI_Win_flush");
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Accumulate(src_ptr,1,src_type,lproc,displ,1,dst_type,
        MPI_SUM,reg_win->win);
    translate_mpi_error(ierr,"comex_accv:MPI_Accumulate");
    ierr = MPI_Win_flush_local(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_accv:MPI_Win_flush_local");
#else
    ierr = MPI_Raccumulate(src_ptr,1,src_type,lproc,displ,1,dst_type,
        MPI_SUM,reg_win->win,&request);
    translate_mpi_error(ierr,"comex_accv:MPI_Raccumulate");
    ierr = MPI_Wait(&request, &status);
    translate_mpi_error(ierr,"comex_accv:MPI_Wait");
#ifdef USE_POST_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_accv:MPI_Win_flush");
#endif
#endif
#else
    ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,reg_win->win);
    translate_mpi_error(ierr,"comex_accv:MPI_Win_lock");
    ierr = MPI_Accumulate(src_ptr,1,src_type,lproc,displ,1,dst_type,
        MPI_SUM,reg_win->win);
    translate_mpi_error(ierr,"comex_accv:MPI_Accumulate");
    ierr = MPI_Win_unlock(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_accv:MPI_Win_unlock");
#endif
    ierr = MPI_Type_free(&src_type);
    translate_mpi_error(ierr,"comex_accv:MPI_Type_free");
    ierr = MPI_Type_free(&dst_type);
    translate_mpi_error(ierr,"comex_accv:MPI_Type_free");
    free(src_ptr);
    return COMEX_SUCCESS;
#endif
}


int comex_fence_all(comex_group_t group)
{
    return comex_wait_all(group);
}


int comex_fence_proc(int proc, comex_group_t group)
{
  comex_igroup_t *igroup = NULL;
  win_link_t *curr_win;
  int ierr;
  igroup = comex_get_igroup_from_group(group);
  if (igroup != NULL) {
    curr_win = igroup->win_list;
    while (curr_win != NULL) {
      ierr = MPI_Win_flush(proc, curr_win->win);
      translate_mpi_error(ierr,"comex_fence_proc:MPI_Win_flush");
      curr_win = curr_win->next;
    }
  }
  return COMEX_SUCCESS;
}


/* comex_barrier is comex_fence_all + MPI_Barrier */
int comex_barrier(comex_group_t group)
{
    MPI_Comm comm;
    int ierr;

    comex_fence_all(group);
    ierr = comex_group_comm(group, &comm);
    assert(COMEX_SUCCESS == ierr);
    MPI_Barrier(comm);

    return COMEX_SUCCESS;
}


void *comex_malloc_local(size_t size)
{
  void *ptr;
  if (size == 0) {
    ptr = NULL;
  } else {
    MPI_Aint tsize;
    int ierr;
    tsize = size;
#if USEX_SICM
    sicm_device_list *idevice = &device_dram;

#if TEST_SICM
    char cdevice[32];
#ifdef TEST_SICM_DEV
    strcpy(cdevice, STR(TEST_SICM_DEV));
    if (!strncmp(cdevice, "knl_hbm", 7)){
       idevice = &device_knl_hbm;
    }else if (!strncmp(cdevice, "ppc_hbm", 7)){
       idevice = &device_ppc_hbm;
    }
#endif
#endif

    sicm_arena arena = sicm_arena_create(0, 0, idevice);
    ptr = sicm_arena_alloc(arena, size);
    if(!ptr){
      fprintf(stderr, "Error in allocating the pool\n");
      exit(11);
    }
#else
    ierr = MPI_Alloc_mem(tsize,MPI_INFO_NULL,&ptr);
    translate_mpi_error(ierr,"comex_malloc_local:MPI_Alloc_mem");
#endif
  }
  return ptr;
}


int comex_free_local(void *ptr)
{
    if (ptr != NULL) {
      int ierr;
#if USEX_SICM
      sicm_free(ptr);
#else
      ierr = MPI_Free_mem(ptr);
      translate_mpi_error(ierr,"comex_free_local:MPI_Free_mem");
#endif 
    }

    return COMEX_SUCCESS;
}


int comex_finalize()
{
    int i, ierr;
    /* it's okay to call multiple times -- extra calls are no-ops */
    if (!initialized) {
        return COMEX_SUCCESS;
    }

    initialized = 0;

    /* Make sure that all outstanding operations are done */
    comex_wait_all(COMEX_GROUP_WORLD);
    
    /* groups */
    comex_group_finalize();

    MPI_Barrier(l_state.world_comm);

    /* destroy the communicators */
#if 0
    ierr = MPI_Comm_free(&l_state.world_comm);
    translate_mpi_error(ierr,"comex_finalize:MPI_Comm_free");
#endif

    /* Clean up request list */
#ifdef USE_MPI_REQUESTS
    for (i=0; i<nb_max_outstanding; i++) {
      free(nb_list[i]);
    }
    free(nb_list);
#endif

#if USE_SICM
    sicm_fini();
#endif    

    return COMEX_SUCCESS;
}


int comex_wait_proc(int proc, comex_group_t group)
{
    assert(0);

    return COMEX_SUCCESS;
}


int comex_wait(comex_request_t* hdl)
{
  int ierr;
#ifndef USE_MPI_DATATYPES
  return COMEX_SUCCESS;
#endif
#ifdef USE_MPI_REQUESTS
#ifdef USE_MPI_FLUSH_LOCAL
  ierr = MPI_Win_flush_local(nb_list[*hdl]->remote_proc,nb_list[*hdl]->win);
  translate_mpi_error(ierr,"comex_wait:MPI_Win_flush_local");
#else
  MPI_Status status;
  ierr = MPI_Wait(&(nb_list[*hdl]->request),&status);
  translate_mpi_error(ierr,"comex_wait:MPI_Wait");
#ifdef USE_POST_MPI_WIN_FLUSH
  ierr = MPI_Win_flush(nb_list[*hdl]->remote_proc,nb_list[*hdl]->win);
  translate_mpi_error(ierr,"comex_wait:MPI_Win_flush_local");
#endif
#endif
  if (nb_list[*hdl]->active == 0) {
    printf("p[%d] comex_wait Error: handle not active\n",l_state.rank);
  }
  nb_list[*hdl]->active = 0;
  if (nb_list[*hdl]->use_type) {
    ierr = MPI_Type_free(&(nb_list[*hdl]->src_type));
    translate_mpi_error(ierr,"comex_wait:MPI_Type_free");
    ierr = MPI_Type_free(&(nb_list[*hdl]->dst_type));
    translate_mpi_error(ierr,"comex_wait:MPI_Type_free");
    nb_list[*hdl]->use_type = 0;
  }
  return COMEX_SUCCESS;
#else
  /* Non-blocking functions not implemented */
  return COMEX_SUCCESS;
#endif
}


/* returns 0 if operation completed */

int comex_test(comex_request_t* hdl, int *status)
{
#ifndef USE_MPI_DATATYPES
  *status = 0;
  return COMEX_SUCCESS;
#endif
#ifdef USE_MPI_REQUESTS
    int flag;
    int ierr;
    MPI_Status stat;
    ierr = MPI_Test(&(nb_list[*hdl]->request),&flag,&stat);
    translate_mpi_error(ierr,"comex_test:MPI_Test");
    if (nb_list[*hdl]->active == 0) {
      printf("comex_test Error: handle not active\n");
    }
    if (flag) {
      /* operation is complete */
      *status = 0;
      nb_list[*hdl]->active = 0;
      if (nb_list[*hdl]->use_type) {
        ierr = MPI_Type_free(&(nb_list[*hdl]->src_type));
        translate_mpi_error(ierr,"comex_wait:MPI_Type_free");
        ierr = MPI_Type_free(&(nb_list[*hdl]->dst_type));
        translate_mpi_error(ierr,"comex_wait:MPI_Type_free");
        nb_list[*hdl]->use_type = 0;
      }
    } else {
      /* operation is incomplete */
      *status = 1;
    }
    return COMEX_SUCCESS;
#else
    *status = 0;
    return COMEX_SUCCESS;
#endif
}


int comex_wait_all(comex_group_t group)
{
    comex_igroup_t *igroup = NULL;
    win_link_t *curr_win;
    int ierr;
    igroup = comex_get_igroup_from_group(group);
    if (igroup != NULL) {
      curr_win = igroup->win_list;
      while (curr_win != NULL) {
#ifdef USE_MPI_REQUESTS
        ierr = MPI_Win_flush_all(curr_win->win);
        translate_mpi_error(ierr,"comex_wait_all:MPI_Win_flush_all");
#else
        ierr = MPI_Win_fence(0,curr_win->win);
        translate_mpi_error(ierr,"comex_wait_all:MPI_Win_fence");
#endif
        curr_win = curr_win->next;
      }
    }
    return COMEX_SUCCESS;
}


int comex_nbput(
        void *src, void *dst, int bytes,
        int proc, comex_group_t group,
        comex_request_t *hdl)
{
#ifndef USE_MPI_DATATYPES
    return comex_put(src, dst, bytes, proc, group);
#endif
#ifdef USE_MPI_REQUESTS
    if (hdl == NULL) {
      return comex_put(src, dst, bytes, proc, group);
    }
    MPI_Aint displ;
    void *ptr;
    int lproc, ierr;
    nb_t *req;
    reg_entry_t *reg_win;
    MPI_Request request;
    reg_win = reg_win_find(proc, dst, 0);
    ptr = reg_win->buf;
    displ = (MPI_Aint)(dst) - (MPI_Aint)(ptr);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    get_nb_request(hdl, &req);
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_nbput:MPI_Win_flush");
#endif
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Put(src, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
        reg_win->win);
    translate_mpi_error(ierr,"comex_nbput:MPI_Put");
    req->remote_proc = lproc;
    req->win = reg_win->win;
#else
    ierr = MPI_Rput(src, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
        reg_win->win, &request);
    translate_mpi_error(ierr,"comex_nbput:MPI_Rput");
#endif
    req->request = request;
    req->use_type = 0;
    req->active = 1;
    return COMEX_SUCCESS;
#else
    return comex_put(src, dst, bytes, proc, group);
#endif
}


int comex_nbget(
        void *src, void *dst, int bytes,
        int proc, comex_group_t group,
        comex_request_t *hdl)
{
#ifndef USE_MPI_DATATYPES
    return comex_get(src, dst, bytes, proc, group);
#endif
#ifdef USE_MPI_REQUESTS
    if (hdl == NULL) {
      return comex_get(src, dst, bytes, proc, group);
    }
    MPI_Aint displ;
    void *ptr;
    int lproc, ierr;
    nb_t *req;
    reg_entry_t *reg_win;
    MPI_Request request;
    reg_win = reg_win_find(proc, src, 0);
    ptr = reg_win->buf;
    displ = (MPI_Aint)(src) - (MPI_Aint)(ptr);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    get_nb_request(hdl, &req);
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_nbget:MPI_Win_flush");
#endif
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Get(dst, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
        reg_win->win);
    translate_mpi_error(ierr,"comex_nbget:MPI_Get");
    req->remote_proc = lproc;
    req->win = reg_win->win;
#else
    ierr = MPI_Rget(dst, bytes, MPI_CHAR, lproc, displ, bytes, MPI_CHAR,
        reg_win->win, &request);
    translate_mpi_error(ierr,"comex_nbget:MPI_Rget");
#endif
    req->request = request;
    req->use_type = 0;
    req->active = 1;
    return COMEX_SUCCESS;
#else
    return comex_get(src, dst, bytes, proc, group);
#endif
}


int comex_nbacc(
        int datatype, void *scale,
        void *src_ptr, void *dst_ptr, int bytes,
        int proc, comex_group_t group,
        comex_request_t *hdl)
{
#ifndef USE_MPI_DATATYPES
    return comex_acc( datatype, scale, src_ptr, dst_ptr,
        bytes, proc, group);
#endif
#ifdef USE_MPI_REQUESTS
    if (hdl == NULL) {
      return comex_acc( datatype, scale,
          src_ptr, dst_ptr, bytes, proc, group);
    }
    MPI_Aint displ;
    void *ptr;
    int count, i, lproc, ierr;
    nb_t *req;
    reg_entry_t *reg_win;
    MPI_Request request;
    MPI_Status status;
    reg_win = reg_win_find(proc, dst_ptr, 0);
    ptr = reg_win->buf;
    displ = (MPI_Aint)(dst_ptr) - (MPI_Aint)(ptr);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    get_nb_request(hdl, &req);
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_nbput:MPI_Win_flush");
#endif
    if (datatype == COMEX_ACC_INT) {
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
          MPI_INT,MPI_SUM,reg_win->win);
      translate_mpi_error(ierr,"comex_nbacc:MPI_Accumulate");
      req->remote_proc = lproc;
      req->win = reg_win->win;
#else
      ierr = MPI_Raccumulate(buf,count,MPI_INT,lproc,displ,count,
          MPI_INT,MPI_SUM,reg_win->win,&request);
      translate_mpi_error(ierr,"comex_nbacc:MPI_Raccumulate");
#endif
      req->request = request;
      req->use_type = 0;
      req->active = 1;
      free(buf);
    } else if (datatype == COMEX_ACC_LNG) {
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
          MPI_LONG,MPI_SUM,reg_win->win);
      translate_mpi_error(ierr,"comex_nbacc:MPI_Accumulate");
      req->remote_proc = lproc;
      req->win = reg_win->win;
#else
      ierr = MPI_Raccumulate(buf,count,MPI_LONG,lproc,displ,count,
          MPI_LONG,MPI_SUM,reg_win->win,&request);
      translate_mpi_error(ierr,"comex_nbacc:MPI_Raccumulate");
#endif
      req->request = request;
      req->use_type = 0;
      req->active = 1;
      free(buf);
    } else if (datatype == COMEX_ACC_FLT) {
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
          MPI_FLOAT,MPI_SUM,reg_win->win);
      translate_mpi_error(ierr,"comex_nbacc:MPI_Accumulate");
      req->remote_proc = lproc;
      req->win = reg_win->win;
#else
      ierr = MPI_Raccumulate(buf,count,MPI_FLOAT,lproc,displ,count,
          MPI_FLOAT,MPI_SUM,reg_win->win,&request);
      translate_mpi_error(ierr,"comex_nbacc:MPI_Raccumulate");
#endif
      req->request = request;
      req->use_type = 0;
      req->active = 1;
      free(buf);
    } else if (datatype == COMEX_ACC_DBL) {
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
          MPI_DOUBLE,MPI_SUM,reg_win->win);
      translate_mpi_error(ierr,"comex_nbacc:MPI_Accumulate");
      req->remote_proc = lproc;
      req->win = reg_win->win;
#else
      ierr = MPI_Raccumulate(buf,count,MPI_DOUBLE,lproc,displ,count,
          MPI_DOUBLE,MPI_SUM,reg_win->win,&request);
      translate_mpi_error(ierr,"comex_nbacc:MPI_Raccumulate");
#endif
      req->request = request;
      req->use_type = 0;
      req->active = 1;
      free(buf);
    } else if (datatype == COMEX_ACC_CPL) {
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
          MPI_FLOAT,MPI_SUM,reg_win->win);
      translate_mpi_error(ierr,"comex_nbacc:MPI_Accumulate");
      req->remote_proc = lproc;
      req->win = reg_win->win;
#else
      ierr = MPI_Raccumulate(buf,count,MPI_FLOAT,lproc,displ,count,
          MPI_FLOAT,MPI_SUM,reg_win->win,&request);
      translate_mpi_error(ierr,"comex_nbacc:MPI_Raccumulate");
#endif
      req->request = request;
      req->use_type = 0;
      req->active = 1;
      free(buf);
    } else if (datatype == COMEX_ACC_DCP) {
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
          MPI_DOUBLE,MPI_SUM,reg_win->win);
      translate_mpi_error(ierr,"comex_nbacc:MPI_Accumulate");
      req->remote_proc = lproc;
      req->win = reg_win->win;
#else
      ierr = MPI_Raccumulate(buf,count,MPI_DOUBLE,lproc,displ,count,
          MPI_DOUBLE,MPI_SUM,reg_win->win,&request);
      translate_mpi_error(ierr,"comex_nbacc:MPI_Raccumulate");
#endif
      req->request = request;
      req->use_type = 0;
      req->active = 1;
      free(buf);
    } else {
      assert(0);
    }
    return COMEX_SUCCESS;
#else
    return comex_acc( datatype, scale,
            src_ptr, dst_ptr, bytes, proc, group);
#endif
}


int comex_nbputs(
        void *src, int *src_stride,
        void *dst, int *dst_stride,
        int *count, int stride_levels, 
        int proc, comex_group_t group,
        comex_request_t *hdl)
{
#ifndef USE_MPI_DATATYPES
    return comex_puts(src, src_stride, dst, dst_stride,
            count, stride_levels, proc, group);
    hdl = NULL;
#else
#ifdef USE_MPI_REQUESTS
    MPI_Datatype src_type, dst_type;
    MPI_Aint displ;
    void *ptr;
    int lproc, ierr;
    reg_entry_t *reg_win;
    nb_t *req;
    MPI_Request request;
    MPI_Status status;
    if (hdl == NULL) {
      return comex_puts(src, src_stride, dst, dst_stride,
          count, stride_levels, proc, group);
    }
    reg_win = reg_win_find(proc, dst, 0);
    ptr = reg_win->buf;
    displ = (MPI_Aint)(dst) - (MPI_Aint)(ptr);

    strided_to_subarray_dtype(src_stride, count, stride_levels,
      MPI_BYTE, &src_type);
    strided_to_subarray_dtype(dst_stride, count, stride_levels,
      MPI_BYTE, &dst_type);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    get_nb_request(hdl, &req);
    ierr = MPI_Type_commit(&src_type);
    translate_mpi_error(ierr,"comex_nbputs:MPI_Type_commit");
    ierr = MPI_Type_commit(&dst_type);
    translate_mpi_error(ierr,"comex_nbputs:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_nbputs:MPI_Win_flush");
#endif
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Put(src, 1, src_type, lproc, displ, 1, dst_type,
        reg_win->win);
    translate_mpi_error(ierr,"comex_nbputs:MPI_Put");
    req->remote_proc = lproc;
    req->win = reg_win->win;
#else
    ierr = MPI_Rput(src, 1, src_type, lproc, displ, 1, dst_type,
        reg_win->win, &request);
    translate_mpi_error(ierr,"comex_nbputs:MPI_Rput");
#endif
    req->request = request;
    req->use_type = 1;
    req->src_type = src_type;
    req->dst_type = dst_type;
    req->remote_proc = lproc;
    req->win = reg_win->win;
    req->active = 1;
    return COMEX_SUCCESS;
#else
    return comex_puts(src, src_stride, dst, dst_stride,
            count, stride_levels, proc, group);
#endif
#endif
}


int comex_nbgets(
        void *src, int *src_stride,
        void *dst, int *dst_stride,
        int *count, int stride_levels, 
        int proc, comex_group_t group,
        comex_request_t *hdl) 
{
#ifndef USE_MPI_DATATYPES
    hdl = NULL;
    return comex_gets(src, src_stride, dst, dst_stride,
            count, stride_levels, proc, group);
#else
#ifdef USE_MPI_REQUESTS
    if (hdl == NULL) {
      return comex_gets(src, src_stride, dst, dst_stride,
          count, stride_levels, proc, group);
    }
    MPI_Datatype src_type, dst_type;
    MPI_Aint displ;
    void *ptr;
    int lproc, ierr;
    reg_entry_t *reg_win;
    MPI_Request request;
    MPI_Status status;
    nb_t *req;
    reg_win = reg_win_find(proc, src, 0);
    ptr = reg_win->buf;
    displ = (MPI_Aint)(src) - (MPI_Aint)(ptr);

    strided_to_subarray_dtype(src_stride, count, stride_levels,
      MPI_BYTE, &src_type);
    strided_to_subarray_dtype(dst_stride, count, stride_levels,
      MPI_BYTE, &dst_type);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    get_nb_request(hdl, &req);
    ierr = MPI_Type_commit(&src_type);
    translate_mpi_error(ierr,"comex_nbgets:MPI_Type_commit");
    ierr = MPI_Type_commit(&dst_type);
    translate_mpi_error(ierr,"comex_nbgets:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_nbgets:MPI_Win_flush");
#endif
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Get(dst, 1, dst_type, lproc, displ, 1, src_type,
        reg_win->win);
    translate_mpi_error(ierr,"comex_nbgets:MPI_Get");
    req->remote_proc = lproc;
    req->win = reg_win->win;
#else
    ierr = MPI_Rget(dst, 1, dst_type, lproc, displ, 1, src_type,
        reg_win->win, &request);
    translate_mpi_error(ierr,"comex_nbgets:MPI_Rget");
#endif
    req->request = request;
    req->use_type = 1;
    req->src_type = src_type;
    req->dst_type = dst_type;
    req->remote_proc = lproc;
    req->win = reg_win->win;
    req->active = 1;
    return COMEX_SUCCESS;
#else
    return comex_gets(src, src_stride, dst, dst_stride,
            count, stride_levels, proc, group);
#endif
#endif
}


int comex_nbaccs(
        int datatype, void *scale,
        void *src, int *src_stride,
        void *dst, int *dst_stride,
        int *count, int stride_levels,
        int proc, comex_group_t group,
        comex_request_t *hdl)
{
#ifndef USE_MPI_DATATYPES
    hdl = NULL;
    return comex_accs(datatype, scale,
            src, src_stride, dst, dst_stride,
            count, stride_levels, proc, group);
#else
#ifdef USE_MPI_REQUESTS
    MPI_Datatype src_type, dst_type;
    MPI_Aint displ;
    void *ptr;
    int lproc, i, ierr;
    void *packbuf;
    int bufsize;
    int new_strides[7], new_count[7];
    reg_entry_t *reg_win;
    MPI_Request request;
    MPI_Status status;
    if (hdl == NULL) {
      return comex_accs(datatype, scale,
          src, src_stride, dst, dst_stride,
          count, stride_levels, proc, group);
    }
    /* If data is contiguous, use comex_acc */
    if (stride_levels == 0) {
      return comex_nbacc(datatype, scale, src, dst, count[0],
          proc, group, hdl);
    }
    nb_t *req;
    reg_win = reg_win_find(proc, dst, 0);
    ptr = reg_win->buf;
    displ = (MPI_Aint)(dst) - (MPI_Aint)(ptr);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    packbuf = malloc_strided_acc_buffer(src, src_stride, count,
        stride_levels, &bufsize, new_strides);

    for (i=0; i<stride_levels+1; i++) new_count[i] = count[i];
    if (datatype == COMEX_ACC_INT) {
      int *buf;
      int iscale = *((int*)scale);
      int nvar = bufsize/sizeof(int);
      buf = (int*)packbuf;
      for (i=0; i<nvar; i++) buf[i] = iscale*buf[i];
      new_count[0] = new_count[0]/sizeof(int);
      strided_to_subarray_dtype(new_strides, new_count, stride_levels,
          MPI_INT, &src_type);
      strided_to_subarray_dtype(dst_stride, new_count, stride_levels,
          MPI_INT, &dst_type);
    } else if (datatype == COMEX_ACC_LNG) {
      long *buf;
      long lscale = *((long*)scale);
      int nvar = bufsize/sizeof(long);
      buf = (long*)packbuf;
      for (i=0; i<nvar; i++) buf[i] = lscale*buf[i];
      new_count[0] = new_count[0]/sizeof(long);
      strided_to_subarray_dtype(new_strides, new_count, stride_levels,
          MPI_LONG, &src_type);
      strided_to_subarray_dtype(dst_stride, new_count, stride_levels,
          MPI_LONG, &dst_type);
    } else if (datatype == COMEX_ACC_FLT) {
      float *buf;
      float fscale = *((float*)scale);
      int nvar = bufsize/sizeof(float);
      buf = (float*)packbuf;
      for (i=0; i<nvar; i++) buf[i] = fscale*buf[i];
      new_count[0] = new_count[0]/sizeof(float);
      strided_to_subarray_dtype(new_strides, new_count, stride_levels,
          MPI_FLOAT, &src_type);
      strided_to_subarray_dtype(dst_stride, new_count, stride_levels,
          MPI_FLOAT, &dst_type);
    } else if (datatype == COMEX_ACC_DBL) {
      double *buf;
      double dscale = *((double*)scale);
      int nvar = bufsize/sizeof(double);
      buf = (double*)packbuf;
      for (i=0; i<nvar; i++) buf[i] = dscale*buf[i];
      new_count[0] = new_count[0]/sizeof(double);
      strided_to_subarray_dtype(new_strides, new_count, stride_levels,
          MPI_DOUBLE, &src_type);
      strided_to_subarray_dtype(dst_stride, new_count, stride_levels,
          MPI_DOUBLE, &dst_type);
    } else if (datatype == COMEX_ACC_CPL) {
      float *buf;
      float crscale = *((float*)scale);
      float ciscale = *((float*)scale+1);
      int nvar = bufsize/(2*sizeof(float));
      buf = (float*)packbuf;
      for (i=0; i<nvar; i++) {
        buf[2*i] = crscale*buf[2*i]-ciscale*buf[2*i+1];
        buf[2*i+1] = ciscale*buf[2*i]+crscale*buf[2*i+1];
      }
      new_count[0] = new_count[0]/sizeof(float);
      strided_to_subarray_dtype(new_strides, new_count, stride_levels,
          MPI_FLOAT, &src_type);
      strided_to_subarray_dtype(dst_stride, new_count, stride_levels,
          MPI_FLOAT, &dst_type);
    } else if (datatype == COMEX_ACC_DCP) {
      double *buf;
      double crscale = *((double*)scale);
      double ciscale = *((double*)scale+1);
      int nvar = bufsize/(2*sizeof(double));
      buf = (double*)packbuf;
      for (i=0; i<nvar; i++) {
        buf[2*i] = crscale*buf[2*i]-ciscale*buf[2*i+1];
        buf[2*i+1] = ciscale*buf[2*i]+crscale*buf[2*i+1];
      }
      new_count[0] = new_count[0]/sizeof(double);
      strided_to_subarray_dtype(new_strides, new_count, stride_levels,
          MPI_DOUBLE, &src_type);
      strided_to_subarray_dtype(dst_stride, new_count, stride_levels,
          MPI_DOUBLE, &dst_type);
    } else {
      assert(0);
    }
    get_nb_request(hdl, &req);
    ierr = MPI_Type_commit(&src_type);
    translate_mpi_error(ierr,"comex_nbaccs:MPI_Type_commit");
    ierr = MPI_Type_commit(&dst_type);
    translate_mpi_error(ierr,"comex_nbaccs:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_nbaccs:MPI_Win_flush");
#endif
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Accumulate(packbuf,1,src_type,lproc,displ,1,dst_type,
        MPI_SUM,reg_win->win);
    translate_mpi_error(ierr,"comex_nbaccs:MPI_Accumulate");
    req->remote_proc = lproc;
    req->win = reg_win->win;
#else
    ierr = MPI_Raccumulate(packbuf,1,src_type,lproc,displ,1,dst_type,
        MPI_SUM,reg_win->win,&request);
    translate_mpi_error(ierr,"comex_nbaccs:MPI_Rget");
#endif
    req->request = request;
    req->use_type = 1;
    req->src_type = src_type;
    req->dst_type = dst_type;
    req->remote_proc = lproc;
    req->win = reg_win->win;
    req->active = 1;
    free(packbuf);

    return COMEX_SUCCESS;
#else
    return comex_accs(datatype, scale,
            src, src_stride, dst, dst_stride,
            count, stride_levels, proc, group);
#endif
#endif
}


int comex_nbputv(
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group,
        comex_request_t* handle)
{
#ifndef USE_MPI_DATATYPES
    return comex_putv(iov, iov_len, proc, group);
#else
#ifdef USE_MPI_REQUESTS
    if (handle == NULL) {
      return comex_putv(iov, iov_len, proc, group);
    }
    MPI_Datatype src_type, dst_type;
    MPI_Aint displ;
    void *ptr, *src_ptr, *dst_ptr;
    int lproc, ierr;
    reg_entry_t *reg_win;
    MPI_Request request;
    MPI_Status status;
    nb_t *req;
    src_ptr = iov[0].src[0];
    dst_ptr = iov[0].dst[0];
    reg_win = reg_win_find(proc, dst_ptr, 0);
    ptr = reg_win->buf;
    displ = 0;
    vector_to_struct_dtype(src_ptr, ptr, iov, iov_len,
        MPI_BYTE, &src_type, &dst_type);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    get_nb_request(handle, &req);
    ierr = MPI_Type_commit(&src_type);
    translate_mpi_error(ierr,"comex_nbputv:MPI_Type_commit");
    ierr = MPI_Type_commit(&dst_type);
    translate_mpi_error(ierr,"comex_nbputv:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_nbputv:MPI_Win_flush");
#endif
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Put(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
        reg_win->win);
    translate_mpi_error(ierr,"comex_nbputv:MPI_Put");
    req->remote_proc = lproc;
    req->win = reg_win->win;
#else
    ierr = MPI_Rput(src_ptr, 1, src_type, lproc, displ, 1, dst_type,
        reg_win->win, &request);
    translate_mpi_error(ierr,"comex_nbputv:MPI_Rput");
#endif
    req->request = request;
    req->use_type = 1;
    req->src_type = src_type;
    req->dst_type = dst_type;
    req->remote_proc = lproc;
    req->win = reg_win->win;
    req->active = 1;
    return COMEX_SUCCESS;
#else
    return comex_putv(iov, iov_len, proc, group);
#endif
#endif
}


int comex_nbgetv(
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group,
        comex_request_t* handle)
{
#ifndef USE_MPI_DATATYPES
    return comex_getv(iov, iov_len, proc, group);
#else
#ifdef USE_MPI_REQUESTS
    if (handle == NULL) {
      return comex_getv(iov, iov_len, proc, group);
    }
    MPI_Datatype src_type, dst_type;
    MPI_Aint displ;
    void *ptr, *src_ptr, *dst_ptr;
    int lproc, ierr;
    reg_entry_t *reg_win;
    MPI_Request request;
    MPI_Status status;
    nb_t *req;
    src_ptr = iov[0].src[0];
    dst_ptr = iov[0].dst[0];
    reg_win = reg_win_find(proc, src_ptr, 0);
    ptr = reg_win->buf;
    displ = 0;
    vector_to_struct_dtype(ptr, dst_ptr, iov, iov_len,
        MPI_BYTE, &src_type, &dst_type);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    get_nb_request(handle, &req);
    ierr = MPI_Type_commit(&src_type);
    translate_mpi_error(ierr,"comex_nbgetv:MPI_Type_commit");
    ierr = MPI_Type_commit(&dst_type);
    translate_mpi_error(ierr,"comex_nbgetv:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_nbgetv:MPI_Win_flush");
#endif
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Get(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
        reg_win->win);
    translate_mpi_error(ierr,"comex_nbgetv:MPI_Get");
    req->remote_proc = lproc;
    req->win = reg_win->win;
#else
    ierr = MPI_Rget(dst_ptr, 1, dst_type, lproc, displ, 1, src_type,
        reg_win->win, &request);
    translate_mpi_error(ierr,"comex_nbgetv:MPI_Rget");
#endif
    req->request = request;
    req->use_type = 1;
    req->src_type = src_type;
    req->dst_type = dst_type;
    req->remote_proc = lproc;
    req->win = reg_win->win;
    req->active = 1;
    return COMEX_SUCCESS;
#else
    return comex_getv(iov, iov_len, proc, group);
#endif
#endif
}


int comex_nbaccv(
        int datatype, void *scale,
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group,
        comex_request_t* handle)
{
#ifndef USE_MPI_DATATYPES
    return comex_accv(datatype, scale, iov, iov_len, proc, group);
#else
#ifdef USE_MPI_REQUESTS
    if (handle == NULL) {
      return comex_accv(datatype, scale, iov, iov_len, proc, group);
    }
    MPI_Datatype src_type, dst_type;
    MPI_Aint displ;
    MPI_Datatype base_type;
    void *ptr, *src_ptr, *dst_ptr;
    MPI_Request request;
    MPI_Status status;
    nb_t *req;
    int lproc, size, ierr;
    if (datatype == COMEX_ACC_INT) {
      size = sizeof(int);
      base_type = MPI_INT;
    } else if (datatype == COMEX_ACC_LNG) {
      size = sizeof(long);
      base_type = MPI_LONG;
    } else if (datatype == COMEX_ACC_FLT) {
      size = sizeof(float);
      base_type = MPI_FLOAT;
    } else if (datatype == COMEX_ACC_DBL) {
      size = sizeof(double);
      base_type = MPI_DOUBLE;
    } else if (datatype == COMEX_ACC_CPL) {
      size = 2*sizeof(float);
      base_type = MPI_FLOAT;
    } else if (datatype == COMEX_ACC_DCP) {
      size = 2*sizeof(double);
      base_type = MPI_DOUBLE;
    }
    reg_entry_t *reg_win;
    dst_ptr = iov[0].dst[0];
    reg_win = reg_win_find(proc, dst_ptr, 0);
    ptr = reg_win->buf;
    displ = 0;
    src_ptr = create_vector_buf_and_dtypes(ptr, iov,
        iov_len, size, scale, base_type, &src_type, &dst_type);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
    get_nb_request(handle, &req);
    ierr = MPI_Type_commit(&src_type);
    translate_mpi_error(ierr,"comex_nbaccv:MPI_Type_commit");
    ierr = MPI_Type_commit(&dst_type);
    translate_mpi_error(ierr,"comex_nbaccv:MPI_Type_commit");
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc, reg_win->win);
    translate_mpi_error(ierr,"comex_nbaccv:MPI_Win_flush");
#endif
#ifdef USE_MPI_FLUSH_LOCAL
    ierr = MPI_Accumulate(src_ptr,1,src_type,lproc,displ,1,dst_type,
        MPI_SUM,reg_win->win);
    translate_mpi_error(ierr,"comex_nbaccv:MPI_Accumulate");
    req->remote_proc = lproc;
    req->win = reg_win->win;
#else
    ierr = MPI_Raccumulate(src_ptr,1,src_type,lproc,displ,1,dst_type,
        MPI_SUM,reg_win->win,&request);
    translate_mpi_error(ierr,"comex_nbaccv:MPI_Raccumulate");
#endif
    req->request = request;
    req->use_type = 1;
    req->src_type = src_type;
    req->dst_type = dst_type;
    req->remote_proc = lproc;
    req->win = reg_win->win;
    req->active = 1;
    free(src_ptr);
    return COMEX_SUCCESS;
#else
    return comex_accv(datatype, scale, iov, iov_len, proc, group);
#endif
#endif
}


int comex_rmw(
        int op, void *ploc, void *prem, int extra,
        int proc, comex_group_t group)
{
    MPI_Aint displ;
    void *ptr;
    int lproc, ierr;
    reg_entry_t *reg_win;
    reg_win = reg_win_find(proc, prem, 0);
    ptr = reg_win->buf;
    if (ptr == NULL) return COMEX_FAILURE;
    displ = (MPI_Aint)(prem) - (MPI_Aint)(ptr);
    if (!(get_local_rank_from_win(reg_win->win, proc, &lproc)
          == COMEX_SUCCESS)) {
      assert(0);
    }
#ifdef USE_PRIOR_MPI_WIN_FLUSH
    ierr = MPI_Win_flush(lproc,reg_win->win);
    translate_mpi_error(ierr,"comex_rmw:MPI_Win_flush");
#endif
    if (op == COMEX_FETCH_AND_ADD) {
      int incr = extra;
#ifdef USE_MPI_REQUESTS
      ierr = MPI_Fetch_and_op(&incr,ploc,MPI_INT,lproc,displ,
          MPI_SUM,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Fetch_and_op");
      ierr = MPI_Win_flush(lproc,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Win_flush");
#else
      ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Win_lock");
      ierr = MPI_Fetch_and_op(&incr,ploc,MPI_INT,lproc,displ,
          MPI_SUM,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Fetch_and_op");
      ierr = MPI_Win_unlock(lproc,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Win_unlock");
#endif
    } else if (op == COMEX_FETCH_AND_ADD_LONG) {
      long incr = extra;
#ifdef USE_MPI_REQUESTS
      ierr = MPI_Fetch_and_op(&incr,ploc,MPI_LONG,lproc,displ,
          MPI_SUM,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Fetch_and_op");
      ierr = MPI_Win_flush(lproc,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Win_flush");
#else
      ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Win_lock");
      ierr = MPI_Fetch_and_op(&incr,ploc,MPI_LONG,lproc,displ,
          MPI_SUM,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Fetch_and_op");
      ierr = MPI_Win_unlock(lproc,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Win_unlock");
#endif
    } else if (op == COMEX_SWAP) {
      int incr = *((int*)ploc);
#ifdef USE_MPI_REQUESTS
      ierr = MPI_Fetch_and_op(&incr,ploc,MPI_INT,lproc,displ,
          MPI_REPLACE,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Fetch_and_op");
      ierr = MPI_Win_flush(lproc,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Win_flush");
#else
      ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Win_lock");
      ierr = MPI_Fetch_and_op(&incr,ploc,MPI_INT,lproc,displ,
          MPI_REPLACE,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Fetch_and_op");
      ierr = MPI_Win_unlock(lproc,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Win_unlock");
#endif
    } else if (op == COMEX_SWAP_LONG) {
      long incr = *((long*)ploc);
#ifdef USE_MPI_REQUESTS
      ierr = MPI_Fetch_and_op(&incr,ploc,MPI_LONG,lproc,displ,
          MPI_REPLACE,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Fetch_and_op");
      ierr = MPI_Win_flush(lproc,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Win_flush");
#else
      ierr = MPI_Win_lock(MPI_LOCK_SHARED,lproc,0,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Win_lock");
      ierr = MPI_Fetch_and_op(&incr,ploc,MPI_LONG,lproc,displ,
          MPI_REPLACE,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Fetch_and_op");
      ierr = MPI_Win_unlock(lproc,reg_win->win);
      translate_mpi_error(ierr,"comex_rmw:MPI_Win_unlock");
#endif
    } else  {
        assert(0);
    }
    
    return COMEX_SUCCESS;
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
int comex_create_mutexes(int num)
{
  int i, j, k, idx, isize, nsize, ierr;
  comex_igroup_t *igroup = NULL;
  MPI_Comm comm;
  int *sbuf;
  int me = l_state.rank;
  int nproc = l_state.size;

  igroup = comex_get_igroup_from_group(COMEX_GROUP_WORLD);
  comm = igroup->comm;

  sbuf = (int*)malloc(nproc*sizeof(int));
  _mutex_num = (int*)malloc(nproc*sizeof(int));
  for (i=0; i<nproc; i++) sbuf[i] = 0;
  sbuf[me] = num;

  ierr = MPI_Allreduce(sbuf, _mutex_num, nproc, MPI_INT, MPI_SUM, comm);
  translate_mpi_error(ierr,"comex_create_mutexes:MPI_Allreduce");
  free(sbuf);

  _mutex_total = 0;
  for (i=0; i<nproc; i++) _mutex_total += _mutex_num[i];

  if (_mutex_list != NULL) comex_destroy_mutexes();
  _mutex_list = (MPI_Win*)malloc(_mutex_total*sizeof(MPI_Win));
  _mutex_buf = (int**)malloc(_mutex_total*sizeof(int*));
  isize = sizeof(int);

  /* create windows for mutexes */
  idx = 0;
  for (i=0; i<nproc; i++) {
    for (j=0; j<_mutex_num[i]; j++) {
      if (i == me) {
        nsize = isize*nproc;
        ierr = MPI_Alloc_mem(nsize,MPI_INFO_NULL,&_mutex_buf[idx]);
        translate_mpi_error(ierr,"comex_create_mutexes:MPI_Alloc_mem");
        sbuf = _mutex_buf[idx];
        for (k=0; k<nproc; k++) sbuf[k] = 0;
      } else {
        nsize = 0;
        _mutex_buf[idx] = NULL;
      }
      ierr = MPI_Win_create(_mutex_buf[idx],nsize,1,MPI_INFO_NULL,comm,
          &_mutex_list[idx]);
      translate_mpi_error(ierr,"comex_create_mutexes:MPI_Win_create");
      idx++;
    } 
  }

  return COMEX_SUCCESS;
}


/**
 * destroy all mutexes
 */
int comex_destroy_mutexes()
{
  int i, ierr;
  if (_mutex_list == NULL) return COMEX_SUCCESS;
  for (i=0; i<_mutex_total; i++) {
    ierr = MPI_Win_free(&_mutex_list[i]);
    translate_mpi_error(ierr,"comex_destroy_mutexes:MPI_Win_free");
    if (_mutex_buf[i] != NULL) MPI_Free_mem(_mutex_buf[i]);
  }
  free(_mutex_list);
  free(_mutex_buf);
  free(_mutex_num);
  _mutex_list = NULL;
  _mutex_buf = NULL;
  _mutex_num = NULL;
  _mutex_total = 0;

  return COMEX_SUCCESS;
}


/**
 * lock a mutex on some processor
 * mutex: index of mutex on processor proc
 * proc: rank of process containing mutex
 */
int comex_lock(int mutex, int proc)
{
  int i, ierr;
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
  ierr = MPI_Win_lock(MPI_LOCK_EXCLUSIVE,proc,0,_mutex_list[idx]);
  translate_mpi_error(ierr,"comex_lock:MPI_Win_lock");
  ierr = MPI_Get(waitlistcopy,nproc,MPI_INT,proc,0,nproc,MPI_INT,_mutex_list[idx]);
  translate_mpi_error(ierr,"comex_lock:MPI_Get");
  ierr = MPI_Put(&lock,1,MPI_INT,proc,me*sizeof(int),1,MPI_INT,_mutex_list[idx]);
  translate_mpi_error(ierr,"comex_lock:MPI_Put");
  ierr = MPI_Win_unlock(proc,_mutex_list[idx]);
  translate_mpi_error(ierr,"comex_lock:MPI_Win_unlock");

  /* Check to see if lock is already held */
  ok = 1;
  for (i=0; i<nproc && ok == 1; i++) {
    if (waitlistcopy[i] != 0 && i != me) ok = 0;
  }

  /* Wait until some one else frees the lock */
  if (!ok) {
    ierr = MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, idx, MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
    translate_mpi_error(ierr,"comex_lock:MPI_Recv");
  }

  free(waitlistcopy);
  return COMEX_SUCCESS;
}


/**
 * unlock a mutex on some processor
 * mutex: index of mutex on processor proc
 * proc: rank of process containing mutex
 */
int comex_unlock(int mutex, int proc)
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
  return COMEX_SUCCESS;
}


int comex_malloc(void *ptrs[], size_t size, comex_group_t group)
{
#if USE_SICM && TEST_SICM
  char cdevice[32];
#  ifdef TEST_SICM_DEV
  strcpy(cdevice,STR(TEST_SICM_DEV));
#  else
  strcpy(cdevice,"dram");
#  endif
  return comex_malloc_mem_dev(ptrs, size, group, cdevice);
#else
    comex_igroup_t *igroup = NULL;
    reg_entry_t *reg_entries = NULL;
    MPI_Comm comm = MPI_COMM_NULL;
    int i, ierr;
    int comm_rank = -1;
    int comm_size = -1;
    int tsize;
    reg_entry_t src;

    igroup = comex_get_igroup_from_group(group);

    /* preconditions */
    COMEX_ASSERT(ptrs);

    comm = igroup->comm;
    assert(comm != MPI_COMM_NULL);
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

#if DEBUG
    printf("[%d] comex_malloc(ptrs=%p, size=%lu, group=%d)\n",
        comm_rank, ptrs, (long unsigned)size, group);
#endif

    /* is this needed? */
    /* comex_barrier(group); */

    /* allocate ret_entry_t object for this process */
    reg_entries = malloc(sizeof(reg_entry_t)*comm_size);
    /*assert(reg_entries) */
    reg_entries[comm_rank].rank = comm_rank;
    reg_entries[comm_rank].len = size;

    /* allocate and register segment. We need to allocate something even if size
       is zero so allocate a nominal amount so that routines don't break. Count
       on the fact that the length is zero to keep things from breaking down */
    if (size > 0) {
      tsize = size;
    } else {
      tsize = 8;
    }
#ifdef USE_MPI_WIN_ALLOC
    MPI_Win_allocate(sizeof(char)*tsize,1,MPI_INFO_NULL,comm,&reg_entries[comm_rank].buf,
        &reg_entries[comm_rank].win);
#else
    MPI_Alloc_mem(tsize,MPI_INFO_NULL,&reg_entries[comm_rank].buf);
#endif
    MPI_Win_create(reg_entries[comm_rank].buf,tsize,1,MPI_INFO_NULL,comm,
        &reg_entries[comm_rank].win);

#ifdef USE_MPI_REQUESTS
    MPI_Win_lock_all(0,reg_entries[comm_rank].win);
    /* Use MPI_MODE_NOCHECK instead of 0 */
#endif


    /* exchange buffer address */
    /* @TODO: Consider using MPI_IN_PLACE? */
    memcpy(&src, &reg_entries[comm_rank], sizeof(reg_entry_t));
    MPI_Allgather(&src, sizeof(reg_entry_t), MPI_BYTE, reg_entries,
            sizeof(reg_entry_t), MPI_BYTE, comm);

    /* assign the ptr array to return to caller */
    for (i=0; i<comm_size; ++i) {
      ptrs[i] = reg_entries[i].buf;
      int world_rank;
      ierr = comex_group_translate_world(group,i,&world_rank);
      assert(COMEX_SUCCESS == ierr);
      if (i != comm_rank) {
        reg_entries[i].win = reg_entries[comm_rank].win;
      }
      /* probably want to use commicator rank instead of world rank*/
      reg_win_insert(world_rank,  reg_entries[i].buf, reg_entries[i].len,
          reg_entries[i].win, igroup);
    }
    comex_igroup_add_win(group,reg_entries[comm_rank].win);

    comex_wait_all(group);
    /* MPI_Win_fence(0,reg_entries[comm_rank].win); */
    MPI_Barrier(comm);

    return COMEX_SUCCESS;
#endif
}

int comex_malloc_mem_dev(void *ptrs[], size_t size, comex_group_t group,
        const char* device)
{
#if (!defined(USE_SICM) || !USE_SICM)
    return comex_malloc(ptrs,size,group);
#else
    comex_igroup_t *igroup = NULL;
    reg_entry_t *reg_entries = NULL;
    MPI_Comm comm = MPI_COMM_NULL;
    int i, ierr;
    int comm_rank = -1;
    int comm_size = -1;
    int tsize;
    reg_entry_t src;

    /* This will become more complicated */
    sicm_device_list *idevice = &device_dram;

    if (!strncmp(device,"dram",4)) {
      idevice = &device_dram;
    } else if (!strncmp(device,"knl_hbm",7)) {
      idevice = &device_knl_hbm;
    } else if (!strncmp(device,"ppc_hbm",7)) {
      idevice = &device_ppc_hbm;
    }

    igroup = comex_get_igroup_from_group(group);

    /* preconditions */
    COMEX_ASSERT(ptrs);

    comm = igroup->comm;
    assert(comm != MPI_COMM_NULL);
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

#if DEBUG
    printf("[%d] comex_malloc(ptrs=%p, size=%lu, group=%d)\n",
        comm_rank, ptrs, (long unsigned)size, group);
#endif

    /* is this needed? */
    /* comex_barrier(group); */

    /* allocate ret_entry_t object for this process */
    reg_entries = malloc(sizeof(reg_entry_t)*comm_size);
    /*assert(reg_entries) */
    reg_entries[comm_rank].rank = comm_rank;
    reg_entries[comm_rank].len = size;

    /* allocate and register segment. We need to allocate something even if size
       is zero so allocate a nominal amount so that routines don't break. Count
       on the fact that the length is zero to keep things from breaking down */
    if (size > 0) {
      tsize = size;
    } else {
      tsize = 8;
    }
    sicm_arena arena = sicm_arena_create(0, 0, idevice);
    reg_entries[comm_rank].buf = sicm_arena_alloc(arena, size);
    if(!(reg_entries[comm_rank].buf)){
        fprintf(stderr, "Error in allocating the pool\n");
        exit(11);
    }
    MPI_Win_create(reg_entries[comm_rank].buf,tsize,1,MPI_INFO_NULL,comm,
        &reg_entries[comm_rank].win);

#ifdef USE_MPI_REQUESTS
    MPI_Win_lock_all(0,reg_entries[comm_rank].win);
    /* Use MPI_MODE_NOCHECK instead of 0 */
#endif


    /* exchange buffer address */
    /* @TODO: Consider using MPI_IN_PLACE? */
    memcpy(&src, &reg_entries[comm_rank], sizeof(reg_entry_t));
    MPI_Allgather(&src, sizeof(reg_entry_t), MPI_BYTE, reg_entries,
            sizeof(reg_entry_t), MPI_BYTE, comm);

    /* assign the ptr array to return to caller */
    for (i=0; i<comm_size; ++i) {
      ptrs[i] = reg_entries[i].buf;
      int world_rank;
      ierr = comex_group_translate_world(group,i,&world_rank);
      assert(COMEX_SUCCESS == ierr);
      if (i != comm_rank) {
        reg_entries[i].win = reg_entries[comm_rank].win;
      }
      /* probably want to use commicator rank instead of world rank*/
      reg_win_insert(world_rank,  reg_entries[i].buf, reg_entries[i].len,
          reg_entries[i].win, igroup);
    }
    comex_igroup_add_win(group,reg_entries[comm_rank].win);

    comex_wait_all(group);
    /* MPI_Win_fence(0,reg_entries[comm_rank].win); */
    MPI_Barrier(comm);

    return COMEX_SUCCESS;
#endif
}


int comex_free(void *ptr, comex_group_t group)
{
#if USE_SICM && TEST_SICM
    return comex_free_dev(ptr, group);
#else
    comex_igroup_t *igroup = NULL;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Win window;
    int comm_rank, world_rank;
    int comm_size;
    void **allgather_ptrs = NULL;
    int i, ierr;
    reg_entry_t *reg_win;

    /* preconditions */
    assert(ptr != NULL);

    igroup = comex_get_igroup_from_group(group);
    comm = igroup->comm;
    assert(comm != MPI_COMM_NULL);
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    
    /* Find the window that this buffer belongs to */
    comex_group_translate_world(group,comm_rank,&world_rank);
    reg_win = reg_win_find(world_rank, ptr, 0);
    window = reg_win->win;

#ifndef USE_MPI_WIN_ALLOC
    /* Save pointer to memory */
    void* buf = reg_win->buf;
#endif

    /* allocate receive buffer for exchange of pointers */
    allgather_ptrs = (long **)malloc(sizeof(void *) * comm_size);
    assert(allgather_ptrs);

    /* exchange of pointers */
    MPI_Allgather(&ptr, sizeof(void *), MPI_BYTE,
            allgather_ptrs, sizeof(void *), MPI_BYTE, comm);

    /* Get rid of pointers for this window */
    for (i=0; i < comm_size; i++) {
      int world_rank;
      ierr = comex_group_translate_world(group, i, &world_rank);
      assert(COMEX_SUCCESS == ierr);
      /* probably should use rank for communicator, not world rank*/
      reg_win_delete(world_rank,allgather_ptrs[i]);
    }

    /* Remove window from group list */
    comex_igroup_delete_win(group, window);

    /* remove my ptr from reg cache and free ptr */
    /* comex_free_local(ptr); */
    free(allgather_ptrs);

    /* free up window */
#ifdef USE_MPI_REQUESTS
    MPI_Win_unlock_all(window);
#endif
    MPI_Win_free(&window);
#ifndef USE_MPI_WIN_ALLOC
    /* Clear memory for this window */
    MPI_Free_mem(buf);
#endif

    /* Is this needed? */
    MPI_Barrier(comm);

    return COMEX_SUCCESS;
#endif
}

int comex_free_dev(void *ptr, comex_group_t group)
{
#if (!defined(USE_SICM) || !USE_SICM)
    return comex_free(ptr,group);
#else
    comex_igroup_t *igroup = NULL;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Win window;
    int comm_rank, world_rank;
    int comm_size;
    void **allgather_ptrs = NULL;
    int i, ierr;
    reg_entry_t *reg_win;

    /* preconditions */
    assert(ptr != NULL);

    igroup = comex_get_igroup_from_group(group);
    comm = igroup->comm;
    assert(comm != MPI_COMM_NULL);
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    
    /* Find the window that this buffer belongs to */
    comex_group_translate_world(group,comm_rank,&world_rank);
    reg_win = reg_win_find(world_rank, ptr, 0);
    window = reg_win->win;

#ifndef USE_MPI_WIN_ALLOC
    /* Save pointer to memory */
    void* buf = reg_win->buf;
#endif

    /* allocate receive buffer for exchange of pointers */
    allgather_ptrs = (void **)malloc(sizeof(void *) * comm_size);
    assert(allgather_ptrs);

    /* exchange of pointers */
    MPI_Allgather(&ptr, sizeof(void *), MPI_BYTE,
            allgather_ptrs, sizeof(void *), MPI_BYTE, comm);

    /* Get rid of pointers for this window */
    for (i=0; i < comm_size; i++) {
      int world_rank;
      ierr = comex_group_translate_world(group, i, &world_rank);
      assert(COMEX_SUCCESS == ierr);
      /* probably should use rank for communicator, not world rank*/
      reg_win_delete(world_rank,allgather_ptrs[i]);
    }

    /* Remove window from group list */
    comex_igroup_delete_win(group, window);

    /* remove my ptr from reg cache and free ptr */
    /* comex_free_local(ptr); */
    free(allgather_ptrs);

    /* free up window */
#ifdef USE_MPI_REQUESTS
    MPI_Win_unlock_all(window);
#endif
    MPI_Win_free(&window);
    /* Clear memory for this window */
    sicm_free(buf);

    /* Is this needed? */
    MPI_Barrier(comm);

    return COMEX_SUCCESS;
#endif
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
#define ACC(WHICH, COMEX_TYPE, C_TYPE)                                  \
    if (datatype == COMEX_TYPE) {                                       \
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
    ACC(REG, COMEX_ACC_DBL, double)
    ACC(REG, COMEX_ACC_FLT, float)
    ACC(REG, COMEX_ACC_INT, int)
    ACC(REG, COMEX_ACC_LNG, long)
    ACC(CPL, COMEX_ACC_DCP, DoubleComplex)
    ACC(CPL, COMEX_ACC_CPL, SingleComplex)
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

