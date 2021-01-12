/*
#if HAVE_CONFIG_H
#   include "config.h"
#endif
*/

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "armci.h"
#include "parmci.h"
#include "cmx.h"
#include "reg_entry.h"
#include "groups.h"

typedef struct _nb_t {
  struct _nb_t *next;
  cmx_request_t request;
  int id;
} nb_t;

extern int ARMCI_Default_Proc_Group;

/**
 * Set up some data structures for non-blocking communication
 */
static nb_t *_nb_list = NULL;

/**
 * Useful variable for debugging
 */
static int _armci_me;

/**
 * This function checks to see if the data copy is contiguous for both the src
 * and destination buffers. If it is, then a contiguous operation can be used
 * instead of a strided operation. This function is intended for arrays of
 * dimension greater than 1 (contiguous operations can always be used for 1
 * dimensional arrays). This operation does not identify all contiguous cases,
 * since no information is available about the last dimension.
 * src_stride: physical dimensions of source buffer
 * dst_stride: physical dimensions of destination buffer
 * count: number of elements being moved in each dimension
 * n_stride: number of strides (array dimension minus one)
 */
int armci_check_contiguous(int *src_stride, int *dst_stride,
    int *count, int n_stride)
{
  int i;
  int ret = 1;
  int stridelen = 1;
  /* NOTE: The count array contains the length of the final dimension and could
   * be used to evaluate some corner cases that are not picked up by this
   * algorithm
   */
  for (i=0; i<n_stride; i++) {
    stridelen *= count[i];
    if (stridelen < src_stride[i] || stridelen < dst_stride[i]) {
      ret = 0;
      break;
    }
  }
  return ret;
}

/**
 * Dummy function for use in debugging
 */
int armci_checkt_contiguous(int *src_stride, int *dst_stride,
    int *count, int n_stride)
{
  return 0;
}

/**
 * Convert an armci_giov_t list to a cmx_giov_t list. If rem = 0, then the
 * remote processor is the source of the data, otherwise the remote processor is
 * the destination of the data. Ref is the origin of the buffer on the remote
 * processor.
 */
#if 0
typedef struct {
  void **src_ptr_array;
  void **dst_ptr_array;
  int  ptr_array_len;
  int bytes;
} armci_giov_t;

typedef struct {
  void **loc; /**< array of local starting addresses */
  cmxInt *rem; /**< array of remote offsets */
  cmxInt count; /**< size of address arrays (src[count],dst[count]) */
  cmxInt bytes; /**< length in bytes for each src[i]/dst[i] pair */
} cmx_giov_t;
#endif
static void convert_giov(armci_giov_t *a, cmx_giov_t *b, int len, void *ref, int rem)
{
  int i,j, nelems;
  if (rem == 0) {
    for (i=0; i<len; ++i) {
      nelems = a[i].ptr_array_len;
      b[i].count = a[i].ptr_array_len;
      b[i].bytes = a[i].bytes;
      b[i].loc = (void**)malloc(nelems*sizeof(void*));
      b[i].rem = (cmxInt*)malloc(nelems*sizeof(cmxInt));
      for (j=0; j<nelems; j++) {
        b[i].loc[j] = a[i].dst_ptr_array[j];
        b[i].rem[j] = (cmxInt)((MPI_Aint)a[i].src_ptr_array[j]-(MPI_Aint)ref);
      }
    }
  } else {
    for (i=0; i<len; ++i) {
      nelems = a[i].ptr_array_len;
      b[i].count = a[i].ptr_array_len;
      b[i].bytes = a[i].bytes;
      b[i].loc = (void**)malloc(nelems*sizeof(void*));
      b[i].rem = (cmxInt*)malloc(nelems*sizeof(cmxInt));
      for (j=0; j<nelems; j++) {
        b[i].loc[j] = a[i].src_ptr_array[j];
        b[i].rem[j] = (cmxInt)((MPI_Aint)a[i].dst_ptr_array[j]-(MPI_Aint)ref);
      }
    }
  }
}

static void free_giov(cmx_giov_t *a, int len)
{
  int i, j;
  for (i=0; i<len; ++i) {
    free(a[i].loc);
    free(a[i].rem);
  }
  free(a);
}

int convert_optype(int op){
  int ret;
  if (op == ARMCI_ACC_INT) {
    ret = CMX_ACC_INT;
  } else if (op == ARMCI_ACC_DBL) {
    ret = CMX_ACC_DBL;
  } else if (op == ARMCI_ACC_FLT) {
    ret = CMX_ACC_FLT;
  } else if (op == ARMCI_ACC_CPL) {
    ret = CMX_ACC_CPL;
  } else if (op == ARMCI_ACC_DCP) {
    ret = CMX_ACC_DCP;
  } else if (op == ARMCI_ACC_LNG) {
    ret = CMX_ACC_LNG;
  } else if (op == ARMCI_SWAP) {
    ret = CMX_SWAP;
  } else if (op == ARMCI_SWAP_LONG) {
    ret = CMX_SWAP_LONG;
  } else if (op == ARMCI_FETCH_AND_ADD) {
    ret = CMX_FETCH_AND_ADD;
  } else if (op == ARMCI_FETCH_AND_ADD_LONG) {
    ret = CMX_FETCH_AND_ADD_LONG;
  } else {
    ret -1;
  }
  return ret;
}

/**
 * Create a new non-blocking request and return both an integer handle and the
 * request data structure to the calling program
 */
void get_nb_request(armci_hdl_t *handle, nb_t **req)
{
  int maxID = 0; 
  nb_t *curr_req = _nb_list;
  nb_t *prev_req = NULL;
  while (curr_req != NULL) {
    if (curr_req->id > maxID) maxID = curr_req->id;
    prev_req = curr_req;
    curr_req = curr_req->next;
  }
  maxID++;
  *req = (nb_t*)malloc(sizeof(nb_t));
  (*req)->id = maxID;
  (*req)->next = NULL;
  *handle = maxID;
  if (prev_req) {
    prev_req->next = *req;
  } else {
    _nb_list = *req;
  }
}

/**
 * Delete a non-blocking request and remove it from the link list
 */
void delete_nb_request(armci_hdl_t *handle)
{
  nb_t *curr_req = _nb_list;
  nb_t *prev_req = NULL;
  while (curr_req != NULL) {
    if (curr_req->id == *handle) break;
    prev_req = curr_req;
    curr_req = curr_req->next;
  }
  if (curr_req == NULL) {
    printf("Could not find request handle for delete\n");
  } else {
    if (prev_req != NULL) {
      prev_req->next = curr_req->next;
    }
    if (_nb_list == curr_req) _nb_list = curr_req->next;
    free(curr_req);
  }
}

/**
 * Recover a non-blocking request from the handle
 */
nb_t* find_nb_request(armci_hdl_t *handle) {
  nb_t *curr_req = _nb_list;
  while (curr_req != NULL) {
    if (curr_req->id == *handle) return curr_req;
    curr_req = curr_req->next;
  }
  return NULL;
}

int PARMCI_Acc(int optype, void *scale, void *src, void *dst, int bytes, int proc)
{
  reg_entry_t *entry; 
  void *buf;
  MPI_Aint offset;
  int lproc;
  entry = reg_entry_find(proc,dst,bytes);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  optype = convert_optype(optype);
  offset = (MPI_Aint)dst-(MPI_Aint)buf;
  return cmx_acc(optype, scale, src, offset, bytes, lproc, *(entry->hdl));
}


int PARMCI_AccS(int optype, void *scale, void *src_ptr, int *src_stride_arr, void *dst_ptr, int *dst_stride_arr, int *count, int stride_levels, int proc)
{
  int iret;
  reg_entry_t *entry; 
  void *buf;
  MPI_Aint offset;
  int lproc;
  entry = reg_entry_find(proc,dst_ptr,0);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  optype = convert_optype(optype);
  offset = (MPI_Aint)dst_ptr-(MPI_Aint)buf;
  /* check if data is contiguous */
  if (armci_check_contiguous(src_stride_arr, dst_stride_arr, count, stride_levels)) {
    int i;
    int lcount = 1;
    for (i=0; i<=stride_levels; i++) lcount *= count[i];
    iret = cmx_acc(optype, scale, src_ptr, offset, lcount, lproc, *(entry->hdl));
  } else {
    iret = cmx_accs(optype, scale, src_ptr, src_stride_arr, offset,
        dst_stride_arr, count, stride_levels, lproc, *(entry->hdl));
  }
  return iret;
}


int PARMCI_AccV(int op, void *scale, armci_giov_t *darr, int len, int proc)
{
  int rc;
  reg_entry_t *entry; 
  void *buf;
  cmx_giov_t *adarr = malloc(sizeof(cmx_giov_t) * len);
  int lproc;
  /* find location of buffer on remote processor. Start by finding a buffer
   * location on the remote array */
  buf = (darr[0].dst_ptr_array)[0];
  entry = reg_entry_find(proc,buf,0);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  convert_giov(darr, adarr, len, buf, 1);
  op = convert_optype(op);
  rc = cmx_accv(op, scale, adarr, len, lproc, *(entry->hdl));
  free_giov(adarr, len);
  return rc;
}


/* fence is always on the world group */
void PARMCI_AllFence()
{
  int ierr = cmx_fence_all(CMX_GROUP_WORLD);
  assert(CMX_SUCCESS == ierr);
}


void PARMCI_Barrier()
{
  int ierr;
  cmx_group_t grp = armci_get_cmx_group(ARMCI_Default_Proc_Group);
  ierr = cmx_barrier(grp);
  assert(CMX_SUCCESS == ierr);
}


int PARMCI_Create_mutexes(int num)
{
  return cmx_create_mutexes(num);
}


int PARMCI_Destroy_mutexes()
{
  return cmx_destroy_mutexes();
}


/* fence is always on the world group */
void PARMCI_Fence(int proc)
{
  cmx_fence_proc(proc, CMX_GROUP_WORLD);
}


void PARMCI_Finalize()
{
  int i;
  reg_entries_destroy();
  armci_group_finalize();
  cmx_finalize();
}


int PARMCI_Free(void *ptr)
{
  reg_entry_t *entry; 
  int rank, size, ierr, i;
  void **allgather_ptrs = NULL;
  void *buf;
  MPI_Comm comm;
  cmx_group_rank(CMX_GROUP_WORLD, &rank);
  cmx_group_size(CMX_GROUP_WORLD, &size);
  cmx_group_comm(CMX_GROUP_WORLD, &comm);
  entry = reg_entry_find(rank,ptr,0);
  buf = entry->buf;
  ierr = cmx_free(*(entry->hdl));
  /* Need to clean up all entries corresponding to this allocation
   * in reg_entry lists. Allocate receive buffer for exchange of pointers */
  allgather_ptrs = (void **)malloc(sizeof(void *) * size);
  assert(allgather_ptrs);
  /* exchange pointers */
  MPI_Allgather(&buf, sizeof(void*), MPI_BYTE,
                allgather_ptrs, sizeof(void*), MPI_BYTE, comm);
  for (i=0; i<size; i++) {
    reg_entry_delete(i,allgather_ptrs[i]);
  }
  return ierr;
}

int PARMCI_Free_memdev(void *ptr)
{
  PARMCI_Free(ptr);
  /*
  return comex_free_dev(ptr, ARMCI_Default_Proc_Group);
  */
}


int ARMCI_Free_group(void *ptr, ARMCI_Group *group)
{
  reg_entry_t *entry; 
  int rank, world_rank, size, ierr, i;
  void **allgather_ptrs = NULL;
  void *buf;
  MPI_Comm comm;
  cmx_group_t grp = armci_get_cmx_group(*group);
  cmx_group_rank(grp, &rank);
  cmx_group_size(grp, &size);
  cmx_group_comm(grp, &comm);
  cmx_group_translate_world(grp, rank, &world_rank);
  entry = reg_entry_find(world_rank,ptr,0);
  buf = entry->buf;
  ierr = cmx_free(*(entry->hdl));
  /* Need to clean up all entries corresponding to this allocation
   * in reg_entry lists. Allocate receive buffer for exchange of pointers */
  allgather_ptrs = (void **)malloc(sizeof(void *) * size);
  assert(allgather_ptrs);
  /* exchange pointers */
  MPI_Allgather(&buf, sizeof(void*), MPI_BYTE,
                allgather_ptrs, sizeof(void*), MPI_BYTE, comm);
  for (i=0; i<size; i++) {
    cmx_group_translate_world(grp, i, &world_rank);
    reg_entry_delete(world_rank,allgather_ptrs[i]);
  }
  return ierr;
}


int PARMCI_Free_local(void *ptr)
{
  return cmx_free_local(ptr);
}


int PARMCI_Get(void *src, void *dst, int bytes, int proc)
{
  reg_entry_t *entry; 
  void *buf;
  MPI_Aint offset;
  int lproc;
  entry = reg_entry_find(proc,src,bytes);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  offset = (MPI_Aint)src-(MPI_Aint)buf;
  return cmx_get(dst, offset, bytes, lproc, *(entry->hdl));
}


int PARMCI_GetS(void *src_ptr, int *src_stride_arr, void *dst_ptr, int *dst_stride_arr, int *count, int stride_levels, int proc)
{
  int iret;
  reg_entry_t *entry; 
  void *buf;
  MPI_Aint offset;
  int lproc;
  entry = reg_entry_find(proc,src_ptr,0);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  offset = (MPI_Aint)src_ptr-(MPI_Aint)buf;
  /* check if data is contiguous */
  if (armci_check_contiguous(src_stride_arr, dst_stride_arr, count, stride_levels)) {
    int i;
    int lcount = 1;
    for (i=0; i<=stride_levels; i++) lcount *= count[i];
    iret = cmx_get(dst_ptr, offset, lcount, lproc, *(entry->hdl));
  } else {
    iret = cmx_gets(dst_ptr, dst_stride_arr, offset, src_stride_arr,
        count, stride_levels, lproc, *(entry->hdl));
  }
  return iret;
}


int PARMCI_GetV(armci_giov_t *darr, int len, int proc)
{
  int rc;
  cmx_giov_t *adarr = malloc(sizeof(cmx_giov_t) * len);
  reg_entry_t *entry; 
  void *buf;
  int lproc;
  /* find location of buffer on remote processor. Start by finding a buffer
   * location on the remote array */
  buf = (darr[0].src_ptr_array)[0];
  entry = reg_entry_find(proc,buf,0);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  convert_giov(darr, adarr, len, buf, 0);
  rc = cmx_getv(adarr, len, lproc, *(entry->hdl));
  free_giov(adarr, len);
  return rc;
}


double PARMCI_GetValueDouble(void *src, int proc)
{
  int ierr;
  double val;
  ierr = PARMCI_Get(src, &val, sizeof(double), proc);
  assert(CMX_SUCCESS == ierr);
  return val;
}


float PARMCI_GetValueFloat(void *src, int proc)
{
  int ierr;
  float val;
  ierr = PARMCI_Get(src, &val, sizeof(float), proc);
  assert(CMX_SUCCESS == ierr);
  return val;
}


int PARMCI_GetValueInt(void *src, int proc)
{
  int ierr;
  int val;
  ierr = PARMCI_Get(src, &val, sizeof(int), proc);
  assert(CMX_SUCCESS == ierr);
  return val;
}


long PARMCI_GetValueLong(void *src, int proc)
{
  int ierr;
  long val;
  ierr = PARMCI_Get(src, &val, sizeof(long), proc);
  assert(CMX_SUCCESS == ierr);
  return val;
}


int PARMCI_Init()
{
  int i, size;
  int rc = cmx_init();
  armci_group_init();
  cmx_group_size(CMX_GROUP_WORLD,&size);
  cmx_group_rank(CMX_GROUP_WORLD,&_armci_me);
  reg_entry_init(size);
  return rc;
}


int PARMCI_Init_args(int *argc, char ***argv)
{
  int i, size;
  int rc = cmx_init_args(argc, argv);
  armci_group_init();
  cmx_group_size(CMX_GROUP_WORLD,&size);
  cmx_group_rank(CMX_GROUP_WORLD,&_armci_me);
  reg_entry_init(size);
  return rc;
}


int PARMCI_Initialized()
{
  return cmx_initialized();
}


void PARMCI_Lock(int mutex, int proc)
{
  cmx_lock(mutex, proc);
}

int ARMCI_Malloc_group(void **ptr_arr, armci_size_t bytes, ARMCI_Group *group)
{
  cmx_handle_t *handle = NULL;
  reg_entry_t *reg_entries = NULL;
  reg_entry_t entry;
  MPI_Comm comm = MPI_COMM_NULL;
  void *buf;
  armci_igroup_t *igroup;
  int i, ret;
  int comm_rank = -1;
  int comm_size = -1;
  int tsize;
  cmx_group_t cmx_grp;

  igroup = armci_get_igroup_from_group(*group);
  cmx_grp = igroup->group;
  handle = (cmx_handle_t*)malloc(sizeof(cmx_handle_t));

  /* ptr_array should already have been allocated externally */
  CMX_ASSERT(ptr_arr);

  cmx_group_comm(cmx_grp, &comm);
  assert(comm != MPI_COMM_NULL);
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  /* allocate ret_entry_t object for this process */
  reg_entries = malloc(sizeof(reg_entry_t)*comm_size);

  ret = cmx_malloc(handle, bytes, cmx_grp);
  cmx_access(*handle,&buf);
  reg_entries[comm_rank].buf = buf;
  reg_entries[comm_rank].rank = comm_rank;
  reg_entries[comm_rank].len = bytes;
  reg_entries[comm_rank].hdl = handle;
  /* exchange buffer address */
  memcpy(&entry, &reg_entries[comm_rank], sizeof(reg_entry_t));
  MPI_Allgather(&entry, sizeof(reg_entry_t), MPI_BYTE, reg_entries,
      sizeof(reg_entry_t), MPI_BYTE, comm);
  for (i=0; i<comm_size; i++) {
    reg_entry_t *node;
    int ierr, world_rank;
    reg_entries[i].hdl = handle;
    cmx_group_translate_world(cmx_grp, i, &world_rank);
    node = reg_entry_insert(world_rank, reg_entries[i].buf,
        reg_entries[i].len, reg_entries[i].hdl);
    ptr_arr[i] = reg_entries[i].buf;
  }
  free(reg_entries);
  if (ret == CMX_SUCCESS) ret = 0;
  else ret = 1;
  return ret;
}

int ARMCI_Malloc_group_memdev(void **ptr_arr, armci_size_t bytes,
        ARMCI_Group *group, const char *device)
{
  ARMCI_Malloc_group(ptr_arr, bytes, group);
  /*
  return comex_malloc_mem_dev(ptr_arr, bytes, *group,device);
  */
}

int PARMCI_Malloc(void **ptr_arr, armci_size_t bytes)
{
  return ARMCI_Malloc_group(ptr_arr, bytes, &ARMCI_Default_Proc_Group);
}

int PARMCI_Malloc_memdev(void **ptr_arr, armci_size_t bytes, const char *device)
{
  PARMCI_Malloc(ptr_arr, bytes);
  /*
  return comex_malloc_mem_dev(ptr_arr, bytes, ARMCI_Default_Proc_Group,device);
  */
}


void* PARMCI_Malloc_local(armci_size_t bytes)
{
    return cmx_malloc_local(bytes);
}


void* PARMCI_Memat(armci_meminfo_t *meminfo, long offset)
{
    void *ptr=NULL;
    int rank;

    cmx_group_rank(CMX_GROUP_WORLD, &rank);

    if(meminfo==NULL) cmx_error("PARMCI_Memat: Invalid arg #1 (NULL ptr)",0);

    if(meminfo->cpid==rank) { ptr = meminfo->addr; return ptr; }

    ptr = meminfo->addr;

    return ptr;
}


void PARMCI_Memget(size_t bytes, armci_meminfo_t *meminfo, int memflg)
{
    void *myptr=NULL;
    void *armci_ptr=NULL; /* legal ARCMI ptr used in ARMCI data xfer ops*/
    size_t size = bytes;
    int rank;

    cmx_group_rank(CMX_GROUP_WORLD, &rank);

    if(size<=0) cmx_error("PARMCI_Memget: size must be > 0", (int)size);
    if(meminfo==NULL) cmx_error("PARMCI_Memget: Invalid arg #2 (NULL ptr)",0);
    if(memflg!=0) cmx_error("PARMCI_Memget: Invalid memflg", memflg);

    armci_ptr = myptr = cmx_malloc_local(size);
    if(size) if(!myptr) cmx_error("PARMCI_Memget failed", (int)size);

    /* fill the meminfo structure */
    meminfo->armci_addr = armci_ptr;
    meminfo->addr       = myptr;
    meminfo->size       = size;
    meminfo->cpid       = rank;
    /* meminfo->attr       = NULL; */
}


void PARMCI_Memdt(armci_meminfo_t *meminfo, long offset)
{
}


void PARMCI_Memctl(armci_meminfo_t *meminfo)
{
    int rank;

    cmx_group_rank(CMX_GROUP_WORLD, &rank);

    if(meminfo==NULL) cmx_error("PARMCI_Memget: Invalid arg #2 (NULL ptr)",0);

    /* only the creator can delete the segment */
    if(meminfo->cpid == rank)
    {
        void *ptr = meminfo->addr;
        cmx_free_local(ptr);
    }

    meminfo->addr       = NULL;
    meminfo->armci_addr = NULL;
    /* if(meminfo->attr!=NULL) free(meminfo->attr); */
}

int PARMCI_NbAcc(int optype, void *scale, void *src, void *dst, int bytes, int proc,
    armci_hdl_t *nb_handle)
{
  int iret;
  reg_entry_t *entry; 
  void *buf;
  MPI_Aint offset;
  nb_t *req;
  int lproc;
  if (nb_handle == NULL) {
    return PARMCI_Acc(optype, scale, src, dst, bytes, proc);
  }
  entry = reg_entry_find(proc,dst,bytes);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  optype = convert_optype(optype);
  offset = (MPI_Aint)dst-(MPI_Aint)buf;
  get_nb_request(nb_handle, &req);
  iret = cmx_nbacc(optype, scale, src, offset, bytes, lproc, *(entry->hdl),
      &(req->request));
  return iret;
}



int PARMCI_NbAccS(int optype, void *scale, void *src_ptr, int *src_stride_arr, void *dst_ptr, int *dst_stride_arr, int *count, int stride_levels, int proc, armci_hdl_t *nb_handle)
{
  int iret;
  reg_entry_t *entry; 
  void *buf;
  MPI_Aint offset;
  nb_t *req;
  int lproc;
  if (nb_handle == NULL) {
    return PARMCI_AccS(optype, scale, src_ptr, src_stride_arr, dst_ptr,
        dst_stride_arr, count, stride_levels, proc);
  }
  entry = reg_entry_find(proc,dst_ptr,0);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  optype = convert_optype(optype);
  offset = (MPI_Aint)dst_ptr-(MPI_Aint)buf;
  get_nb_request(nb_handle, &req);
  /* check if data is contiguous */
  if (armci_check_contiguous(src_stride_arr, dst_stride_arr, count, stride_levels)) {
    int i;
    int lcount = 1;
    for (i=0; i<=stride_levels; i++) lcount *= count[i];
    iret = cmx_nbacc(optype, scale, src_ptr, offset, lcount, lproc,
        *(entry->hdl), &(req->request));
  } else {
    iret = cmx_nbaccs(optype, scale, src_ptr, src_stride_arr, offset,
        dst_stride_arr, count, stride_levels, lproc,
        *(entry->hdl), &(req->request));
  }
  return iret;
}


int PARMCI_NbAccV(int op, void *scale, armci_giov_t *darr, int len, int proc, armci_hdl_t *nb_handle)
{
  int rc;
  reg_entry_t *entry; 
  void *buf;
  nb_t *req;
  cmx_giov_t *adarr;
  int lproc;
  if (nb_handle == NULL) {
    return PARMCI_AccV(op, scale, darr, len, proc);
  }
  adarr = malloc(sizeof(cmx_giov_t) * len);
  /* find location of buffer on remote processor. Start by finding a buffer
   * location on the remote array */
  buf = (darr[0].dst_ptr_array)[0];
  entry = reg_entry_find(proc,buf,0);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  convert_giov(darr, adarr, len, buf, 1);
  op = convert_optype(op);
  get_nb_request(nb_handle, &req);
  rc = cmx_nbaccv(op, scale, adarr, len, lproc, *(entry->hdl), &(req->request));
  free_giov(adarr, len);
  return rc;
}

int PARMCI_NbGet(void *src, void *dst, int bytes, int proc, armci_hdl_t *nb_handle)
{
  int iret;
  reg_entry_t *entry; 
  void *buf;
  MPI_Aint offset;
  nb_t *req;
  int lproc;
  if (nb_handle == NULL) {
    return PARMCI_Get(src, dst, bytes, proc);
  }
  entry = reg_entry_find(proc,src,bytes);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  offset = (MPI_Aint)src-(MPI_Aint)buf;
  get_nb_request(nb_handle, &req);
  iret = cmx_nbget(dst, offset, bytes, lproc, *(entry->hdl), &(req->request));
  return iret;
}


int PARMCI_NbGetS(void *src_ptr, int *src_stride_arr, void *dst_ptr, int *dst_stride_arr, int *count, int stride_levels, int proc, armci_hdl_t *nb_handle)
{
  int iret;
  reg_entry_t *entry; 
  void *buf;
  MPI_Aint offset;
  nb_t *req;
  int lproc;
  if (nb_handle == NULL) {
    return PARMCI_GetS(src_ptr, src_stride_arr, dst_ptr, dst_stride_arr,
        count, stride_levels, proc);
  }
  entry = reg_entry_find(proc,src_ptr,0);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  offset = (MPI_Aint)src_ptr-(MPI_Aint)buf;
  get_nb_request(nb_handle, &req);
  /* check if data is contiguous */
  if (armci_check_contiguous(src_stride_arr, dst_stride_arr, count, stride_levels)) {
    int i;
    int lcount = 1;
    for (i=0; i<=stride_levels; i++) lcount *= count[i];
    iret = cmx_nbget(dst_ptr, offset, lcount, lproc,
        *(entry->hdl), &(req->request));
  } else {
    iret = cmx_nbgets(dst_ptr, dst_stride_arr, offset, src_stride_arr,
        count, stride_levels, lproc, *(entry->hdl), &(req->request));
  }
  return iret;
}


int PARMCI_NbGetV(armci_giov_t *darr, int len, int proc, armci_hdl_t *nb_handle)
{
  int rc;
  reg_entry_t *entry; 
  void *buf;
  nb_t *req;
  cmx_giov_t *adarr;
  int lproc;
  if (nb_handle == NULL) {
    return PARMCI_GetV(darr, len, proc);
  }
  adarr = malloc(sizeof(cmx_giov_t) * len);
  /* find location of buffer on remote processor. Start by finding a buffer
   * location on the remote array */
  buf = (darr[0].src_ptr_array)[0];
  entry = reg_entry_find(proc,buf,0);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  convert_giov(darr, adarr, len, buf, 0);
  get_nb_request(nb_handle, &req);
  rc = cmx_nbgetv(adarr, len, lproc, *(entry->hdl), &(req->request));
  free_giov(adarr, len);
  return rc;
}


int PARMCI_NbPut(void *src, void *dst, int bytes, int proc, armci_hdl_t *nb_handle)
{
  int iret;
  reg_entry_t *entry; 
  void *buf;
  MPI_Aint offset;
  nb_t *req;
  int lproc;
  if (nb_handle == NULL) {
    return PARMCI_Put(src, dst, bytes, proc);
  }
  entry = reg_entry_find(proc,dst,bytes);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  offset = (MPI_Aint)dst-(MPI_Aint)buf;
  get_nb_request(nb_handle, &req);
  iret = cmx_nbput(src, offset, bytes, lproc, *(entry->hdl), &(req->request));
  return iret;
}


int PARMCI_NbPutS(void *src_ptr, int *src_stride_arr, void *dst_ptr, int *dst_stride_arr, int *count, int stride_levels, int proc, armci_hdl_t *nb_handle)
{
  int iret;
  reg_entry_t *entry; 
  void *buf;
  MPI_Aint offset;
  nb_t *req;
  int lproc;
  if (nb_handle == NULL) {
    return PARMCI_PutS(src_ptr, src_stride_arr, dst_ptr, dst_stride_arr,
        count, stride_levels, proc);
  }
  entry = reg_entry_find(proc,dst_ptr,0);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  offset = (MPI_Aint)dst_ptr-(MPI_Aint)buf;
  get_nb_request(nb_handle, &req);
  /* check if data is contiguous */
  if (armci_check_contiguous(src_stride_arr, dst_stride_arr, count, stride_levels)) {
    int i;
    int lcount = 1;
    for (i=0; i<=stride_levels; i++) lcount *= count[i];
    iret = cmx_nbput(src_ptr, offset, lcount, lproc,
        *(entry->hdl), &(req->request));
  } else {
    iret = cmx_nbputs(src_ptr, src_stride_arr, offset, dst_stride_arr,
        count, stride_levels, lproc, *(entry->hdl), &(req->request));
  }
  return iret;
}


int PARMCI_NbPutV(armci_giov_t *darr, int len, int proc, armci_hdl_t *nb_handle)
{
  int rc;
  reg_entry_t *entry; 
  void *buf;
  nb_t *req;
  cmx_giov_t *adarr;
  int lproc;
  if (nb_handle == NULL) {
    return PARMCI_PutV(darr, len, proc);
  }
  adarr = malloc(sizeof(cmx_giov_t) * len);
  /* find location of buffer on remote processor. Start by finding a buffer
   * location on the remote array */
  buf = (darr[0].dst_ptr_array)[0];
  entry = reg_entry_find(proc,buf,0);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  convert_giov(darr, adarr, len, buf, 1);
  get_nb_request(nb_handle, &req);
  rc = cmx_nbputv(adarr, len, lproc, *(entry->hdl), &(req->request));
  free_giov(adarr, len);
  return rc;
}


int PARMCI_NbPutValueDouble(double src, void *dst, int proc, armci_hdl_t *nb_handle)
{
  return PARMCI_NbPut(&src, dst, sizeof(double), proc, nb_handle);
}


int PARMCI_NbPutValueFloat(float src, void *dst, int proc, armci_hdl_t *nb_handle)
{
  return PARMCI_NbPut(&src, dst, sizeof(float), proc, nb_handle);
}


int PARMCI_NbPutValueInt(int src, void *dst, int proc, armci_hdl_t *nb_handle)
{
  return PARMCI_NbPut(&src, dst, sizeof(int), proc, nb_handle);
}


int PARMCI_NbPutValueLong(long src, void *dst, int proc, armci_hdl_t *nb_handle)
{
  return PARMCI_NbPut(&src, dst, sizeof(long), proc, nb_handle);
}


int PARMCI_Put(void *src, void *dst, int bytes, int proc)
{
  reg_entry_t *entry; 
  void *buf;
  MPI_Aint offset;
  int lproc;
  entry = reg_entry_find(proc,dst,bytes);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  offset = (MPI_Aint)dst-(MPI_Aint)buf;
  return cmx_put(src, offset, bytes, lproc, *(entry->hdl));
}


int PARMCI_PutS(void *src_ptr, int *src_stride_arr, void *dst_ptr, int *dst_stride_arr, int *count, int stride_levels, int proc)
{
  int iret;
  reg_entry_t *entry; 
  void *buf;
  MPI_Aint offset;
  nb_t *req;
  int lproc;
  entry = reg_entry_find(proc,dst_ptr,0);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  offset = (MPI_Aint)dst_ptr-(MPI_Aint)buf;
  /* check if data is contiguous */
  if (armci_check_contiguous(src_stride_arr, dst_stride_arr, count, stride_levels)) {
    int i;
    int lcount = 1;
    for (i=0; i<=stride_levels; i++) lcount *= count[i];
    iret = cmx_put(src_ptr, offset, lcount, lproc, *(entry->hdl));
  } else {
    iret = cmx_puts(src_ptr, src_stride_arr, offset, dst_stride_arr,
        count, stride_levels, lproc, *(entry->hdl));
  }
  return iret;
}


int PARMCI_PutS_flag(void *src_ptr, int *src_stride_arr, void *dst_ptr, int *dst_stride_arr, int *count, int stride_levels, int *flag, int val, int proc)
{
    assert(0);
    return 0;
}


int PARMCI_PutS_flag_dir(void *src_ptr, int *src_stride_arr, void *dst_ptr, int *dst_stride_arr, int *count, int stride_levels, int *flag, int val, int proc)
{
    assert(0);
    return 0;
}


int PARMCI_PutV(armci_giov_t *darr, int len, int proc)
{
  int rc;
  reg_entry_t *entry; 
  void *buf;
  cmx_giov_t *adarr = malloc(sizeof(cmx_giov_t) * len);
  int lproc;
  /* find location of buffer on remote processor. Start by finding a buffer
   * location on the remote array */
  buf = (darr[0].dst_ptr_array)[0];
  entry = reg_entry_find(proc,buf,0);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  convert_giov(darr, adarr, len, buf, 1);
  rc = cmx_putv(adarr, len, lproc, *(entry->hdl));
  free_giov(adarr, len);
  return rc;
}


int PARMCI_PutValueDouble(double src, void *dst, int proc)
{
  return PARMCI_Put(&src, dst, sizeof(double), proc);
}


int PARMCI_PutValueFloat(float src, void *dst, int proc)
{
  return PARMCI_Put(&src, dst, sizeof(float), proc);
}


int PARMCI_PutValueInt(int src, void *dst, int proc)
{
  return PARMCI_Put(&src, dst, sizeof(int), proc);
}


int PARMCI_PutValueLong(long src, void *dst, int proc)
{
  return PARMCI_Put(&src, dst, sizeof(long), proc);
}


int PARMCI_Put_flag(void *src, void *dst, int bytes, int *f, int v, int proc)
{
  assert(0);
  return 0;
}


int PARMCI_Rmw(int op, void *ploc, void *prem, int extra, int proc)
{
  reg_entry_t *entry; 
  void *buf;
  MPI_Aint offset;
  int lproc;
  entry = reg_entry_find(proc,prem,0);
  buf = entry->buf;
  cmx_group_translate_ranks(1, CMX_GROUP_WORLD, &proc, entry->hdl->group, &lproc);
  offset = (MPI_Aint)prem-(MPI_Aint)buf;
  op = convert_optype(op);
  return cmx_rmw(op, ploc, offset, extra, lproc, *(entry->hdl));
}


int PARMCI_Test(armci_hdl_t *nb_handle)
{
  int status, ierr;
  nb_t *req = find_nb_request(nb_handle);
  if (req == NULL) return 0;
  ierr = cmx_test(&(req->request), &status);
  if (status == 0) {
    /* operation was completed so free handle */
    delete_nb_request(nb_handle);
  }
  assert(CMX_SUCCESS == ierr);
  return status;
}


void PARMCI_Unlock(int mutex, int proc)
{
    cmx_unlock(mutex, proc);
}


int PARMCI_Wait(armci_hdl_t *nb_handle)
{
  int ret;
  nb_t *req = find_nb_request(nb_handle);
  ret = cmx_wait(&(req->request));
  delete_nb_request(nb_handle);
  return CMX_SUCCESS;
}


int PARMCI_WaitAll()
{
    return cmx_wait_all(CMX_GROUP_WORLD);
}


int PARMCI_WaitProc(int proc)
{
    return cmx_wait_proc(proc, CMX_GROUP_WORLD);
}


int parmci_notify(int proc)
{
    assert(0);
    return 0;
}


int parmci_notify_wait(int proc, int *pval)
{
    assert(0);
    return 0;
}


int armci_domain_nprocs(armci_domain_t domain, int id)
{
    return 1;
}


int armci_domain_id(armci_domain_t domain, int glob_proc_id)
{
    return glob_proc_id;
}


int armci_domain_glob_proc_id(armci_domain_t domain, int id, int loc_proc_id)
{
    return id;
}


int armci_domain_my_id(armci_domain_t domain)
{
    int rank;

    assert(cmx_initialized());
    cmx_group_rank(CMX_GROUP_WORLD, &rank);

    return rank;
}


int armci_domain_count(armci_domain_t domain)
{
    int size;

    assert(cmx_initialized());
    cmx_group_size(CMX_GROUP_WORLD, &size);

    return size;
}


int armci_domain_same_id(armci_domain_t domain, int proc)
{
    int rank;

    cmx_group_rank(CMX_GROUP_WORLD, &rank);

    return (proc == rank);
}


void ARMCI_Error(char *msg, int code)
{
    cmx_error(msg, code);
}


void ARMCI_Set_shm_limit(unsigned long shmemlimit)
{
    /* ignore */
}


/* Shared memory not implemented */
int ARMCI_Uses_shm()
{
    return 0;
}


int ARMCI_Uses_shm_group()
{
    return 0;
}


/* Is it memory copy? */
void PARMCI_Copy(void *src, void *dst, int n)
{
    assert(0);
    memcpy(dst, src, sizeof(int) * n);
}


/* Group Functions */
int ARMCI_Uses_shm_grp(ARMCI_Group *group)
{
    assert(0);
    return 0;
}


void ARMCI_Cleanup()
{
  PARMCI_Finalize();
}


/* JAD technically not an error to have empty impl of aggregate methods */
void ARMCI_SET_AGGREGATE_HANDLE(armci_hdl_t* handle)
{
}


void ARMCI_UNSET_AGGREGATE_HANDLE(armci_hdl_t* handle)
{
}


/* Always return 0, since shared memory not implemented yet */
int ARMCI_Same_node(int proc)
{
    return 0;
}



