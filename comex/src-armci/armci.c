#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "armci.h"
#include "parmci.h"
#include "comex.h"


extern int ARMCI_Default_Proc_Group;
MPI_Comm ARMCI_COMM_WORLD;

int _number_of_procs_per_node = 1;
int _my_node_id;
ARMCI_Group ARMCI_Node_group;

/**
 * Initialize parameters that can be used by the armci_domain_xxx function
 */
void armci_init_domains(MPI_Comm comm)
{
  int i, status;
  char name[MPI_MAX_PROCESSOR_NAME];
  char *namebuf, *buf_ptr, *prev_ptr;
  int namelen, rank, size, nprocs;
  int *nodeid, *nodesize;
  int ncnt;

  status = MPI_Comm_rank(comm, &rank);
  assert(MPI_SUCCESS == status);
  status = MPI_Comm_size(comm, &size);
  assert(MPI_SUCCESS == status);

  /* determine number of processors per node. First find node name */
  namebuf = (char*)malloc(MPI_MAX_PROCESSOR_NAME*size*sizeof(char));
  nodeid = (int*)malloc(size*sizeof(int));
  nodesize = (int*)malloc(size*sizeof(int));
  MPI_Get_processor_name(name, &namelen);
  status = MPI_Allgather(name,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,namebuf,
      MPI_MAX_PROCESSOR_NAME,MPI_CHAR,comm);
  assert(MPI_SUCCESS == status);

  /* Bin all processors with the same node name */
  ncnt = 0;
  nodeid[0] = ncnt;
  nodesize[0] = 1;
  prev_ptr = namebuf;
  buf_ptr = namebuf+MPI_MAX_PROCESSOR_NAME;
  for (i=1; i<size; i++) {
    namelen = strlen(buf_ptr);
    if (strncmp(prev_ptr,buf_ptr,namelen) != 0) {
      ncnt++;
      nodesize[ncnt]=0;
    }
    nodeid[i] = ncnt;
    nodesize[ncnt]++;
    prev_ptr += MPI_MAX_PROCESSOR_NAME;
    buf_ptr += MPI_MAX_PROCESSOR_NAME;
  }
  ncnt++;
  /* check to see if all nodes have the same number of processors */
  status = 1;
  nprocs = nodesize[0];
  for (i=1; i<ncnt; i++) {
    if (nodesize[i] != nprocs) status = 0;
  }
  /* uneven number of processors per node so bail out and assume each
   * processor is a node */
  if (!status) {
    _number_of_procs_per_node = 1;
    _my_node_id = rank;
  } else {
    /* Same number of processors for all nodes so set domain variables */
    _number_of_procs_per_node = nprocs;
    _my_node_id = rank/_number_of_procs_per_node;
  }

  free(namebuf);
  free(nodeid);
  free(nodesize);
  status = MPI_Barrier(comm);
  assert(MPI_SUCCESS == status);
  /* Create a comex group on the node */
  if (_number_of_procs_per_node > 1) {
    int *nodelist = (int*)malloc(_number_of_procs_per_node*sizeof(int));
    for (i=0; i<_number_of_procs_per_node; i++)
      nodelist[i] = _my_node_id*_number_of_procs_per_node+i;
    comex_group_create(_number_of_procs_per_node, nodelist,
        COMEX_GROUP_WORLD, &ARMCI_Node_group);
  }
}

/**
 * This function checks to see if the data copy is contiguous for both the src
 * and destination buffers. If it is, then a contiguous operation can be used
 * instead of a strided operation. This function is intended for arrays of
 * dimension greater than 1 (contiguous operations can always be used for 1
 * dimensional arrays).
 * 
 * The current implementation tries to identify all contiguous cases by using
 * all information from the stride and count arrays. The old implementation did
 * not identify all cases of contiguous data transfers.
 *
 * src_stride: physical dimensions of source buffer
 * dst_stride: physical dimensions of destination buffer
 * count: number of elements being moved in each dimension
 * n_stride: number of strides (array dimension minus one)
 */
int armci_check_contiguous(int *src_stride, int *dst_stride,
    int *count, int n_stride)
{
#if 1
  /* This is code from the merge between CMX and the current develop branch
   * (2018/7/5) */
  int i;
  int ret = 1;
  int stridelen = 1;
  int gap = 0;
  int src_ld[7], dst_ld[7];
  /**
   * Calculate physical dimensions of buffers from stride arrays
   */
  src_ld[0] = src_stride[0];
  dst_ld[0] = dst_stride[0];
  for (i=1; i<n_stride; i++) {
    src_ld[i] = src_stride[i]/src_stride[i-1];
    dst_ld[i] = dst_stride[i]/dst_stride[i-1];
  }
  /* NOTE: The count array contains the length of the final dimension and can
   * be used to evaluate some corner cases
   */
  for (i=0; i<n_stride; i++) {
    /* check for overflow */
    int tmp = stridelen * count[i];
    if (stridelen != 0 && tmp / stridelen != count[i]) {
      ret = 0;
      break;
    }
    stridelen = tmp;
    if ((count[i] < src_ld[i] || count[i] < dst_ld[i])
        && gap == 1) {
      /* Data is definitely strided in memory */
      ret = 0;
      break;
    } else if ((count[i] < src_ld[i] || count[i] < dst_ld[i]) &&
        gap == 0) {
      /* First dimension that doesn't match physical dimension */
      gap = 1;
    } else if (count[i] != 1 && gap == 1) {
      /* Found a mismatch between requested block and physical dimensions
       * indicating a possible stride in memory
       * */
      ret = 0;
      break;
    }
  }
  /**
   * Everything looks good up to this point but need to verify that last
   * dimension is 1 if a mismatch between requested block and physical
   * array dimensions has been found previously
   */
  if (gap == 1 && ret == 1 && n_stride > 0) {
    if (count[n_stride] != 1) ret = 0;
  }
  return ret;
#else
  int i;
  int ret = 1;
  int stridelen = 1;
  /* NOTE: The count array contains the length of the final dimension and could
   * be used to evaluate some corner cases that are not picked up by this
   * algorithm
   */ 
  for (i=0; i<n_stride; i++) {
    /* check for overflow */
    int tmp = stridelen * count[i];
    if (stridelen != 0 && tmp / stridelen != count[i]) {
      ret = 0;
      break;
    }
    stridelen = tmp;
    if (stridelen < src_stride[i] || stridelen < dst_stride[i]) {
      ret = 0;
      break;
    }
  }
  return ret;
#endif
}

/**
 * Dummy function for use in debugging
 */
int armci_checkt_contiguous(int *src_stride, int *dst_stride,
    int *count, int n_stride)
{
  return 0;
}

static void convert_giov(armci_giov_t *a, comex_giov_t *b, int len)
{
    int i;
    for (i=0; i<len; ++i) {
        b[i].src = a[i].src_ptr_array;
        b[i].dst = a[i].dst_ptr_array;
        b[i].count = a[i].ptr_array_len;
        b[i].bytes = a[i].bytes;
    }
}




int PARMCI_Acc(int optype, void *scale, void *src, void *dst, int bytes, int proc)
{
    return comex_acc(optype, scale, src, dst, bytes, proc, COMEX_GROUP_WORLD);
}


int PARMCI_AccS(int optype, void *scale, void *src_ptr, int *src_stride_arr, void *dst_ptr, int *dst_stride_arr, int *count, int stride_levels, int proc)
{
  int iret;
  /* check if data is contiguous */
  if (armci_check_contiguous(src_stride_arr, dst_stride_arr, count, stride_levels)) {
    int i;
    int lcount = 1;
    for (i=0; i<=stride_levels; i++) lcount *= count[i];
    iret = comex_acc(optype, scale, src_ptr, dst_ptr, lcount, proc, COMEX_GROUP_WORLD);
  } else {
    iret = comex_accs(optype, scale, src_ptr, src_stride_arr, dst_ptr,
        dst_stride_arr, count, stride_levels, proc, COMEX_GROUP_WORLD);
  }
  return iret;
}


int PARMCI_AccV(int op, void *scale, armci_giov_t *darr, int len, int proc)
{
    int rc;
    comex_giov_t *adarr = malloc(sizeof(comex_giov_t) * len);
    convert_giov(darr, adarr, len);
    rc = comex_accv(op, scale, adarr, len, proc, COMEX_GROUP_WORLD);
    free(adarr);
    return rc;
}


/* fence is always on the world group */
void PARMCI_AllFence()
{
    int rc;
    rc = comex_fence_all(COMEX_GROUP_WORLD);
    assert(COMEX_SUCCESS == rc);
}


void PARMCI_GroupFence(ARMCI_Group *group)
{
  int rc;
  if (*group > 0) {
    rc = comex_fence_all(*group);
  } else {
    rc = comex_fence_all(COMEX_GROUP_WORLD);
  }
  assert(COMEX_SUCCESS == rc);
}


void PARMCI_Barrier()
{
    int rc;
    rc = comex_barrier(ARMCI_Default_Proc_Group);
    assert(COMEX_SUCCESS == rc);
}


int PARMCI_Create_mutexes(int num)
{
    return comex_create_mutexes(num);
}


int PARMCI_Destroy_mutexes()
{
    return comex_destroy_mutexes();
}


/* fence is always on the world group */
void PARMCI_Fence(int proc)
{
    comex_fence_proc(proc, COMEX_GROUP_WORLD);
}


void PARMCI_Finalize()
{
    comex_finalize();
}


int PARMCI_Free(void *ptr)
{
    return comex_free(ptr, ARMCI_Default_Proc_Group);
}

int PARMCI_Free_memdev(void *ptr)
{
    return comex_free_dev(ptr, ARMCI_Default_Proc_Group);
}

int ARMCI_Free_group(void *ptr, ARMCI_Group *group)
{
    return comex_free(ptr, *group);
}


int PARMCI_Free_local(void *ptr)
{
    return comex_free_local(ptr);
}


int PARMCI_Get(void *src, void *dst, int bytes, int proc)
{
    return comex_get(src, dst, bytes, proc, COMEX_GROUP_WORLD);
}


int PARMCI_GetS(void *src_ptr, int *src_stride_arr, void *dst_ptr, int *dst_stride_arr, int *count, int stride_levels, int proc)
{
  int iret;
  /* check if data is contiguous */
  if (armci_check_contiguous(src_stride_arr, dst_stride_arr, count, stride_levels)) {
    int i;
    int lcount = 1;
    for (i=0; i<=stride_levels; i++) lcount *= count[i];
    iret = comex_get(src_ptr, dst_ptr, lcount, proc, COMEX_GROUP_WORLD);
  } else {
    iret = comex_gets(src_ptr, src_stride_arr, dst_ptr, dst_stride_arr,
        count, stride_levels, proc, COMEX_GROUP_WORLD);
  }
  return iret;
}


int PARMCI_GetV(armci_giov_t *darr, int len, int proc)
{
    int rc;
    comex_giov_t *adarr = malloc(sizeof(comex_giov_t) * len);
    convert_giov(darr, adarr, len);
    rc = comex_getv(adarr, len, proc, COMEX_GROUP_WORLD);
    free(adarr);
    return rc;
}


double PARMCI_GetValueDouble(void *src, int proc)
{
    int rc;
    double val;
    rc = comex_get(src, &val, sizeof(double), proc, COMEX_GROUP_WORLD);
    assert(COMEX_SUCCESS == rc);
    return val;
}


float PARMCI_GetValueFloat(void *src, int proc)
{
    int rc;
    float val;
    rc = comex_get(src, &val, sizeof(float), proc, COMEX_GROUP_WORLD);
    assert(COMEX_SUCCESS == rc);
    return val;
}


int PARMCI_GetValueInt(void *src, int proc)
{
    int rc;
    int val;
    rc = comex_get(src, &val, sizeof(int), proc, COMEX_GROUP_WORLD);
    assert(COMEX_SUCCESS == rc);
    return val;
}


long PARMCI_GetValueLong(void *src, int proc)
{
    int rc;
    long val;
    rc = comex_get(src, &val, sizeof(long), proc, COMEX_GROUP_WORLD);
    assert(COMEX_SUCCESS == rc);
    return val;
}


int PARMCI_Init()
{
    int rc = comex_init();
    assert(COMEX_SUCCESS == rc);
    rc = comex_group_comm(COMEX_GROUP_WORLD, &ARMCI_COMM_WORLD);
    assert(COMEX_SUCCESS == rc);
    ARMCI_Default_Proc_Group = 0;
    armci_init_domains(ARMCI_COMM_WORLD);
    return rc;
}


int PARMCI_Init_args(int *argc, char ***argv)
{
    int rc = comex_init_args(argc, argv);
    assert(COMEX_SUCCESS == rc);
    rc = comex_group_comm(COMEX_GROUP_WORLD, &ARMCI_COMM_WORLD);
    assert(COMEX_SUCCESS == rc);
    armci_init_domains(ARMCI_COMM_WORLD);
    ARMCI_Default_Proc_Group = 0;
    return rc;
}


int PARMCI_Initialized()
{
    return comex_initialized();
}


void PARMCI_Lock(int mutex, int proc)
{
    comex_lock(mutex, proc);
}


int PARMCI_Malloc(void **ptr_arr, armci_size_t bytes)
{
    return comex_malloc(ptr_arr, bytes, ARMCI_Default_Proc_Group);
}

int PARMCI_Malloc_memdev(void **ptr_arr, armci_size_t bytes, const char *device)
{
    return comex_malloc_mem_dev(ptr_arr, bytes, ARMCI_Default_Proc_Group,device);
}


int ARMCI_Malloc_group(void **ptr_arr, armci_size_t bytes, ARMCI_Group *group)
{
    return comex_malloc(ptr_arr, bytes, *group);
}

int ARMCI_Malloc_group_memdev(void **ptr_arr, armci_size_t bytes,
    ARMCI_Group *group, const char *device)
{
    return comex_malloc_mem_dev(ptr_arr, bytes, *group,device);
}


void* PARMCI_Malloc_local(armci_size_t bytes)
{
    return comex_malloc_local(bytes);
}


void* PARMCI_Memat(armci_meminfo_t *meminfo, long offset)
{
    void *ptr=NULL;
    int rank;

    comex_group_rank(COMEX_GROUP_WORLD, &rank);

    if(meminfo==NULL) comex_error("PARMCI_Memat: Invalid arg #1 (NULL ptr)",0);

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

    comex_group_rank(COMEX_GROUP_WORLD, &rank);

    if(size<=0) comex_error("PARMCI_Memget: size must be > 0", (int)size);
    if(meminfo==NULL) comex_error("PARMCI_Memget: Invalid arg #2 (NULL ptr)",0);
    if(memflg!=0) comex_error("PARMCI_Memget: Invalid memflg", memflg);

    armci_ptr = myptr = comex_malloc_local(size);
    if(size) if(!myptr) comex_error("PARMCI_Memget failed", (int)size);

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

    comex_group_rank(COMEX_GROUP_WORLD, &rank);

    if(meminfo==NULL) comex_error("PARMCI_Memget: Invalid arg #2 (NULL ptr)",0);

    /* only the creator can delete the segment */
    if(meminfo->cpid == rank)
    {
        void *ptr = meminfo->addr;
        comex_free_local(ptr);
    }

    meminfo->addr       = NULL;
    meminfo->armci_addr = NULL;
    /* if(meminfo->attr!=NULL) free(meminfo->attr); */
}



int PARMCI_NbAccS(int optype, void *scale, void *src_ptr, int *src_stride_arr, void *dst_ptr, int *dst_stride_arr, int *count, int stride_levels, int proc, armci_hdl_t *nb_handle)
{
  int iret;
  /* check if data is contiguous */
  if (armci_check_contiguous(src_stride_arr, dst_stride_arr, count, stride_levels)) {
    int i;
    int lcount = 1;
    for (i=0; i<=stride_levels; i++) lcount *= count[i];
    iret = comex_nbacc(optype, scale, src_ptr, dst_ptr, lcount, proc,
        COMEX_GROUP_WORLD, nb_handle);
  } else {
    iret = comex_nbaccs(optype, scale, src_ptr, src_stride_arr, dst_ptr,
        dst_stride_arr, count, stride_levels, proc, COMEX_GROUP_WORLD, nb_handle);
  }
  return iret;
}


int PARMCI_NbAccV(int op, void *scale, armci_giov_t *darr, int len, int proc, armci_hdl_t *nb_handle)
{
    int rc;
    comex_giov_t *adarr = malloc(sizeof(comex_giov_t) * len);
    convert_giov(darr, adarr, len);
    rc = comex_nbaccv(op, scale, adarr, len, proc, COMEX_GROUP_WORLD, nb_handle);
    free(adarr);
    return rc;
}


int PARMCI_NbGet(void *src, void *dst, int bytes, int proc, armci_hdl_t *nb_handle)
{
    return comex_nbget(src, dst, bytes, proc, COMEX_GROUP_WORLD, nb_handle);
}


int PARMCI_NbGetS(void *src_ptr, int *src_stride_arr, void *dst_ptr, int *dst_stride_arr, int *count, int stride_levels, int proc, armci_hdl_t *nb_handle)
{
  int iret;
  /* check if data is contiguous */
  if (armci_check_contiguous(src_stride_arr, dst_stride_arr, count, stride_levels)) {
    int i;
    int lcount = 1;
    for (i=0; i<=stride_levels; i++) lcount *= count[i];
    iret = comex_nbget(src_ptr, dst_ptr, lcount, proc,
        COMEX_GROUP_WORLD, nb_handle);
  } else {
    iret = comex_nbgets(src_ptr, src_stride_arr, dst_ptr, dst_stride_arr,
        count, stride_levels, proc, COMEX_GROUP_WORLD, nb_handle);
  }
  return iret;
}


int PARMCI_NbGetV(armci_giov_t *darr, int len, int proc, armci_hdl_t *nb_handle)
{
    int rc;
    comex_giov_t *adarr = malloc(sizeof(comex_giov_t) * len);
    convert_giov(darr, adarr, len);
    rc = comex_nbgetv(adarr, len, proc, COMEX_GROUP_WORLD, nb_handle);
    free(adarr);
    return rc;
}


int PARMCI_NbPut(void *src, void *dst, int bytes, int proc, armci_hdl_t *nb_handle)
{
    return comex_nbput(src, dst, bytes, proc, COMEX_GROUP_WORLD, nb_handle);
}


int PARMCI_NbPutS(void *src_ptr, int *src_stride_arr, void *dst_ptr, int *dst_stride_arr, int *count, int stride_levels, int proc, armci_hdl_t *nb_handle)
{
  int iret;
  /* check if data is contiguous */
  if (armci_check_contiguous(src_stride_arr, dst_stride_arr, count, stride_levels)) {
    int i;
    int lcount = 1;
    for (i=0; i<=stride_levels; i++) lcount *= count[i];
    iret = comex_nbput(src_ptr, dst_ptr, lcount, proc,
        COMEX_GROUP_WORLD, nb_handle);
  } else {
    iret = comex_nbputs(src_ptr, src_stride_arr, dst_ptr, dst_stride_arr,
        count, stride_levels, proc, COMEX_GROUP_WORLD, nb_handle);
  }
  return iret;
}


int PARMCI_NbPutV(armci_giov_t *darr, int len, int proc, armci_hdl_t *nb_handle)
{
    int rc;
    comex_giov_t *adarr = malloc(sizeof(comex_giov_t) * len);
    convert_giov(darr, adarr, len);
    rc = comex_nbputv(adarr, len, proc, COMEX_GROUP_WORLD, nb_handle);
    free(adarr);
    return rc;
}


int PARMCI_NbPutValueDouble(double src, void *dst, int proc, armci_hdl_t *nb_handle)
{
    return comex_nbput(&src, dst, sizeof(double), proc, COMEX_GROUP_WORLD, nb_handle);
}


int PARMCI_NbPutValueFloat(float src, void *dst, int proc, armci_hdl_t *nb_handle)
{
    return comex_nbput(&src, dst, sizeof(float), proc, COMEX_GROUP_WORLD, nb_handle);
}


int PARMCI_NbPutValueInt(int src, void *dst, int proc, armci_hdl_t *nb_handle)
{
    return comex_nbput(&src, dst, sizeof(int), proc, COMEX_GROUP_WORLD, nb_handle);
}


int PARMCI_NbPutValueLong(long src, void *dst, int proc, armci_hdl_t *nb_handle)
{
    return comex_nbput(&src, dst, sizeof(long), proc, COMEX_GROUP_WORLD, nb_handle);
}


int PARMCI_Put(void *src, void *dst, int bytes, int proc)
{
    return comex_put(src, dst, bytes, proc, COMEX_GROUP_WORLD);
}


int PARMCI_PutS(void *src_ptr, int *src_stride_arr, void *dst_ptr, int *dst_stride_arr, int *count, int stride_levels, int proc)
{
  int iret;
  /* check if data is contiguous */
  if (armci_check_contiguous(src_stride_arr, dst_stride_arr, count, stride_levels)) {
    int i;
    int lcount = 1;
    for (i=0; i<=stride_levels; i++) lcount *= count[i];
    iret = comex_put(src_ptr, dst_ptr, lcount, proc, COMEX_GROUP_WORLD);
  } else {
    iret = comex_puts(src_ptr, src_stride_arr, dst_ptr, dst_stride_arr,
        count, stride_levels, proc, COMEX_GROUP_WORLD);
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
    comex_giov_t *adarr = malloc(sizeof(comex_giov_t) * len);
    convert_giov(darr, adarr, len);
    rc = comex_putv(adarr, len, proc, COMEX_GROUP_WORLD);
    free(adarr);
    return rc;
}


int PARMCI_PutValueDouble(double src, void *dst, int proc)
{
    return comex_put(&src, dst, sizeof(double), proc, COMEX_GROUP_WORLD);
}


int PARMCI_PutValueFloat(float src, void *dst, int proc)
{
    return comex_put(&src, dst, sizeof(float), proc, COMEX_GROUP_WORLD);
}


int PARMCI_PutValueInt(int src, void *dst, int proc)
{
    return comex_put(&src, dst, sizeof(int), proc, COMEX_GROUP_WORLD);
}


int PARMCI_PutValueLong(long src, void *dst, int proc)
{
    return comex_put(&src, dst, sizeof(long), proc, COMEX_GROUP_WORLD);
}


int PARMCI_Put_flag(void *src, void *dst, int bytes, int *f, int v, int proc)
{
    assert(0);
    return 0;
}


int PARMCI_Rmw(int op, void *ploc, void *prem, int extra, int proc)
{
    return comex_rmw(op, ploc, prem, extra, proc, COMEX_GROUP_WORLD);
}


int PARMCI_Test(armci_hdl_t *nb_handle)
{
    int rc;
    int status;
    rc = comex_test(nb_handle, &status);
    assert(COMEX_SUCCESS == rc);
    return status;
}


void PARMCI_Unlock(int mutex, int proc)
{
    comex_unlock(mutex, proc);
}


int PARMCI_Wait(armci_hdl_t *nb_handle)
{
    return comex_wait(nb_handle);
}


int PARMCI_WaitAll()
{
    return comex_wait_all(COMEX_GROUP_WORLD);
}


int PARMCI_WaitProc(int proc)
{
    return comex_wait_proc(proc, COMEX_GROUP_WORLD);
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

/**
 * Return number of processes on node id
 */
int armci_domain_nprocs(armci_domain_t domain, int id)
{
    return _number_of_procs_per_node;
}


/**
 * Return ID of node corresponding to glob_proc_id
 */
int armci_domain_id(armci_domain_t domain, int glob_proc_id)
{
    return glob_proc_id/_number_of_procs_per_node;
}

/**
 * Return global rank of local proc (loc_proc_id) on node id
 */
int armci_domain_glob_proc_id(armci_domain_t domain, int id, int loc_proc_id)
{
    return id*_number_of_procs_per_node+loc_proc_id;
}


/**
 * Return ID of node containing calling process
 */
int armci_domain_my_id(armci_domain_t domain)
{
    int rank;

    assert(comex_initialized());

    return _my_node_id;
}

/**
 * Return number of nodes in the entire system
 */
int armci_domain_count(armci_domain_t domain)
{
    int size;

    assert(comex_initialized());
    comex_group_size(COMEX_GROUP_WORLD, &size);

    return size/_number_of_procs_per_node;
}


int armci_domain_same_id(armci_domain_t domain, int proc)
{
    int rank;

    comex_group_rank(COMEX_GROUP_WORLD, &rank);

    return (proc/_number_of_procs_per_node == rank/_number_of_procs_per_node);
}


void ARMCI_Error(char *msg, int code)
{
    comex_error(msg, code);
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
    comex_finalize();
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



