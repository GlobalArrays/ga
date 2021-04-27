/**
 * Device cache.
 *
 * Defensive programming via copious COMEX_ASSERT statements is encouraged.
 */
/* C headers */
#include <fcntl.h>
#include <unistd.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/errno.h>

/* 3rd party headers */
#include <mpi.h>

/* our headers */
#include "dev_cache.h"
#include "comex_impl.h"
#include "comex_structs.h"

/* The purpose of this code is to wrap cudaIpc code so that the calling program
 * can access cudaMalloc'ed segments from any processor on the same CMP node. It
 * mimics the reg_cache code */
#define STATIC static inline

extern comex_group_world_t g_state;
extern void _fence_master(int master_rank);
extern nb_t* nb_wait_for_handle();
extern void nb_recv(void *buf, int count, int source, nb_t *nb);
extern void nb_send_header(void *buf, int count, int dest, nb_t *nb);
extern void nb_wait_for_all(nb_t *nb);

/* the static members in this module */
static dev_entry_t **dev_cache = NULL; /**< list of caches (one per process) */
static int dev_nprocs = 0; /**< number of caches (one per process) */

static cudaIpcMemHandle_t *mem_handles = NULL;

static char *_mem_list_names;

/*
void dev_cache_init(int op, comex_group_t group);
void dev_cache_insert(void *ptr, comex_group_t group);
void dev_cache_exchange(void *ptr, void **ptrs, comex_igroup_t group);
void dev_cache_launch(void *ptr, int op);
void dev_cache_open(int rank, void *ptr);
void dev_cache_close(int rank, void *ptr);
void dev_cache_remove(void *ptr);
void dev_cache_destroy();
*/

/**
 * Create internal data structures for the registration cache.
 *
 * @param[in] nprocs    number of device caches to create i.e. one per
 *                      process
 * @param[in] rank      rank of calling process
 * @param[in] comm      world communicator
 *
 * @pre this function is called once to initialize the internal data
 * structures and cannot be called again until reg_cache_destroy() has been
 * called
 *
 * @see dev_cache_destroy()
 *
 * @return RR_SUCCESS on success
 */
void dev_cache_init(comex_group_t group)
{
  int i = 0;
  int fd = 0;
  int retval = 0;
  int size;
  char name[31];
  comex_igroup_t *igroup = comex_get_igroup_from_group(group);

  /* preconditions */
  COMEX_ASSERT(NULL == dev_cache);
  COMEX_ASSERT(0 == dev_nprocs);

  /* keep the number of caches around for later use */
  dev_nprocs = igroup->size;

  /* allocate the registration cache list: */
  dev_cache = (dev_entry_t **)malloc(sizeof(dev_entry_t*) * dev_nprocs); 

  size = 31*dev_nprocs;
  _mem_list_names = (char*)malloc(size);
  memset(_mem_list_names,0,size);
  sprintf(name,"/CMX_______________%10d\n",igroup->rank);
  MPI_Allgather(name, 31, MPI_CHAR, _mem_list_names, 31, MPI_CHAR,
      igroup->comm);
  /* create shared memory segment */
  fd = shm_open(name, O_CREAT|O_EXCL|O_RDWR, S_IRUSR|S_IWUSR);
  if (-1 == fd && EEXIST == errno) {
    retval = shm_unlink(name);
    if (-1 == retval) {
      comex_error("dev_cache_init: shm_unlink", retval);
    }
  }

  /* try a second time */
  if (-1 == fd) {
    fd = shm_open(name, O_CREAT|O_EXCL|O_RDWR, S_IRUSR|S_IWUSR);
  }

  /* finally report error if needed */
  if (-1 == fd) {
    comex_error("dev_cache_init: shm_open", retval);
  }

  /* set the size of my shared memory object */
  size = dev_nprocs *sizeof(cudaIpcMemHandle_t);
  retval = ftruncate(fd, size);

  /* map into local address space */
  mem_handles = (cudaIpcMemHandle_t*)mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);

  /* close file descriptor */
  retval = close(fd);
  if (-1 == retval) {
    comex_error("dev_cache_init: close", -1);
  }

  /* initialize the registration cache list: */
  for (i = 0; i < dev_nprocs; ++i) {
    dev_cache[i] = NULL;
  }
}


/**
 * Deregister and destroy all cache entries and associated buffers.
 *
 * @pre this function is called once to destroy the internal data structures
 * and cannot be called again until dev_cache_init() has been called
 *
 * @see dev_cache_init()
 */
void dev_cache_destroy()
{
    int i = 0;
    int size = 0;
    int retval = 0;

    /* preconditions */
    COMEX_ASSERT(NULL != dev_cache);
    COMEX_ASSERT(0 != dev_nprocs);

    for (i = 0; i < dev_nprocs; ++i) {
        dev_entry_t *runner = dev_cache[i];

        while (runner) {
            dev_entry_t *previous = runner; /* pointer to previous runner */

            /* get next runner */
            runner = runner->next;
            /* destroy the entry */
            free(previous);
        }
    }

    /* free registration cache list */
    free(dev_cache);
    dev_cache = NULL;

    /* free memory handles list */
    /* unmap the memory */
    size = 31*dev_nprocs;
    retval = munmap(mem_handles, size);
    if (-1 == retval) {
      comex_error("dev_cache_destroy: munmap", retval);
    }

    /* remove the shared memory object */
    retval = shm_unlink(_mem_list_names);
    if (-1 == retval) {
      comex_error("dev_cash_destroy: shm_unlink", retval);
    }

    free(_mem_list_names);
    /* reset the number of caches */
    dev_nprocs = 0;
}

/**
 * Exchange pointers between all ranks in the allocation.
 *
 * @param[in] ptr pointer to device allocation on calling processor
 * @param[out] ptrs list of pointers to all allocations
 * @param[in] group group containing all processes in the allocation
 */
void dev_cache_exchange(void *ptr, void ***ptrs, comex_group_t group)
{
  int prog_rank;
  comex_igroup_t *igroup = comex_get_igroup_from_group(group);
  int fd = 0;
  int retval = 0;
  int rank = igroup->rank;
  int nprocs = igroup->size;
  MPI_Comm comm = igroup->comm;
  int wrank, tmp, size, i;
  nb_t *nb = NULL;
  void *g_ptr;

  /* get world rank */
  comex_group_translate_world(group,rank, &wrank);
  prog_rank = g_state.master[wrank];

  cudaIpcMemHandle_t *g_mem_handles;
  if (ptr != NULL) {
    cudaIpcGetMemHandle(&mem_handles[wrank],ptr);
  }
  /* send this handle to the progress rank and get a mapped address back */
  /* start by copying this handle to progress rank */
  fd = shm_open(&_mem_list_names[31*prog_rank], O_CREAT|O_EXCL|O_RDWR, S_IRUSR|S_IWUSR);
  size = nprocs*sizeof(void*);
  g_mem_handles = (cudaIpcMemHandle_t*)mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  g_mem_handles[rank] = mem_handles[wrank];
  retval = close(fd);
  /* copy memory handle to other processors on SMP node */
  MPI_Comm_size(comm, &nprocs);
  for (i=0; i<nprocs; i++) {
    /* get world rank of process */
    comex_group_translate_world(group,i, &tmp);
    if (i == rank) {
      /* copy cudaIpcMemHandle_t to progrss rank */
      fd = shm_open(&_mem_list_names[31*prog_rank], O_CREAT|O_EXCL|O_RDWR, S_IRUSR|S_IWUSR);
      g_mem_handles = (cudaIpcMemHandle_t*)mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
      g_mem_handles[rank] = mem_handles[rank];
      retval = close(fd);
    } else if (g_state.master[tmp] == prog_rank) {
      /* both calling process and process i have same progress rank copy mem
       * handle so copy cudaIpcMemHandle_t to remote process */
      fd = shm_open(&_mem_list_names[31*tmp], O_CREAT|O_EXCL|O_RDWR, S_IRUSR|S_IWUSR);
      g_mem_handles = (cudaIpcMemHandle_t*)mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
      g_mem_handles[rank] = mem_handles[rank];
      retval = close(fd);
    }
  }
  /* need to sync across all processors, including progress ranks */
  _fence_master(prog_rank);
  comex_fence_all(group);
  comex_wait_all(group);
  /* everybody should have a memory handle for all allocations on the same SMP
   * node. Now need to come * up with a global list of pointers. Each process
   * needs to get a pointer from its progress rank*/
  *ptrs = (void**)malloc(sizeof(void*));
  nb = nb_wait_for_handle();
  header_t *header = (header_t*)malloc(sizeof(header_t));
  COMEX_ASSERT(header);
  MAYBE_MEMSET(header, 0, sizeof(header_t));
  header->operation = OP_MALLOC_DEV;
  header->rank = wrank;
  header->length = sizeof(void*);
  /* prepost receive */
  nb_recv(&g_ptr, size, prog_rank, nb);
  nb_send_header(header, sizeof(header_t), prog_rank, nb);
  nb_wait_for_all(nb);
  MPI_Allgather(&g_ptr, sizeof(void*), MPI_BYTE, *ptrs, sizeof(void*), MPI_BYTE,
      igroup->comm);
  
}

#if 0
/**
 * Create a new registration entry based on the given members.
 *
 * @pre 0 <= rank && rank < reg_nprocs
 * @pre NULL != buf
 * @pre 0 <= len
 * @pre reg_cache_init() was previously called
 * @pre NULL == reg_cache_find(rank, buf, 0)
 * @pre NULL == reg_cache_find_intersection(rank, buf, 0)
 *
 * @return pointer to new node
 */
void dev_cache_insert(int rank, void *buf)
{
    dev_cache_t *node = NULL;

    if (buf == NULL) {
      return;
    }
    /* preconditions */
    COMEX_ASSERT(0 <= rank && rank < dev_nprocs);
    /* TODO: May need to do something about this
    COMEX_ASSERT(NULL == dev_cache_find(rank, buf, len, dev_id));
    */

    /* allocate the new entry */
    node = (dev_entry_t *)malloc(sizeof(dev_entry_t));
    COMEX_ASSERT(node);

    /* initialize the new entry */
    node->rank = rank;
    node->ptr = buf;
    node->next = NULL;

    /* push new entry to tail of linked list */
    if (NULL == reg_cache[rank]) {
        reg_cache[rank] = node;
    }
    else {
        reg_entry_t *runner = reg_cache[rank];
        while (runner->next) {
            runner = runner->next;
        }
        runner->next = node;
    }

    return node;
}


/**
 * Locate a registration cache entry which contains the given segment
 * completely.
 *
 * @param[in] rank  rank of the process
 * @param[in] buf   starting address of the buffer
 * @parma[in] len   length of the buffer
 * @parma[in] dev_id  device ID (if used)
 * 
 * @pre 0 <= rank && rank < reg_nprocs
 * @pre reg_cache_init() was previously called
 *
 * @return the reg cache entry, or NULL on failure
 */
void dev_cache_open(int rank, void *buf)
{
    reg_entry_t *entry = NULL;
    reg_entry_t *runner = NULL;

    if (buf == NULL) return entry;
#if DEBUG
    printf("[%d] reg_cache_find(rank=%d, buf=%p, len=%d)\n",
            g_state.rank, rank, buf, len);
#endif

    /* preconditions */
    COMEX_ASSERT(NULL != reg_cache);
    COMEX_ASSERT(0 <= rank && rank < reg_nprocs);

    runner = reg_cache[rank];

    while (runner && NULL == entry) {
        if (RR_SUCCESS == reg_entry_contains(runner, buf, len, dev_id)) {
            entry = runner;
#if DEBUG
            printf("[%d] reg_cache_find entry found\n"
                    "reg_entry=%p buf=%p len=%d\n"
                    "rank=%d buf=%p len=%zu name=%s mapped=%p\n",
                    g_state.rank, runner, buf, len,
                    runner->rank, runner->buf, runner->len,
                    runner->name, runner->mapped);
#endif
        }
        runner = runner->next;
    }

#ifndef NDEBUG
    /* we COMEX_ASSERT that the found entry was unique */
    while (runner) {
        if (RR_SUCCESS == reg_entry_contains(runner, buf, len, dev_id)) {
#if DEBUG
            printf("[%d] reg_cache_find duplicate found\n"
                    "reg_entry=%p buf=%p len=%d\n"
                    "rank=%d buf=%p len=%zu name=%s mapped=%p\n",
                    g_state.rank, runner, buf, len,
                    runner->rank, runner->buf, runner->len,
                    runner->name, runner->mapped);
#endif
            COMEX_ASSERT(0);
        }
        runner = runner->next;
    }
#endif

    return entry;
}

/**
 * Removes the reg cache entry associated with the given rank and buffer.
 *
 * If this process owns the buffer, it will unregister the buffer, as well.
 *
 * @param[in] rank
 * @param[in] buf
 * @param[in] dev_id device ID, if applicable
 *
 * @pre 0 <= rank && rank < reg_nprocs
 * @pre NULL != buf
 * @pre reg_cache_init() was previously called
 * @pre NULL != reg_cache_find(rank, buf, 0)
 *
 * @return RR_SUCCESS on success
 *         RR_FAILURE otherwise
 */
reg_return_t
reg_cache_delete(int rank, void *buf, int dev_id)
{
    reg_return_t status = RR_FAILURE;
    reg_entry_t *runner = NULL;
    reg_entry_t *previous_runner = NULL;
    if (buf == NULL) return RR_SUCCESS;

#if DEBUG
    printf("[%d] reg_cache_delete(rank=%d, buf=%p)\n",
            g_state.rank, rank, buf);
#endif

    /* preconditions */
    COMEX_ASSERT(NULL != reg_cache);
    COMEX_ASSERT(0 <= rank && rank < reg_nprocs);
    COMEX_ASSERT(NULL != buf);
    COMEX_ASSERT(NULL != reg_cache_find(rank, buf, 0, dev_id));

    /* this is more restrictive than reg_cache_find() in that we locate
     * exactlty the same region starting address */
    runner = reg_cache[rank];
    while (runner) {
        if (runner->buf == buf && runner->dev_id == dev_id) {
            break;
        }
        previous_runner = runner;
        runner = runner->next;
    }
    /* we should have found an entry */
    if (NULL == runner) {
      printf("p[%d] (reg_cache_delete) rank: %d buf: %p\n",g_state.rank,rank,buf);
        COMEX_ASSERT(0);
        return RR_FAILURE;
    }

    /* pop the entry out of the linked list */
    if (previous_runner) {
        previous_runner->next = runner->next;
    }
    else {
        reg_cache[rank] = reg_cache[rank]->next;
    }

    status = reg_entry_destroy(rank, runner);

    return status;
}


reg_return_t reg_cache_nullify(reg_entry_t *node)
{
#if DEBUG
    printf("[%d] reg_cache_nullify(node=%p)\n",
            g_state.rank, node);
#endif

    node->next = NULL;
    node->buf = NULL;
    node->len = 0;
    node->mapped = NULL;
    node->rank = -1;
    node->use_dev = 0;
    node->dev_id = -1;
    (void)memset(node->name, 0, SHM_NAME_SIZE);

    return RR_SUCCESS;
}
#endif
