/* cmx header file */
#ifndef _CMX_IMPL_H
#define _CMX_IMPL_H

#include <mpi.h>

#include <stdlib.h>
#include <vector>
#include <complex>
#include "defines.hpp"
#include "p_group.hpp"
#include "reg_cache.hpp"
#include "p_environment.hpp"

#define XDEBUG

namespace CMX {

enum _cmx_types{_CMX_UNKNOWN = 0,
  _CMX_INT,
  _CMX_LONG,
  _CMX_FLOAT,
  _CMX_DOUBLE,
  _CMX_COMPLEX,
  _CMX_DCOMPLEX};

template<typename _type>
class p_CMX {

public:

  /**
   * Basic constructor
   * @param[in] group home group for allocation
   * @param[in] size size of allocation, in bytes
   */
  p_CMX(p_Group *group, cmxInt size){
    if constexpr(std::is_same_v<_type,int>) {
      p_datatype=_CMX_INT;
    } else if constexpr(std::is_same_v<_type,long>) {
      p_datatype=_CMX_LONG;
    } else if constexpr(std::is_same_v<_type,float>) {
      p_datatype=_CMX_FLOAT;
    } else if constexpr(std::is_same_v<_type,double>) {
      p_datatype=_CMX_DOUBLE;
    } else if constexpr(std::is_same_v<_type,std::complex<float> >) {
      p_datatype=_CMX_COMPLEX;
    } else if constexpr(std::is_same_v<_type,std::complex<double> >) {
      p_datatype=_CMX_DCOMPLEX;
    }

    p_environment = p_Environment::instance();

#ifdef DEBUG
    printf("Initialize p_CMX p_datatype: %d\n",p_datatype);
    switch(p_datatype) {
      case _CMX_UNKNOWN:
        printf("UNKNOWN datatype\n");
        break;
      case _CMX_INT:
        printf("int datatype\n");
        break;
      case _CMX_LONG:
        printf("long datatype\n");
        break;
      case _CMX_FLOAT:
        printf("float datatype\n");
        break;
      case _CMX_DOUBLE:
        printf("double datatype\n");
        break;
      case _CMX_COMPLEX:
        printf("single complex datatype\n");
        break;
      case _CMX_DCOMPLEX:
        printf("double complex datatype\n");
        break;
      default:
        printf("UNASSIGNED datatype\n");
        break;
    }
#endif

#if 0

    reg_entry_t *reg_entries = NULL;
    reg_entry_t my_reg;
    size_t size_entries = 0;
    int my_master = -1;
    int my_world_rank = -1;
    int i = 0;
    int is_notifier = 0;
    int reg_entries_local_count = 0;
    reg_entry_t *reg_entries_local = NULL;
    int status = 0;

    /* preconditions */
    CMX_ASSERT(cmx_hdl);
   
#if DEBUG
    fprintf(stderr, "[%d] cmx_malloc(ptrs=%p, size=%lu, group=%d)\n",
            g_state.rank, ptrs, (long unsigned)size, group);
#endif

    /* is this needed? */
    cmx_barrier(igroup);

    my_world_rank = _get_world_rank(igroup, igroup->rank);
    my_master = g_state.master[my_world_rank];

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_malloc my_master=%d\n", g_state.rank, my_master);
#endif

    int smallest_rank_with_same_hostid, largest_rank_with_same_hostid; 
    int num_progress_ranks_per_node, is_node_ranks_packed;
    num_progress_ranks_per_node = get_num_progress_ranks_per_node();
    is_node_ranks_packed = get_progress_rank_distribution_on_node();
    smallest_rank_with_same_hostid = _smallest_world_rank_with_same_hostid(igroup);
    largest_rank_with_same_hostid = _largest_world_rank_with_same_hostid(igroup);
    is_notifier = g_state.rank == get_my_master_rank_with_same_hostid(g_state.rank,
        g_state.node_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
        num_progress_ranks_per_node, is_node_ranks_packed);
    if (is_notifier) {
        reg_entries_local = malloc(sizeof(reg_entry_t)*g_state.node_size);
    }

    /* allocate space for registration cache entries */
    size_entries = sizeof(reg_entry_t) * igroup->size;
    reg_entries = malloc(size_entries);
    MAYBE_MEMSET(reg_entries, 0, sizeof(reg_entry_t)*igroup->size);
#if DEBUG
    fprintf(stderr, "[%d] cmx_malloc lr_same_hostid=%d\n", 
      g_state.rank, largest_rank_with_same_hostid);
    fprintf(stderr, "[%d] cmx_malloc igroup size=%d\n", g_state.rank, igroup->size);
    fprintf(stderr, "[%d] cmx_malloc node_size=%d\n", g_state.rank, g_state.node_size);
    fprintf(stderr, "[%d] cmx_malloc is_notifier=%d\n", g_state.rank, is_notifier);
    fprintf(stderr, "[%d] rank, igroup size[5%d]\n",
            g_state.rank, igroup->size);
#endif
#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_malloc allocated reg entries\n",
            g_state.rank);
#endif

    /* allocate and register segment */
    MAYBE_MEMSET(&my_reg, 0, sizeof(reg_entry_t));
    if (0 == size) {
        reg_cache_nullify(&my_reg);
    }
    else {
        my_reg = *_cmx_malloc_local(sizeof(char)*size);
    }

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_malloc allocated and registered local shmem\n",
            g_state.rank);
#endif

    /* exchange buffer address via reg entries */
    reg_entries[igroup->rank] = my_reg;
    status = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            reg_entries, sizeof(reg_entry_t), MPI_BYTE, igroup->comm);
    _translate_mpi_error(status, "cmx_malloc:MPI_Allgather");
#if DEBUG
    fprintf(stderr, "[%d] cmx_malloc allgather status [%d]\n",
            g_state.rank, status);
#endif
    CMX_ASSERT(MPI_SUCCESS == status);

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_malloc allgather reg entries\n",
            g_state.rank);
#endif

    /* insert reg entries into local registration cache */
    for (i=0; i<igroup->size; ++i) {
        if (NULL == reg_entries[i].buf) {
            /* a proc did not allocate (size==0) */
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_malloc found NULL buf at %d\n",
                    g_state.rank, i);
#endif
        }
        else if (g_state.rank == reg_entries[i].rank) {
            /* we already registered our own memory, but PR hasn't */
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_malloc found self at %d\n",
                    g_state.rank, i);
#endif
            if (is_notifier) {
                /* does this need to be a memcpy?? */
                reg_entries_local[reg_entries_local_count++] = reg_entries[i];
            }
        }
        // else if (g_state.hostid[reg_entries[i].rank]
        //         == g_state.hostid[my_world_rank]) 

        else if (g_state.master[reg_entries[i].rank] == 
           g_state.master[get_my_master_rank_with_same_hostid(g_state.rank,
           g_state.node_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
           num_progress_ranks_per_node, is_node_ranks_packed)] )
            {
            /* same SMP node, need to mmap */
            /* open remote shared memory object */
            void *memory = _shm_attach(reg_entries[i].name, reg_entries[i].len);
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_malloc registering "
                    "rank=%d buf=%p len=%lu name=%s map=%p\n",
                    g_state.rank,
                    reg_entries[i].rank,
                    reg_entries[i].buf,
                    reg_entries[i].len,
                    reg_entries[i].name,
                    memory);
#endif
            (void)reg_cache_insert(
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
    for (i=0; i<igroup->size; ++i) {
      cmx_alloc_t *link = (cmx_alloc_t*)malloc(sizeof(cmx_alloc_t));
      cmx_hdl->list = link;
      link->buf = reg_entries[i].buf;
      link->size = (cmxInt)reg_entries[i].len;
      link->rank = reg_entries[i].rank;
      link->next = prev;
      prev = link;
    }
    cmx_hdl->group = igroup;
    cmx_hdl->rank = igroup->rank;
    cmx_hdl->buf = my_reg.mapped;
    cmx_hdl->bytes = my_reg.len;

    /* send reg entries to my master */
    /* first non-master rank in an SMP node sends the message to master */
    if (is_notifier) {
        _cmx_request nb;
        int reg_entries_local_size = 0;
        int message_size = 0;
        char *message = NULL;
        header_t *header = NULL;

        nb_handle_init(&nb);
        reg_entries_local_size = sizeof(reg_entry_t)*reg_entries_local_count;
        message_size = sizeof(header_t) + reg_entries_local_size;
        message = malloc(message_size);
        CMX_ASSERT(message);
        header = (header_t*)message;
        header->operation = OP_MALLOC;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = 0;
        header->length = reg_entries_local_count;
        (void)memcpy(message+sizeof(header_t), reg_entries_local, reg_entries_local_size);
        nb_recv(NULL, 0, my_master, &nb); /* prepost ack */
        nb_send_header(message, message_size, my_master, &nb);
        nb_wait_for_all(&nb);
        free(reg_entries_local);
    }

    free(reg_entries);

    cmx_barrier(igroup);

    return CMX_SUCCESS;
#endif
  };

  /**
   * Simple destructor
   */
  ~p_CMX()
  {
  }

private:

  int p_datatype = _CMX_UNKNOWN;

  p_Environment p_environment;

};
}
#endif /* _CMX_IMPL_H */
