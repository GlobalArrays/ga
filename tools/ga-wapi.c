
#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include <mpi.h>
#include "ga-papi.h"
#include "typesf2c.h"
#include "ga-wprof.h" // New profile headers

static int me;
static int nproc;

void wnga_abs_value(Integer g_a)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_abs_value(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_ABS_VALUE, ltme, 0);
}


void wnga_abs_value_patch(Integer g_a, Integer *lo, Integer *hi)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_abs_value_patch(g_a, lo, hi);
   ltme += I_Wtime();
   update_local_entry(PNGA_ABS_VALUE_PATCH, ltme, 0);
}


void wnga_acc(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha)
{
   unsigned long long ltme, sz;
   GET_SIZE(g_a, lo, hi, sz);
   ltme=- I_Wtime();
   pnga_acc(g_a, lo, hi, buf, ld, alpha);
   ltme += I_Wtime();
   update_local_entry(PNGA_ACC, ltme, sz);
}


void wnga_access_block_grid_idx(Integer g_a, Integer *subscript, AccessIndex *index, Integer *ld)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_access_block_grid_idx(g_a, subscript, index, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_ACCESS_BLOCK_GRID_IDX, ltme, 0);
}


void wnga_access_block_grid_ptr(Integer g_a, Integer *index, void *ptr, Integer *ld)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_access_block_grid_ptr(g_a, index, ptr, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_ACCESS_BLOCK_GRID_PTR, ltme, 0);    
}


void wnga_access_block_idx(Integer g_a, Integer idx, AccessIndex *index, Integer *ld)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_access_block_idx(g_a, idx, index, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_ACCESS_BLOCK_IDX, ltme, 0);
}


void wnga_access_block_ptr(Integer g_a, Integer idx, void *ptr, Integer *ld)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_access_block_ptr(g_a, idx, ptr, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_ACCESS_BLOCK_PTR, ltme, 0);
}


void wnga_access_block_segment_idx(Integer g_a, Integer proc, AccessIndex *index, Integer *len)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_access_block_segment_idx(g_a, proc, index, len);
   ltme += I_Wtime();
   update_local_entry(PNGA_ACCESS_BLOCK_SEGMENT_IDX, ltme, 0);
}


void wnga_access_block_segment_ptr(Integer g_a, Integer proc, void *ptr, Integer *len)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_access_block_segment_ptr(g_a, proc, ptr, len);
   ltme += I_Wtime();
   update_local_entry(PNGA_ACCESS_BLOCK_SEGMENT_PTR, ltme, 0);
}


void wnga_access_ghost_element(Integer g_a, AccessIndex *index, Integer subscript[], Integer ld[])
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_access_ghost_element(g_a, index, subscript, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_ACCESS_GHOST_ELEMENT, ltme, 0);    
}


void wnga_access_ghost_element_ptr(Integer g_a, void *ptr, Integer subscript[], Integer ld[])
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_access_ghost_element_ptr(g_a, ptr, subscript, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_ACCESS_GHOST_ELEMENT_PTR, ltme, 0);    
}


void wnga_access_ghost_ptr(Integer g_a, Integer dims[], void *ptr, Integer ld[])
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_access_ghost_ptr(g_a, dims, ptr, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_ACCESS_GHOST_PTR, ltme, 0);        
}

void wnga_access_ghosts(Integer g_a, Integer dims[], AccessIndex *index, Integer ld[])
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_access_ghosts(g_a, dims, index, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_ACCESS_GHOSTS, ltme, 0);            
}


void wnga_access_idx(Integer g_a, Integer *lo, Integer *hi, AccessIndex *index, Integer *ld)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_access_idx(g_a, lo, hi, index, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_ACCESS_IDX, ltme, 0);
}


void wnga_access_ptr(Integer g_a, Integer *lo, Integer *hi, void *ptr, Integer *ld)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_access_ptr(g_a, lo, hi, ptr, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_ACCESS_PTR, ltme, 0);
}


void wnga_add(void *alpha, Integer g_a, void *beta, Integer g_b, Integer g_c)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_add(alpha, g_a, beta, g_b, g_c);
   ltme += I_Wtime();
   update_local_entry(PNGA_ADD, ltme, 0);
}


void wnga_add_constant(Integer g_a, void *alpha)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_add_constant(g_a, alpha);
   ltme += I_Wtime();
   update_local_entry(PNGA_ADD_CONSTANT, ltme, 0);
}


void wnga_add_constant_patch(Integer g_a, Integer *lo, Integer *hi, void *alpha)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_add_constant_patch(g_a, lo, hi, alpha);
   ltme += I_Wtime();
   update_local_entry(PNGA_ADD_CONSTANT_PATCH, ltme, 0);
}


void wnga_add_diagonal(Integer g_a, Integer g_v)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_add_diagonal(g_a, g_v);
   ltme += I_Wtime();
   update_local_entry(PNGA_ADD_DIAGONAL, ltme, 0);
}


void wnga_add_patch(void *alpha, Integer g_a, Integer *alo, Integer *ahi, void *beta, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_add_patch(alpha, g_a, alo, ahi, beta, g_b, blo, bhi, g_c, clo, chi);
   ltme += I_Wtime();
   update_local_entry(PNGA_ADD_PATCH, ltme, 0);
}

logical wnga_allocate(Integer g_a)
{
   logical return_value;
   unsigned long long ltme, sz;
   ltme=- I_Wtime();
   return_value = pnga_allocate(g_a);
   ltme += I_Wtime();
   GET_LOCAL_MSIZE(g_a, sz);
   update_local_entry(PNGA_ALLOCATE, ltme, sz);
   return return_value;
}


void wnga_bin_index(Integer g_bin, Integer g_cnt, Integer g_off, Integer *values, Integer *subs, Integer n, Integer sortit)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_bin_index(g_bin, g_cnt, g_off, values, subs, n, sortit);
   ltme += I_Wtime();
   update_local_entry(PNGA_BIN_INDEX, ltme, 0);    
}


void wnga_bin_sorter(Integer g_bin, Integer g_cnt, Integer g_off)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_bin_sorter(g_bin, g_cnt, g_off);
   ltme += I_Wtime();
   update_local_entry(PNGA_BIN_SORTER, ltme, 0);
}


void wnga_brdcst(Integer type, void *buf, Integer len, Integer originator)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_brdcst(type, buf, len, originator);
   ltme += I_Wtime();
   update_local_entry(PNGA_BRDCST, ltme, len);
}


void wnga_check_handle(Integer g_a, char *string)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_check_handle(g_a, string);
   ltme += I_Wtime();
   update_local_entry(PNGA_CHECK_HANDLE, ltme, 0);
}


Integer wnga_cluster_nnodes()
{
   Integer return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_cluster_nnodes();
   ltme += I_Wtime();
   update_local_entry(PNGA_CLUSTER_NNODES, ltme, 0);    
   return return_value;
}


Integer wnga_cluster_nodeid()
{
   Integer return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_cluster_nodeid();
   ltme += I_Wtime();
   update_local_entry(PNGA_CLUSTER_NODEID, ltme, 0);   
   return return_value;
}


Integer wnga_cluster_nprocs(Integer node)
{
   Integer return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_cluster_nprocs(node);
   ltme += I_Wtime();
   update_local_entry(PNGA_CLUSTER_NPROCS, ltme, 0);       
   return return_value;
}


Integer wnga_cluster_proc_nodeid(Integer proc)
{
   Integer return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_cluster_proc_nodeid(proc);
   ltme += I_Wtime();
   update_local_entry(PNGA_CLUSTER_PROC_NODEID, ltme, 0);
   return return_value;
}

Integer wnga_cluster_procid(Integer node, Integer loc_proc_id)
{
   Integer return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_cluster_procid(node, loc_proc_id);
   ltme += I_Wtime();
   update_local_entry(PNGA_CLUSTER_PROCID, ltme, 0);    
   return return_value;
}


logical wnga_comp_patch(Integer andim, Integer *alo, Integer *ahi, Integer bndim, Integer *blo, Integer *bhi)
{
   logical return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_comp_patch(andim, alo, ahi, bndim, blo, bhi);
   ltme += I_Wtime();
   update_local_entry(PNGA_COMP_PATCH, ltme, 0);    
   return return_value;
}

logical wnga_compare_distr(Integer g_a, Integer g_b)
{
   logical return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_compare_distr(g_a, g_b);
   ltme += I_Wtime();
   update_local_entry(PNGA_COMPARE_DISTR, ltme, 0);    
   return return_value;
}


void wnga_copy(Integer g_a, Integer g_b)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_copy(g_a, g_b);
   ltme += I_Wtime();
   update_local_entry(PNGA_COPY, ltme, 0);
}


void wnga_copy_patch(char *trans, Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_copy_patch(trans, g_a, alo, ahi, g_b, blo, bhi);
   ltme += I_Wtime();
   update_local_entry(PNGA_COPY_PATCH, ltme, 0);    
}


void wnga_copy_patch_dp(char *t_a, Integer g_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_copy_patch_dp(t_a, g_a, ailo, aihi, ajlo, ajhi, g_b, bilo, bihi, bjlo, bjhi);
   ltme += I_Wtime();
   update_local_entry(PNGA_COPY_PATCH_DP, ltme, 0);
}


logical wnga_create(Integer type, Integer ndim, Integer *dims, char *name, Integer *chunk, Integer *g_a)
{
   logical return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_create(type, ndim, dims, name, chunk, g_a);
   ltme += I_Wtime();
   uint64_t sz;
   GET_LOCAL_MSIZE(*g_a, sz);
   update_local_entry(PNGA_CREATE, ltme, sz);
   return return_value;
}


logical wnga_create_bin_range(Integer g_bin, Integer g_cnt, Integer g_off, Integer *g_range)
{
   logical return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_create_bin_range(g_bin, g_cnt, g_off, g_range);
   ltme += I_Wtime();
   update_local_entry(PNGA_CREATE_BIN_RANGE, ltme, 0);    
   return return_value;
}


logical wnga_create_config(Integer type, Integer ndim, Integer *dims, char *name, Integer *chunk, Integer p_handle, Integer *g_a)
{
   logical return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_create_config(type, ndim, dims, name, chunk, p_handle, g_a);
   ltme += I_Wtime();
   uint64_t sz ;
   GET_LOCAL_MSIZE(*g_a, sz);
   update_local_entry(PNGA_CREATE_CONFIG, ltme, sz);
    return return_value;
}


logical wnga_create_ghosts(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *chunk, Integer *g_a)
{
   logical return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_create_ghosts(type, ndim, dims, width, name, chunk, g_a);
   ltme += I_Wtime();
   uint64_t sz ;
   GET_LOCAL_MSIZE(*g_a, sz);
   update_local_entry(PNGA_CREATE_GHOSTS, ltme, sz);    
   return return_value;
}


logical wnga_create_ghosts_config(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *chunk, Integer p_handle, Integer *g_a)
{
   logical return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_create_ghosts_config(type, ndim, dims, width, name, chunk, p_handle, g_a);
   ltme += I_Wtime();
   uint64_t sz;
   GET_LOCAL_MSIZE(*g_a, sz);
   update_local_entry(PNGA_CREATE_GHOSTS_CONFIG, ltme, sz);    
   return return_value;
}


logical wnga_create_ghosts_irreg(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *map, Integer *block, Integer *g_a)
{
   logical return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_create_ghosts_irreg(type, ndim, dims, width, name, map, block, g_a);
   ltme += I_Wtime();
   uint64_t sz;
   GET_LOCAL_MSIZE(*g_a, sz);
   update_local_entry(PNGA_CREATE_GHOSTS_IRREG, ltme, sz);    
   return return_value;
}


logical wnga_create_ghosts_irreg_config(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *map, Integer *block, Integer p_handle, Integer *g_a)
{
   logical return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_create_ghosts_irreg_config(type, ndim, dims, width, name, map, block, p_handle, g_a);
   ltme += I_Wtime();
   uint64_t sz;
   GET_LOCAL_MSIZE(*g_a, sz);
   update_local_entry(PNGA_CREATE_GHOSTS_IRREG_CONFIG, ltme, sz);    
   return return_value;
}


Integer wnga_create_handle()
{
   Integer return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_create_handle();
   ltme += I_Wtime();
   update_local_entry(PNGA_CREATE_HANDLE, ltme, 0);
   return return_value;
}


logical wnga_create_irreg(Integer type, Integer ndim, Integer *dims, char *name, Integer *map, Integer *block, Integer *g_a)
{
   logical return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_create_irreg(type, ndim, dims, name, map, block, g_a);
   ltme += I_Wtime();
   uint64_t sz;
   GET_LOCAL_MSIZE(*g_a, sz);
   update_local_entry(PNGA_CREATE_IRREG, ltme, sz);
   return return_value;
}


logical wnga_create_irreg_config(Integer type, Integer ndim, Integer *dims, char *name, Integer *map, Integer *block, Integer p_handle, Integer *g_a)
{
   logical return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_create_irreg_config(type, ndim, dims, name, map, block, p_handle, g_a);
   ltme += I_Wtime();
   uint64_t sz;
   GET_LOCAL_MSIZE(*g_a, sz);
   update_local_entry(PNGA_CREATE_IRREG_CONFIG, ltme, sz);    
   return return_value;
}


logical wnga_create_mutexes(Integer num)
{
   logical return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_create_mutexes(num);
   ltme += I_Wtime();
   update_local_entry(PNGA_CREATE_CREATE_MUTEXES, ltme, 0);
   return return_value;
}


DoublePrecision wnga_ddot_patch_dp(Integer g_a, char *t_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, char *t_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi)
{
   DoublePrecision return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_ddot_patch_dp(g_a, t_a, ailo, aihi, ajlo, ajhi, g_b, t_b, bilo, bihi, bjlo, bjhi);
   ltme += I_Wtime();
   update_local_entry(PNGA_DDOT_PATCH_DP, ltme, 0);    
   return return_value;
}


int wnga_deregister_type(int type)
{
   int return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_deregister_type(type);
   ltme += I_Wtime();
   update_local_entry(PNGA_DEREGISTER_TYPE, ltme, 0);
   return return_value;
}


logical wnga_destroy(Integer g_a)
{
   logical return_value;
   unsigned long long ltme, sz;
   ltme=- I_Wtime();
   return_value = pnga_destroy(g_a);
   ltme += I_Wtime();
   GET_LOCAL_MSIZE(g_a, sz);
   update_local_entry(PNGA_DESTROY, ltme, sz);    
   return return_value;
}


logical wnga_destroy_mutexes()
{
   logical return_value;
   unsigned long long ltme;
   ltme=- I_Wtime();
   return_value = pnga_destroy_mutexes();
   ltme += I_Wtime();
   update_local_entry(PNGA_DESTROY_MUTEXES, ltme, 0);    
   return return_value;
}


void wnga_diag(Integer g_a, Integer g_s, Integer g_v, DoublePrecision *eval)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_diag(g_a, g_s, g_v, eval);
   ltme += I_Wtime();
   update_local_entry(PNGA_DIAG, ltme, 0);    
}


void wnga_diag_reuse(Integer reuse, Integer g_a, Integer g_s, Integer g_v, DoublePrecision *eval)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_diag_reuse(reuse, g_a, g_s, g_v, eval);
   ltme += I_Wtime();
   update_local_entry(PNGA_DIAG_REUSE, ltme, 0);    
}


void wnga_diag_seq(Integer g_a, Integer g_s, Integer g_v, DoublePrecision *eval)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_diag_seq(g_a, g_s, g_v, eval);
   ltme += I_Wtime();
   update_local_entry(PNGA_DIAG_SEQ, ltme, 0);
}


void wnga_diag_std(Integer g_a, Integer g_v, DoublePrecision *eval)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_diag_std(g_a, g_v, eval);
   ltme += I_Wtime();
   update_local_entry(PNGA_DIAG_STD, ltme, 0);
}


void wnga_diag_std_seq(Integer g_a, Integer g_v, DoublePrecision *eval)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_diag_std_seq(g_a, g_v, eval);
   ltme += I_Wtime();
   update_local_entry(PNGA_DIAG_STD_SEQ, ltme, 0);
}


void wnga_distribution(Integer g_a, Integer proc, Integer *lo, Integer *hi)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_distribution(g_a, proc, lo, hi);
   ltme += I_Wtime();
   update_local_entry(PNGA_DISTRIBUTION, ltme, 0);
}


void wnga_dot(int type, Integer g_a, Integer g_b, void *value)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_dot(type, g_a, g_b, value);
   ltme += I_Wtime();
   update_local_entry(PNGA_DOT, ltme, 0);    
}


void wnga_dot_patch(Integer g_a, char *t_a, Integer *alo, Integer *ahi, Integer g_b, char *t_b, Integer *blo, Integer *bhi, void *retval)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_dot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi, retval);
   ltme += I_Wtime();
   update_local_entry(PNGA_DOT_PATCH, ltme, 0);
}


logical wnga_duplicate(Integer g_a, Integer *g_b, char *array_name)
{
   logical return_value;
   unsigned long long ltme, sz;
   ltme=- I_Wtime();
   return_value = pnga_duplicate(g_a, g_b, array_name);
   ltme += I_Wtime();
   GET_LOCAL_MSIZE(*g_b, sz)
   update_local_entry(PNGA_DUPLICATE, ltme, sz);    
   return return_value;
}


void wnga_elem_divide(Integer g_a, Integer g_b, Integer g_c)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_elem_divide(g_a, g_b, g_c);
   ltme += I_Wtime();
   update_local_entry(PNGA_ELEM_DIVIDE, ltme, 0);
}


void wnga_elem_divide_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_elem_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
   ltme += I_Wtime();
   update_local_entry(PNGA_ELEM_DIVIDE_PATCH, ltme, 0);    
}


void wnga_elem_maximum(Integer g_a, Integer g_b, Integer g_c)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_elem_maximum(g_a, g_b, g_c);
   ltme += I_Wtime();
   update_local_entry(PNGA_ELEM_MAXIMUM, ltme, 0);
}


void wnga_elem_maximum_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_elem_maximum_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
   ltme += I_Wtime();
   update_local_entry(PNGA_ELEM_MAXIMUM_PATCH, ltme, 0);
}


void wnga_elem_minimum(Integer g_a, Integer g_b, Integer g_c)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_elem_minimum(g_a, g_b, g_c);
   ltme += I_Wtime();
   update_local_entry(PNGA_ELEM_MINIMUM, ltme, 0);    
}


void wnga_elem_minimum_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_elem_minimum_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
   ltme += I_Wtime();
   update_local_entry(PNGA_ELEM_MINIMUM_PATCH, ltme, 0);    
}


void wnga_elem_multiply(Integer g_a, Integer g_b, Integer g_c)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_elem_multiply(g_a, g_b, g_c);
   ltme += I_Wtime();
   update_local_entry(PNGA_ELEM_MULTIPLY, ltme, 0);    
}


void wnga_elem_multiply_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_elem_multiply_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
   ltme += I_Wtime();
   update_local_entry(PNGA_ELEM_MULTIPLY_PATCH, ltme, 0);    
}


void wnga_elem_step_divide_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_elem_step_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
   ltme += I_Wtime();
   update_local_entry(PNGA_ELEM_STEP_DIVIDE_PATCH, ltme, 0);    
}


void wnga_elem_stepb_divide_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_elem_stepb_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
   ltme += I_Wtime();
   update_local_entry(PNGA_ELEM_STEPB_DIVIDE_PATCH, ltme, 0);
}


void wnga_error(char *string, Integer icode)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_error(string, icode);
   ltme += I_Wtime();
   update_local_entry(PNGA_ERROR, ltme, 0); 
}


void wnga_fence()
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_fence();
   ltme += I_Wtime();
   update_local_entry(PNGA_FENCE, ltme, 0);
}


void wnga_fill(Integer g_a, void *val)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_fill(g_a, val);
   ltme += I_Wtime();
   update_local_entry(PNGA_FILL, ltme, 0);
}


void wnga_fill_patch(Integer g_a, Integer *lo, Integer *hi, void *val)
{
   unsigned long long ltme;
   ltme=- I_Wtime();
   pnga_fill_patch(g_a, lo, hi, val);
   ltme += I_Wtime();
   update_local_entry(PNGA_FILL_PATCH, ltme, 0);
}


void wnga_gather(Integer g_a, void *v, Integer subscript[], Integer c_flag, Integer nv)
{
   unsigned long long ltme, sz, tp;
   ltme=- I_Wtime();
   pnga_gather(g_a, v, subscript, c_flag, nv);
   ltme += I_Wtime();
   OBTAIN_SIZE(tp, c_flag);
   sz = tp * nv;
   update_local_entry(PNGA_GATHER, ltme, sz);    
}


void wnga_gather2d(Integer g_a, void *v, Integer *i, Integer *j, Integer nv)
{
   unsigned long long ltme, sz, tp;
   ltme=- I_Wtime();
   pnga_gather2d(g_a, v, i, j, nv);
   ltme += I_Wtime();
   OBTAIN_ESIZE(tp, g_a);
   sz = tp * nv;
   update_local_entry(PNGA_GATHER2D, ltme, sz);
}


void wnga_get(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
   unsigned long long ltme, sz; /* TODO: Calculate the bytes */
   GET_SIZE(g_a, lo, hi, sz);
   ltme=- I_Wtime();
   pnga_get(g_a, lo, hi, buf, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_GET, ltme, sz);
}


void wnga_get_block_info(Integer g_a, Integer *num_blocks, Integer *block_dims)
{
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   pnga_get_block_info(g_a, num_blocks, block_dims);
   ltme += I_Wtime();
   update_local_entry(PNGA_GET_BLOCK_INFO, ltme, 0);
}


logical wnga_get_debug()
{
   logical return_value;
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   return_value = pnga_get_debug();
   ltme += I_Wtime();
   update_local_entry(PNGA_GET_DEBUG, ltme, 0);    
   return return_value;
}


void wnga_get_diag(Integer g_a, Integer g_v)
{
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   pnga_get_diag(g_a, g_v);
   ltme += I_Wtime();
   update_local_entry(PNGA_GET_DIAG, ltme, 0); 
}


Integer wnga_get_dimension(Integer g_a)
{
   Integer return_value;
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   return_value = pnga_get_dimension(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_GET_DIMENSION, ltme, 0);
   return return_value;
}


void wnga_get_field(Integer g_a, Integer *lo, Integer *hi, Integer foff, Integer fsize, void *buf, Integer *ld)
{
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   pnga_get_field(g_a, lo, hi, foff, fsize, buf, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_GET_FIELD, ltme, 0);    
}


void wnga_get_ghost_block(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   pnga_get_ghost_block(g_a, lo, hi, buf, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_GET_GHOST_BLOCK, ltme, 0);    
}


Integer wnga_get_pgroup(Integer g_a)
{
   Integer return_value;
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   return_value = pnga_get_pgroup(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_GET_PGROUP, ltme, 0);
   return return_value;
}


Integer wnga_get_pgroup_size(Integer grp_id)
{
   Integer return_value;
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   return_value = pnga_get_pgroup_size(grp_id);
   ltme += I_Wtime();
   update_local_entry(PNGA_GET_PGROUP_SIZE, ltme, 0);
   return return_value;
}


void wnga_get_proc_grid(Integer g_a, Integer *dims)
{
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   pnga_get_proc_grid(g_a, dims);
   ltme += I_Wtime();
   update_local_entry(PNGA_GET_PROC_GRID, ltme, 0);
}


void wnga_get_proc_index(Integer g_a, Integer iproc, Integer *index)
{
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   pnga_get_proc_index(g_a, iproc, index);
   ltme += I_Wtime();
   update_local_entry(PNGA_GET_PROC_INDEX, ltme, 0);
}


void wnga_ghost_barrier()
{
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   pnga_ghost_barrier();
   ltme += I_Wtime();
   update_local_entry(PNGA_GHOST_BARRIER, ltme, 0);    
}


void wnga_gop(Integer type, void *x, Integer n, char *op)
{
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   pnga_gop(type, x, n, op);
   ltme += I_Wtime();
   update_local_entry(PNGA_GOP, ltme, 0);

}


logical wnga_has_ghosts(Integer g_a)
{
   logical return_value;
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   return_value = pnga_has_ghosts(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_HAS_GHOSTS, ltme, 0);
   return return_value;
}


void wnga_init_fence()
{
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   pnga_init_fence();
   ltme += I_Wtime();
   update_local_entry(PNGA_INIT_FENCE, ltme, 0);
}


void wnga_inquire(Integer g_a, Integer *type, Integer *ndim, Integer *dims)
{
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   pnga_inquire(g_a, type, ndim, dims);
   ltme += I_Wtime();
   update_local_entry(PNGA_INQUIRE, ltme, 0);
}


Integer wnga_inquire_memory()
{
   Integer return_value;
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   return_value = pnga_inquire_memory();
   ltme += I_Wtime();
   update_local_entry(PNGA_INQUIRE_MEMORY, ltme, 0);
   return return_value;
}


void wnga_inquire_name(Integer g_a, char **array_name)
{
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   pnga_inquire_name(g_a, array_name);
   ltme += I_Wtime();
   update_local_entry(PNGA_INQUIRE_NAME, ltme, 0);
}


void wnga_inquire_type(Integer g_a, Integer *type)
{
   unsigned long long ltme; /* TODO: Calculate the bytes */
   ltme=- I_Wtime();
   pnga_inquire_type(g_a, type);
   ltme += I_Wtime();
   update_local_entry(PNGA_INQUIRE_TYPE, ltme, 0);    
}


logical wnga_is_mirrored(Integer g_a)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_is_mirrored(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_IS_MIRRORED, ltme, 0);    
   return return_value;
}


void wnga_list_nodeid(Integer *list, Integer nprocs)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_list_nodeid(list, nprocs);
   ltme += I_Wtime();
   update_local_entry(PNGA_LIST_NODEID, ltme, 0);
}


Integer wnga_llt_solve(Integer g_a, Integer g_b)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_llt_solve(g_a, g_b);
   ltme += I_Wtime();
   update_local_entry(PNGA_LLT_SOLVE, ltme, 0);
   return return_value;
}


logical wnga_locate(Integer g_a, Integer *subscript, Integer *owner)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_locate(g_a, subscript, owner);
   ltme += I_Wtime();
   update_local_entry(PNGA_LOCATE, ltme, 0);
   return return_value;
}


logical wnga_locate_nnodes(Integer g_a, Integer *lo, Integer *hi, Integer *np)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_locate_nnodes(g_a, lo, hi, np);
   ltme += I_Wtime();
   update_local_entry(PNGA_LOCATE_NNODES, ltme, 0);    
   return return_value;
}


Integer wnga_locate_num_blocks(Integer g_a, Integer *lo, Integer *hi)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_locate_num_blocks(g_a, lo, hi);
   ltme += I_Wtime();
   update_local_entry(PNGA_LOCATE_NUM_BLOCKS, ltme, 0);    
   return return_value;
}


logical wnga_locate_region(Integer g_a, Integer *lo, Integer *hi, Integer *map, Integer *proclist, Integer *np)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_locate_region(g_a, lo, hi, map, proclist, np);
   ltme += I_Wtime();
   update_local_entry(PNGA_LOCATE_REGION, ltme, 0);    
   return return_value;
}


void wnga_lock(Integer mutex)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_lock(mutex);
   ltme += I_Wtime();
   update_local_entry(PNGA_LOCK, ltme, 0);    
}


void wnga_lu_solve(char *tran, Integer g_a, Integer g_b)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_lu_solve(tran, g_a, g_b);
   ltme += I_Wtime();
   update_local_entry(PNGA_LU_SOLVE, ltme, 0); 
}


void wnga_lu_solve_alt(Integer tran, Integer g_a, Integer g_b)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_lu_solve_alt(tran, g_a, g_b);
   ltme += I_Wtime();
   update_local_entry(PNGA_LU_SOLVE_ALT, ltme, 0);    
}


void wnga_lu_solve_seq(char *trans, Integer g_a, Integer g_b)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_lu_solve_seq(trans, g_a, g_b);
   ltme += I_Wtime();
   update_local_entry(PNGA_LU_SOLVE_SEQ, ltme, 0);     
}


void wnga_mask_sync(Integer begin, Integer end)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_mask_sync(begin, end);
   ltme += I_Wtime();
   update_local_entry(PNGA_MASK_SYNC, ltme, 0);
}


void wnga_matmul(char *transa, char *transb, void *alpha, void *beta, Integer g_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi, Integer g_c, Integer cilo, Integer cihi, Integer cjlo, Integer cjhi)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_matmul(transa, transb, alpha, beta, g_a, ailo, aihi, ajlo, ajhi, g_b, bilo, bihi, bjlo, bjhi, g_c, cilo, cihi, cjlo, cjhi);
   ltme += I_Wtime();
   update_local_entry(PNGA_MATMUL, ltme, 0);
}


void wnga_matmul_mirrored(char *transa, char *transb, void *alpha, void *beta, Integer g_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi, Integer g_c, Integer cilo, Integer cihi, Integer cjlo, Integer cjhi)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_matmul_mirrored(transa, transb, alpha, beta, g_a, ailo, aihi, ajlo, ajhi, g_b, bilo, bihi, bjlo, bjhi, g_c, cilo, cihi, cjlo, cjhi);
   ltme += I_Wtime();
   update_local_entry(PNGA_MATMUL_MIRRORED, ltme, 0);
}


void wnga_matmul_patch(char *transa, char *transb, void *alpha, void *beta, Integer g_a, Integer alo[], Integer ahi[], Integer g_b, Integer blo[], Integer bhi[], Integer g_c, Integer clo[], Integer chi[])
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_matmul_patch(transa, transb, alpha, beta, g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
   ltme += I_Wtime();
   update_local_entry(PNGA_MATMUL_PATCH, ltme, 0);    
}


void wnga_median(Integer g_a, Integer g_b, Integer g_c, Integer g_m)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_median(g_a, g_b, g_c, g_m);
   ltme += I_Wtime();
   update_local_entry(PNGA_MEDIAN, ltme, 0);    
}


void wnga_median_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi, Integer g_m, Integer *mlo, Integer *mhi)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_median_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi, g_m, mlo, mhi);
   ltme += I_Wtime();
   update_local_entry(PNGA_MEDIAN_PATCH, ltme, 0);    
}


Integer wnga_memory_avail()
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_memory_avail();
   ltme += I_Wtime();
   update_local_entry(PNGA_MEMORY_AVAIL, ltme, 0);
   return return_value;
}


Integer wnga_memory_avail_type(Integer datatype)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_memory_avail_type(datatype);
   ltme += I_Wtime();
   update_local_entry(PNGA_MEMORY_AVAIL_TYPE, ltme, 0);    
   return return_value;
}


logical wnga_memory_limited()
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_memory_limited();
   ltme += I_Wtime();
   update_local_entry(PNGA_MEMORY_LIMITED, ltme, 0);    
   return return_value;
}


void wnga_merge_distr_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_merge_distr_patch(g_a, alo, ahi, g_b, blo, bhi);
   ltme += I_Wtime();
   update_local_entry(PNGA_MERGE_DISTR_PATCH, ltme, 0);
}


void wnga_merge_mirrored(Integer g_a)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_merge_mirrored(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_MERGE_MIRRORED, ltme, 0);
}


void wnga_msg_brdcst(Integer type, void *buffer, Integer len, Integer root)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_msg_brdcst(type, buffer, len, root);
   ltme += I_Wtime();
   update_local_entry(PNGA_MSG_BRDCST, ltme, len);    
}


void wnga_msg_pgroup_sync(Integer grp_id)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_msg_pgroup_sync(grp_id);
   ltme += I_Wtime();
   update_local_entry(PNGA_MSG_PGROUP_SYNC, ltme, 0);    
}


void wnga_msg_sync()
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_msg_sync();
   ltme += I_Wtime();
   update_local_entry(PNGA_MSG_SYNC, ltme, 0);
}


void wnga_nbacc(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha, Integer *nbhndl)
{
   unsigned long long ltme, sz; 
   GET_SIZE(g_a, lo, hi, sz);
   ltme=- I_Wtime();
   pnga_nbacc(g_a, lo, hi, buf, ld, alpha, nbhndl);
   ltme += I_Wtime();
   update_local_entry(PNGA_NBACC, ltme, sz);
}


void wnga_nbget(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle)
{
   unsigned long long ltme, sz; /* TODO: Calculate the size */
   GET_SIZE(g_a, lo, hi, sz);
   ltme=- I_Wtime();
   pnga_nbget(g_a, lo, hi, buf, ld, nbhandle);
   ltme += I_Wtime();
   update_local_entry(PNGA_NBGET, ltme, sz);
}


void wnga_nbget_field(Integer g_a, Integer *lo, Integer *hi, Integer foff, Integer fsize, void *buf, Integer *ld, Integer *nbhandle)
{
   unsigned long long ltme, sz; /* TODO: Calculate the size */
   GET_SIZE(g_a, lo, hi, sz);
   ltme=- I_Wtime();
   pnga_nbget_field(g_a, lo, hi, foff, fsize, buf, ld, nbhandle);
   ltme += I_Wtime();
   update_local_entry(PNGA_NBGET_FIELD, ltme, sz);    
}


void wnga_nbget_ghost_dir(Integer g_a, Integer *mask, Integer *nbhandle)
{
   unsigned long long ltme; /* TODO: Calculate the size */
   ltme=- I_Wtime();
   pnga_nbget_ghost_dir(g_a, mask, nbhandle);
   ltme += I_Wtime();
   update_local_entry(PNGA_NBGET_GHOST_DIR, ltme, 0);    
}


void wnga_nblock(Integer g_a, Integer *nblock)
{
   unsigned long long ltme; /* TODO: Calculate the size */
   ltme=- I_Wtime();
   pnga_nblock(g_a, nblock);
   ltme += I_Wtime();
   update_local_entry(PNGA_NBLOCK, ltme, 0);    
}


void wnga_nbput(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle)
{
   unsigned long long ltme, sz; /* TODO: Calculate the size */
   GET_SIZE(g_a, lo, hi, sz);
   ltme=- I_Wtime();
   pnga_nbput(g_a, lo, hi, buf, ld, nbhandle);
   ltme += I_Wtime();
   update_local_entry(PNGA_NBPUT, ltme, sz);        
}


void wnga_nbput_field(Integer g_a, Integer *lo, Integer *hi, Integer foff, Integer fsize, void *buf, Integer *ld, Integer *nbhandle)
{
   unsigned long long ltme, sz; /* TODO: Calculate the size */
   GET_SIZE(g_a, lo, hi, sz);
   ltme=- I_Wtime();
   pnga_nbput_field(g_a, lo, hi, foff, fsize, buf, ld, nbhandle);
   ltme += I_Wtime();
   update_local_entry(PNGA_NBPUT_FIELD, ltme, sz);
}


Integer wnga_nbtest(Integer *nbhandle)
{
   Integer return_value;
   unsigned long long ltme; /* TODO: Calculate the size */
   ltme=- I_Wtime();
   return_value = pnga_nbtest(nbhandle);
   ltme += I_Wtime();
   update_local_entry(PNGA_NBTEST, ltme, 0);    
   return return_value;
}


void wnga_nbwait(Integer *nbhandle)
{
   unsigned long long ltme; /* TODO: Calculate the size */
   ltme=- I_Wtime();
   pnga_nbwait(nbhandle);
   ltme += I_Wtime();
   update_local_entry(PNGA_NBWAIT, ltme, 0);    
}


Integer wnga_ndim(Integer g_a)
{
   Integer return_value;
   unsigned long long ltme; /* TODO: Calculate the size */
   ltme=- I_Wtime();
   return_value = pnga_ndim(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_NDIM, ltme, 0);    
   return return_value;
}


Integer wnga_nnodes()
{
   Integer return_value;
   unsigned long long ltme; /* TODO: Calculate the size */
   ltme=- I_Wtime();
   return_value = pnga_nnodes();
   ltme += I_Wtime();
   update_local_entry(PNGA_NNODES, ltme, 0);    
   return return_value;
}


Integer wnga_nodeid()
{
   Integer return_value;
   unsigned long long ltme; /* TODO: Calculate the size */
   ltme=- I_Wtime();
   return_value = pnga_nodeid();
   ltme += I_Wtime();
   update_local_entry(PNGA_NODEID, ltme, 0);    
   return return_value;
}


void wnga_norm1(Integer g_a, double *nm)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_norm1(g_a, nm);
   ltme += I_Wtime();
   update_local_entry(PNGA_NORM1, ltme, 0);    
}


void wnga_norm_infinity(Integer g_a, double *nm)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_norm_infinity(g_a, nm);
   ltme += I_Wtime();
   update_local_entry(PNGA_NORM_INFINITY, ltme, 0);    
}

logical wnga_overlay(Integer g_a, Integer g_parent)
{
   logical result;
   unsigned long long ltme, sz;
   ltme=- I_Wtime();
   result = pnga_overlay(g_a, g_parent);
   ltme += I_Wtime();
   GET_LOCAL_MSIZE(g_a, sz);
   update_local_entry(PNGA_OVERLAY, ltme, sz);
   return result;

}

void wnga_pack(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi, Integer *icount)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_pack(g_a, g_b, g_sbit, lo, hi, icount);
   ltme += I_Wtime();
   update_local_entry(PNGA_PACK, ltme, 0);    
}


void wnga_patch_enum(Integer g_a, Integer lo, Integer hi, void *start, void *stride)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_patch_enum(g_a, lo, hi, start, stride);
   ltme += I_Wtime();
   update_local_entry(PNGA_PATCH_ENUM, ltme, 0);    
}


logical wnga_patch_intersect(Integer *lo, Integer *hi, Integer *lop, Integer *hip, Integer ndim)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_patch_intersect(lo, hi, lop, hip, ndim);
   ltme += I_Wtime();
   update_local_entry(PNGA_PATCH_INTERSECT, ltme, 0);    
   return return_value;
}


void wnga_periodic(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha, Integer op_code)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_periodic(g_a, lo, hi, buf, ld, alpha, op_code);
   ltme += I_Wtime();
   update_local_entry(PNGA_PERIODIC, ltme, 0);    
}


Integer wnga_pgroup_absolute_id(Integer grp, Integer pid)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_pgroup_absolute_id(grp, pid);
   ltme += I_Wtime();
   update_local_entry(PNGA_PGROUP_ABSOLUTE_ID, ltme, 0);    
   return return_value;
}


void wnga_pgroup_brdcst(Integer grp_id, Integer type, void *buf, Integer len, Integer originator)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_pgroup_brdcst(grp_id, type, buf, len, originator);
   ltme += I_Wtime();
   update_local_entry(PNGA_PGROUP_BRDCST, ltme, 0);    
}


Integer wnga_pgroup_create(Integer *list, Integer count)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_pgroup_create(list, count);
   ltme += I_Wtime();
   update_local_entry(PNGA_PGROUP_CREATE, ltme, 0);    
   return return_value;
}


logical wnga_pgroup_destroy(Integer grp)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_pgroup_destroy(grp);
   ltme += I_Wtime();
   update_local_entry(PNGA_PGROUP_DESTROY, ltme, 0);    
   return return_value;
}


Integer wnga_pgroup_get_default()
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_pgroup_get_default();
   ltme += I_Wtime();
   update_local_entry(PNGA_PGROUP_GET_DEFAULT, ltme, 0);    
   return return_value;
}


Integer wnga_pgroup_get_mirror()
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_pgroup_get_mirror();
   ltme += I_Wtime();
   update_local_entry(PNGA_PGROUP_GET_MIRROR, ltme, 0);    
   return return_value;
}


Integer wnga_pgroup_get_world()
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_pgroup_get_world();
   ltme += I_Wtime();
   update_local_entry(PNGA_PGROUP_GET_WORLD, ltme, 0);    
   return return_value;
}


void wnga_pgroup_gop(Integer p_grp, Integer type, void *x, Integer n, char *op)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_pgroup_gop(p_grp, type, x, n, op);
   ltme += I_Wtime();
   update_local_entry(PNGA_PGROUP_GOP, ltme, 0);    
}


Integer wnga_pgroup_nnodes(Integer grp)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_pgroup_nnodes(grp);
   ltme += I_Wtime();
   update_local_entry(PNGA_PGROUP_NNODES, ltme, 0);    
   return return_value;
}


Integer wnga_pgroup_nodeid(Integer grp)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_pgroup_nodeid(grp);
   ltme += I_Wtime();
   update_local_entry(PNGA_PGROUP_NODEID, ltme, 0);    
   return return_value;
}


void wnga_pgroup_set_default(Integer grp)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_pgroup_set_default(grp);
   ltme += I_Wtime();
   update_local_entry(PNGA_PGROUP_SET_DEFAULT, ltme, 0);    
}


Integer wnga_pgroup_split(Integer grp, Integer grp_num)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_pgroup_split(grp, grp_num);
   ltme += I_Wtime();
   update_local_entry(PNGA_PGROUP_SPLIT, ltme, 0);
   return return_value;
}


Integer wnga_pgroup_split_irreg(Integer grp, Integer mycolor)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_pgroup_split_irreg(grp, mycolor);
   ltme += I_Wtime();
   update_local_entry(PNGA_PGROUP_SPLIT_IRREG, ltme, 0);    
   return return_value;
}


void wnga_pgroup_sync(Integer grp_id)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_pgroup_sync(grp_id);
   ltme += I_Wtime();
   update_local_entry(PNGA_PGROUP_SYNC, ltme, 0);    
}


void wnga_print(Integer g_a)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_print(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_PRINT, ltme, 0);    
}


void wnga_print_distribution(int fstyle, Integer g_a)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_print_distribution(fstyle, g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_PRINT_DISTRIBUTION, ltme, 0);    
}


void wnga_print_file(FILE *file, Integer g_a)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_print_file(file, g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_PRINT_FILE, ltme, 0);    
}


void wnga_print_patch(Integer g_a, Integer *lo, Integer *hi, Integer pretty)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_print_patch(g_a, lo, hi, pretty);
   ltme += I_Wtime();
   update_local_entry(PNGA_PRINT_PATCH, ltme, 0);    
}


void wnga_print_patch2d(Integer g_a, Integer ilo, Integer ihi, Integer jlo, Integer jhi, Integer pretty)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_print_patch2d(g_a, ilo, ihi, jlo, jhi, pretty);
   ltme += I_Wtime();
   update_local_entry(PNGA_PRINT_PATCH2D, ltme, 0);     
}


void wnga_print_patch_file(FILE *file, Integer g_a, Integer *lo, Integer *hi, Integer pretty)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_print_patch_file(file, g_a, lo, hi, pretty);
   ltme += I_Wtime();
   update_local_entry(PNGA_PRINT_PATCH_FILE, ltme, 0);    
}


void wnga_print_patch_file2d(FILE *file, Integer g_a, Integer ilo, Integer ihi, Integer jlo, Integer jhi, Integer pretty)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_print_patch_file2d(file, g_a, ilo, ihi, jlo, jhi, pretty);
   ltme += I_Wtime();
   update_local_entry(PNGA_PRINT_PATCH_FILE2D, ltme, 0);    
}


void wnga_print_stats()
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_print_stats();
   ltme += I_Wtime();
   update_local_entry(PNGA_PRINT_STATS, ltme, 0);    
}


void wnga_proc_topology(Integer g_a, Integer proc, Integer *subscript)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_proc_topology(g_a, proc, subscript);
   ltme += I_Wtime();
   update_local_entry(PNGA_PROC_TOPOLOGY, ltme, 0);    
}


void wnga_put(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
   unsigned long long ltme, sz;
   GET_SIZE(g_a, lo, hi, sz); 
   ltme=- I_Wtime();
   pnga_put(g_a, lo, hi, buf, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_PUT, ltme, sz);     
}


void wnga_put_field(Integer g_a, Integer *lo, Integer *hi, Integer foff, Integer fsize, void *buf, Integer *ld)
{
   unsigned long long ltme, sz;
   GET_SIZE(g_a, lo, hi, sz); 
   ltme=- I_Wtime();
   pnga_put_field(g_a, lo, hi, foff, fsize, buf, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_PUT_FIELD, ltme, sz);

}


void wnga_randomize(Integer g_a, void *val)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_randomize(g_a, val);
   ltme += I_Wtime();
   update_local_entry(PNGA_RANDOMIZE, ltme, 0);    
}


Integer wnga_read_inc(Integer g_a, Integer *subscript, Integer inc)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_read_inc(g_a, subscript, inc);
   ltme += I_Wtime();
   update_local_entry(PNGA_READ_INC, ltme, 0);    
   return return_value;
}


void wnga_recip(Integer g_a)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_recip(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_RECIP, ltme, 0);    
}


void wnga_recip_patch(Integer g_a, Integer *lo, Integer *hi)
{
   unsigned long long ltme, sz;
   GET_SIZE(g_a, lo, hi, sz); 
   ltme=- I_Wtime();
   pnga_recip_patch(g_a, lo, hi);
   ltme += I_Wtime();
   update_local_entry(PNGA_RECIP_PATCH, ltme, sz);    
}


int wnga_register_type(size_t size)
{
   int return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_register_type(size);
   ltme += I_Wtime();
   update_local_entry(PNGA_REGISTER_TYPE, ltme, 0);
   return return_value;
}


void wnga_release(Integer g_a, Integer *lo, Integer *hi)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_release(g_a, lo, hi);
   ltme += I_Wtime();
   update_local_entry(PNGA_RELEASE, ltme, 0);    
}


void wnga_release_block(Integer g_a, Integer iblock)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_release_block(g_a, iblock);
   ltme += I_Wtime();
   update_local_entry(PNGA_RELEASE_BLOCK, ltme, 0);
}


void wnga_release_block_grid(Integer g_a, Integer *index)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_release_block_grid(g_a, index);
   ltme += I_Wtime();
   update_local_entry(PNGA_RELEASE_BLOCK_GRID, ltme, 0);    
}


void wnga_release_block_segment(Integer g_a, Integer iproc)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_release_block_segment(g_a, iproc);
   ltme += I_Wtime();
   update_local_entry(PNGA_RELEASE_BLOCK_SEGMENT, ltme, 0);    
}


void wnga_release_ghost_element(Integer g_a, Integer subscript[])
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_release_ghost_element(g_a, subscript);
   ltme += I_Wtime();
   update_local_entry(PNGA_RELEASE_GHOST_ELEMENT, ltme, 0);    
}


void wnga_release_ghosts(Integer g_a)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_release_ghosts(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_RELEASE_GHOSTS, ltme, 0);    
}


void wnga_release_update(Integer g_a, Integer *lo, Integer *hi)
{
   unsigned long long ltme, sz; 
   GET_SIZE(g_a, lo, hi, sz);
   ltme=- I_Wtime();
   pnga_release_update(g_a, lo, hi);
   ltme += I_Wtime();
   update_local_entry(PNGA_RELEASE_UDPATE, ltme, sz);    
}


void wnga_release_update_block(Integer g_a, Integer iblock)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_release_update_block(g_a, iblock);
   ltme += I_Wtime();
   update_local_entry(PNGA_RELEASE_UDPATE_BLOCK, ltme, 0);    
}


void wnga_release_update_block_grid(Integer g_a, Integer *index)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_release_update_block_grid(g_a, index);
   ltme += I_Wtime();
   update_local_entry(PNGA_RELEASE_UDPATE_BLOCK_GRID, ltme, 0);    
}


void wnga_release_update_block_segment(Integer g_a, Integer iproc)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_release_update_block_segment(g_a, iproc);
   ltme += I_Wtime();
   update_local_entry(PNGA_RELEASE_UDPATE_BLOCK_SEGMENT, ltme, 0);    
}


void wnga_release_update_ghost_element(Integer g_a, Integer subscript[])
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_release_update_ghost_element(g_a, subscript);
   ltme += I_Wtime();
   update_local_entry(PNGA_RELEASE_UDPATE_GHOST_ELEMENT, ltme, 0);
}


void wnga_release_update_ghosts(Integer g_a)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_release_update_ghosts(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_RELEASE_UDPATE_GHOSTS, ltme, 0);    
}


void wnga_scale(Integer g_a, void *alpha)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_scale(g_a, alpha);
   ltme += I_Wtime();
   update_local_entry(PNGA_SCALE, ltme, 0);    
}


void wnga_scale_cols(Integer g_a, Integer g_v)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_scale_cols(g_a, g_v);
   ltme += I_Wtime();
   update_local_entry(PNGA_SCALE_COLS, ltme, 0);    
}


void wnga_scale_patch(Integer g_a, Integer *lo, Integer *hi, void *alpha)
{
   unsigned long long ltme, sz; 
   GET_SIZE(g_a, lo, hi, sz);
   ltme=- I_Wtime();
   pnga_scale_patch(g_a, lo, hi, alpha);
   ltme += I_Wtime();
   update_local_entry(PNGA_SCALE_PATCH, ltme, sz);    
}


void wnga_scale_rows(Integer g_a, Integer g_v)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_scale_rows(g_a, g_v);
   ltme += I_Wtime();
   update_local_entry(PNGA_SCALE_ROWS, ltme, 0);    
}


void wnga_scan_add(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi, Integer excl)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_scan_add(g_a, g_b, g_sbit, lo, hi, excl);
   ltme += I_Wtime();
   update_local_entry(PNGA_SCAN_ADD, ltme, 0);    
}


void wnga_scan_copy(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_scan_copy(g_a, g_b, g_sbit, lo, hi);
   ltme += I_Wtime();
   update_local_entry(PNGA_SCAN_COPY, ltme, 0);    
}


void wnga_scatter(Integer g_a, void *v, Integer *subscript, Integer c_flag, Integer nv)
{
   unsigned long long ltme, sz;
   OBTAIN_SIZE(sz, c_flag);
   sz *= nv; 
   ltme=- I_Wtime();
   pnga_scatter(g_a, v, subscript, c_flag, nv);
   ltme += I_Wtime();
   update_local_entry(PNGA_SCATTER, ltme, sz);    
}


void wnga_scatter2d(Integer g_a, void *v, Integer *i, Integer *j, Integer nv)
{
   unsigned long long ltme, sz;
   OBTAIN_ESIZE(sz, g_a); 
   ltme=- I_Wtime();
   pnga_scatter2d(g_a, v, i, j, nv);
   ltme += I_Wtime();
   sz *= nv;
   update_local_entry(PNGA_SCATTER2D, ltme, sz);    
}


void wnga_scatter_acc(Integer g_a, void *v, Integer subscript[], Integer c_flag, Integer nv, void *alpha)
{
   unsigned long long ltme, sz; 
   OBTAIN_SIZE(sz, c_flag);
   sz *= nv;
   ltme=- I_Wtime();
   pnga_scatter_acc(g_a, v, subscript, c_flag, nv, alpha);
   ltme += I_Wtime();
   update_local_entry(PNGA_SCATTER_ACC, ltme, sz);    
}


void wnga_scatter_acc2d(Integer g_a, void *v, Integer *i, Integer *j, Integer nv, void *alpha)
{
   unsigned long long ltme, sz;
   OBTAIN_ESIZE(sz, g_a); 
   ltme=- I_Wtime();
   pnga_scatter_acc2d(g_a, v, i, j, nv, alpha);
   ltme += I_Wtime();
   sz *= nv;
   update_local_entry(PNGA_SCATTER_ACC2D, ltme, sz);    
}


void wnga_select_elem(Integer g_a, char *op, void *val, Integer *subscript)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_select_elem(g_a, op, val, subscript);
   ltme += I_Wtime();
   update_local_entry(PNGA_SELECT_ELEM, ltme, 0);    
}


void wnga_set_array_name(Integer g_a, char *array_name)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_set_array_name(g_a, array_name);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_ARRAY_NAME, ltme, 0);    
}


void wnga_set_block_cyclic(Integer g_a, Integer *dims)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_set_block_cyclic(g_a, dims);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_BLOCK_CYCLIC, ltme, 0);    
}


void wnga_set_block_cyclic_proc_grid(Integer g_a, Integer *dims, Integer *proc_grid)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_set_block_cyclic_proc_grid(g_a, dims, proc_grid);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_BLOCK_CYCLIC_PROC_GRID, ltme, 0);    
}


void wnga_set_chunk(Integer g_a, Integer *chunk)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_set_chunk(g_a, chunk);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_CHUNK, ltme, 0);    
}


void wnga_set_data(Integer g_a, Integer ndim, Integer *dims, Integer type)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_set_data(g_a, ndim, dims, type);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_DATA, ltme, 0);    
}


void wnga_set_debug(logical flag)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_set_debug(flag);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_DEBUG, ltme, 0);    
}


void wnga_set_diagonal(Integer g_a, Integer g_v)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_set_diagonal(g_a, g_v);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_DIAGONAL, ltme, 0);    
}


void wnga_set_ghost_corner_flag(Integer g_a, logical flag)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_set_ghost_corner_flag(g_a, flag);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_GHOST_CORNER_FLAG, ltme, 0);    
}


logical wnga_set_ghost_info(Integer g_a)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_set_ghost_info(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_GHOST_INFO, ltme, 0);    
   return return_value;
}


void wnga_set_ghosts(Integer g_a, Integer *width)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_set_ghosts(g_a, width);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_GHOSTS, ltme, 0);    
}


void wnga_set_irreg_distr(Integer g_a, Integer *map, Integer *block)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_set_irreg_distr(g_a, map, block);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_IRREG_DISTR, ltme, 0);    
}


void wnga_set_irreg_flag(Integer g_a, logical flag)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_set_irreg_flag(g_a, flag);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_IRREG_FLAG, ltme, 0);    
}


void wnga_set_memory_limit(Integer mem_limit)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_set_memory_limit(mem_limit);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_MEMORY_LIMIT, ltme, 0);    
}


void wnga_set_pgroup(Integer g_a, Integer p_handle)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_set_pgroup(g_a, p_handle);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_PGROUP, ltme, 0);    
}


void wnga_set_restricted(Integer g_a, Integer *list, Integer size)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_set_restricted(g_a, list, size);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_RESTRICTED, ltme, 0);    
}


void wnga_set_restricted_range(Integer g_a, Integer lo_proc, Integer hi_proc)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_set_restricted_range(g_a, lo_proc, hi_proc);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_RESTRICTED_RANGE, ltme, 0);    
}


logical wnga_set_update4_info(Integer g_a)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_set_update4_info(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_UPDATE4_INFO, ltme, 0);    
   return return_value;
}


logical wnga_set_update5_info(Integer g_a)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_set_update5_info(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_SET_UPDATE5_INFO, ltme, 0);    
   return return_value;
}


void wnga_shift_diagonal(Integer g_a, void *c)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_shift_diagonal(g_a, c);
   ltme += I_Wtime();
   update_local_entry(PNGA_SHIFT_DIAGONAL, ltme, 0);    
}


Integer wnga_solve(Integer g_a, Integer g_b)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_solve(g_a, g_b);
   ltme += I_Wtime();
   update_local_entry(PNGA_SOLVE, ltme, 0);    
   return return_value;
}


Integer wnga_spd_invert(Integer g_a)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_spd_invert(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_SPD_INVERT, ltme, 0);    
   return return_value;
}


void wnga_step_bound_info(Integer g_xx, Integer g_vv, Integer g_xxll, Integer g_xxuu, void *boundmin, void *wolfemin, void *boundmax)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_step_bound_info(g_xx, g_vv, g_xxll, g_xxuu, boundmin, wolfemin, boundmax);
   ltme += I_Wtime();
   update_local_entry(PNGA_STEP_BOUND_INFO, ltme, 0);    
}


void wnga_step_bound_info_patch(Integer g_xx, Integer *xxlo, Integer *xxhi, Integer g_vv, Integer *vvlo, Integer *vvhi, Integer g_xxll, Integer *xxlllo, Integer *xxllhi, Integer g_xxuu, Integer *xxuulo, Integer *xxuuhi, void *boundmin, void *wolfemin, void *boundmax)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_step_bound_info_patch(g_xx, xxlo, xxhi, g_vv, vvlo, vvhi, g_xxll, xxlllo, xxllhi, g_xxuu, xxuulo, xxuuhi, boundmin, wolfemin, boundmax);
   ltme += I_Wtime();
   update_local_entry(PNGA_STEP_BOUND_INFO_PATCH, ltme, 0);    
}


void wnga_step_mask_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_step_mask_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
   ltme += I_Wtime();
   update_local_entry(PNGA_STEP_MASK_PATCH, ltme, 0);    
}


void wnga_step_max(Integer g_a, Integer g_b, void *retval)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_step_max(g_a, g_b, retval);
   ltme += I_Wtime();
   update_local_entry(PNGA_STEP_MAX, ltme, 0);    
}


void wnga_step_max_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, void *result)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_step_max_patch(g_a, alo, ahi, g_b, blo, bhi, result);
   ltme += I_Wtime();
   update_local_entry(PNGA_STEP_MAX_PATCH, ltme, 0);    
}


void wnga_strided_acc(Integer g_a, Integer *lo, Integer *hi, Integer *skip, void *buf, Integer *ld, void *alpha)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_strided_acc(g_a, lo, hi, skip, buf, ld, alpha);
   ltme += I_Wtime();
   update_local_entry(PNGA_STRIDED_ACC, ltme, 0);    
}


void wnga_strided_get(Integer g_a, Integer *lo, Integer *hi, Integer *skip, void *buf, Integer *ld)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_strided_get(g_a, lo, hi, skip, buf, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_STRIDED_GET, ltme, 0);
}


void wnga_strided_put(Integer g_a, Integer *lo, Integer *hi, Integer *skip, void *buf, Integer *ld)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_strided_put(g_a, lo, hi, skip, buf, ld);
   ltme += I_Wtime();
   update_local_entry(PNGA_STRIDED_PUT, ltme, 0);    
}


void wnga_summarize(Integer verbose)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_summarize(verbose);
   ltme += I_Wtime();
   update_local_entry(PNGA_SUMMARIZE, ltme, 0);    
}


void wnga_symmetrize(Integer g_a)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_symmetrize(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_SYMMETRIZE, ltme, 0);    
}


void wnga_sync()
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_sync();
   ltme += I_Wtime();
   update_local_entry(PNGA_SYNC, ltme, 0);    
}


double wnga_timer()
{
   double return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_timer();
   ltme += I_Wtime();
   update_local_entry(PNGA_TIMER, ltme, 0);    
   return return_value;
}


Integer wnga_total_blocks(Integer g_a)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_total_blocks(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_TOTAL_BLOCKS, ltme, 0);
   return return_value;
}


void wnga_transpose(Integer g_a, Integer g_b)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_transpose(g_a, g_b);
   ltme += I_Wtime();
   update_local_entry(PNGA_TRANSPOSE, ltme, 0);    
}


Integer wnga_type_c2f(Integer type)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_type_c2f(type);
   ltme += I_Wtime();
   update_local_entry(PNGA_TYPE_C2F, ltme, 0);    
   return return_value;
}


Integer wnga_type_f2c(Integer type)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_type_f2c(type);
   ltme += I_Wtime();
   update_local_entry(PNGA_TYPE_F2C, ltme, 0);    
   return return_value;
}


void wnga_unlock(Integer mutex)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_unlock(mutex);
   ltme += I_Wtime();
   update_local_entry(PNGA_UNLOCK, ltme, 0);    
}


void wnga_unpack(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi, Integer *icount)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_unpack(g_a, g_b, g_sbit, lo, hi, icount);
   ltme += I_Wtime();
   update_local_entry(PNGA_UNPACK, ltme, 0);    
}


void wnga_update1_ghosts(Integer g_a)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_update1_ghosts(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_UPDATE1_GHOSTS, ltme, 0);    
}


logical wnga_update2_ghosts(Integer g_a)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_update2_ghosts(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_UPDATE2_GHOSTS, ltme, 0);    
   return return_value;
}


logical wnga_update3_ghosts(Integer g_a)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_update3_ghosts(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_UPDATE3_GHOSTS, ltme, 0);    
   return return_value;
}


logical wnga_update44_ghosts(Integer g_a)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_update44_ghosts(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_UPDATED44_GHOSTS, ltme, 0);    
   return return_value;
}


logical wnga_update4_ghosts(Integer g_a)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_update4_ghosts(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_UPDATE4_GHOSTS, ltme, 0);    
   return return_value;
}


logical wnga_update55_ghosts(Integer g_a)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_update55_ghosts(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_UPDATE55_GHOSTS, ltme, 0);    
   return return_value;
}


logical wnga_update5_ghosts(Integer g_a)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_update5_ghosts(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_UPDATE55_GHOSTS, ltme, 0);    
   return return_value;
}


logical wnga_update6_ghosts(Integer g_a)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_update6_ghosts(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_UPDATE6_GHOSTS, ltme, 0);    
   return return_value;
}


logical wnga_update7_ghosts(Integer g_a)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_update7_ghosts(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_UPDATE7_GHOSTS, ltme, 0);    
   return return_value;
}


logical wnga_update_ghost_dir(Integer g_a, Integer pdim, Integer pdir, logical pflag)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_update_ghost_dir(g_a, pdim, pdir, pflag);
   ltme += I_Wtime();
   update_local_entry(PNGA_UPDATE_GHOST_DIR, ltme, 0);    
   return return_value;
}


void wnga_update_ghosts(Integer g_a)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_update_ghosts(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_UPDATE_GHOSTS, ltme, 0);    
}


logical wnga_uses_ma()
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_uses_ma();
   ltme += I_Wtime();
   update_local_entry(PNGA_USES_MA, ltme, 0);    
   return return_value;
}


logical wnga_uses_proc_grid(Integer g_a)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_uses_proc_grid(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_USES_PROC_GRID, ltme, 0);    
   return return_value;
}


logical wnga_valid_handle(Integer g_a)
{
   logical return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_valid_handle(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_VALID_HANDLE, ltme, 0);    
   return return_value;
}


Integer wnga_verify_handle(Integer g_a)
{
   Integer return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_verify_handle(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_VERIFY_HANDLE, ltme, 0);
   return return_value;
}


DoublePrecision wnga_wtime()
{
   DoublePrecision return_value;
   unsigned long long ltme; 
   ltme=- I_Wtime();
   return_value = pnga_wtime();
   ltme += I_Wtime();
   update_local_entry(PNGA_WTIME, ltme, 0);    
   return return_value;
}


void wnga_zero(Integer g_a)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_zero(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_ZERO, ltme, 0);    
}


void wnga_zero_diagonal(Integer g_a)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_zero_diagonal(g_a);
   ltme += I_Wtime();
   update_local_entry(PNGA_ZERO_DIAGONAL, ltme, 0);    
}


void wnga_zero_patch(Integer g_a, Integer *lo, Integer *hi)
{
   unsigned long long ltme; 
   ltme=- I_Wtime();
   pnga_zero_patch(g_a, lo, hi);
   ltme += I_Wtime();
   update_local_entry(PNGA_ZERO_PATCH, ltme, 0);    
}

void wnga_initialize()
{
    
    pnga_initialize();
    update_local_entry(PNGA_INITIALIZE, 0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    init_ga_prof_struct(me, nproc);
}

void wnga_initialize_ltd(Integer limit)
{

    pnga_initialize_ltd(limit);
    update_local_entry(PNGA_INITIALIZE_LTD, 0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    init_ga_prof_struct(me, nproc);
}

void wnga_terminate()
{
    int lme, lnproc, i;
    static int terminate_flag = 0;
    ++terminate_flag;
    MPI_Comm comm = GA_MPI_Comm();
    update_local_entry(PNGA_TERMINATE, 0, 0);
    pnga_terminate();

    if(terminate_flag == 1){
       for(i = 0; i < WPROF_TOTAL; ++i) update_global_entry(i, comm);
       print_ga_prof_stats(HUMAN_FMT, stdout, comm);
    }
}

