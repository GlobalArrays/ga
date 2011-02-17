
#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "papi.h"
#include "typesf2c.h"

static long count_pnga_abs_value = 0;
static long count_pnga_abs_value_patch = 0;
static long count_pnga_acc = 0;
static long count_pnga_access_block_grid_idx = 0;
static long count_pnga_access_block_grid_ptr = 0;
static long count_pnga_access_block_idx = 0;
static long count_pnga_access_block_ptr = 0;
static long count_pnga_access_block_segment_idx = 0;
static long count_pnga_access_block_segment_ptr = 0;
static long count_pnga_access_ghost_element = 0;
static long count_pnga_access_ghost_element_ptr = 0;
static long count_pnga_access_ghost_ptr = 0;
static long count_pnga_access_ghosts = 0;
static long count_pnga_access_idx = 0;
static long count_pnga_access_ptr = 0;
static long count_pnga_add = 0;
static long count_pnga_add_constant = 0;
static long count_pnga_add_constant_patch = 0;
static long count_pnga_add_diagonal = 0;
static long count_pnga_add_patch = 0;
static long count_pnga_allocate = 0;
static long count_pnga_bin_index = 0;
static long count_pnga_bin_sorter = 0;
static long count_pnga_brdcst = 0;
static long count_pnga_check_handle = 0;
static long count_pnga_cluster_nnodes = 0;
static long count_pnga_cluster_nodeid = 0;
static long count_pnga_cluster_nprocs = 0;
static long count_pnga_cluster_proc_nodeid = 0;
static long count_pnga_cluster_procid = 0;
static long count_pnga_comp_patch = 0;
static long count_pnga_compare_distr = 0;
static long count_pnga_copy = 0;
static long count_pnga_copy_patch = 0;
static long count_pnga_copy_patch_dp = 0;
static long count_pnga_create = 0;
static long count_pnga_create_bin_range = 0;
static long count_pnga_create_config = 0;
static long count_pnga_create_ghosts = 0;
static long count_pnga_create_ghosts_config = 0;
static long count_pnga_create_ghosts_irreg = 0;
static long count_pnga_create_ghosts_irreg_config = 0;
static long count_pnga_create_handle = 0;
static long count_pnga_create_irreg = 0;
static long count_pnga_create_irreg_config = 0;
static long count_pnga_create_mutexes = 0;
static long count_pnga_ddot_patch_dp = 0;
static long count_pnga_destroy = 0;
static long count_pnga_destroy_mutexes = 0;
static long count_pnga_diag = 0;
static long count_pnga_diag_reuse = 0;
static long count_pnga_diag_seq = 0;
static long count_pnga_diag_std = 0;
static long count_pnga_diag_std_seq = 0;
static long count_pnga_distribution = 0;
static long count_pnga_dot = 0;
static long count_pnga_dot_patch = 0;
static long count_pnga_duplicate = 0;
static long count_pnga_elem_divide = 0;
static long count_pnga_elem_divide_patch = 0;
static long count_pnga_elem_maximum = 0;
static long count_pnga_elem_maximum_patch = 0;
static long count_pnga_elem_minimum = 0;
static long count_pnga_elem_minimum_patch = 0;
static long count_pnga_elem_multiply = 0;
static long count_pnga_elem_multiply_patch = 0;
static long count_pnga_elem_step_divide_patch = 0;
static long count_pnga_elem_stepb_divide_patch = 0;
static long count_pnga_error = 0;
static long count_pnga_fence = 0;
static long count_pnga_fill = 0;
static long count_pnga_fill_patch = 0;
static long count_pnga_gather = 0;
static long count_pnga_gather2d = 0;
static long count_pnga_get = 0;
static long count_pnga_get_block_info = 0;
static long count_pnga_get_debug = 0;
static long count_pnga_get_diag = 0;
static long count_pnga_get_dimension = 0;
static long count_pnga_get_ghost_block = 0;
static long count_pnga_get_pgroup = 0;
static long count_pnga_get_pgroup_size = 0;
static long count_pnga_get_proc_grid = 0;
static long count_pnga_get_proc_index = 0;
static long count_pnga_ghost_barrier = 0;
static long count_pnga_gop = 0;
static long count_pnga_has_ghosts = 0;
static long count_pnga_init_fence = 0;
static long count_pnga_initialize = 0;
static long count_pnga_initialize_ltd = 0;
static long count_pnga_inquire = 0;
static long count_pnga_inquire_memory = 0;
static long count_pnga_inquire_name = 0;
static long count_pnga_inquire_type = 0;
static long count_pnga_is_mirrored = 0;
static long count_pnga_list_nodeid = 0;
static long count_pnga_llt_solve = 0;
static long count_pnga_locate = 0;
static long count_pnga_locate_nnodes = 0;
static long count_pnga_locate_num_blocks = 0;
static long count_pnga_locate_region = 0;
static long count_pnga_lock = 0;
static long count_pnga_lu_solve = 0;
static long count_pnga_lu_solve_alt = 0;
static long count_pnga_lu_solve_seq = 0;
static long count_pnga_mask_sync = 0;
static long count_pnga_matmul = 0;
static long count_pnga_matmul_mirrored = 0;
static long count_pnga_matmul_patch = 0;
static long count_pnga_median = 0;
static long count_pnga_median_patch = 0;
static long count_pnga_memory_avail = 0;
static long count_pnga_memory_avail_type = 0;
static long count_pnga_memory_limited = 0;
static long count_pnga_merge_distr_patch = 0;
static long count_pnga_merge_mirrored = 0;
static long count_pnga_msg_brdcst = 0;
static long count_pnga_msg_pgroup_sync = 0;
static long count_pnga_msg_sync = 0;
static long count_pnga_nbacc = 0;
static long count_pnga_nbget = 0;
static long count_pnga_nbget_ghost_dir = 0;
static long count_pnga_nblock = 0;
static long count_pnga_nbput = 0;
static long count_pnga_nbtest = 0;
static long count_pnga_nbwait = 0;
static long count_pnga_ndim = 0;
static long count_pnga_nnodes = 0;
static long count_pnga_nodeid = 0;
static long count_pnga_norm1 = 0;
static long count_pnga_norm_infinity = 0;
static long count_pnga_pack = 0;
static long count_pnga_patch_enum = 0;
static long count_pnga_patch_intersect = 0;
static long count_pnga_periodic = 0;
static long count_pnga_pgroup_absolute_id = 0;
static long count_pnga_pgroup_brdcst = 0;
static long count_pnga_pgroup_create = 0;
static long count_pnga_pgroup_destroy = 0;
static long count_pnga_pgroup_get_default = 0;
static long count_pnga_pgroup_get_mirror = 0;
static long count_pnga_pgroup_get_world = 0;
static long count_pnga_pgroup_gop = 0;
static long count_pnga_pgroup_nnodes = 0;
static long count_pnga_pgroup_nodeid = 0;
static long count_pnga_pgroup_set_default = 0;
static long count_pnga_pgroup_split = 0;
static long count_pnga_pgroup_split_irreg = 0;
static long count_pnga_pgroup_sync = 0;
static long count_pnga_print = 0;
static long count_pnga_print_distribution = 0;
static long count_pnga_print_file = 0;
static long count_pnga_print_patch = 0;
static long count_pnga_print_patch2d = 0;
static long count_pnga_print_patch_file = 0;
static long count_pnga_print_patch_file2d = 0;
static long count_pnga_print_stats = 0;
static long count_pnga_proc_topology = 0;
static long count_pnga_put = 0;
static long count_pnga_randomize = 0;
static long count_pnga_read_inc = 0;
static long count_pnga_recip = 0;
static long count_pnga_recip_patch = 0;
static long count_pnga_release = 0;
static long count_pnga_release_block = 0;
static long count_pnga_release_block_grid = 0;
static long count_pnga_release_block_segment = 0;
static long count_pnga_release_ghost_element = 0;
static long count_pnga_release_ghosts = 0;
static long count_pnga_release_update = 0;
static long count_pnga_release_update_block = 0;
static long count_pnga_release_update_block_grid = 0;
static long count_pnga_release_update_block_segment = 0;
static long count_pnga_release_update_ghost_element = 0;
static long count_pnga_release_update_ghosts = 0;
static long count_pnga_scale = 0;
static long count_pnga_scale_cols = 0;
static long count_pnga_scale_patch = 0;
static long count_pnga_scale_rows = 0;
static long count_pnga_scan_add = 0;
static long count_pnga_scan_copy = 0;
static long count_pnga_scatter = 0;
static long count_pnga_scatter2d = 0;
static long count_pnga_scatter_acc = 0;
static long count_pnga_scatter_acc2d = 0;
static long count_pnga_select_elem = 0;
static long count_pnga_set_array_name = 0;
static long count_pnga_set_block_cyclic = 0;
static long count_pnga_set_block_cyclic_proc_grid = 0;
static long count_pnga_set_chunk = 0;
static long count_pnga_set_data = 0;
static long count_pnga_set_debug = 0;
static long count_pnga_set_diagonal = 0;
static long count_pnga_set_ghost_corner_flag = 0;
static long count_pnga_set_ghost_info = 0;
static long count_pnga_set_ghosts = 0;
static long count_pnga_set_irreg_distr = 0;
static long count_pnga_set_irreg_flag = 0;
static long count_pnga_set_memory_limit = 0;
static long count_pnga_set_pgroup = 0;
static long count_pnga_set_restricted = 0;
static long count_pnga_set_restricted_range = 0;
static long count_pnga_set_update4_info = 0;
static long count_pnga_set_update5_info = 0;
static long count_pnga_shift_diagonal = 0;
static long count_pnga_solve = 0;
static long count_pnga_spd_invert = 0;
static long count_pnga_step_bound_info = 0;
static long count_pnga_step_bound_info_patch = 0;
static long count_pnga_step_mask_patch = 0;
static long count_pnga_step_max = 0;
static long count_pnga_step_max_patch = 0;
static long count_pnga_strided_acc = 0;
static long count_pnga_strided_get = 0;
static long count_pnga_strided_put = 0;
static long count_pnga_summarize = 0;
static long count_pnga_symmetrize = 0;
static long count_pnga_sync = 0;
static long count_pnga_terminate = 0;
static long count_pnga_timer = 0;
static long count_pnga_total_blocks = 0;
static long count_pnga_transpose = 0;
static long count_pnga_type_c2f = 0;
static long count_pnga_type_f2c = 0;
static long count_pnga_unlock = 0;
static long count_pnga_unpack = 0;
static long count_pnga_update1_ghosts = 0;
static long count_pnga_update2_ghosts = 0;
static long count_pnga_update3_ghosts = 0;
static long count_pnga_update44_ghosts = 0;
static long count_pnga_update4_ghosts = 0;
static long count_pnga_update55_ghosts = 0;
static long count_pnga_update5_ghosts = 0;
static long count_pnga_update6_ghosts = 0;
static long count_pnga_update7_ghosts = 0;
static long count_pnga_update_ghost_dir = 0;
static long count_pnga_update_ghosts = 0;
static long count_pnga_uses_ma = 0;
static long count_pnga_uses_proc_grid = 0;
static long count_pnga_valid_handle = 0;
static long count_pnga_verify_handle = 0;
static long count_pnga_wtime = 0;
static long count_pnga_zero = 0;
static long count_pnga_zero_diagonal = 0;
static long count_pnga_zero_patch = 0;


void wnga_abs_value(Integer g_a)
{
    ++count_pnga_abs_value;
    pnga_abs_value(g_a);
}


void wnga_abs_value_patch(Integer g_a, Integer *lo, Integer *hi)
{
    ++count_pnga_abs_value_patch;
    pnga_abs_value_patch(g_a, lo, hi);
}


void wnga_acc(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha)
{
    ++count_pnga_acc;
    pnga_acc(g_a, lo, hi, buf, ld, alpha);
}


void wnga_access_block_grid_idx(Integer g_a, Integer *subscript, AccessIndex *index, Integer *ld)
{
    ++count_pnga_access_block_grid_idx;
    pnga_access_block_grid_idx(g_a, subscript, index, ld);
}


void wnga_access_block_grid_ptr(Integer g_a, Integer *index, void *ptr, Integer *ld)
{
    ++count_pnga_access_block_grid_ptr;
    pnga_access_block_grid_ptr(g_a, index, ptr, ld);
}


void wnga_access_block_idx(Integer g_a, Integer idx, AccessIndex *index, Integer *ld)
{
    ++count_pnga_access_block_idx;
    pnga_access_block_idx(g_a, idx, index, ld);
}


void wnga_access_block_ptr(Integer g_a, Integer idx, void *ptr, Integer *ld)
{
    ++count_pnga_access_block_ptr;
    pnga_access_block_ptr(g_a, idx, ptr, ld);
}


void wnga_access_block_segment_idx(Integer g_a, Integer proc, AccessIndex *index, Integer *len)
{
    ++count_pnga_access_block_segment_idx;
    pnga_access_block_segment_idx(g_a, proc, index, len);
}


void wnga_access_block_segment_ptr(Integer g_a, Integer proc, void *ptr, Integer *len)
{
    ++count_pnga_access_block_segment_ptr;
    pnga_access_block_segment_ptr(g_a, proc, ptr, len);
}


void wnga_access_ghost_element(Integer g_a, AccessIndex *index, Integer subscript[], Integer ld[])
{
    ++count_pnga_access_ghost_element;
    pnga_access_ghost_element(g_a, index, subscript, ld);
}


void wnga_access_ghost_element_ptr(Integer g_a, void *ptr, Integer subscript[], Integer ld[])
{
    ++count_pnga_access_ghost_element_ptr;
    pnga_access_ghost_element_ptr(g_a, ptr, subscript, ld);
}


void wnga_access_ghost_ptr(Integer g_a, Integer dims[], void *ptr, Integer ld[])
{
    ++count_pnga_access_ghost_ptr;
    pnga_access_ghost_ptr(g_a, dims, ptr, ld);
}


void wnga_access_ghosts(Integer g_a, Integer dims[], AccessIndex *index, Integer ld[])
{
    ++count_pnga_access_ghosts;
    pnga_access_ghosts(g_a, dims, index, ld);
}


void wnga_access_idx(Integer g_a, Integer *lo, Integer *hi, AccessIndex *index, Integer *ld)
{
    ++count_pnga_access_idx;
    pnga_access_idx(g_a, lo, hi, index, ld);
}


void wnga_access_ptr(Integer g_a, Integer *lo, Integer *hi, void *ptr, Integer *ld)
{
    ++count_pnga_access_ptr;
    pnga_access_ptr(g_a, lo, hi, ptr, ld);
}


void wnga_add(void *alpha, Integer g_a, void *beta, Integer g_b, Integer g_c)
{
    ++count_pnga_add;
    pnga_add(alpha, g_a, beta, g_b, g_c);
}


void wnga_add_constant(Integer g_a, void *alpha)
{
    ++count_pnga_add_constant;
    pnga_add_constant(g_a, alpha);
}


void wnga_add_constant_patch(Integer g_a, Integer *lo, Integer *hi, void *alpha)
{
    ++count_pnga_add_constant_patch;
    pnga_add_constant_patch(g_a, lo, hi, alpha);
}


void wnga_add_diagonal(Integer g_a, Integer g_v)
{
    ++count_pnga_add_diagonal;
    pnga_add_diagonal(g_a, g_v);
}


void wnga_add_patch(void *alpha, Integer g_a, Integer *alo, Integer *ahi, void *beta, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    ++count_pnga_add_patch;
    pnga_add_patch(alpha, g_a, alo, ahi, beta, g_b, blo, bhi, g_c, clo, chi);
}


logical wnga_allocate(Integer g_a)
{
    ++count_pnga_allocate;
    return pnga_allocate(g_a);
}


void wnga_bin_index(Integer g_bin, Integer g_cnt, Integer g_off, Integer *values, Integer *subs, Integer n, Integer sortit)
{
    ++count_pnga_bin_index;
    pnga_bin_index(g_bin, g_cnt, g_off, values, subs, n, sortit);
}


void wnga_bin_sorter(Integer g_bin, Integer g_cnt, Integer g_off)
{
    ++count_pnga_bin_sorter;
    pnga_bin_sorter(g_bin, g_cnt, g_off);
}


void wnga_brdcst(Integer type, void *buf, Integer len, Integer originator)
{
    ++count_pnga_brdcst;
    pnga_brdcst(type, buf, len, originator);
}


void wnga_check_handle(Integer g_a, char *string)
{
    ++count_pnga_check_handle;
    pnga_check_handle(g_a, string);
}


Integer wnga_cluster_nnodes()
{
    ++count_pnga_cluster_nnodes;
    return pnga_cluster_nnodes();
}


Integer wnga_cluster_nodeid()
{
    ++count_pnga_cluster_nodeid;
    return pnga_cluster_nodeid();
}


Integer wnga_cluster_nprocs(Integer node)
{
    ++count_pnga_cluster_nprocs;
    return pnga_cluster_nprocs(node);
}


Integer wnga_cluster_proc_nodeid(Integer proc)
{
    ++count_pnga_cluster_proc_nodeid;
    return pnga_cluster_proc_nodeid(proc);
}


Integer wnga_cluster_procid(Integer node, Integer loc_proc_id)
{
    ++count_pnga_cluster_procid;
    return pnga_cluster_procid(node, loc_proc_id);
}


logical wnga_comp_patch(Integer andim, Integer *alo, Integer *ahi, Integer bndim, Integer *blo, Integer *bhi)
{
    ++count_pnga_comp_patch;
    return pnga_comp_patch(andim, alo, ahi, bndim, blo, bhi);
}


logical wnga_compare_distr(Integer g_a, Integer g_b)
{
    ++count_pnga_compare_distr;
    return pnga_compare_distr(g_a, g_b);
}


void wnga_copy(Integer g_a, Integer g_b)
{
    ++count_pnga_copy;
    pnga_copy(g_a, g_b);
}


void wnga_copy_patch(char *trans, Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi)
{
    ++count_pnga_copy_patch;
    pnga_copy_patch(trans, g_a, alo, ahi, g_b, blo, bhi);
}


void wnga_copy_patch_dp(char *t_a, Integer g_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi)
{
    ++count_pnga_copy_patch_dp;
    pnga_copy_patch_dp(t_a, g_a, ailo, aihi, ajlo, ajhi, g_b, bilo, bihi, bjlo, bjhi);
}


logical wnga_create(Integer type, Integer ndim, Integer *dims, char *name, Integer *chunk, Integer *g_a)
{
    ++count_pnga_create;
    return pnga_create(type, ndim, dims, name, chunk, g_a);
}


logical wnga_create_bin_range(Integer g_bin, Integer g_cnt, Integer g_off, Integer *g_range)
{
    ++count_pnga_create_bin_range;
    return pnga_create_bin_range(g_bin, g_cnt, g_off, g_range);
}


logical wnga_create_config(Integer type, Integer ndim, Integer *dims, char *name, Integer *chunk, Integer p_handle, Integer *g_a)
{
    ++count_pnga_create_config;
    return pnga_create_config(type, ndim, dims, name, chunk, p_handle, g_a);
}


logical wnga_create_ghosts(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *chunk, Integer *g_a)
{
    ++count_pnga_create_ghosts;
    return pnga_create_ghosts(type, ndim, dims, width, name, chunk, g_a);
}


logical wnga_create_ghosts_config(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *chunk, Integer p_handle, Integer *g_a)
{
    ++count_pnga_create_ghosts_config;
    return pnga_create_ghosts_config(type, ndim, dims, width, name, chunk, p_handle, g_a);
}


logical wnga_create_ghosts_irreg(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *map, Integer *block, Integer *g_a)
{
    ++count_pnga_create_ghosts_irreg;
    return pnga_create_ghosts_irreg(type, ndim, dims, width, name, map, block, g_a);
}


logical wnga_create_ghosts_irreg_config(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *map, Integer *block, Integer p_handle, Integer *g_a)
{
    ++count_pnga_create_ghosts_irreg_config;
    return pnga_create_ghosts_irreg_config(type, ndim, dims, width, name, map, block, p_handle, g_a);
}


Integer wnga_create_handle()
{
    ++count_pnga_create_handle;
    return pnga_create_handle();
}


logical wnga_create_irreg(Integer type, Integer ndim, Integer *dims, char *name, Integer *map, Integer *block, Integer *g_a)
{
    ++count_pnga_create_irreg;
    return pnga_create_irreg(type, ndim, dims, name, map, block, g_a);
}


logical wnga_create_irreg_config(Integer type, Integer ndim, Integer *dims, char *name, Integer *map, Integer *block, Integer p_handle, Integer *g_a)
{
    ++count_pnga_create_irreg_config;
    return pnga_create_irreg_config(type, ndim, dims, name, map, block, p_handle, g_a);
}


logical wnga_create_mutexes(Integer num)
{
    ++count_pnga_create_mutexes;
    return pnga_create_mutexes(num);
}


DoublePrecision wnga_ddot_patch_dp(Integer g_a, char *t_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, char *t_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi)
{
    ++count_pnga_ddot_patch_dp;
    return pnga_ddot_patch_dp(g_a, t_a, ailo, aihi, ajlo, ajhi, g_b, t_b, bilo, bihi, bjlo, bjhi);
}


logical wnga_destroy(Integer g_a)
{
    ++count_pnga_destroy;
    return pnga_destroy(g_a);
}


logical wnga_destroy_mutexes()
{
    ++count_pnga_destroy_mutexes;
    return pnga_destroy_mutexes();
}


void wnga_diag(Integer g_a, Integer g_s, Integer g_v, DoublePrecision *eval)
{
    ++count_pnga_diag;
    pnga_diag(g_a, g_s, g_v, eval);
}


void wnga_diag_reuse(Integer reuse, Integer g_a, Integer g_s, Integer g_v, DoublePrecision *eval)
{
    ++count_pnga_diag_reuse;
    pnga_diag_reuse(reuse, g_a, g_s, g_v, eval);
}


void wnga_diag_seq(Integer g_a, Integer g_s, Integer g_v, DoublePrecision *eval)
{
    ++count_pnga_diag_seq;
    pnga_diag_seq(g_a, g_s, g_v, eval);
}


void wnga_diag_std(Integer g_a, Integer g_v, DoublePrecision *eval)
{
    ++count_pnga_diag_std;
    pnga_diag_std(g_a, g_v, eval);
}


void wnga_diag_std_seq(Integer g_a, Integer g_v, DoublePrecision *eval)
{
    ++count_pnga_diag_std_seq;
    pnga_diag_std_seq(g_a, g_v, eval);
}


void wnga_distribution(Integer g_a, Integer proc, Integer *lo, Integer *hi)
{
    ++count_pnga_distribution;
    pnga_distribution(g_a, proc, lo, hi);
}


void wnga_dot(int type, Integer g_a, Integer g_b, void *value)
{
    ++count_pnga_dot;
    pnga_dot(type, g_a, g_b, value);
}


void wnga_dot_patch(Integer g_a, char *t_a, Integer *alo, Integer *ahi, Integer g_b, char *t_b, Integer *blo, Integer *bhi, void *retval)
{
    ++count_pnga_dot_patch;
    pnga_dot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi, retval);
}


logical wnga_duplicate(Integer g_a, Integer *g_b, char *array_name)
{
    ++count_pnga_duplicate;
    return pnga_duplicate(g_a, g_b, array_name);
}


void wnga_elem_divide(Integer g_a, Integer g_b, Integer g_c)
{
    ++count_pnga_elem_divide;
    pnga_elem_divide(g_a, g_b, g_c);
}


void wnga_elem_divide_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    ++count_pnga_elem_divide_patch;
    pnga_elem_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_elem_maximum(Integer g_a, Integer g_b, Integer g_c)
{
    ++count_pnga_elem_maximum;
    pnga_elem_maximum(g_a, g_b, g_c);
}


void wnga_elem_maximum_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    ++count_pnga_elem_maximum_patch;
    pnga_elem_maximum_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_elem_minimum(Integer g_a, Integer g_b, Integer g_c)
{
    ++count_pnga_elem_minimum;
    pnga_elem_minimum(g_a, g_b, g_c);
}


void wnga_elem_minimum_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    ++count_pnga_elem_minimum_patch;
    pnga_elem_minimum_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_elem_multiply(Integer g_a, Integer g_b, Integer g_c)
{
    ++count_pnga_elem_multiply;
    pnga_elem_multiply(g_a, g_b, g_c);
}


void wnga_elem_multiply_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    ++count_pnga_elem_multiply_patch;
    pnga_elem_multiply_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_elem_step_divide_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    ++count_pnga_elem_step_divide_patch;
    pnga_elem_step_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_elem_stepb_divide_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    ++count_pnga_elem_stepb_divide_patch;
    pnga_elem_stepb_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_error(char *string, Integer icode)
{
    ++count_pnga_error;
    pnga_error(string, icode);
}


void wnga_fence()
{
    ++count_pnga_fence;
    pnga_fence();
}


void wnga_fill(Integer g_a, void *val)
{
    ++count_pnga_fill;
    pnga_fill(g_a, val);
}


void wnga_fill_patch(Integer g_a, Integer *lo, Integer *hi, void *val)
{
    ++count_pnga_fill_patch;
    pnga_fill_patch(g_a, lo, hi, val);
}


void wnga_gather(Integer g_a, void *v, Integer subscript[], Integer nv)
{
    ++count_pnga_gather;
    pnga_gather(g_a, v, subscript, nv);
}


void wnga_gather2d(Integer g_a, void *v, Integer *i, Integer *j, Integer nv)
{
    ++count_pnga_gather2d;
    pnga_gather2d(g_a, v, i, j, nv);
}


void wnga_get(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
    ++count_pnga_get;
    pnga_get(g_a, lo, hi, buf, ld);
}


void wnga_get_block_info(Integer g_a, Integer *num_blocks, Integer *block_dims)
{
    ++count_pnga_get_block_info;
    pnga_get_block_info(g_a, num_blocks, block_dims);
}


logical wnga_get_debug()
{
    ++count_pnga_get_debug;
    return pnga_get_debug();
}


void wnga_get_diag(Integer g_a, Integer g_v)
{
    ++count_pnga_get_diag;
    pnga_get_diag(g_a, g_v);
}


Integer wnga_get_dimension(Integer g_a)
{
    ++count_pnga_get_dimension;
    return pnga_get_dimension(g_a);
}


void wnga_get_ghost_block(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
    ++count_pnga_get_ghost_block;
    pnga_get_ghost_block(g_a, lo, hi, buf, ld);
}


Integer wnga_get_pgroup(Integer g_a)
{
    ++count_pnga_get_pgroup;
    return pnga_get_pgroup(g_a);
}


Integer wnga_get_pgroup_size(Integer grp_id)
{
    ++count_pnga_get_pgroup_size;
    return pnga_get_pgroup_size(grp_id);
}


void wnga_get_proc_grid(Integer g_a, Integer *dims)
{
    ++count_pnga_get_proc_grid;
    pnga_get_proc_grid(g_a, dims);
}


void wnga_get_proc_index(Integer g_a, Integer iproc, Integer *index)
{
    ++count_pnga_get_proc_index;
    pnga_get_proc_index(g_a, iproc, index);
}


void wnga_ghost_barrier()
{
    ++count_pnga_ghost_barrier;
    pnga_ghost_barrier();
}


void wnga_gop(Integer type, void *x, Integer n, char *op)
{
    ++count_pnga_gop;
    pnga_gop(type, x, n, op);
}


logical wnga_has_ghosts(Integer g_a)
{
    ++count_pnga_has_ghosts;
    return pnga_has_ghosts(g_a);
}


void wnga_init_fence()
{
    ++count_pnga_init_fence;
    pnga_init_fence();
}


void wnga_initialize()
{
    ++count_pnga_initialize;
    pnga_initialize();
}


void wnga_initialize_ltd(Integer limit)
{
    ++count_pnga_initialize_ltd;
    pnga_initialize_ltd(limit);
}


void wnga_inquire(Integer g_a, Integer *type, Integer *ndim, Integer *dims)
{
    ++count_pnga_inquire;
    pnga_inquire(g_a, type, ndim, dims);
}


Integer wnga_inquire_memory()
{
    ++count_pnga_inquire_memory;
    return pnga_inquire_memory();
}


void wnga_inquire_name(Integer g_a, char **array_name)
{
    ++count_pnga_inquire_name;
    pnga_inquire_name(g_a, array_name);
}


void wnga_inquire_type(Integer g_a, Integer *type)
{
    ++count_pnga_inquire_type;
    pnga_inquire_type(g_a, type);
}


logical wnga_is_mirrored(Integer g_a)
{
    ++count_pnga_is_mirrored;
    return pnga_is_mirrored(g_a);
}


void wnga_list_nodeid(Integer *list, Integer nprocs)
{
    ++count_pnga_list_nodeid;
    pnga_list_nodeid(list, nprocs);
}


Integer wnga_llt_solve(Integer g_a, Integer g_b)
{
    ++count_pnga_llt_solve;
    return pnga_llt_solve(g_a, g_b);
}


logical wnga_locate(Integer g_a, Integer *subscript, Integer *owner)
{
    ++count_pnga_locate;
    return pnga_locate(g_a, subscript, owner);
}


logical wnga_locate_nnodes(Integer g_a, Integer *lo, Integer *hi, Integer *np)
{
    ++count_pnga_locate_nnodes;
    return pnga_locate_nnodes(g_a, lo, hi, np);
}


Integer wnga_locate_num_blocks(Integer g_a, Integer *lo, Integer *hi)
{
    ++count_pnga_locate_num_blocks;
    return pnga_locate_num_blocks(g_a, lo, hi);
}


logical wnga_locate_region(Integer g_a, Integer *lo, Integer *hi, Integer *map, Integer *proclist, Integer *np)
{
    ++count_pnga_locate_region;
    return pnga_locate_region(g_a, lo, hi, map, proclist, np);
}


void wnga_lock(Integer mutex)
{
    ++count_pnga_lock;
    pnga_lock(mutex);
}


void wnga_lu_solve(char *tran, Integer g_a, Integer g_b)
{
    ++count_pnga_lu_solve;
    pnga_lu_solve(tran, g_a, g_b);
}


void wnga_lu_solve_alt(Integer tran, Integer g_a, Integer g_b)
{
    ++count_pnga_lu_solve_alt;
    pnga_lu_solve_alt(tran, g_a, g_b);
}


void wnga_lu_solve_seq(char *trans, Integer g_a, Integer g_b)
{
    ++count_pnga_lu_solve_seq;
    pnga_lu_solve_seq(trans, g_a, g_b);
}


void wnga_mask_sync(Integer begin, Integer end)
{
    ++count_pnga_mask_sync;
    pnga_mask_sync(begin, end);
}


void wnga_matmul(char *transa, char *transb, void *alpha, void *beta, Integer g_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi, Integer g_c, Integer cilo, Integer cihi, Integer cjlo, Integer cjhi)
{
    ++count_pnga_matmul;
    pnga_matmul(transa, transb, alpha, beta, g_a, ailo, aihi, ajlo, ajhi, g_b, bilo, bihi, bjlo, bjhi, g_c, cilo, cihi, cjlo, cjhi);
}


void wnga_matmul_mirrored(char *transa, char *transb, void *alpha, void *beta, Integer g_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi, Integer g_c, Integer cilo, Integer cihi, Integer cjlo, Integer cjhi)
{
    ++count_pnga_matmul_mirrored;
    pnga_matmul_mirrored(transa, transb, alpha, beta, g_a, ailo, aihi, ajlo, ajhi, g_b, bilo, bihi, bjlo, bjhi, g_c, cilo, cihi, cjlo, cjhi);
}


void wnga_matmul_patch(char *transa, char *transb, void *alpha, void *beta, Integer g_a, Integer alo[], Integer ahi[], Integer g_b, Integer blo[], Integer bhi[], Integer g_c, Integer clo[], Integer chi[])
{
    ++count_pnga_matmul_patch;
    pnga_matmul_patch(transa, transb, alpha, beta, g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_median(Integer g_a, Integer g_b, Integer g_c, Integer g_m)
{
    ++count_pnga_median;
    pnga_median(g_a, g_b, g_c, g_m);
}


void wnga_median_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi, Integer g_m, Integer *mlo, Integer *mhi)
{
    ++count_pnga_median_patch;
    pnga_median_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi, g_m, mlo, mhi);
}


Integer wnga_memory_avail()
{
    ++count_pnga_memory_avail;
    return pnga_memory_avail();
}


Integer wnga_memory_avail_type(Integer datatype)
{
    ++count_pnga_memory_avail_type;
    return pnga_memory_avail_type(datatype);
}


logical wnga_memory_limited()
{
    ++count_pnga_memory_limited;
    return pnga_memory_limited();
}


void wnga_merge_distr_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi)
{
    ++count_pnga_merge_distr_patch;
    pnga_merge_distr_patch(g_a, alo, ahi, g_b, blo, bhi);
}


void wnga_merge_mirrored(Integer g_a)
{
    ++count_pnga_merge_mirrored;
    pnga_merge_mirrored(g_a);
}


void wnga_msg_brdcst(Integer type, void *buffer, Integer len, Integer root)
{
    ++count_pnga_msg_brdcst;
    pnga_msg_brdcst(type, buffer, len, root);
}


void wnga_msg_pgroup_sync(Integer grp_id)
{
    ++count_pnga_msg_pgroup_sync;
    pnga_msg_pgroup_sync(grp_id);
}


void wnga_msg_sync()
{
    ++count_pnga_msg_sync;
    pnga_msg_sync();
}


void wnga_nbacc(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha, Integer *nbhndl)
{
    ++count_pnga_nbacc;
    pnga_nbacc(g_a, lo, hi, buf, ld, alpha, nbhndl);
}


void wnga_nbget(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle)
{
    ++count_pnga_nbget;
    pnga_nbget(g_a, lo, hi, buf, ld, nbhandle);
}


void wnga_nbget_ghost_dir(Integer g_a, Integer *mask, Integer *nbhandle)
{
    ++count_pnga_nbget_ghost_dir;
    pnga_nbget_ghost_dir(g_a, mask, nbhandle);
}


void wnga_nblock(Integer g_a, Integer *nblock)
{
    ++count_pnga_nblock;
    pnga_nblock(g_a, nblock);
}


void wnga_nbput(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle)
{
    ++count_pnga_nbput;
    pnga_nbput(g_a, lo, hi, buf, ld, nbhandle);
}


Integer wnga_nbtest(Integer *nbhandle)
{
    ++count_pnga_nbtest;
    return pnga_nbtest(nbhandle);
}


void wnga_nbwait(Integer *nbhandle)
{
    ++count_pnga_nbwait;
    pnga_nbwait(nbhandle);
}


Integer wnga_ndim(Integer g_a)
{
    ++count_pnga_ndim;
    return pnga_ndim(g_a);
}


Integer wnga_nnodes()
{
    ++count_pnga_nnodes;
    return pnga_nnodes();
}


Integer wnga_nodeid()
{
    ++count_pnga_nodeid;
    return pnga_nodeid();
}


void wnga_norm1(Integer g_a, double *nm)
{
    ++count_pnga_norm1;
    pnga_norm1(g_a, nm);
}


void wnga_norm_infinity(Integer g_a, double *nm)
{
    ++count_pnga_norm_infinity;
    pnga_norm_infinity(g_a, nm);
}


void wnga_pack(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi, Integer *icount)
{
    ++count_pnga_pack;
    pnga_pack(g_a, g_b, g_sbit, lo, hi, icount);
}


void wnga_patch_enum(Integer g_a, Integer lo, Integer hi, void *start, void *stride)
{
    ++count_pnga_patch_enum;
    pnga_patch_enum(g_a, lo, hi, start, stride);
}


logical wnga_patch_intersect(Integer *lo, Integer *hi, Integer *lop, Integer *hip, Integer ndim)
{
    ++count_pnga_patch_intersect;
    return pnga_patch_intersect(lo, hi, lop, hip, ndim);
}


void wnga_periodic(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha, Integer op_code)
{
    ++count_pnga_periodic;
    pnga_periodic(g_a, lo, hi, buf, ld, alpha, op_code);
}


Integer wnga_pgroup_absolute_id(Integer grp, Integer pid)
{
    ++count_pnga_pgroup_absolute_id;
    return pnga_pgroup_absolute_id(grp, pid);
}


void wnga_pgroup_brdcst(Integer grp_id, Integer type, void *buf, Integer len, Integer originator)
{
    ++count_pnga_pgroup_brdcst;
    pnga_pgroup_brdcst(grp_id, type, buf, len, originator);
}


Integer wnga_pgroup_create(Integer *list, Integer count)
{
    ++count_pnga_pgroup_create;
    return pnga_pgroup_create(list, count);
}


logical wnga_pgroup_destroy(Integer grp)
{
    ++count_pnga_pgroup_destroy;
    return pnga_pgroup_destroy(grp);
}


Integer wnga_pgroup_get_default()
{
    ++count_pnga_pgroup_get_default;
    return pnga_pgroup_get_default();
}


Integer wnga_pgroup_get_mirror()
{
    ++count_pnga_pgroup_get_mirror;
    return pnga_pgroup_get_mirror();
}


Integer wnga_pgroup_get_world()
{
    ++count_pnga_pgroup_get_world;
    return pnga_pgroup_get_world();
}


void wnga_pgroup_gop(Integer p_grp, Integer type, void *x, Integer n, char *op)
{
    ++count_pnga_pgroup_gop;
    pnga_pgroup_gop(p_grp, type, x, n, op);
}


Integer wnga_pgroup_nnodes(Integer grp)
{
    ++count_pnga_pgroup_nnodes;
    return pnga_pgroup_nnodes(grp);
}


Integer wnga_pgroup_nodeid(Integer grp)
{
    ++count_pnga_pgroup_nodeid;
    return pnga_pgroup_nodeid(grp);
}


void wnga_pgroup_set_default(Integer grp)
{
    ++count_pnga_pgroup_set_default;
    pnga_pgroup_set_default(grp);
}


Integer wnga_pgroup_split(Integer grp, Integer grp_num)
{
    ++count_pnga_pgroup_split;
    return pnga_pgroup_split(grp, grp_num);
}


Integer wnga_pgroup_split_irreg(Integer grp, Integer mycolor)
{
    ++count_pnga_pgroup_split_irreg;
    return pnga_pgroup_split_irreg(grp, mycolor);
}


void wnga_pgroup_sync(Integer grp_id)
{
    ++count_pnga_pgroup_sync;
    pnga_pgroup_sync(grp_id);
}


void wnga_print(Integer g_a)
{
    ++count_pnga_print;
    pnga_print(g_a);
}


void wnga_print_distribution(int fstyle, Integer g_a)
{
    ++count_pnga_print_distribution;
    pnga_print_distribution(fstyle, g_a);
}


void wnga_print_file(FILE *file, Integer g_a)
{
    ++count_pnga_print_file;
    pnga_print_file(file, g_a);
}


void wnga_print_patch(Integer g_a, Integer *lo, Integer *hi, Integer pretty)
{
    ++count_pnga_print_patch;
    pnga_print_patch(g_a, lo, hi, pretty);
}


void wnga_print_patch2d(Integer g_a, Integer ilo, Integer ihi, Integer jlo, Integer jhi, Integer pretty)
{
    ++count_pnga_print_patch2d;
    pnga_print_patch2d(g_a, ilo, ihi, jlo, jhi, pretty);
}


void wnga_print_patch_file(FILE *file, Integer g_a, Integer *lo, Integer *hi, Integer pretty)
{
    ++count_pnga_print_patch_file;
    pnga_print_patch_file(file, g_a, lo, hi, pretty);
}


void wnga_print_patch_file2d(FILE *file, Integer g_a, Integer ilo, Integer ihi, Integer jlo, Integer jhi, Integer pretty)
{
    ++count_pnga_print_patch_file2d;
    pnga_print_patch_file2d(file, g_a, ilo, ihi, jlo, jhi, pretty);
}


void wnga_print_stats()
{
    ++count_pnga_print_stats;
    pnga_print_stats();
}


void wnga_proc_topology(Integer g_a, Integer proc, Integer *subscript)
{
    ++count_pnga_proc_topology;
    pnga_proc_topology(g_a, proc, subscript);
}


void wnga_put(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
    ++count_pnga_put;
    pnga_put(g_a, lo, hi, buf, ld);
}


void wnga_randomize(Integer g_a, void *val)
{
    ++count_pnga_randomize;
    pnga_randomize(g_a, val);
}


Integer wnga_read_inc(Integer g_a, Integer *subscript, Integer inc)
{
    ++count_pnga_read_inc;
    return pnga_read_inc(g_a, subscript, inc);
}


void wnga_recip(Integer g_a)
{
    ++count_pnga_recip;
    pnga_recip(g_a);
}


void wnga_recip_patch(Integer g_a, Integer *lo, Integer *hi)
{
    ++count_pnga_recip_patch;
    pnga_recip_patch(g_a, lo, hi);
}


void wnga_release(Integer g_a, Integer *lo, Integer *hi)
{
    ++count_pnga_release;
    pnga_release(g_a, lo, hi);
}


void wnga_release_block(Integer g_a, Integer iblock)
{
    ++count_pnga_release_block;
    pnga_release_block(g_a, iblock);
}


void wnga_release_block_grid(Integer g_a, Integer *index)
{
    ++count_pnga_release_block_grid;
    pnga_release_block_grid(g_a, index);
}


void wnga_release_block_segment(Integer g_a, Integer iproc)
{
    ++count_pnga_release_block_segment;
    pnga_release_block_segment(g_a, iproc);
}


void wnga_release_ghost_element(Integer g_a, Integer subscript[])
{
    ++count_pnga_release_ghost_element;
    pnga_release_ghost_element(g_a, subscript);
}


void wnga_release_ghosts(Integer g_a)
{
    ++count_pnga_release_ghosts;
    pnga_release_ghosts(g_a);
}


void wnga_release_update(Integer g_a, Integer *lo, Integer *hi)
{
    ++count_pnga_release_update;
    pnga_release_update(g_a, lo, hi);
}


void wnga_release_update_block(Integer g_a, Integer iblock)
{
    ++count_pnga_release_update_block;
    pnga_release_update_block(g_a, iblock);
}


void wnga_release_update_block_grid(Integer g_a, Integer *index)
{
    ++count_pnga_release_update_block_grid;
    pnga_release_update_block_grid(g_a, index);
}


void wnga_release_update_block_segment(Integer g_a, Integer iproc)
{
    ++count_pnga_release_update_block_segment;
    pnga_release_update_block_segment(g_a, iproc);
}


void wnga_release_update_ghost_element(Integer g_a, Integer subscript[])
{
    ++count_pnga_release_update_ghost_element;
    pnga_release_update_ghost_element(g_a, subscript);
}


void wnga_release_update_ghosts(Integer g_a)
{
    ++count_pnga_release_update_ghosts;
    pnga_release_update_ghosts(g_a);
}


void wnga_scale(Integer g_a, void *alpha)
{
    ++count_pnga_scale;
    pnga_scale(g_a, alpha);
}


void wnga_scale_cols(Integer g_a, Integer g_v)
{
    ++count_pnga_scale_cols;
    pnga_scale_cols(g_a, g_v);
}


void wnga_scale_patch(Integer g_a, Integer *lo, Integer *hi, void *alpha)
{
    ++count_pnga_scale_patch;
    pnga_scale_patch(g_a, lo, hi, alpha);
}


void wnga_scale_rows(Integer g_a, Integer g_v)
{
    ++count_pnga_scale_rows;
    pnga_scale_rows(g_a, g_v);
}


void wnga_scan_add(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi, Integer excl)
{
    ++count_pnga_scan_add;
    pnga_scan_add(g_a, g_b, g_sbit, lo, hi, excl);
}


void wnga_scan_copy(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi)
{
    ++count_pnga_scan_copy;
    pnga_scan_copy(g_a, g_b, g_sbit, lo, hi);
}


void wnga_scatter(Integer g_a, void *v, Integer *subscript, Integer nv)
{
    ++count_pnga_scatter;
    pnga_scatter(g_a, v, subscript, nv);
}


void wnga_scatter2d(Integer g_a, void *v, Integer *i, Integer *j, Integer nv)
{
    ++count_pnga_scatter2d;
    pnga_scatter2d(g_a, v, i, j, nv);
}


void wnga_scatter_acc(Integer g_a, void *v, Integer subscript[], Integer nv, void *alpha)
{
    ++count_pnga_scatter_acc;
    pnga_scatter_acc(g_a, v, subscript, nv, alpha);
}


void wnga_scatter_acc2d(Integer g_a, void *v, Integer *i, Integer *j, Integer nv, void *alpha)
{
    ++count_pnga_scatter_acc2d;
    pnga_scatter_acc2d(g_a, v, i, j, nv, alpha);
}


void wnga_select_elem(Integer g_a, char *op, void *val, Integer *subscript)
{
    ++count_pnga_select_elem;
    pnga_select_elem(g_a, op, val, subscript);
}


void wnga_set_array_name(Integer g_a, char *array_name)
{
    ++count_pnga_set_array_name;
    pnga_set_array_name(g_a, array_name);
}


void wnga_set_block_cyclic(Integer g_a, Integer *dims)
{
    ++count_pnga_set_block_cyclic;
    pnga_set_block_cyclic(g_a, dims);
}


void wnga_set_block_cyclic_proc_grid(Integer g_a, Integer *dims, Integer *proc_grid)
{
    ++count_pnga_set_block_cyclic_proc_grid;
    pnga_set_block_cyclic_proc_grid(g_a, dims, proc_grid);
}


void wnga_set_chunk(Integer g_a, Integer *chunk)
{
    ++count_pnga_set_chunk;
    pnga_set_chunk(g_a, chunk);
}


void wnga_set_data(Integer g_a, Integer ndim, Integer *dims, Integer type)
{
    ++count_pnga_set_data;
    pnga_set_data(g_a, ndim, dims, type);
}


void wnga_set_debug(logical flag)
{
    ++count_pnga_set_debug;
    pnga_set_debug(flag);
}


void wnga_set_diagonal(Integer g_a, Integer g_v)
{
    ++count_pnga_set_diagonal;
    pnga_set_diagonal(g_a, g_v);
}


void wnga_set_ghost_corner_flag(Integer g_a, logical flag)
{
    ++count_pnga_set_ghost_corner_flag;
    pnga_set_ghost_corner_flag(g_a, flag);
}


logical wnga_set_ghost_info(Integer g_a)
{
    ++count_pnga_set_ghost_info;
    return pnga_set_ghost_info(g_a);
}


void wnga_set_ghosts(Integer g_a, Integer *width)
{
    ++count_pnga_set_ghosts;
    pnga_set_ghosts(g_a, width);
}


void wnga_set_irreg_distr(Integer g_a, Integer *mapc, Integer *nblock)
{
    ++count_pnga_set_irreg_distr;
    pnga_set_irreg_distr(g_a, mapc, nblock);
}


void wnga_set_irreg_flag(Integer g_a, logical flag)
{
    ++count_pnga_set_irreg_flag;
    pnga_set_irreg_flag(g_a, flag);
}


void wnga_set_memory_limit(Integer mem_limit)
{
    ++count_pnga_set_memory_limit;
    pnga_set_memory_limit(mem_limit);
}


void wnga_set_pgroup(Integer g_a, Integer p_handle)
{
    ++count_pnga_set_pgroup;
    pnga_set_pgroup(g_a, p_handle);
}


void wnga_set_restricted(Integer g_a, Integer *list, Integer size)
{
    ++count_pnga_set_restricted;
    pnga_set_restricted(g_a, list, size);
}


void wnga_set_restricted_range(Integer g_a, Integer lo_proc, Integer hi_proc)
{
    ++count_pnga_set_restricted_range;
    pnga_set_restricted_range(g_a, lo_proc, hi_proc);
}


logical wnga_set_update4_info(Integer g_a)
{
    ++count_pnga_set_update4_info;
    return pnga_set_update4_info(g_a);
}


logical wnga_set_update5_info(Integer g_a)
{
    ++count_pnga_set_update5_info;
    return pnga_set_update5_info(g_a);
}


void wnga_shift_diagonal(Integer g_a, void *c)
{
    ++count_pnga_shift_diagonal;
    pnga_shift_diagonal(g_a, c);
}


Integer wnga_solve(Integer g_a, Integer g_b)
{
    ++count_pnga_solve;
    return pnga_solve(g_a, g_b);
}


Integer wnga_spd_invert(Integer g_a)
{
    ++count_pnga_spd_invert;
    return pnga_spd_invert(g_a);
}


void wnga_step_bound_info(Integer g_xx, Integer g_vv, Integer g_xxll, Integer g_xxuu, void *boundmin, void *wolfemin, void *boundmax)
{
    ++count_pnga_step_bound_info;
    pnga_step_bound_info(g_xx, g_vv, g_xxll, g_xxuu, boundmin, wolfemin, boundmax);
}


void wnga_step_bound_info_patch(Integer g_xx, Integer *xxlo, Integer *xxhi, Integer g_vv, Integer *vvlo, Integer *vvhi, Integer g_xxll, Integer *xxlllo, Integer *xxllhi, Integer g_xxuu, Integer *xxuulo, Integer *xxuuhi, void *boundmin, void *wolfemin, void *boundmax)
{
    ++count_pnga_step_bound_info_patch;
    pnga_step_bound_info_patch(g_xx, xxlo, xxhi, g_vv, vvlo, vvhi, g_xxll, xxlllo, xxllhi, g_xxuu, xxuulo, xxuuhi, boundmin, wolfemin, boundmax);
}


void wnga_step_mask_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    ++count_pnga_step_mask_patch;
    pnga_step_mask_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_step_max(Integer g_a, Integer g_b, void *retval)
{
    ++count_pnga_step_max;
    pnga_step_max(g_a, g_b, retval);
}


void wnga_step_max_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, void *result)
{
    ++count_pnga_step_max_patch;
    pnga_step_max_patch(g_a, alo, ahi, g_b, blo, bhi, result);
}


void wnga_strided_acc(Integer g_a, Integer *lo, Integer *hi, Integer *skip, void *buf, Integer *ld, void *alpha)
{
    ++count_pnga_strided_acc;
    pnga_strided_acc(g_a, lo, hi, skip, buf, ld, alpha);
}


void wnga_strided_get(Integer g_a, Integer *lo, Integer *hi, Integer *skip, void *buf, Integer *ld)
{
    ++count_pnga_strided_get;
    pnga_strided_get(g_a, lo, hi, skip, buf, ld);
}


void wnga_strided_put(Integer g_a, Integer *lo, Integer *hi, Integer *skip, void *buf, Integer *ld)
{
    ++count_pnga_strided_put;
    pnga_strided_put(g_a, lo, hi, skip, buf, ld);
}


void wnga_summarize(Integer verbose)
{
    ++count_pnga_summarize;
    pnga_summarize(verbose);
}


void wnga_symmetrize(Integer g_a)
{
    ++count_pnga_symmetrize;
    pnga_symmetrize(g_a);
}


void wnga_sync()
{
    ++count_pnga_sync;
    pnga_sync();
}


double wnga_timer()
{
    ++count_pnga_timer;
    return pnga_timer();
}


Integer wnga_total_blocks(Integer g_a)
{
    ++count_pnga_total_blocks;
    return pnga_total_blocks(g_a);
}


void wnga_transpose(Integer g_a, Integer g_b)
{
    ++count_pnga_transpose;
    pnga_transpose(g_a, g_b);
}


Integer wnga_type_c2f(Integer type)
{
    ++count_pnga_type_c2f;
    return pnga_type_c2f(type);
}


Integer wnga_type_f2c(Integer type)
{
    ++count_pnga_type_f2c;
    return pnga_type_f2c(type);
}


void wnga_unlock(Integer mutex)
{
    ++count_pnga_unlock;
    pnga_unlock(mutex);
}


void wnga_unpack(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi, Integer *icount)
{
    ++count_pnga_unpack;
    pnga_unpack(g_a, g_b, g_sbit, lo, hi, icount);
}


void wnga_update1_ghosts(Integer g_a)
{
    ++count_pnga_update1_ghosts;
    pnga_update1_ghosts(g_a);
}


logical wnga_update2_ghosts(Integer g_a)
{
    ++count_pnga_update2_ghosts;
    return pnga_update2_ghosts(g_a);
}


logical wnga_update3_ghosts(Integer g_a)
{
    ++count_pnga_update3_ghosts;
    return pnga_update3_ghosts(g_a);
}


logical wnga_update44_ghosts(Integer g_a)
{
    ++count_pnga_update44_ghosts;
    return pnga_update44_ghosts(g_a);
}


logical wnga_update4_ghosts(Integer g_a)
{
    ++count_pnga_update4_ghosts;
    return pnga_update4_ghosts(g_a);
}


logical wnga_update55_ghosts(Integer g_a)
{
    ++count_pnga_update55_ghosts;
    return pnga_update55_ghosts(g_a);
}


logical wnga_update5_ghosts(Integer g_a)
{
    ++count_pnga_update5_ghosts;
    return pnga_update5_ghosts(g_a);
}


logical wnga_update6_ghosts(Integer g_a)
{
    ++count_pnga_update6_ghosts;
    return pnga_update6_ghosts(g_a);
}


logical wnga_update7_ghosts(Integer g_a)
{
    ++count_pnga_update7_ghosts;
    return pnga_update7_ghosts(g_a);
}


logical wnga_update_ghost_dir(Integer g_a, Integer pdim, Integer pdir, logical pflag)
{
    ++count_pnga_update_ghost_dir;
    return pnga_update_ghost_dir(g_a, pdim, pdir, pflag);
}


void wnga_update_ghosts(Integer g_a)
{
    ++count_pnga_update_ghosts;
    pnga_update_ghosts(g_a);
}


logical wnga_uses_ma()
{
    ++count_pnga_uses_ma;
    return pnga_uses_ma();
}


logical wnga_uses_proc_grid(Integer g_a)
{
    ++count_pnga_uses_proc_grid;
    return pnga_uses_proc_grid(g_a);
}


logical wnga_valid_handle(Integer g_a)
{
    ++count_pnga_valid_handle;
    return pnga_valid_handle(g_a);
}


Integer wnga_verify_handle(Integer g_a)
{
    ++count_pnga_verify_handle;
    return pnga_verify_handle(g_a);
}


DoublePrecision wnga_wtime()
{
    ++count_pnga_wtime;
    return pnga_wtime();
}


void wnga_zero(Integer g_a)
{
    ++count_pnga_zero;
    pnga_zero(g_a);
}


void wnga_zero_diagonal(Integer g_a)
{
    ++count_pnga_zero_diagonal;
    pnga_zero_diagonal(g_a);
}


void wnga_zero_patch(Integer g_a, Integer *lo, Integer *hi)
{
    ++count_pnga_zero_patch;
    pnga_zero_patch(g_a, lo, hi);
}

void wnga_terminate()
{
    ++count_pnga_terminate;
    /* don't dump info if terminate more than once */
    if (1 == count_pnga_terminate) {

        if (count_pnga_abs_value) {
            printf("pnga_abs_value %ld\n", count_pnga_abs_value);
        }

        if (count_pnga_abs_value_patch) {
            printf("pnga_abs_value_patch %ld\n", count_pnga_abs_value_patch);
        }

        if (count_pnga_acc) {
            printf("pnga_acc %ld\n", count_pnga_acc);
        }

        if (count_pnga_access_block_grid_idx) {
            printf("pnga_access_block_grid_idx %ld\n", count_pnga_access_block_grid_idx);
        }

        if (count_pnga_access_block_grid_ptr) {
            printf("pnga_access_block_grid_ptr %ld\n", count_pnga_access_block_grid_ptr);
        }

        if (count_pnga_access_block_idx) {
            printf("pnga_access_block_idx %ld\n", count_pnga_access_block_idx);
        }

        if (count_pnga_access_block_ptr) {
            printf("pnga_access_block_ptr %ld\n", count_pnga_access_block_ptr);
        }

        if (count_pnga_access_block_segment_idx) {
            printf("pnga_access_block_segment_idx %ld\n", count_pnga_access_block_segment_idx);
        }

        if (count_pnga_access_block_segment_ptr) {
            printf("pnga_access_block_segment_ptr %ld\n", count_pnga_access_block_segment_ptr);
        }

        if (count_pnga_access_ghost_element) {
            printf("pnga_access_ghost_element %ld\n", count_pnga_access_ghost_element);
        }

        if (count_pnga_access_ghost_element_ptr) {
            printf("pnga_access_ghost_element_ptr %ld\n", count_pnga_access_ghost_element_ptr);
        }

        if (count_pnga_access_ghost_ptr) {
            printf("pnga_access_ghost_ptr %ld\n", count_pnga_access_ghost_ptr);
        }

        if (count_pnga_access_ghosts) {
            printf("pnga_access_ghosts %ld\n", count_pnga_access_ghosts);
        }

        if (count_pnga_access_idx) {
            printf("pnga_access_idx %ld\n", count_pnga_access_idx);
        }

        if (count_pnga_access_ptr) {
            printf("pnga_access_ptr %ld\n", count_pnga_access_ptr);
        }

        if (count_pnga_add) {
            printf("pnga_add %ld\n", count_pnga_add);
        }

        if (count_pnga_add_constant) {
            printf("pnga_add_constant %ld\n", count_pnga_add_constant);
        }

        if (count_pnga_add_constant_patch) {
            printf("pnga_add_constant_patch %ld\n", count_pnga_add_constant_patch);
        }

        if (count_pnga_add_diagonal) {
            printf("pnga_add_diagonal %ld\n", count_pnga_add_diagonal);
        }

        if (count_pnga_add_patch) {
            printf("pnga_add_patch %ld\n", count_pnga_add_patch);
        }

        if (count_pnga_allocate) {
            printf("pnga_allocate %ld\n", count_pnga_allocate);
        }

        if (count_pnga_bin_index) {
            printf("pnga_bin_index %ld\n", count_pnga_bin_index);
        }

        if (count_pnga_bin_sorter) {
            printf("pnga_bin_sorter %ld\n", count_pnga_bin_sorter);
        }

        if (count_pnga_brdcst) {
            printf("pnga_brdcst %ld\n", count_pnga_brdcst);
        }

        if (count_pnga_check_handle) {
            printf("pnga_check_handle %ld\n", count_pnga_check_handle);
        }

        if (count_pnga_cluster_nnodes) {
            printf("pnga_cluster_nnodes %ld\n", count_pnga_cluster_nnodes);
        }

        if (count_pnga_cluster_nodeid) {
            printf("pnga_cluster_nodeid %ld\n", count_pnga_cluster_nodeid);
        }

        if (count_pnga_cluster_nprocs) {
            printf("pnga_cluster_nprocs %ld\n", count_pnga_cluster_nprocs);
        }

        if (count_pnga_cluster_proc_nodeid) {
            printf("pnga_cluster_proc_nodeid %ld\n", count_pnga_cluster_proc_nodeid);
        }

        if (count_pnga_cluster_procid) {
            printf("pnga_cluster_procid %ld\n", count_pnga_cluster_procid);
        }

        if (count_pnga_comp_patch) {
            printf("pnga_comp_patch %ld\n", count_pnga_comp_patch);
        }

        if (count_pnga_compare_distr) {
            printf("pnga_compare_distr %ld\n", count_pnga_compare_distr);
        }

        if (count_pnga_copy) {
            printf("pnga_copy %ld\n", count_pnga_copy);
        }

        if (count_pnga_copy_patch) {
            printf("pnga_copy_patch %ld\n", count_pnga_copy_patch);
        }

        if (count_pnga_copy_patch_dp) {
            printf("pnga_copy_patch_dp %ld\n", count_pnga_copy_patch_dp);
        }

        if (count_pnga_create) {
            printf("pnga_create %ld\n", count_pnga_create);
        }

        if (count_pnga_create_bin_range) {
            printf("pnga_create_bin_range %ld\n", count_pnga_create_bin_range);
        }

        if (count_pnga_create_config) {
            printf("pnga_create_config %ld\n", count_pnga_create_config);
        }

        if (count_pnga_create_ghosts) {
            printf("pnga_create_ghosts %ld\n", count_pnga_create_ghosts);
        }

        if (count_pnga_create_ghosts_config) {
            printf("pnga_create_ghosts_config %ld\n", count_pnga_create_ghosts_config);
        }

        if (count_pnga_create_ghosts_irreg) {
            printf("pnga_create_ghosts_irreg %ld\n", count_pnga_create_ghosts_irreg);
        }

        if (count_pnga_create_ghosts_irreg_config) {
            printf("pnga_create_ghosts_irreg_config %ld\n", count_pnga_create_ghosts_irreg_config);
        }

        if (count_pnga_create_handle) {
            printf("pnga_create_handle %ld\n", count_pnga_create_handle);
        }

        if (count_pnga_create_irreg) {
            printf("pnga_create_irreg %ld\n", count_pnga_create_irreg);
        }

        if (count_pnga_create_irreg_config) {
            printf("pnga_create_irreg_config %ld\n", count_pnga_create_irreg_config);
        }

        if (count_pnga_create_mutexes) {
            printf("pnga_create_mutexes %ld\n", count_pnga_create_mutexes);
        }

        if (count_pnga_ddot_patch_dp) {
            printf("pnga_ddot_patch_dp %ld\n", count_pnga_ddot_patch_dp);
        }

        if (count_pnga_destroy) {
            printf("pnga_destroy %ld\n", count_pnga_destroy);
        }

        if (count_pnga_destroy_mutexes) {
            printf("pnga_destroy_mutexes %ld\n", count_pnga_destroy_mutexes);
        }

        if (count_pnga_diag) {
            printf("pnga_diag %ld\n", count_pnga_diag);
        }

        if (count_pnga_diag_reuse) {
            printf("pnga_diag_reuse %ld\n", count_pnga_diag_reuse);
        }

        if (count_pnga_diag_seq) {
            printf("pnga_diag_seq %ld\n", count_pnga_diag_seq);
        }

        if (count_pnga_diag_std) {
            printf("pnga_diag_std %ld\n", count_pnga_diag_std);
        }

        if (count_pnga_diag_std_seq) {
            printf("pnga_diag_std_seq %ld\n", count_pnga_diag_std_seq);
        }

        if (count_pnga_distribution) {
            printf("pnga_distribution %ld\n", count_pnga_distribution);
        }

        if (count_pnga_dot) {
            printf("pnga_dot %ld\n", count_pnga_dot);
        }

        if (count_pnga_dot_patch) {
            printf("pnga_dot_patch %ld\n", count_pnga_dot_patch);
        }

        if (count_pnga_duplicate) {
            printf("pnga_duplicate %ld\n", count_pnga_duplicate);
        }

        if (count_pnga_elem_divide) {
            printf("pnga_elem_divide %ld\n", count_pnga_elem_divide);
        }

        if (count_pnga_elem_divide_patch) {
            printf("pnga_elem_divide_patch %ld\n", count_pnga_elem_divide_patch);
        }

        if (count_pnga_elem_maximum) {
            printf("pnga_elem_maximum %ld\n", count_pnga_elem_maximum);
        }

        if (count_pnga_elem_maximum_patch) {
            printf("pnga_elem_maximum_patch %ld\n", count_pnga_elem_maximum_patch);
        }

        if (count_pnga_elem_minimum) {
            printf("pnga_elem_minimum %ld\n", count_pnga_elem_minimum);
        }

        if (count_pnga_elem_minimum_patch) {
            printf("pnga_elem_minimum_patch %ld\n", count_pnga_elem_minimum_patch);
        }

        if (count_pnga_elem_multiply) {
            printf("pnga_elem_multiply %ld\n", count_pnga_elem_multiply);
        }

        if (count_pnga_elem_multiply_patch) {
            printf("pnga_elem_multiply_patch %ld\n", count_pnga_elem_multiply_patch);
        }

        if (count_pnga_elem_step_divide_patch) {
            printf("pnga_elem_step_divide_patch %ld\n", count_pnga_elem_step_divide_patch);
        }

        if (count_pnga_elem_stepb_divide_patch) {
            printf("pnga_elem_stepb_divide_patch %ld\n", count_pnga_elem_stepb_divide_patch);
        }

        if (count_pnga_error) {
            printf("pnga_error %ld\n", count_pnga_error);
        }

        if (count_pnga_fence) {
            printf("pnga_fence %ld\n", count_pnga_fence);
        }

        if (count_pnga_fill) {
            printf("pnga_fill %ld\n", count_pnga_fill);
        }

        if (count_pnga_fill_patch) {
            printf("pnga_fill_patch %ld\n", count_pnga_fill_patch);
        }

        if (count_pnga_gather) {
            printf("pnga_gather %ld\n", count_pnga_gather);
        }

        if (count_pnga_gather2d) {
            printf("pnga_gather2d %ld\n", count_pnga_gather2d);
        }

        if (count_pnga_get) {
            printf("pnga_get %ld\n", count_pnga_get);
        }

        if (count_pnga_get_block_info) {
            printf("pnga_get_block_info %ld\n", count_pnga_get_block_info);
        }

        if (count_pnga_get_debug) {
            printf("pnga_get_debug %ld\n", count_pnga_get_debug);
        }

        if (count_pnga_get_diag) {
            printf("pnga_get_diag %ld\n", count_pnga_get_diag);
        }

        if (count_pnga_get_dimension) {
            printf("pnga_get_dimension %ld\n", count_pnga_get_dimension);
        }

        if (count_pnga_get_ghost_block) {
            printf("pnga_get_ghost_block %ld\n", count_pnga_get_ghost_block);
        }

        if (count_pnga_get_pgroup) {
            printf("pnga_get_pgroup %ld\n", count_pnga_get_pgroup);
        }

        if (count_pnga_get_pgroup_size) {
            printf("pnga_get_pgroup_size %ld\n", count_pnga_get_pgroup_size);
        }

        if (count_pnga_get_proc_grid) {
            printf("pnga_get_proc_grid %ld\n", count_pnga_get_proc_grid);
        }

        if (count_pnga_get_proc_index) {
            printf("pnga_get_proc_index %ld\n", count_pnga_get_proc_index);
        }

        if (count_pnga_ghost_barrier) {
            printf("pnga_ghost_barrier %ld\n", count_pnga_ghost_barrier);
        }

        if (count_pnga_gop) {
            printf("pnga_gop %ld\n", count_pnga_gop);
        }

        if (count_pnga_has_ghosts) {
            printf("pnga_has_ghosts %ld\n", count_pnga_has_ghosts);
        }

        if (count_pnga_init_fence) {
            printf("pnga_init_fence %ld\n", count_pnga_init_fence);
        }

        if (count_pnga_initialize) {
            printf("pnga_initialize %ld\n", count_pnga_initialize);
        }

        if (count_pnga_initialize_ltd) {
            printf("pnga_initialize_ltd %ld\n", count_pnga_initialize_ltd);
        }

        if (count_pnga_inquire) {
            printf("pnga_inquire %ld\n", count_pnga_inquire);
        }

        if (count_pnga_inquire_memory) {
            printf("pnga_inquire_memory %ld\n", count_pnga_inquire_memory);
        }

        if (count_pnga_inquire_name) {
            printf("pnga_inquire_name %ld\n", count_pnga_inquire_name);
        }

        if (count_pnga_inquire_type) {
            printf("pnga_inquire_type %ld\n", count_pnga_inquire_type);
        }

        if (count_pnga_is_mirrored) {
            printf("pnga_is_mirrored %ld\n", count_pnga_is_mirrored);
        }

        if (count_pnga_list_nodeid) {
            printf("pnga_list_nodeid %ld\n", count_pnga_list_nodeid);
        }

        if (count_pnga_llt_solve) {
            printf("pnga_llt_solve %ld\n", count_pnga_llt_solve);
        }

        if (count_pnga_locate) {
            printf("pnga_locate %ld\n", count_pnga_locate);
        }

        if (count_pnga_locate_nnodes) {
            printf("pnga_locate_nnodes %ld\n", count_pnga_locate_nnodes);
        }

        if (count_pnga_locate_num_blocks) {
            printf("pnga_locate_num_blocks %ld\n", count_pnga_locate_num_blocks);
        }

        if (count_pnga_locate_region) {
            printf("pnga_locate_region %ld\n", count_pnga_locate_region);
        }

        if (count_pnga_lock) {
            printf("pnga_lock %ld\n", count_pnga_lock);
        }

        if (count_pnga_lu_solve) {
            printf("pnga_lu_solve %ld\n", count_pnga_lu_solve);
        }

        if (count_pnga_lu_solve_alt) {
            printf("pnga_lu_solve_alt %ld\n", count_pnga_lu_solve_alt);
        }

        if (count_pnga_lu_solve_seq) {
            printf("pnga_lu_solve_seq %ld\n", count_pnga_lu_solve_seq);
        }

        if (count_pnga_mask_sync) {
            printf("pnga_mask_sync %ld\n", count_pnga_mask_sync);
        }

        if (count_pnga_matmul) {
            printf("pnga_matmul %ld\n", count_pnga_matmul);
        }

        if (count_pnga_matmul_mirrored) {
            printf("pnga_matmul_mirrored %ld\n", count_pnga_matmul_mirrored);
        }

        if (count_pnga_matmul_patch) {
            printf("pnga_matmul_patch %ld\n", count_pnga_matmul_patch);
        }

        if (count_pnga_median) {
            printf("pnga_median %ld\n", count_pnga_median);
        }

        if (count_pnga_median_patch) {
            printf("pnga_median_patch %ld\n", count_pnga_median_patch);
        }

        if (count_pnga_memory_avail) {
            printf("pnga_memory_avail %ld\n", count_pnga_memory_avail);
        }

        if (count_pnga_memory_avail_type) {
            printf("pnga_memory_avail_type %ld\n", count_pnga_memory_avail_type);
        }

        if (count_pnga_memory_limited) {
            printf("pnga_memory_limited %ld\n", count_pnga_memory_limited);
        }

        if (count_pnga_merge_distr_patch) {
            printf("pnga_merge_distr_patch %ld\n", count_pnga_merge_distr_patch);
        }

        if (count_pnga_merge_mirrored) {
            printf("pnga_merge_mirrored %ld\n", count_pnga_merge_mirrored);
        }

        if (count_pnga_msg_brdcst) {
            printf("pnga_msg_brdcst %ld\n", count_pnga_msg_brdcst);
        }

        if (count_pnga_msg_pgroup_sync) {
            printf("pnga_msg_pgroup_sync %ld\n", count_pnga_msg_pgroup_sync);
        }

        if (count_pnga_msg_sync) {
            printf("pnga_msg_sync %ld\n", count_pnga_msg_sync);
        }

        if (count_pnga_nbacc) {
            printf("pnga_nbacc %ld\n", count_pnga_nbacc);
        }

        if (count_pnga_nbget) {
            printf("pnga_nbget %ld\n", count_pnga_nbget);
        }

        if (count_pnga_nbget_ghost_dir) {
            printf("pnga_nbget_ghost_dir %ld\n", count_pnga_nbget_ghost_dir);
        }

        if (count_pnga_nblock) {
            printf("pnga_nblock %ld\n", count_pnga_nblock);
        }

        if (count_pnga_nbput) {
            printf("pnga_nbput %ld\n", count_pnga_nbput);
        }

        if (count_pnga_nbtest) {
            printf("pnga_nbtest %ld\n", count_pnga_nbtest);
        }

        if (count_pnga_nbwait) {
            printf("pnga_nbwait %ld\n", count_pnga_nbwait);
        }

        if (count_pnga_ndim) {
            printf("pnga_ndim %ld\n", count_pnga_ndim);
        }

        if (count_pnga_nnodes) {
            printf("pnga_nnodes %ld\n", count_pnga_nnodes);
        }

        if (count_pnga_nodeid) {
            printf("pnga_nodeid %ld\n", count_pnga_nodeid);
        }

        if (count_pnga_norm1) {
            printf("pnga_norm1 %ld\n", count_pnga_norm1);
        }

        if (count_pnga_norm_infinity) {
            printf("pnga_norm_infinity %ld\n", count_pnga_norm_infinity);
        }

        if (count_pnga_pack) {
            printf("pnga_pack %ld\n", count_pnga_pack);
        }

        if (count_pnga_patch_enum) {
            printf("pnga_patch_enum %ld\n", count_pnga_patch_enum);
        }

        if (count_pnga_patch_intersect) {
            printf("pnga_patch_intersect %ld\n", count_pnga_patch_intersect);
        }

        if (count_pnga_periodic) {
            printf("pnga_periodic %ld\n", count_pnga_periodic);
        }

        if (count_pnga_pgroup_absolute_id) {
            printf("pnga_pgroup_absolute_id %ld\n", count_pnga_pgroup_absolute_id);
        }

        if (count_pnga_pgroup_brdcst) {
            printf("pnga_pgroup_brdcst %ld\n", count_pnga_pgroup_brdcst);
        }

        if (count_pnga_pgroup_create) {
            printf("pnga_pgroup_create %ld\n", count_pnga_pgroup_create);
        }

        if (count_pnga_pgroup_destroy) {
            printf("pnga_pgroup_destroy %ld\n", count_pnga_pgroup_destroy);
        }

        if (count_pnga_pgroup_get_default) {
            printf("pnga_pgroup_get_default %ld\n", count_pnga_pgroup_get_default);
        }

        if (count_pnga_pgroup_get_mirror) {
            printf("pnga_pgroup_get_mirror %ld\n", count_pnga_pgroup_get_mirror);
        }

        if (count_pnga_pgroup_get_world) {
            printf("pnga_pgroup_get_world %ld\n", count_pnga_pgroup_get_world);
        }

        if (count_pnga_pgroup_gop) {
            printf("pnga_pgroup_gop %ld\n", count_pnga_pgroup_gop);
        }

        if (count_pnga_pgroup_nnodes) {
            printf("pnga_pgroup_nnodes %ld\n", count_pnga_pgroup_nnodes);
        }

        if (count_pnga_pgroup_nodeid) {
            printf("pnga_pgroup_nodeid %ld\n", count_pnga_pgroup_nodeid);
        }

        if (count_pnga_pgroup_set_default) {
            printf("pnga_pgroup_set_default %ld\n", count_pnga_pgroup_set_default);
        }

        if (count_pnga_pgroup_split) {
            printf("pnga_pgroup_split %ld\n", count_pnga_pgroup_split);
        }

        if (count_pnga_pgroup_split_irreg) {
            printf("pnga_pgroup_split_irreg %ld\n", count_pnga_pgroup_split_irreg);
        }

        if (count_pnga_pgroup_sync) {
            printf("pnga_pgroup_sync %ld\n", count_pnga_pgroup_sync);
        }

        if (count_pnga_print) {
            printf("pnga_print %ld\n", count_pnga_print);
        }

        if (count_pnga_print_distribution) {
            printf("pnga_print_distribution %ld\n", count_pnga_print_distribution);
        }

        if (count_pnga_print_file) {
            printf("pnga_print_file %ld\n", count_pnga_print_file);
        }

        if (count_pnga_print_patch) {
            printf("pnga_print_patch %ld\n", count_pnga_print_patch);
        }

        if (count_pnga_print_patch2d) {
            printf("pnga_print_patch2d %ld\n", count_pnga_print_patch2d);
        }

        if (count_pnga_print_patch_file) {
            printf("pnga_print_patch_file %ld\n", count_pnga_print_patch_file);
        }

        if (count_pnga_print_patch_file2d) {
            printf("pnga_print_patch_file2d %ld\n", count_pnga_print_patch_file2d);
        }

        if (count_pnga_print_stats) {
            printf("pnga_print_stats %ld\n", count_pnga_print_stats);
        }

        if (count_pnga_proc_topology) {
            printf("pnga_proc_topology %ld\n", count_pnga_proc_topology);
        }

        if (count_pnga_put) {
            printf("pnga_put %ld\n", count_pnga_put);
        }

        if (count_pnga_randomize) {
            printf("pnga_randomize %ld\n", count_pnga_randomize);
        }

        if (count_pnga_read_inc) {
            printf("pnga_read_inc %ld\n", count_pnga_read_inc);
        }

        if (count_pnga_recip) {
            printf("pnga_recip %ld\n", count_pnga_recip);
        }

        if (count_pnga_recip_patch) {
            printf("pnga_recip_patch %ld\n", count_pnga_recip_patch);
        }

        if (count_pnga_release) {
            printf("pnga_release %ld\n", count_pnga_release);
        }

        if (count_pnga_release_block) {
            printf("pnga_release_block %ld\n", count_pnga_release_block);
        }

        if (count_pnga_release_block_grid) {
            printf("pnga_release_block_grid %ld\n", count_pnga_release_block_grid);
        }

        if (count_pnga_release_block_segment) {
            printf("pnga_release_block_segment %ld\n", count_pnga_release_block_segment);
        }

        if (count_pnga_release_ghost_element) {
            printf("pnga_release_ghost_element %ld\n", count_pnga_release_ghost_element);
        }

        if (count_pnga_release_ghosts) {
            printf("pnga_release_ghosts %ld\n", count_pnga_release_ghosts);
        }

        if (count_pnga_release_update) {
            printf("pnga_release_update %ld\n", count_pnga_release_update);
        }

        if (count_pnga_release_update_block) {
            printf("pnga_release_update_block %ld\n", count_pnga_release_update_block);
        }

        if (count_pnga_release_update_block_grid) {
            printf("pnga_release_update_block_grid %ld\n", count_pnga_release_update_block_grid);
        }

        if (count_pnga_release_update_block_segment) {
            printf("pnga_release_update_block_segment %ld\n", count_pnga_release_update_block_segment);
        }

        if (count_pnga_release_update_ghost_element) {
            printf("pnga_release_update_ghost_element %ld\n", count_pnga_release_update_ghost_element);
        }

        if (count_pnga_release_update_ghosts) {
            printf("pnga_release_update_ghosts %ld\n", count_pnga_release_update_ghosts);
        }

        if (count_pnga_scale) {
            printf("pnga_scale %ld\n", count_pnga_scale);
        }

        if (count_pnga_scale_cols) {
            printf("pnga_scale_cols %ld\n", count_pnga_scale_cols);
        }

        if (count_pnga_scale_patch) {
            printf("pnga_scale_patch %ld\n", count_pnga_scale_patch);
        }

        if (count_pnga_scale_rows) {
            printf("pnga_scale_rows %ld\n", count_pnga_scale_rows);
        }

        if (count_pnga_scan_add) {
            printf("pnga_scan_add %ld\n", count_pnga_scan_add);
        }

        if (count_pnga_scan_copy) {
            printf("pnga_scan_copy %ld\n", count_pnga_scan_copy);
        }

        if (count_pnga_scatter) {
            printf("pnga_scatter %ld\n", count_pnga_scatter);
        }

        if (count_pnga_scatter2d) {
            printf("pnga_scatter2d %ld\n", count_pnga_scatter2d);
        }

        if (count_pnga_scatter_acc) {
            printf("pnga_scatter_acc %ld\n", count_pnga_scatter_acc);
        }

        if (count_pnga_scatter_acc2d) {
            printf("pnga_scatter_acc2d %ld\n", count_pnga_scatter_acc2d);
        }

        if (count_pnga_select_elem) {
            printf("pnga_select_elem %ld\n", count_pnga_select_elem);
        }

        if (count_pnga_set_array_name) {
            printf("pnga_set_array_name %ld\n", count_pnga_set_array_name);
        }

        if (count_pnga_set_block_cyclic) {
            printf("pnga_set_block_cyclic %ld\n", count_pnga_set_block_cyclic);
        }

        if (count_pnga_set_block_cyclic_proc_grid) {
            printf("pnga_set_block_cyclic_proc_grid %ld\n", count_pnga_set_block_cyclic_proc_grid);
        }

        if (count_pnga_set_chunk) {
            printf("pnga_set_chunk %ld\n", count_pnga_set_chunk);
        }

        if (count_pnga_set_data) {
            printf("pnga_set_data %ld\n", count_pnga_set_data);
        }

        if (count_pnga_set_debug) {
            printf("pnga_set_debug %ld\n", count_pnga_set_debug);
        }

        if (count_pnga_set_diagonal) {
            printf("pnga_set_diagonal %ld\n", count_pnga_set_diagonal);
        }

        if (count_pnga_set_ghost_corner_flag) {
            printf("pnga_set_ghost_corner_flag %ld\n", count_pnga_set_ghost_corner_flag);
        }

        if (count_pnga_set_ghost_info) {
            printf("pnga_set_ghost_info %ld\n", count_pnga_set_ghost_info);
        }

        if (count_pnga_set_ghosts) {
            printf("pnga_set_ghosts %ld\n", count_pnga_set_ghosts);
        }

        if (count_pnga_set_irreg_distr) {
            printf("pnga_set_irreg_distr %ld\n", count_pnga_set_irreg_distr);
        }

        if (count_pnga_set_irreg_flag) {
            printf("pnga_set_irreg_flag %ld\n", count_pnga_set_irreg_flag);
        }

        if (count_pnga_set_memory_limit) {
            printf("pnga_set_memory_limit %ld\n", count_pnga_set_memory_limit);
        }

        if (count_pnga_set_pgroup) {
            printf("pnga_set_pgroup %ld\n", count_pnga_set_pgroup);
        }

        if (count_pnga_set_restricted) {
            printf("pnga_set_restricted %ld\n", count_pnga_set_restricted);
        }

        if (count_pnga_set_restricted_range) {
            printf("pnga_set_restricted_range %ld\n", count_pnga_set_restricted_range);
        }

        if (count_pnga_set_update4_info) {
            printf("pnga_set_update4_info %ld\n", count_pnga_set_update4_info);
        }

        if (count_pnga_set_update5_info) {
            printf("pnga_set_update5_info %ld\n", count_pnga_set_update5_info);
        }

        if (count_pnga_shift_diagonal) {
            printf("pnga_shift_diagonal %ld\n", count_pnga_shift_diagonal);
        }

        if (count_pnga_solve) {
            printf("pnga_solve %ld\n", count_pnga_solve);
        }

        if (count_pnga_spd_invert) {
            printf("pnga_spd_invert %ld\n", count_pnga_spd_invert);
        }

        if (count_pnga_step_bound_info) {
            printf("pnga_step_bound_info %ld\n", count_pnga_step_bound_info);
        }

        if (count_pnga_step_bound_info_patch) {
            printf("pnga_step_bound_info_patch %ld\n", count_pnga_step_bound_info_patch);
        }

        if (count_pnga_step_mask_patch) {
            printf("pnga_step_mask_patch %ld\n", count_pnga_step_mask_patch);
        }

        if (count_pnga_step_max) {
            printf("pnga_step_max %ld\n", count_pnga_step_max);
        }

        if (count_pnga_step_max_patch) {
            printf("pnga_step_max_patch %ld\n", count_pnga_step_max_patch);
        }

        if (count_pnga_strided_acc) {
            printf("pnga_strided_acc %ld\n", count_pnga_strided_acc);
        }

        if (count_pnga_strided_get) {
            printf("pnga_strided_get %ld\n", count_pnga_strided_get);
        }

        if (count_pnga_strided_put) {
            printf("pnga_strided_put %ld\n", count_pnga_strided_put);
        }

        if (count_pnga_summarize) {
            printf("pnga_summarize %ld\n", count_pnga_summarize);
        }

        if (count_pnga_symmetrize) {
            printf("pnga_symmetrize %ld\n", count_pnga_symmetrize);
        }

        if (count_pnga_sync) {
            printf("pnga_sync %ld\n", count_pnga_sync);
        }

        if (count_pnga_terminate) {
            printf("pnga_terminate %ld\n", count_pnga_terminate);
        }

        if (count_pnga_timer) {
            printf("pnga_timer %ld\n", count_pnga_timer);
        }

        if (count_pnga_total_blocks) {
            printf("pnga_total_blocks %ld\n", count_pnga_total_blocks);
        }

        if (count_pnga_transpose) {
            printf("pnga_transpose %ld\n", count_pnga_transpose);
        }

        if (count_pnga_type_c2f) {
            printf("pnga_type_c2f %ld\n", count_pnga_type_c2f);
        }

        if (count_pnga_type_f2c) {
            printf("pnga_type_f2c %ld\n", count_pnga_type_f2c);
        }

        if (count_pnga_unlock) {
            printf("pnga_unlock %ld\n", count_pnga_unlock);
        }

        if (count_pnga_unpack) {
            printf("pnga_unpack %ld\n", count_pnga_unpack);
        }

        if (count_pnga_update1_ghosts) {
            printf("pnga_update1_ghosts %ld\n", count_pnga_update1_ghosts);
        }

        if (count_pnga_update2_ghosts) {
            printf("pnga_update2_ghosts %ld\n", count_pnga_update2_ghosts);
        }

        if (count_pnga_update3_ghosts) {
            printf("pnga_update3_ghosts %ld\n", count_pnga_update3_ghosts);
        }

        if (count_pnga_update44_ghosts) {
            printf("pnga_update44_ghosts %ld\n", count_pnga_update44_ghosts);
        }

        if (count_pnga_update4_ghosts) {
            printf("pnga_update4_ghosts %ld\n", count_pnga_update4_ghosts);
        }

        if (count_pnga_update55_ghosts) {
            printf("pnga_update55_ghosts %ld\n", count_pnga_update55_ghosts);
        }

        if (count_pnga_update5_ghosts) {
            printf("pnga_update5_ghosts %ld\n", count_pnga_update5_ghosts);
        }

        if (count_pnga_update6_ghosts) {
            printf("pnga_update6_ghosts %ld\n", count_pnga_update6_ghosts);
        }

        if (count_pnga_update7_ghosts) {
            printf("pnga_update7_ghosts %ld\n", count_pnga_update7_ghosts);
        }

        if (count_pnga_update_ghost_dir) {
            printf("pnga_update_ghost_dir %ld\n", count_pnga_update_ghost_dir);
        }

        if (count_pnga_update_ghosts) {
            printf("pnga_update_ghosts %ld\n", count_pnga_update_ghosts);
        }

        if (count_pnga_uses_ma) {
            printf("pnga_uses_ma %ld\n", count_pnga_uses_ma);
        }

        if (count_pnga_uses_proc_grid) {
            printf("pnga_uses_proc_grid %ld\n", count_pnga_uses_proc_grid);
        }

        if (count_pnga_valid_handle) {
            printf("pnga_valid_handle %ld\n", count_pnga_valid_handle);
        }

        if (count_pnga_verify_handle) {
            printf("pnga_verify_handle %ld\n", count_pnga_verify_handle);
        }

        if (count_pnga_wtime) {
            printf("pnga_wtime %ld\n", count_pnga_wtime);
        }

        if (count_pnga_zero) {
            printf("pnga_zero %ld\n", count_pnga_zero);
        }

        if (count_pnga_zero_diagonal) {
            printf("pnga_zero_diagonal %ld\n", count_pnga_zero_diagonal);
        }

        if (count_pnga_zero_patch) {
            printf("pnga_zero_patch %ld\n", count_pnga_zero_patch);
        }

    }
}

