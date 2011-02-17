
#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "papi.h"
#include "typesf2c.h"


void wnga_abs_value(Integer g_a)
{
    pnga_abs_value(g_a);
}


void wnga_abs_value_patch(Integer g_a, Integer *lo, Integer *hi)
{
    pnga_abs_value_patch(g_a, lo, hi);
}


void wnga_acc(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha)
{
    pnga_acc(g_a, lo, hi, buf, ld, alpha);
}


void wnga_access_block_grid_idx(Integer g_a, Integer *subscript, AccessIndex *index, Integer *ld)
{
    pnga_access_block_grid_idx(g_a, subscript, index, ld);
}


void wnga_access_block_grid_ptr(Integer g_a, Integer *index, void *ptr, Integer *ld)
{
    pnga_access_block_grid_ptr(g_a, index, ptr, ld);
}


void wnga_access_block_idx(Integer g_a, Integer idx, AccessIndex *index, Integer *ld)
{
    pnga_access_block_idx(g_a, idx, index, ld);
}


void wnga_access_block_ptr(Integer g_a, Integer idx, void *ptr, Integer *ld)
{
    pnga_access_block_ptr(g_a, idx, ptr, ld);
}


void wnga_access_block_segment_idx(Integer g_a, Integer proc, AccessIndex *index, Integer *len)
{
    pnga_access_block_segment_idx(g_a, proc, index, len);
}


void wnga_access_block_segment_ptr(Integer g_a, Integer proc, void *ptr, Integer *len)
{
    pnga_access_block_segment_ptr(g_a, proc, ptr, len);
}


void wnga_access_ghost_element(Integer g_a, AccessIndex *index, Integer subscript[], Integer ld[])
{
    pnga_access_ghost_element(g_a, index, subscript, ld);
}


void wnga_access_ghost_element_ptr(Integer g_a, void *ptr, Integer subscript[], Integer ld[])
{
    pnga_access_ghost_element_ptr(g_a, ptr, subscript, ld);
}


void wnga_access_ghost_ptr(Integer g_a, Integer dims[], void *ptr, Integer ld[])
{
    pnga_access_ghost_ptr(g_a, dims, ptr, ld);
}


void wnga_access_ghosts(Integer g_a, Integer dims[], AccessIndex *index, Integer ld[])
{
    pnga_access_ghosts(g_a, dims, index, ld);
}


void wnga_access_idx(Integer g_a, Integer *lo, Integer *hi, AccessIndex *index, Integer *ld)
{
    pnga_access_idx(g_a, lo, hi, index, ld);
}


void wnga_access_ptr(Integer g_a, Integer *lo, Integer *hi, void *ptr, Integer *ld)
{
    pnga_access_ptr(g_a, lo, hi, ptr, ld);
}


void wnga_add(void *alpha, Integer g_a, void *beta, Integer g_b, Integer g_c)
{
    pnga_add(alpha, g_a, beta, g_b, g_c);
}


void wnga_add_constant(Integer g_a, void *alpha)
{
    pnga_add_constant(g_a, alpha);
}


void wnga_add_constant_patch(Integer g_a, Integer *lo, Integer *hi, void *alpha)
{
    pnga_add_constant_patch(g_a, lo, hi, alpha);
}


void wnga_add_diagonal(Integer g_a, Integer g_v)
{
    pnga_add_diagonal(g_a, g_v);
}


void wnga_add_patch(void *alpha, Integer g_a, Integer *alo, Integer *ahi, void *beta, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    pnga_add_patch(alpha, g_a, alo, ahi, beta, g_b, blo, bhi, g_c, clo, chi);
}


logical wnga_allocate(Integer g_a)
{
    return pnga_allocate(g_a);
}


void wnga_bin_index(Integer g_bin, Integer g_cnt, Integer g_off, Integer *values, Integer *subs, Integer n, Integer sortit)
{
    pnga_bin_index(g_bin, g_cnt, g_off, values, subs, n, sortit);
}


void wnga_bin_sorter(Integer g_bin, Integer g_cnt, Integer g_off)
{
    pnga_bin_sorter(g_bin, g_cnt, g_off);
}


void wnga_brdcst(Integer type, void *buf, Integer len, Integer originator)
{
    pnga_brdcst(type, buf, len, originator);
}


void wnga_check_handle(Integer g_a, char *string)
{
    pnga_check_handle(g_a, string);
}


Integer wnga_cluster_nnodes()
{
    return pnga_cluster_nnodes();
}


Integer wnga_cluster_nodeid()
{
    return pnga_cluster_nodeid();
}


Integer wnga_cluster_nprocs(Integer node)
{
    return pnga_cluster_nprocs(node);
}


Integer wnga_cluster_proc_nodeid(Integer proc)
{
    return pnga_cluster_proc_nodeid(proc);
}


Integer wnga_cluster_procid(Integer node, Integer loc_proc_id)
{
    return pnga_cluster_procid(node, loc_proc_id);
}


logical wnga_comp_patch(Integer andim, Integer *alo, Integer *ahi, Integer bndim, Integer *blo, Integer *bhi)
{
    return pnga_comp_patch(andim, alo, ahi, bndim, blo, bhi);
}


logical wnga_compare_distr(Integer g_a, Integer g_b)
{
    return pnga_compare_distr(g_a, g_b);
}


void wnga_copy(Integer g_a, Integer g_b)
{
    pnga_copy(g_a, g_b);
}


void wnga_copy_patch(char *trans, Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi)
{
    pnga_copy_patch(trans, g_a, alo, ahi, g_b, blo, bhi);
}


void wnga_copy_patch_dp(char *t_a, Integer g_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi)
{
    pnga_copy_patch_dp(t_a, g_a, ailo, aihi, ajlo, ajhi, g_b, bilo, bihi, bjlo, bjhi);
}


logical wnga_create(Integer type, Integer ndim, Integer *dims, char *name, Integer *chunk, Integer *g_a)
{
    return pnga_create(type, ndim, dims, name, chunk, g_a);
}


logical wnga_create_bin_range(Integer g_bin, Integer g_cnt, Integer g_off, Integer *g_range)
{
    return pnga_create_bin_range(g_bin, g_cnt, g_off, g_range);
}


logical wnga_create_config(Integer type, Integer ndim, Integer *dims, char *name, Integer *chunk, Integer p_handle, Integer *g_a)
{
    return pnga_create_config(type, ndim, dims, name, chunk, p_handle, g_a);
}


logical wnga_create_ghosts(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *chunk, Integer *g_a)
{
    return pnga_create_ghosts(type, ndim, dims, width, name, chunk, g_a);
}


logical wnga_create_ghosts_config(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *chunk, Integer p_handle, Integer *g_a)
{
    return pnga_create_ghosts_config(type, ndim, dims, width, name, chunk, p_handle, g_a);
}


logical wnga_create_ghosts_irreg(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *map, Integer *block, Integer *g_a)
{
    return pnga_create_ghosts_irreg(type, ndim, dims, width, name, map, block, g_a);
}


logical wnga_create_ghosts_irreg_config(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *map, Integer *block, Integer p_handle, Integer *g_a)
{
    return pnga_create_ghosts_irreg_config(type, ndim, dims, width, name, map, block, p_handle, g_a);
}


Integer wnga_create_handle()
{
    return pnga_create_handle();
}


logical wnga_create_irreg(Integer type, Integer ndim, Integer *dims, char *name, Integer *map, Integer *block, Integer *g_a)
{
    return pnga_create_irreg(type, ndim, dims, name, map, block, g_a);
}


logical wnga_create_irreg_config(Integer type, Integer ndim, Integer *dims, char *name, Integer *map, Integer *block, Integer p_handle, Integer *g_a)
{
    return pnga_create_irreg_config(type, ndim, dims, name, map, block, p_handle, g_a);
}


logical wnga_create_mutexes(Integer num)
{
    return pnga_create_mutexes(num);
}


DoublePrecision wnga_ddot_patch_dp(Integer g_a, char *t_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, char *t_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi)
{
    return pnga_ddot_patch_dp(g_a, t_a, ailo, aihi, ajlo, ajhi, g_b, t_b, bilo, bihi, bjlo, bjhi);
}


logical wnga_destroy(Integer g_a)
{
    return pnga_destroy(g_a);
}


logical wnga_destroy_mutexes()
{
    return pnga_destroy_mutexes();
}


void wnga_diag(Integer g_a, Integer g_s, Integer g_v, DoublePrecision *eval)
{
    pnga_diag(g_a, g_s, g_v, eval);
}


void wnga_diag_reuse(Integer reuse, Integer g_a, Integer g_s, Integer g_v, DoublePrecision *eval)
{
    pnga_diag_reuse(reuse, g_a, g_s, g_v, eval);
}


void wnga_diag_seq(Integer g_a, Integer g_s, Integer g_v, DoublePrecision *eval)
{
    pnga_diag_seq(g_a, g_s, g_v, eval);
}


void wnga_diag_std(Integer g_a, Integer g_v, DoublePrecision *eval)
{
    pnga_diag_std(g_a, g_v, eval);
}


void wnga_diag_std_seq(Integer g_a, Integer g_v, DoublePrecision *eval)
{
    pnga_diag_std_seq(g_a, g_v, eval);
}


void wnga_distribution(Integer g_a, Integer proc, Integer *lo, Integer *hi)
{
    pnga_distribution(g_a, proc, lo, hi);
}


void wnga_dot(int type, Integer g_a, Integer g_b, void *value)
{
    pnga_dot(type, g_a, g_b, value);
}


void wnga_dot_patch(Integer g_a, char *t_a, Integer *alo, Integer *ahi, Integer g_b, char *t_b, Integer *blo, Integer *bhi, void *retval)
{
    pnga_dot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi, retval);
}


logical wnga_duplicate(Integer g_a, Integer *g_b, char *array_name)
{
    return pnga_duplicate(g_a, g_b, array_name);
}


void wnga_elem_divide(Integer g_a, Integer g_b, Integer g_c)
{
    pnga_elem_divide(g_a, g_b, g_c);
}


void wnga_elem_divide_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    pnga_elem_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_elem_maximum(Integer g_a, Integer g_b, Integer g_c)
{
    pnga_elem_maximum(g_a, g_b, g_c);
}


void wnga_elem_maximum_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    pnga_elem_maximum_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_elem_minimum(Integer g_a, Integer g_b, Integer g_c)
{
    pnga_elem_minimum(g_a, g_b, g_c);
}


void wnga_elem_minimum_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    pnga_elem_minimum_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_elem_multiply(Integer g_a, Integer g_b, Integer g_c)
{
    pnga_elem_multiply(g_a, g_b, g_c);
}


void wnga_elem_multiply_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    pnga_elem_multiply_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_elem_step_divide_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    pnga_elem_step_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_elem_stepb_divide_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    pnga_elem_stepb_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_error(char *string, Integer icode)
{
    pnga_error(string, icode);
}


void wnga_fence()
{
    pnga_fence();
}


void wnga_fill(Integer g_a, void *val)
{
    pnga_fill(g_a, val);
}


void wnga_fill_patch(Integer g_a, Integer *lo, Integer *hi, void *val)
{
    pnga_fill_patch(g_a, lo, hi, val);
}


void wnga_gather(Integer g_a, void *v, Integer subscript[], Integer nv)
{
    pnga_gather(g_a, v, subscript, nv);
}


void wnga_gather2d(Integer g_a, void *v, Integer *i, Integer *j, Integer nv)
{
    pnga_gather2d(g_a, v, i, j, nv);
}


void wnga_get(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
    pnga_get(g_a, lo, hi, buf, ld);
}


void wnga_get_block_info(Integer g_a, Integer *num_blocks, Integer *block_dims)
{
    pnga_get_block_info(g_a, num_blocks, block_dims);
}


logical wnga_get_debug()
{
    return pnga_get_debug();
}


void wnga_get_diag(Integer g_a, Integer g_v)
{
    pnga_get_diag(g_a, g_v);
}


Integer wnga_get_dimension(Integer g_a)
{
    return pnga_get_dimension(g_a);
}


void wnga_get_ghost_block(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
    pnga_get_ghost_block(g_a, lo, hi, buf, ld);
}


Integer wnga_get_pgroup(Integer g_a)
{
    return pnga_get_pgroup(g_a);
}


Integer wnga_get_pgroup_size(Integer grp_id)
{
    return pnga_get_pgroup_size(grp_id);
}


void wnga_get_proc_grid(Integer g_a, Integer *dims)
{
    pnga_get_proc_grid(g_a, dims);
}


void wnga_get_proc_index(Integer g_a, Integer iproc, Integer *index)
{
    pnga_get_proc_index(g_a, iproc, index);
}


void wnga_ghost_barrier()
{
    pnga_ghost_barrier();
}


void wnga_gop(Integer type, void *x, Integer n, char *op)
{
    pnga_gop(type, x, n, op);
}


logical wnga_has_ghosts(Integer g_a)
{
    return pnga_has_ghosts(g_a);
}


void wnga_init_fence()
{
    pnga_init_fence();
}


void wnga_initialize()
{
    pnga_initialize();
}


void wnga_initialize_ltd(Integer limit)
{
    pnga_initialize_ltd(limit);
}


void wnga_inquire(Integer g_a, Integer *type, Integer *ndim, Integer *dims)
{
    pnga_inquire(g_a, type, ndim, dims);
}


Integer wnga_inquire_memory()
{
    return pnga_inquire_memory();
}


void wnga_inquire_name(Integer g_a, char **array_name)
{
    pnga_inquire_name(g_a, array_name);
}


void wnga_inquire_type(Integer g_a, Integer *type)
{
    pnga_inquire_type(g_a, type);
}


logical wnga_is_mirrored(Integer g_a)
{
    return pnga_is_mirrored(g_a);
}


void wnga_list_nodeid(Integer *list, Integer nprocs)
{
    pnga_list_nodeid(list, nprocs);
}


Integer wnga_llt_solve(Integer g_a, Integer g_b)
{
    return pnga_llt_solve(g_a, g_b);
}


logical wnga_locate(Integer g_a, Integer *subscript, Integer *owner)
{
    return pnga_locate(g_a, subscript, owner);
}


logical wnga_locate_nnodes(Integer g_a, Integer *lo, Integer *hi, Integer *np)
{
    return pnga_locate_nnodes(g_a, lo, hi, np);
}


Integer wnga_locate_num_blocks(Integer g_a, Integer *lo, Integer *hi)
{
    return pnga_locate_num_blocks(g_a, lo, hi);
}


logical wnga_locate_region(Integer g_a, Integer *lo, Integer *hi, Integer *map, Integer *proclist, Integer *np)
{
    return pnga_locate_region(g_a, lo, hi, map, proclist, np);
}


void wnga_lock(Integer mutex)
{
    pnga_lock(mutex);
}


void wnga_lu_solve(char *tran, Integer g_a, Integer g_b)
{
    pnga_lu_solve(tran, g_a, g_b);
}


void wnga_lu_solve_alt(Integer tran, Integer g_a, Integer g_b)
{
    pnga_lu_solve_alt(tran, g_a, g_b);
}


void wnga_lu_solve_seq(char *trans, Integer g_a, Integer g_b)
{
    pnga_lu_solve_seq(trans, g_a, g_b);
}


void wnga_mask_sync(Integer begin, Integer end)
{
    pnga_mask_sync(begin, end);
}


void wnga_matmul(char *transa, char *transb, void *alpha, void *beta, Integer g_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi, Integer g_c, Integer cilo, Integer cihi, Integer cjlo, Integer cjhi)
{
    pnga_matmul(transa, transb, alpha, beta, g_a, ailo, aihi, ajlo, ajhi, g_b, bilo, bihi, bjlo, bjhi, g_c, cilo, cihi, cjlo, cjhi);
}


void wnga_matmul_mirrored(char *transa, char *transb, void *alpha, void *beta, Integer g_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi, Integer g_c, Integer cilo, Integer cihi, Integer cjlo, Integer cjhi)
{
    pnga_matmul_mirrored(transa, transb, alpha, beta, g_a, ailo, aihi, ajlo, ajhi, g_b, bilo, bihi, bjlo, bjhi, g_c, cilo, cihi, cjlo, cjhi);
}


void wnga_matmul_patch(char *transa, char *transb, void *alpha, void *beta, Integer g_a, Integer alo[], Integer ahi[], Integer g_b, Integer blo[], Integer bhi[], Integer g_c, Integer clo[], Integer chi[])
{
    pnga_matmul_patch(transa, transb, alpha, beta, g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_median(Integer g_a, Integer g_b, Integer g_c, Integer g_m)
{
    pnga_median(g_a, g_b, g_c, g_m);
}


void wnga_median_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi, Integer g_m, Integer *mlo, Integer *mhi)
{
    pnga_median_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi, g_m, mlo, mhi);
}


Integer wnga_memory_avail()
{
    return pnga_memory_avail();
}


Integer wnga_memory_avail_type(Integer datatype)
{
    return pnga_memory_avail_type(datatype);
}


logical wnga_memory_limited()
{
    return pnga_memory_limited();
}


void wnga_merge_distr_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi)
{
    pnga_merge_distr_patch(g_a, alo, ahi, g_b, blo, bhi);
}


void wnga_merge_mirrored(Integer g_a)
{
    pnga_merge_mirrored(g_a);
}


void wnga_msg_brdcst(Integer type, void *buffer, Integer len, Integer root)
{
    pnga_msg_brdcst(type, buffer, len, root);
}


void wnga_msg_pgroup_sync(Integer grp_id)
{
    pnga_msg_pgroup_sync(grp_id);
}


void wnga_msg_sync()
{
    pnga_msg_sync();
}


void wnga_nbacc(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha, Integer *nbhndl)
{
    pnga_nbacc(g_a, lo, hi, buf, ld, alpha, nbhndl);
}


void wnga_nbget(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle)
{
    pnga_nbget(g_a, lo, hi, buf, ld, nbhandle);
}


void wnga_nbget_ghost_dir(Integer g_a, Integer *mask, Integer *nbhandle)
{
    pnga_nbget_ghost_dir(g_a, mask, nbhandle);
}


void wnga_nblock(Integer g_a, Integer *nblock)
{
    pnga_nblock(g_a, nblock);
}


void wnga_nbput(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle)
{
    pnga_nbput(g_a, lo, hi, buf, ld, nbhandle);
}


Integer wnga_nbtest(Integer *nbhandle)
{
    return pnga_nbtest(nbhandle);
}


void wnga_nbwait(Integer *nbhandle)
{
    pnga_nbwait(nbhandle);
}


Integer wnga_ndim(Integer g_a)
{
    return pnga_ndim(g_a);
}


Integer wnga_nnodes()
{
    return pnga_nnodes();
}


Integer wnga_nodeid()
{
    return pnga_nodeid();
}


void wnga_norm1(Integer g_a, double *nm)
{
    pnga_norm1(g_a, nm);
}


void wnga_norm_infinity(Integer g_a, double *nm)
{
    pnga_norm_infinity(g_a, nm);
}


void wnga_pack(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi, Integer *icount)
{
    pnga_pack(g_a, g_b, g_sbit, lo, hi, icount);
}


void wnga_patch_enum(Integer g_a, Integer lo, Integer hi, void *start, void *stride)
{
    pnga_patch_enum(g_a, lo, hi, start, stride);
}


logical wnga_patch_intersect(Integer *lo, Integer *hi, Integer *lop, Integer *hip, Integer ndim)
{
    return pnga_patch_intersect(lo, hi, lop, hip, ndim);
}


void wnga_periodic(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha, Integer op_code)
{
    pnga_periodic(g_a, lo, hi, buf, ld, alpha, op_code);
}


Integer wnga_pgroup_absolute_id(Integer grp, Integer pid)
{
    return pnga_pgroup_absolute_id(grp, pid);
}


void wnga_pgroup_brdcst(Integer grp_id, Integer type, void *buf, Integer len, Integer originator)
{
    pnga_pgroup_brdcst(grp_id, type, buf, len, originator);
}


Integer wnga_pgroup_create(Integer *list, Integer count)
{
    return pnga_pgroup_create(list, count);
}


logical wnga_pgroup_destroy(Integer grp)
{
    return pnga_pgroup_destroy(grp);
}


Integer wnga_pgroup_get_default()
{
    return pnga_pgroup_get_default();
}


Integer wnga_pgroup_get_mirror()
{
    return pnga_pgroup_get_mirror();
}


Integer wnga_pgroup_get_world()
{
    return pnga_pgroup_get_world();
}


void wnga_pgroup_gop(Integer p_grp, Integer type, void *x, Integer n, char *op)
{
    pnga_pgroup_gop(p_grp, type, x, n, op);
}


Integer wnga_pgroup_nnodes(Integer grp)
{
    return pnga_pgroup_nnodes(grp);
}


Integer wnga_pgroup_nodeid(Integer grp)
{
    return pnga_pgroup_nodeid(grp);
}


void wnga_pgroup_set_default(Integer grp)
{
    pnga_pgroup_set_default(grp);
}


Integer wnga_pgroup_split(Integer grp, Integer grp_num)
{
    return pnga_pgroup_split(grp, grp_num);
}


Integer wnga_pgroup_split_irreg(Integer grp, Integer mycolor)
{
    return pnga_pgroup_split_irreg(grp, mycolor);
}


void wnga_pgroup_sync(Integer grp_id)
{
    pnga_pgroup_sync(grp_id);
}


void wnga_print(Integer g_a)
{
    pnga_print(g_a);
}


void wnga_print_distribution(int fstyle, Integer g_a)
{
    pnga_print_distribution(fstyle, g_a);
}


void wnga_print_file(FILE *file, Integer g_a)
{
    pnga_print_file(file, g_a);
}


void wnga_print_patch(Integer g_a, Integer *lo, Integer *hi, Integer pretty)
{
    pnga_print_patch(g_a, lo, hi, pretty);
}


void wnga_print_patch2d(Integer g_a, Integer ilo, Integer ihi, Integer jlo, Integer jhi, Integer pretty)
{
    pnga_print_patch2d(g_a, ilo, ihi, jlo, jhi, pretty);
}


void wnga_print_patch_file(FILE *file, Integer g_a, Integer *lo, Integer *hi, Integer pretty)
{
    pnga_print_patch_file(file, g_a, lo, hi, pretty);
}


void wnga_print_patch_file2d(FILE *file, Integer g_a, Integer ilo, Integer ihi, Integer jlo, Integer jhi, Integer pretty)
{
    pnga_print_patch_file2d(file, g_a, ilo, ihi, jlo, jhi, pretty);
}


void wnga_print_stats()
{
    pnga_print_stats();
}


void wnga_proc_topology(Integer g_a, Integer proc, Integer *subscript)
{
    pnga_proc_topology(g_a, proc, subscript);
}


void wnga_put(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
    pnga_put(g_a, lo, hi, buf, ld);
}


void wnga_randomize(Integer g_a, void *val)
{
    pnga_randomize(g_a, val);
}


Integer wnga_read_inc(Integer g_a, Integer *subscript, Integer inc)
{
    return pnga_read_inc(g_a, subscript, inc);
}


void wnga_recip(Integer g_a)
{
    pnga_recip(g_a);
}


void wnga_recip_patch(Integer g_a, Integer *lo, Integer *hi)
{
    pnga_recip_patch(g_a, lo, hi);
}


void wnga_release(Integer g_a, Integer *lo, Integer *hi)
{
    pnga_release(g_a, lo, hi);
}


void wnga_release_block(Integer g_a, Integer iblock)
{
    pnga_release_block(g_a, iblock);
}


void wnga_release_block_grid(Integer g_a, Integer *index)
{
    pnga_release_block_grid(g_a, index);
}


void wnga_release_block_segment(Integer g_a, Integer iproc)
{
    pnga_release_block_segment(g_a, iproc);
}


void wnga_release_ghost_element(Integer g_a, Integer subscript[])
{
    pnga_release_ghost_element(g_a, subscript);
}


void wnga_release_ghosts(Integer g_a)
{
    pnga_release_ghosts(g_a);
}


void wnga_release_update(Integer g_a, Integer *lo, Integer *hi)
{
    pnga_release_update(g_a, lo, hi);
}


void wnga_release_update_block(Integer g_a, Integer iblock)
{
    pnga_release_update_block(g_a, iblock);
}


void wnga_release_update_block_grid(Integer g_a, Integer *index)
{
    pnga_release_update_block_grid(g_a, index);
}


void wnga_release_update_block_segment(Integer g_a, Integer iproc)
{
    pnga_release_update_block_segment(g_a, iproc);
}


void wnga_release_update_ghost_element(Integer g_a, Integer subscript[])
{
    pnga_release_update_ghost_element(g_a, subscript);
}


void wnga_release_update_ghosts(Integer g_a)
{
    pnga_release_update_ghosts(g_a);
}


void wnga_scale(Integer g_a, void *alpha)
{
    pnga_scale(g_a, alpha);
}


void wnga_scale_cols(Integer g_a, Integer g_v)
{
    pnga_scale_cols(g_a, g_v);
}


void wnga_scale_patch(Integer g_a, Integer *lo, Integer *hi, void *alpha)
{
    pnga_scale_patch(g_a, lo, hi, alpha);
}


void wnga_scale_rows(Integer g_a, Integer g_v)
{
    pnga_scale_rows(g_a, g_v);
}


void wnga_scan_add(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi, Integer excl)
{
    pnga_scan_add(g_a, g_b, g_sbit, lo, hi, excl);
}


void wnga_scan_copy(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi)
{
    pnga_scan_copy(g_a, g_b, g_sbit, lo, hi);
}


void wnga_scatter(Integer g_a, void *v, Integer *subscript, Integer nv)
{
    pnga_scatter(g_a, v, subscript, nv);
}


void wnga_scatter2d(Integer g_a, void *v, Integer *i, Integer *j, Integer nv)
{
    pnga_scatter2d(g_a, v, i, j, nv);
}


void wnga_scatter_acc(Integer g_a, void *v, Integer subscript[], Integer nv, void *alpha)
{
    pnga_scatter_acc(g_a, v, subscript, nv, alpha);
}


void wnga_scatter_acc2d(Integer g_a, void *v, Integer *i, Integer *j, Integer nv, void *alpha)
{
    pnga_scatter_acc2d(g_a, v, i, j, nv, alpha);
}


void wnga_select_elem(Integer g_a, char *op, void *val, Integer *subscript)
{
    pnga_select_elem(g_a, op, val, subscript);
}


void wnga_set_array_name(Integer g_a, char *array_name)
{
    pnga_set_array_name(g_a, array_name);
}


void wnga_set_block_cyclic(Integer g_a, Integer *dims)
{
    pnga_set_block_cyclic(g_a, dims);
}


void wnga_set_block_cyclic_proc_grid(Integer g_a, Integer *dims, Integer *proc_grid)
{
    pnga_set_block_cyclic_proc_grid(g_a, dims, proc_grid);
}


void wnga_set_chunk(Integer g_a, Integer *chunk)
{
    pnga_set_chunk(g_a, chunk);
}


void wnga_set_data(Integer g_a, Integer ndim, Integer *dims, Integer type)
{
    pnga_set_data(g_a, ndim, dims, type);
}


void wnga_set_debug(logical flag)
{
    pnga_set_debug(flag);
}


void wnga_set_diagonal(Integer g_a, Integer g_v)
{
    pnga_set_diagonal(g_a, g_v);
}


void wnga_set_ghost_corner_flag(Integer g_a, logical flag)
{
    pnga_set_ghost_corner_flag(g_a, flag);
}


logical wnga_set_ghost_info(Integer g_a)
{
    return pnga_set_ghost_info(g_a);
}


void wnga_set_ghosts(Integer g_a, Integer *width)
{
    pnga_set_ghosts(g_a, width);
}


void wnga_set_irreg_distr(Integer g_a, Integer *mapc, Integer *nblock)
{
    pnga_set_irreg_distr(g_a, mapc, nblock);
}


void wnga_set_irreg_flag(Integer g_a, logical flag)
{
    pnga_set_irreg_flag(g_a, flag);
}


void wnga_set_memory_limit(Integer mem_limit)
{
    pnga_set_memory_limit(mem_limit);
}


void wnga_set_pgroup(Integer g_a, Integer p_handle)
{
    pnga_set_pgroup(g_a, p_handle);
}


void wnga_set_restricted(Integer g_a, Integer *list, Integer size)
{
    pnga_set_restricted(g_a, list, size);
}


void wnga_set_restricted_range(Integer g_a, Integer lo_proc, Integer hi_proc)
{
    pnga_set_restricted_range(g_a, lo_proc, hi_proc);
}


logical wnga_set_update4_info(Integer g_a)
{
    return pnga_set_update4_info(g_a);
}


logical wnga_set_update5_info(Integer g_a)
{
    return pnga_set_update5_info(g_a);
}


void wnga_shift_diagonal(Integer g_a, void *c)
{
    pnga_shift_diagonal(g_a, c);
}


Integer wnga_solve(Integer g_a, Integer g_b)
{
    return pnga_solve(g_a, g_b);
}


Integer wnga_spd_invert(Integer g_a)
{
    return pnga_spd_invert(g_a);
}


void wnga_step_bound_info(Integer g_xx, Integer g_vv, Integer g_xxll, Integer g_xxuu, void *boundmin, void *wolfemin, void *boundmax)
{
    pnga_step_bound_info(g_xx, g_vv, g_xxll, g_xxuu, boundmin, wolfemin, boundmax);
}


void wnga_step_bound_info_patch(Integer g_xx, Integer *xxlo, Integer *xxhi, Integer g_vv, Integer *vvlo, Integer *vvhi, Integer g_xxll, Integer *xxlllo, Integer *xxllhi, Integer g_xxuu, Integer *xxuulo, Integer *xxuuhi, void *boundmin, void *wolfemin, void *boundmax)
{
    pnga_step_bound_info_patch(g_xx, xxlo, xxhi, g_vv, vvlo, vvhi, g_xxll, xxlllo, xxllhi, g_xxuu, xxuulo, xxuuhi, boundmin, wolfemin, boundmax);
}


void wnga_step_mask_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    pnga_step_mask_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
}


void wnga_step_max(Integer g_a, Integer g_b, void *retval)
{
    pnga_step_max(g_a, g_b, retval);
}


void wnga_step_max_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, void *result)
{
    pnga_step_max_patch(g_a, alo, ahi, g_b, blo, bhi, result);
}


void wnga_strided_acc(Integer g_a, Integer *lo, Integer *hi, Integer *skip, void *buf, Integer *ld, void *alpha)
{
    pnga_strided_acc(g_a, lo, hi, skip, buf, ld, alpha);
}


void wnga_strided_get(Integer g_a, Integer *lo, Integer *hi, Integer *skip, void *buf, Integer *ld)
{
    pnga_strided_get(g_a, lo, hi, skip, buf, ld);
}


void wnga_strided_put(Integer g_a, Integer *lo, Integer *hi, Integer *skip, void *buf, Integer *ld)
{
    pnga_strided_put(g_a, lo, hi, skip, buf, ld);
}


void wnga_summarize(Integer verbose)
{
    pnga_summarize(verbose);
}


void wnga_symmetrize(Integer g_a)
{
    pnga_symmetrize(g_a);
}


void wnga_sync()
{
    pnga_sync();
}


void wnga_terminate()
{
    pnga_terminate();
}


double wnga_timer()
{
    return pnga_timer();
}


Integer wnga_total_blocks(Integer g_a)
{
    return pnga_total_blocks(g_a);
}


void wnga_transpose(Integer g_a, Integer g_b)
{
    pnga_transpose(g_a, g_b);
}


Integer wnga_type_c2f(Integer type)
{
    return pnga_type_c2f(type);
}


Integer wnga_type_f2c(Integer type)
{
    return pnga_type_f2c(type);
}


void wnga_unlock(Integer mutex)
{
    pnga_unlock(mutex);
}


void wnga_unpack(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi, Integer *icount)
{
    pnga_unpack(g_a, g_b, g_sbit, lo, hi, icount);
}


void wnga_update1_ghosts(Integer g_a)
{
    pnga_update1_ghosts(g_a);
}


logical wnga_update2_ghosts(Integer g_a)
{
    return pnga_update2_ghosts(g_a);
}


logical wnga_update3_ghosts(Integer g_a)
{
    return pnga_update3_ghosts(g_a);
}


logical wnga_update44_ghosts(Integer g_a)
{
    return pnga_update44_ghosts(g_a);
}


logical wnga_update4_ghosts(Integer g_a)
{
    return pnga_update4_ghosts(g_a);
}


logical wnga_update55_ghosts(Integer g_a)
{
    return pnga_update55_ghosts(g_a);
}


logical wnga_update5_ghosts(Integer g_a)
{
    return pnga_update5_ghosts(g_a);
}


logical wnga_update6_ghosts(Integer g_a)
{
    return pnga_update6_ghosts(g_a);
}


logical wnga_update7_ghosts(Integer g_a)
{
    return pnga_update7_ghosts(g_a);
}


logical wnga_update_ghost_dir(Integer g_a, Integer pdim, Integer pdir, logical pflag)
{
    return pnga_update_ghost_dir(g_a, pdim, pdir, pflag);
}


void wnga_update_ghosts(Integer g_a)
{
    pnga_update_ghosts(g_a);
}


logical wnga_uses_ma()
{
    return pnga_uses_ma();
}


logical wnga_uses_proc_grid(Integer g_a)
{
    return pnga_uses_proc_grid(g_a);
}


logical wnga_valid_handle(Integer g_a)
{
    return pnga_valid_handle(g_a);
}


Integer wnga_verify_handle(Integer g_a)
{
    return pnga_verify_handle(g_a);
}


DoublePrecision wnga_wtime()
{
    return pnga_wtime();
}


void wnga_zero(Integer g_a)
{
    pnga_zero(g_a);
}


void wnga_zero_diagonal(Integer g_a)
{
    pnga_zero_diagonal(g_a);
}


void wnga_zero_patch(Integer g_a, Integer *lo, Integer *hi)
{
    pnga_zero_patch(g_a, lo, hi);
}

