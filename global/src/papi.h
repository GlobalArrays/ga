#ifndef PAPI_H_
#define PAPI_H_

#include "typesf2c.h"

/* Routines from base.c */
extern logical pnga_allocate(Integer *g_a);
extern logical pnga_compare_distr(Integer *g_a, Integer *g_b);
extern logical pnga_create(Integer type, Integer ndim,
                           Integer *dims, char* name,
                           Integer *chunk, Integer *g_a);
extern logical pnga_create_config(Integer type, Integer ndim,
                                  Integer *dims, char* name,
                                  Integer *chunk, Integer p_handle,
                                  Integer *g_a);
extern logical pnga_create_ghosts(Integer type, Integer ndim,
                                  Integer *dims, Integer *width, char* name,
                                  Integer *chunk, Integer *g_a);
extern logical pnga_create_ghosts_irreg(Integer type, Integer ndim,
                                        Integer *dims, Integer *width,
                                        char* name,
                                        Integer *map, Integer *block,
                                        Integer *g_a);
extern logical pnga_create_ghosts_irreg_config(Integer type, Integer ndim,
                                               Integer *dims, Integer *width,
                                               char* name,
                                               Integer *map, Integer *block,
                                               Integer p_handle, Integer *g_a);
extern logical pnga_create_ghosts_config(Integer type, Integer ndim,
                                         Integer *dims, Integer *width,
                                         char* name,
                                         Integer *chunk, Integer p_handle,
                                         Integer *g_a);
extern logical pnga_create_irreg(Integer type, Integer ndim,
                                 Integer *dims, char* name,
                                 Integer *map, Integer *block, Integer *g_a);
extern logical pnga_create_irreg_config(Integer type, Integer ndim,
                                        Integer *dims, char* name,
                                        Integer *map,
                                        Integer *block, Integer p_handle,
                                        Integer *g_a);
extern Integer pnga_create_handle();
extern logical pnga_create_mutexes(Integer *num);
extern logical pnga_destroy(Integer *g_a);
extern logical pnga_destroy_mutexes();
extern void pnga_distribution(Integer *g_a, Integer *proc, Integer *lo, Integer *hi);
extern logical pnga_duplicate(Integer *g_a, Integer *g_b, char *array_name);
extern void pnga_fill(Integer *g_a, void* val);
extern void pnga_get_block_info(Integer *g_a, Integer *num_blocks,
                                Integer *block_dims);
extern logical pnga_get_debug();
extern Integer pnga_get_dimension(Integer *g_a);
extern void pnga_get_proc_grid(Integer *g_a, Integer *dims);
extern void pnga_get_proc_index(Integer *g_a, Integer *iproc, Integer *index);
extern logical pnga_has_ghosts(Integer *g_a);
extern void pnga_initialize();
extern void pnga_initialize_ltd(Integer *limit);
extern void pnga_inquire(Integer *g_a, Integer *type, Integer *ndim,
                         Integer *dims);
extern Integer pnga_inquire_memory();
extern void pnga_inquire_name(Integer *g_a, char **array_name);
extern logical pnga_is_mirrored(Integer *g_a);
extern void pnga_list_nodeid(Integer *list, Integer *nprocs);
extern logical pnga_locate(Integer *g_a, Integer *subscript, Integer *owner);
extern Integer pnga_locate_num_blocks(Integer *g_a, Integer *lo, Integer *hi);
extern logical pnga_locate_region(Integer *g_a, Integer *lo, Integer *hi,
                                  Integer *map, Integer *proclist, Integer *np);
extern void pnga_lock(Integer *mutex);
extern Integer pnga_ndim(Integer *g_a);
extern void pnga_mask_sync(Integer *begin, Integer *end);
extern Integer pnga_memory_avail();
extern logical pnga_memory_limited();
extern void pnga_merge_distr_patch(Integer *g_a, Integer *alo, Integer *ahi,
                                   Integer *g_b, Integer *blo, Integer *bhi);
extern void pnga_merge_mirrored(Integer *g_a);
extern void pnga_nblock(Integer *g_a, Integer *nblock);

extern Integer pnga_nnodes();
extern Integer pnga_nodeid();
extern Integer pnga_pgroup_absolute_id(Integer *grp, Integer *pid);
extern Integer pnga_pgroup_create(Integer *list, Integer *count);
extern logical pnga_pgroup_destroy(Integer *grp);
extern Integer pnga_pgroup_get_default();
extern Integer pnga_pgroup_get_mirror();
extern Integer pnga_pgroup_get_world();
extern void pnga_pgroup_set_default(Integer *grp);
extern Integer pnga_pgroup_split(Integer *grp, Integer *grp_num);
extern Integer pnga_pgroup_split_irreg(Integer *grp, Integer *mycolor);
extern Integer pnga_pgroup_nnodes(Integer *grp);
extern Integer pnga_pgroup_nodeid(Integer *grp);
extern void pnga_proc_topology(Integer* g_a, Integer* proc, Integer* subscript);
extern void pnga_randomize(Integer *g_a, void* val);
extern Integer pnga_get_pgroup(Integer *g_a);
extern Integer pnga_get_pgroup_size(Integer *grp_id);
extern void pnga_set_array_name(Integer *g_a, char *array_name);
extern void pnga_set_block_cyclic(Integer *g_a, Integer *dims);
extern void pnga_set_block_cyclic_proc_grid(Integer *g_a, Integer *dims, Integer *proc_grid);
extern void pnga_set_chunk(Integer *g_a, Integer *chunk);
extern void pnga_set_data(Integer *g_a, Integer *ndim, Integer *dims, Integer *type);
extern void pnga_set_debug(logical *flag);
extern void pnga_set_ghosts(Integer *g_a, Integer *width);
extern void pnga_set_irreg_distr(Integer *g_a, Integer *mapc, Integer *nblock);
extern void pnga_set_irreg_flag(Integer *g_a, logical *flag);
extern void pnga_set_memory_limit(Integer *mem_limit);
extern void pnga_set_pgroup(Integer *g_a, Integer *p_handle);
extern void pnga_set_restricted(Integer *g_a, Integer *list, Integer *size);
extern void pnga_set_restricted_range(Integer *g_a, Integer *lo_proc, Integer *hi_proc);
extern void pnga_terminate();
extern Integer pnga_total_blocks(Integer *g_a);

/* Routines from onesided.c */
extern void pnga_nbput(Integer *g_a, Integer *lo, Integer *hi, void *buf,
                       Integer *ld, Integer *nbhandle);
extern void pnga_put(Integer *g_a, Integer *lo, Integer *hi, void *buf,
                     Integer *ld);

/* Routines from global.util.c */
extern void pnga_error(char *string, Integer icode);

/* Routines from datatypes.c */
extern Integer pnga_type_f2c(Integer type);
extern Integer pnga_type_c2f(Integer type);

/* Routines from collect.c */
void pnga_msg_brdcst(Integer type, void *buffer, Integer len, Integer root);
void pnga_brdcst(Integer *type, void *buf, Integer *len, Integer *originator);
void pnga_pgroup_brdcst(Integer *grp_id, Integer *type, void *buf, Integer *len, Integer *originator);
void pnga_msg_sync();
void pnga_msg_pgroup_sync(Integer *grp_id);
void pnga_pgroup_gop(Integer p_grp, Integer type, void *x, Integer n, char *op);
void pnga_gop(Integer type, void *x, Integer n, char *op);

/* Routines from elem_alg.c */
void pnga_abs_value_patch(Integer *g_a, Integer *lo, Integer *hi);
void pnga_recip_patch(Integer *g_a, Integer *lo, Integer *hi);
void pnga_add_constant_patch(Integer *g_a, Integer *lo, Integer *hi, void *alpha);
void pnga_abs_value(Integer *g_a);
void pnga_add_constant(Integer *g_a, void *alpha);
void pnga_recip(Integer *g_a);
void pnga_elem_multiply(Integer *g_a, Integer *g_b, Integer *g_c);
void pnga_elem_divide(Integer *g_a, Integer *g_b, Integer *g_c);
void pnga_elem_maximum(Integer *g_a, Integer *g_b, Integer *g_c);
void pnga_elem_minimum(Integer *g_a, Integer *g_b, Integer *g_c);
void pnga_elem_multiply_patch(Integer *g_a,Integer *alo,Integer *ahi,Integer *g_b,Integer *blo,Integer *bhi,Integer *g_c,Integer *clo,Integer *chi);
void pnga_elem_divide_patch(Integer *g_a,Integer *alo,Integer *ahi,Integer *g_b,Integer *blo,Integer *bhi,Integer *g_c,Integer *clo,Integer *chi);
void pnga_elem_maximum_patch(Integer *g_a,Integer *alo,Integer *ahi,Integer *g_b,Integer *blo,Integer *bhi,Integer *g_c,Integer *clo,Integer *chi);
void pnga_elem_minimum_patch(Integer *g_a,Integer *alo,Integer *ahi,Integer *g_b,Integer *blo,Integer *bhi,Integer *g_c,Integer *clo,Integer *chi);
void pnga_elem_step_divide_patch(Integer *g_a,Integer *alo,Integer *ahi, Integer *g_b,Integer *blo,Integer *bhi,Integer *g_c, Integer *clo,Integer *chi);
void pnga_elem_stepb_divide_patch(Integer *g_a,Integer *alo,Integer *ahi, Integer *g_b,Integer *blo,Integer *bhi,Integer *g_c, Integer *clo,Integer *chi);
void pnga_step_mask_patch(Integer *g_a,Integer *alo,Integer *ahi, Integer *g_b,Integer *blo,Integer *bhi,Integer *g_c, Integer *clo,Integer *chi);
void pnga_step_bound_info_patch(Integer *g_xx, Integer *xxlo, Integer *xxhi, Integer *g_vv, Integer *vvlo, Integer *vvhi, Integer *g_xxll, Integer *xxlllo, Integer *xxllhi, Integer *g_xxuu, Integer *xxuulo, Integer *xxuuhi, void *boundmin, void* wolfemin, void *boundmax);
void pnga_step_max_patch(Integer *g_a, Integer *alo, Integer *ahi, Integer *g_b, Integer *blo, Integer *bhi, void *result);
void pnga_step_max(Integer *g_a, Integer *g_b, void *retval);
void pnga_step_bound_info(Integer *g_xx, Integer *g_vv, Integer *g_xxll, Integer *g_xxuu, void *boundmin, void *wolfemin, void *boundmax);

#endif /* PAPI_H_ */
