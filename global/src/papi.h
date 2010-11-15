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
                                  Integer *chunk, Integer p_handle, Integer *g_a);
extern logical pnga_create_ghosts(Integer type, Integer ndim,
                                  Integer *dims, Integer *width, char* name,
                                  Integer *chunk, Integer *g_a);
extern logical pnga_create_ghosts_irreg(Integer type, Integer ndim,
                                        Integer *dims, Integer *width, char* name,
                                        Integer *map, Integer *block, Integer *g_a);
extern logical pnga_create_ghosts_irreg_config(Integer type, Integer ndim,
                                               Integer *dims, Integer *width, char* name,
                                               Integer *map, Integer *block,
                                               Integer p_handle, Integer *g_a);
extern logical pnga_create_ghosts_config(Integer type, Integer ndim,
                                         Integer *dims, Integer *width, char* name,
                                         Integer *chunk, Integer p_handle, Integer *g_a);
extern logical pnga_create_irreg(Integer type, Integer ndim,
                                 Integer *dims, char* name,
                                 Integer *map, Integer *block, Integer *g_a);
extern logical pnga_create_irreg_config(Integer type, Integer ndim,
                                        Integer *dims, char* name, Integer *map,
                                        Integer *block, Integer p_handle, Integer *g_a);
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
extern void pnga_inquire(Integer *g_a, Integer *type, Integer *ndim, Integer *dims);
extern Integer pnga_inquire_memory();
extern void pnga_inquire_name(Integer *g_a, char **array_name);
extern logical pnga_is_mirrored(Integer *g_a);
extern void pnga_list_nodeid(Integer *list, Integer *nprocs);
extern logical pnga_locate(Integer *g_a, Integer *subscript, Integer *owner);
extern Integer pnga_locate_num_blocks(Integer *g_a, Integer *lo, Integer *hi);
extern logical pnga_locate_region(Integer *g_a, Integer *lo, Integer *hi, Integer *map,
                                  Integer *proclist, Integer *np);
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
extern void pnga_set_debug(logical *flag);

/* Routines from onesided.c */
extern void pnga_nbput(Integer *g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle);
extern void pnga_put(Integer *g_a, Integer *lo, Integer *hi, void *buf, Integer *ld);

/* Routines from global.util.c */
extern void pnga_error(char *string, Integer icode);

/* Routines from datatypes.c */
extern Integer pnga_type_f2c(Integer type);
extern Integer pnga_type_c2f(Integer type);
#endif /* PAPI_H_ */
