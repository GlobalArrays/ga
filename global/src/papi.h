#ifndef PAPI_H_
#define PAPI_H_

#include "typesf2c.h"

/**
 * Routines from base.c
 */
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
extern void pnga_get_proc_grid(Integer *g_a, Integer *dims);
extern void pnga_get_proc_index(Integer *g_a, Integer *iproc, Integer *index);
extern logical pnga_has_ghosts(Integer *g_a);
extern Integer pnga_nnodes();
extern Integer pnga_nodeid();
extern void pnga_set_debug(logical *flag);

/**
 * Routines from onesided.c
 */

extern void pnga_nbput(Integer *g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle);
extern void pnga_put(Integer *g_a, Integer *lo, Integer *hi, void *buf, Integer *ld);

/**
 * Routines from global.util.c
 */
extern void pnga_error(char *string, Integer icode);

#endif /* PAPI_H_ */
