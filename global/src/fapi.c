/**
 * @file fapi.c
 *
 * Implements the Fortran interface.
 * These calls forward to the (possibly) weak symbols of the internal
 * implementations.
 */
#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif

#include "farg.h"

#if PROFILING_DEFINES
#   include "wapidefs.h"
#endif
#include "wapi.h"

#define FNAM 31
#define FMSG 256

/* copied from globalp.h until global.h is eliminated */
#ifdef FALSE
#undef FALSE
#endif
#ifdef TRUE
#undef TRUE
#endif
#ifdef CRAY_YMP
#define FALSE _btol(0)
#define TRUE  _btol(1)
#else
#define FALSE (logical) 0
#define TRUE  (logical) 1
#endif

/* Routines from base.c */

#define ga_allocate_ F77_FUNC_(ga_allocate,GA_ALLOCATE)
logical FATR ga_allocate_(Integer *g_a)
{
  return wnga_allocate(g_a);
}

#define nga_allocate_ F77_FUNC_(nga_allocate,NGA_ALLOCATE)
logical FATR nga_allocate_(Integer *g_a)
{
  return wnga_allocate(g_a);
}

#define ga_compare_distr_ F77_FUNC_(ga_compare_distr,GA_COMPARE_DISTR)
logical FATR ga_compare_distr_(Integer *g_a, Integer *g_b)
{
  return wnga_compare_distr(g_a, g_b);
}

#define nga_compare_distr_ F77_FUNC_(nga_compare_distr,NGA_COMPARE_DISTR)
logical FATR nga_compare_distr_(Integer *g_a, Integer *g_b)
{
  return wnga_compare_distr(g_a, g_b);
}

#define ga_create_ F77_FUNC_(ga_create,GA_CREATE)
logical FATR ga_create_(
#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
        type, dim1, dim2, array_name, chunk1, chunk2, g_a, slen
#else
        type, dim1, dim2, array_name, slen, chunk1, chunk2, g_a
#endif
        )
Integer *type, *dim1, *dim2, *chunk1, *chunk2, *g_a;
int slen;
char* array_name;
{
  char buf[FNAM];
  Integer ndim, dims[2], chunk[2];
  ga_f2cstring(array_name ,slen, buf, FNAM);
  dims[0] = *dim1;
  dims[1] = *dim2;
  ndim = 2;
  chunk[0] = *chunk1;
  chunk[1] = *chunk2;

  return(wnga_create(*type, ndim, dims, buf, chunk, g_a));
}

#define nga_create_ F77_FUNC_(nga_create,NGA_CREATE)
logical FATR nga_create_(
#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
    Integer *type, Integer *ndim,
    Integer *dims, char* array_name, Integer *chunk,
    Integer *g_a, int slen
#else
    Integer *type, Integer *ndim,
    Integer *dims, char* array_name, int slen,
    Integer *p_handle, Integer *g_a
#endif
    )
{
  char buf[FNAM];
  ga_f2cstring(array_name ,slen, buf, FNAM);

  return (wnga_create(*type, *ndim,  dims, buf, chunk, g_a));
}

#define nga_create_config_ F77_FUNC_(nga_create_config,NGA_CREATE_CONFIG)
logical FATR nga_create_config_(
#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
    Integer *type, Integer *ndim,
    Integer *dims, char* array_name, Integer *chunk,
    Integer *p_handle, Integer *g_a, int
    slen
#else
    Integer *type, Integer *ndim,
    Integer *dims, char* array_name, int slen,
    Integer *chunk, Integer *p_handle,
    Integer *g_a
#endif
    )
{
  char buf[FNAM];
  ga_f2cstring(array_name ,slen, buf, FNAM);

  return (wnga_create_config(*type, *ndim,  dims, buf, chunk, *p_handle,
                             g_a));
}

#define nga_create_ghosts_ F77_FUNC_(nga_create_ghosts,NGA_CREATE_GHOSTS)
logical FATR nga_create_ghosts_(
#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
    Integer *type, Integer *ndim, Integer *dims,
    Integer *width, char* array_name, Integer *chunk, Integer *g_a,
    int slen
#else
    Integer *type, Integer *ndim, Integer *dims,
    Integer *width, char* array_name, int slen,
    Integer *chunk, Integer *g_a
#endif
    )
{
  char buf[FNAM];
  ga_f2cstring(array_name ,slen, buf, FNAM);

  return (wnga_create_ghosts(*type, *ndim,  dims, width, buf, chunk, g_a));
}

#define nga_create_ghosts_config_ F77_FUNC_(nga_create_ghosts_config,NGA_CREATE_GHOSTS_CONFIG)
logical FATR nga_create_ghosts_config_(
#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
    Integer *type, Integer *ndim,
    Integer *dims, Integer *width, char* array_name,
    Integer *chunk, Integer *p_handle,
    Integer *g_a,
    int slen
#else
    Integer *type, Integer *ndim,
    Integer *dims, Integer *width, char* array_name,
    int slen,
    Integer *chunk,
    Integer *p_handle,
    Integer *g_a
#endif
    )
{
  char buf[FNAM];
  ga_f2cstring(array_name ,slen, buf, FNAM);

  return (wnga_create_ghosts_config(*type, *ndim,  dims, width, buf, chunk,
                                    *p_handle, g_a));
}

#define nga_create_ghosts_irreg_ F77_FUNC_(nga_create_ghosts_irreg,NGA_CREATE_GHOSTS_IRREG)
logical FATR nga_create_ghosts_irreg_(
#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
    Integer *type, Integer *ndim,
    Integer *dims, Integer width[], char* array_name, Integer map[],
    Integer block[], Integer *g_a, int slen
#else
    Integer *type, Integer *ndim,
    Integer *dims, Integer width[], char* array_name, int slen,
    Integer map[], Integer block[], Integer *g_a
#endif
    )
{
  char buf[FNAM];
  Integer st;
  ga_f2cstring(array_name ,slen, buf, FNAM);

  st = wnga_create_ghosts_irreg(*type, *ndim,  dims, width,
                                buf, map, block, g_a);
  return st;
}

#define nga_create_ghosts_irreg_config_ F77_FUNC_(nga_create_ghosts_irreg_config,NGA_CREATE_GHOSTS_IRREG_CONFIG)
logical FATR nga_create_ghosts_irreg_config_(
#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
    Integer *type,
    Integer *ndim, Integer *dims, Integer width[], char* array_name,
    Integer map[], Integer block[], Integer *p_handle, Integer *g_a,
    int slen
#else
    Integer *type,
    Integer *ndim, Integer *dims, Integer width[], char* array_name,
    int slen, Integer map[], Integer block[],
    Integer *p_handle, Integer *g_a
#endif
    )
{
  char buf[FNAM];
  Integer st;
  ga_f2cstring(array_name ,slen, buf, FNAM);

  st = wnga_create_ghosts_irreg_config(*type, *ndim,  dims,
      width, buf, map, block, *p_handle, g_a);
  return st;
}

#define ga_create_irreg_ F77_FUNC_(ga_create_irreg,GA_CREATE_IRREG)
logical FATR ga_create_irreg_(
#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
    Integer *type, Integer *dim1, Integer *dim2, char *array_name,
    Integer *map1, Integer *nblock1, Integer *map2, Integer *nblock2,
    Integer *g_a, int slen
#else
    Integer *type, Integer *dim1, Integer *dim2, char *array_name, int
    slen, Integer *map1, Integer *nblock1, Integer *map2, Integer
    *nblock2, Integer *g_a
#endif
    )
{
  char buf[FNAM];
  Integer i, ndim, dims[2], block[2], *map;
  Integer status;
  ga_f2cstring(array_name ,slen, buf, FNAM);
  dims[0] = *dim1;
  dims[1] = *dim2;
  block[0] = *nblock1;
  block[1] = *nblock2;
  ndim = 2;
  map = (Integer*)malloc((wnga_nnodes()+1)*sizeof(Integer));
  for(i=0; i<*nblock1; i++) map[i] = map1[i];
  for(i=0; i<*nblock2; i++) map[i+ *nblock1] = map2[i];
  status = wnga_create_irreg(*type, ndim, dims, buf, map, block, g_a);
  free(map);
  return status;
}

#define nga_create_irreg_ F77_FUNC_(nga_create_irreg,NGA_CREATE_IRREG)
logical FATR nga_create_irreg_(
#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
    Integer *type, Integer *ndim, Integer *dims,
    char* array_name, Integer map[], Integer block[],
    Integer *g_a, int slen
#else
    Integer *type, Integer *ndim, Integer *dims,
    char* array_name, int slen,
    Integer map[], Integer block[], Integer *g_a
#endif
    )
{
  char buf[FNAM];
  Integer st;
  ga_f2cstring(array_name ,slen, buf, FNAM);

  st = wnga_create_irreg(*type, *ndim,  dims, buf, map, block, g_a);
  return st;
}

#define nga_create_irreg_config_ F77_FUNC_(nga_create_irreg_config,NGA_CREATE_IRREG_CONFIG)
logical FATR nga_create_irreg_config_(
#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
    Integer *type, Integer *ndim,
    Integer *dims, char* array_name, Integer map[],
    Integer block[], Integer *p_handle, Integer *g_a,
    int slen
#else
    Integer *type, Integer *ndim,
    Integer *dims, char* array_name, int slen, Integer map[],
    Integer block[], Integer *p_handle, Integer *g_a
#endif
    )
{
  char buf[FNAM];
  Integer st;
  ga_f2cstring(array_name ,slen, buf, FNAM);

  st = wnga_create_irreg_config(*type, *ndim,  dims, buf, map,
      block, *p_handle, g_a);
  return st;
}

#define ga_create_handle_ F77_FUNC_(ga_create_handle,GA_CREATE_HANDLE)
Integer FATR ga_create_handle_()
{
  return wnga_create_handle();
}

#define nga_create_handle_ F77_FUNC_(nga_create_handle,NGA_CREATE_HANDLE)
Integer FATR nga_create_handle_()
{
  return wnga_create_handle();
}

#define ga_create_mutexes_ F77_FUNC_(ga_create_mutexes,GA_CREATE_MUTEXES)
logical FATR ga_create_mutexes_(Integer *num)
{
  return wnga_create_mutexes(num);
}

#define nga_create_mutexes_ F77_FUNC_(nga_create_mutexes,NGA_CREATE_MUTEXES)
logical FATR nga_create_mutexes_(Integer *num)
{
  return wnga_create_mutexes(num);
}

#define ga_destroy_ F77_FUNC_(ga_destroy,GA_DESTROY)
logical FATR ga_destroy_(Integer *g_a)
{
  return wnga_destroy(g_a);
}

#define nga_destroy_ F77_FUNC_(nga_destroy,NGA_DESTROY)
logical FATR nga_destroy_(Integer *g_a)
{
  return wnga_destroy(g_a);
}

#define ga_destroy_mutexes_ F77_FUNC_(ga_destroy_mutexes,GA_DESTROY_MUTEXES)
logical FATR ga_destroy_mutexes_()
{
  return wnga_destroy_mutexes();
}

#define nga_destroy_mutexes_ F77_FUNC_(nga_destroy_mutexes,NGA_DESTROY_MUTEXES)
logical FATR nga_destroy_mutexes_()
{
  return wnga_destroy_mutexes();
}

#define ga_distribution_ F77_FUNC_(ga_distribution,GA_DISTRIBUTION)
void FATR ga_distribution_(Integer *g_a, Integer *proc, Integer *ilo,
                           Integer *ihi, Integer *jlo, Integer *jhi)
{
  Integer lo[2], hi[2];
  wnga_distribution(g_a, proc, lo, hi);
  *ilo = lo[0];
  *jlo = lo[1];
  *ihi = hi[0];
  *jhi = hi[1];
}

#define nga_distribution_ F77_FUNC_(nga_distribution,NGA_DISTRIBUTION)
void FATR nga_distribution_(Integer *g_a, Integer *proc, Integer *lo, Integer *hi)
{
  wnga_distribution(g_a, proc, lo, hi);
}

#define ga_duplicate_ F77_FUNC_(ga_duplicate,GA_DUPLICATE)
logical FATR ga_duplicate_( Integer *g_a, Integer *g_b, char *array_name, int slen)
{
  char buf[FNAM];

  ga_f2cstring(array_name ,slen, buf, FNAM);
  return(wnga_duplicate(g_a, g_b, buf));
}

#define nga_duplicate_ F77_FUNC_(nga_duplicate,NGA_DUPLICATE)
logical FATR nga_duplicate_( Integer *g_a, Integer *g_b, char *array_name, int slen)
{
  char buf[FNAM];

  ga_f2cstring(array_name ,slen, buf, FNAM);
  return(wnga_duplicate(g_a, g_b, buf));
}

#define ga_fill_ F77_FUNC_(ga_fill,GA_FILL)
void FATR ga_fill_(Integer *g_a, void* val)
{
  wnga_fill(g_a, val);
}

#define nga_fill_ F77_FUNC_(nga_fill,NGA_FILL)
void FATR nga_fill_(Integer *g_a, void* val)
{
  wnga_fill(g_a, val);
}

#define ga_get_block_info_ F77_FUNC_(ga_get_block_info,GA_GET_BLOCK_INFO)
void FATR ga_get_block_info_(Integer *g_a, Integer *num_blocks,
                             Integer *block_dims)
{
  wnga_get_block_info(g_a, num_blocks, block_dims);
}

#define nga_get_block_info_ F77_FUNC_(nga_get_block_info,NGA_GET_BLOCK_INFO)
void FATR nga_get_block_info_(Integer *g_a, Integer *num_blocks,
                             Integer *block_dims)
{
  wnga_get_block_info(g_a, num_blocks, block_dims);
}

#define ga_get_debug_ F77_FUNC_(ga_get_debug,GA_GET_DEBUG)
logical FATR ga_get_debug_()
{
  return wnga_get_debug();
}

#define nga_get_debug_ F77_FUNC_(nga_get_debug,NGA_GET_DEBUG)
logical FATR nga_get_debug_()
{
  return wnga_get_debug();
}

#define ga_get_dimension_ F77_FUNC_(ga_get_dimension,GA_GET_DIMENSION)
Integer FATR ga_get_dimension_(Integer *g_a)
{
  return wnga_get_dimension(g_a);
}

#define nga_get_dimension_ F77_FUNC_(nga_get_dimension,NGA_GET_DIMENSION)
Integer FATR nga_get_dimension_(Integer *g_a)
{
  return wnga_get_dimension(g_a);
}

#define ga_get_proc_grid_ F77_FUNC_(ga_get_proc_grid,GA_GET_PROC_GRID)
void FATR ga_get_proc_grid_(Integer *g_a, Integer *dims)
{
  wnga_get_proc_grid(g_a, dims);
}

#define ga_get_proc_index_ F77_FUNC_(ga_get_proc_index,GA_GET_PROC_INDEX)
void FATR ga_get_proc_index_(Integer *g_a, Integer *iproc, Integer *index)
{
  wnga_get_proc_index(g_a, iproc, index);
}

#define nga_get_proc_index_ F77_FUNC_(nga_get_proc_index,NGA_GET_PROC_INDEX)
void FATR nga_get_proc_index_(Integer *g_a, Integer *iproc, Integer *index)
{
  wnga_get_proc_index(g_a, iproc, index);
}

#define nga_get_proc_grid_ F77_FUNC_(nga_get_proc_grid,NGA_GET_PROC_GRID)
void FATR nga_get_proc_grid_(Integer *g_a, Integer *dims)
{
  wnga_get_proc_grid(g_a, dims);
}

#define ga_has_ghosts_ F77_FUNC_(ga_has_ghosts,GA_HAS_GHOSTS)
logical FATR ga_has_ghosts_(Integer *g_a)
{
  return wnga_has_ghosts(g_a);
}

#define nga_has_ghosts_ F77_FUNC_(nga_has_ghosts,NGA_HAS_GHOSTS)
logical FATR nga_has_ghosts_(Integer *g_a)
{
  return wnga_has_ghosts(g_a);
}

#define ga_initialize_ F77_FUNC_(ga_initialize,GA_INITIALIZE)
void FATR ga_initialize_()
{
  wnga_initialize();
}

#define nga_initialize_ F77_FUNC_(nga_initialize,NGA_INITIALIZE)
void FATR nga_initialize_()
{
  wnga_initialize();
}

#define ga_initialize_ltd_ F77_FUNC_(ga_initialize_ltd,GA_INITIALIZE_LTD)
void FATR ga_initialize_ltd_(Integer *limit)
{
  wnga_initialize_ltd(limit);
}

#define nga_initialize_ltd_ F77_FUNC_(nga_initialize_ltd,NGA_INITIALIZE_LTD)
void FATR nga_initialize_ltd_(Integer *limit)
{
  wnga_initialize_ltd(limit);
}

#define ga_inquire_ F77_FUNC_(ga_inquire,GA_INQUIRE)
void FATR ga_inquire_(Integer *g_a, Integer *type, Integer *dim1, Integer *dim2)
{
  Integer dims[2], ndim;
  wnga_inquire(g_a, type, &ndim, dims);
  if (ndim != 2) wnga_error("Wrong array dimension in ga_inquire",ndim);
  *type = pnga_type_c2f(*type);
  *dim1 = dims[0];
  *dim2 = dims[1];
}

#define nga_inquire_ F77_FUNC_(nga_inquire,NGA_INQUIRE)
void FATR nga_inquire_(Integer *g_a, Integer *type, Integer *ndim, Integer *dims)
{
  wnga_inquire(g_a, type, ndim, dims);
  *type = pnga_type_c2f(*type);
}

#define ga_inquire_memory_ F77_FUNC_(ga_inquire_memory,GA_INQUIRE_MEMORY)
Integer FATR ga_inquire_memory_()
{
  return wnga_inquire_memory();
}

#define nga_inquire_memory_ F77_FUNC_(nga_inquire_memory,NGA_INQUIRE_MEMORY)
Integer FATR nga_inquire_memory_()
{
  return wnga_inquire_memory();
}

#define ga_inquire_name_ F77_FUNC_(ga_inquire_name,GA_INQUIRE_NAME)
void FATR ga_inquire_name_(Integer *g_a, char *array_name, int len)
{
  char *c_name;
  wnga_inquire_name(g_a, &c_name);
  ga_c2fstring(c_name, array_name, len);
}

#define nga_inquire_name_ F77_FUNC_(nga_inquire_name,NGA_INQUIRE_NAME)
void FATR nga_inquire_name_(Integer *g_a, char *array_name, int len)
{
  char *c_name;
  wnga_inquire_name(g_a, &c_name);
  ga_c2fstring(c_name, array_name, len);
}

#define ga_is_mirrored_ F77_FUNC_(ga_is_mirrored,GA_IS_MIRRORED)
logical FATR ga_is_mirrored_(Integer *g_a)
{
  return wnga_is_mirrored(g_a);
}

#define nga_is_mirrored_ F77_FUNC_(nga_is_mirrored,NGA_IS_MIRRORED)
logical FATR nga_is_mirrored_(Integer *g_a)
{
  return wnga_is_mirrored(g_a);
}

#define ga_list_nodeid_ F77_FUNC_(ga_list_nodeid,GA_LIST_NODEID)
void FATR ga_list_nodeid_(Integer *list, Integer *nprocs)
{
  wnga_list_nodeid(list, nprocs);
}

#define nga_list_nodeid_ F77_FUNC_(nga_list_nodeid,NGA_LIST_NODEID)
void FATR nga_list_nodeid_(Integer *list, Integer *nprocs)
{
  wnga_list_nodeid(list, nprocs);
}

#define nga_locate_num_blocks_ F77_FUNC_(nga_locate_num_blocks,NGA_LOCATE_NUM_BLOCKS)
Integer FATR nga_locate_num_blocks_(Integer *g_a, Integer *lo, Integer *hi)
{
  return wnga_locate_num_blocks(g_a,lo,hi);
}

#define ga_locate_region_ F77_FUNC_(ga_locate_region,GA_LOCATE_REGION)
logical FATR ga_locate_region_( Integer *g_a,
                                Integer *ilo,
                                Integer *ihi,
                                Integer *jlo,
                                Integer *jhi,
                                Integer map[][5],
                                Integer *np)
{
  logical status = FALSE;
  Integer lo[2], hi[2], p;
  Integer *mapl, *proclist;
  proclist = (Integer*)malloc(wnga_nnodes()*sizeof(Integer));
  mapl = (Integer*)malloc(5*wnga_nnodes()*sizeof(Integer));
  lo[0] = *ilo;
  lo[1] = *jlo;
  hi[0] = *ihi;
  hi[1] = *jhi;
  if (wnga_locate_num_blocks(g_a, lo, hi) == -1) {
    status = wnga_locate_region(g_a, lo, hi, mapl, proclist, np);
    /* need to swap elements (ilo,jlo,ihi,jhi) -> (ilo,ihi,jlo,jhi) */
    for(p = 0; p< *np; p++){
      map[p][0] = mapl[4*p];
      map[p][1] = mapl[4*p + 2];
      map[p][2] = mapl[4*p + 1];
      map[p][3] = mapl[4*p + 3];
      map[p][4] = proclist[p];
    }
  } else {
    wnga_error("Must call nga_locate_region on block-cyclic data distribution",0);
  }
  free(proclist);
  free(mapl);
  return status;
}

#define nga_locate_region_ F77_FUNC_(nga_locate_region,NGA_LOCATE_REGION)
logical FATR nga_locate_region_( Integer *g_a,
                                 Integer *lo,
                                 Integer *hi,
                                 Integer *map,
                                 Integer *proclist,
                                 Integer *np)
{
  return wnga_locate_region(g_a, lo, hi, map, proclist, np);
}

#define ga_lock_ F77_FUNC_(ga_lock,GA_LOCK)
void FATR ga_lock_(Integer *mutex)
{
  wnga_lock(mutex);
}

#define nga_lock_ F77_FUNC_(nga_lock,NGA_LOCK)
void FATR nga_lock_(Integer *mutex)
{
  wnga_lock(mutex);
}

#define ga_locate_ F77_FUNC_(ga_locate,GA_LOCATE)
logical FATR ga_locate_(Integer *g_a, Integer *i, Integer *j, Integer *owner)
{
  Integer subscript[2];
  subscript[0] = *i;
  subscript[1] = *j;
  return wnga_locate(g_a, subscript, owner);
}

#define nga_locate_ F77_FUNC_(nga_locate,NGA_LOCATE)
logical FATR nga_locate_(Integer *g_a, Integer *subscript, Integer *owner)
{
  return wnga_locate(g_a, subscript, owner);
}

#define ga_mask_sync_ F77_FUNC_(ga_mask_sync,GA_MASK_SYNC)
void FATR ga_mask_sync_(Integer *begin, Integer *end)
{
  wnga_mask_sync(begin, end);
}

#define nga_mask_sync_ F77_FUNC_(nga_mask_sync,NGA_MASK_SYNC)
void FATR nga_mask_sync_(Integer *begin, Integer *end)
{
  wnga_mask_sync(begin, end);
}

#define ga_memory_avail_ F77_FUNC_(ga_memory_avail,GA_MEMORY_AVAIL)
Integer FATR ga_memory_avail_()
{
  return wnga_memory_avail();
}

#define nga_memory_avail_ F77_FUNC_(nga_memory_avail,NGA_MEMORY_AVAIL)
Integer FATR nga_memory_avail_()
{
  return wnga_memory_avail();
}

#define ga_memory_limited_ F77_FUNC_(ga_memory_limited,GA_MEMORY_LIMITED)
logical FATR ga_memory_limited_()
{
  return wnga_memory_limited();
}

#define nga_memory_limited_ F77_FUNC_(nga_memory_limited,NGA_MEMORY_LIMITED)
logical FATR nga_memory_limited_()
{
  return wnga_memory_limited();
}

#define nga_merge_distr_patch_ F77_FUNC_(nga_merge_distr_patch,NGA_MERGE_DISTR_PATCH)
void FATR nga_merge_distr_patch_(Integer *g_a, Integer *alo, Integer *ahi,
                                 Integer *g_b, Integer *blo, Integer *bhi)
{
  wnga_merge_distr_patch(g_a, alo, ahi, g_b, blo, bhi);
}

#define ga_merge_mirrored_ F77_FUNC_(ga_merge_mirrored,GA_MERGE_MIRRORED)
void FATR ga_merge_mirrored_(Integer *g_a)
{
  wnga_merge_mirrored(g_a);
}

#define nga_merge_mirrored_ F77_FUNC_(nga_merge_mirrored,NGA_MERGE_MIRRORED)
void FATR nga_merge_mirrored_(Integer *g_a)
{
  wnga_merge_mirrored(g_a);
}

#define ga_nblock_ F77_FUNC_(ga_nblock,GA_NBLOCK)
void FATR ga_nblock_(Integer *g_a, Integer *nblock)
{
  wnga_nblock(g_a, nblock);
}

#define nga_nblock_ F77_FUNC_(nga_nblock,NGA_NBLOCK)
void FATR nga_nblock_(Integer *g_a, Integer *nblock)
{
  wnga_nblock(g_a, nblock);
}

#define ga_ndim_ F77_FUNC_(ga_ndim,GA_NDIM)
Integer FATR ga_ndim_(Integer *g_a)
{
  return wnga_ndim(g_a);
}

#define nga_ndim_ F77_FUNC_(nga_ndim,NGA_NDIM)
Integer FATR nga_ndim_(Integer *g_a)
{
  return wnga_ndim(g_a);
}

#define ga_nnodes_ F77_FUNC_(ga_nnodes,GA_NNODES)
Integer FATR ga_nnodes_()
{
  return wnga_nnodes();
}

#define nga_nnodes_ F77_FUNC_(nga_nnodes,NGA_NNODES)
Integer FATR nga_nnodes_()
{
  return wnga_nnodes();
}

#define ga_nodeid_ F77_FUNC_(ga_nodeid,GA_NODEID)
Integer FATR ga_nodeid_()
{
  return wnga_nodeid();
}

#define nga_nodeid_ F77_FUNC_(nga_nodeid,NGA_NODEID)
Integer FATR nga_nodeid_()
{
  return wnga_nodeid();
}

#define ga_pgroup_absolute_id_ F77_FUNC_(ga_pgroup_absolute_id,GA_PGROUP_ABSOLUTE_ID)
Integer FATR ga_pgroup_absolute_id_(Integer *grp, Integer *pid)
{
  return wnga_pgroup_absolute_id(grp, pid);
}

#define nga_pgroup_absolute_id_ F77_FUNC_(nga_pgroup_absolute_id,NGA_PGROUP_ABSOLUTE_ID)
Integer FATR nga_pgroup_absolute_id_(Integer *grp, Integer *pid)
{
  return wnga_pgroup_absolute_id(grp, pid);
}

#define ga_pgroup_create_ F77_FUNC_(ga_pgroup_create,GA_PGROUP_CREATE)
Integer FATR ga_pgroup_create_(Integer *list, Integer *count)
{
  return wnga_pgroup_create(list, count);
}

#define nga_pgroup_create_ F77_FUNC_(nga_pgroup_create,NGA_PGROUP_CREATE)
Integer FATR nga_pgroup_create_(Integer *list, Integer *count)
{
  return wnga_pgroup_create(list, count);
}

#define ga_pgroup_destroy_ F77_FUNC_(ga_pgroup_destroy,GA_PGROUP_DESTROY)
logical FATR ga_pgroup_destroy_(Integer *grp)
{
  return wnga_pgroup_destroy(grp);
}

#define nga_pgroup_destroy_ F77_FUNC_(nga_pgroup_destroy,NGA_PGROUP_DESTROY)
logical FATR nga_pgroup_destroy_(Integer *grp)
{
  return wnga_pgroup_destroy(grp);
}

#define ga_pgroup_get_default_ F77_FUNC_(ga_pgroup_get_default,GA_PGROUP_GET_DEFAULT)
Integer FATR ga_pgroup_get_default_()
{
  return wnga_pgroup_get_default();
}

#define nga_pgroup_get_default_ F77_FUNC_(nga_pgroup_get_default,NGA_PGROUP_GET_DEFAULT)
Integer FATR nga_pgroup_get_default_()
{
  return wnga_pgroup_get_default();
}

#define ga_pgroup_get_mirror_ F77_FUNC_(ga_pgroup_get_mirror,GA_PGROUP_GET_MIRROR)
Integer FATR ga_pgroup_get_mirror_()
{
  return wnga_pgroup_get_mirror();
}

#define nga_pgroup_get_mirror_ F77_FUNC_(nga_pgroup_get_mirror,NGA_PGROUP_GET_MIRROR)
Integer FATR nga_pgroup_get_mirror_()
{
  return wnga_pgroup_get_mirror();
}

#define ga_pgroup_get_world_ F77_FUNC_(ga_pgroup_get_world,GA_PGROUP_GET_WORLD)
Integer FATR ga_pgroup_get_world_()
{
  return wnga_pgroup_get_world();
}

#define nga_pgroup_get_mirror_ F77_FUNC_(nga_pgroup_get_mirror,NGA_PGROUP_GET_MIRROR)
Integer FATR nga_pgroup_get_world_()
{
  return wnga_pgroup_get_world();
}

#define ga_pgroup_set_default_ F77_FUNC_(ga_pgroup_set_default,GA_PGROUP_SET_DEFAULT)
void FATR ga_pgroup_set_default_(Integer *grp)
{
  wnga_pgroup_set_default(grp);
}

#define nga_pgroup_set_default_ F77_FUNC_(nga_pgroup_set_default,NGA_PGROUP_SET_DEFAULT)
void FATR nga_pgroup_set_default_(Integer *grp)
{
  wnga_pgroup_set_default(grp);
}

#define ga_pgroup_split_ F77_FUNC_(ga_pgroup_split,GA_PGROUP_SPLIT)
Integer FATR ga_pgroup_split_(Integer *grp, Integer *grp_num)
{
  return wnga_pgroup_split(grp, grp_num);
}

#define nga_pgroup_split_ F77_FUNC_(nga_pgroup_split,NGA_PGROUP_SPLIT)
Integer FATR nga_pgroup_split_(Integer *grp, Integer *grp_num)
{
  return wnga_pgroup_split(grp, grp_num);
}

#define ga_pgroup_split_ F77_FUNC_(ga_pgroup_split,GA_PGROUP_SPLIT)
Integer FATR ga_pgroup_split_irreg_(Integer *grp, Integer *mycolor)
{
  return wnga_pgroup_split_irreg(grp, mycolor);
}

#define nga_pgroup_split_irreg_ F77_FUNC_(nga_pgroup_split_irreg,NGA_PGROUP_SPLIT_IRREG)
Integer FATR nga_pgroup_split_irreg_(Integer *grp, Integer *mycolor)
{
  return wnga_pgroup_split_irreg(grp, mycolor);
}

#define ga_pgroup_nnodes_ F77_FUNC_(ga_pgroup_nnodes,GA_PGROUP_NNODES)
Integer FATR ga_pgroup_nnodes_(Integer *grp)
{
  return wnga_pgroup_nnodes(grp);
}

#define nga_pgroup_nnodes_ F77_FUNC_(nga_pgroup_nnodes,NGA_PGROUP_NNODES)
Integer FATR nga_pgroup_nnodes_(Integer *grp)
{
  return wnga_pgroup_nnodes(grp);
}

#define ga_pgroup_nodeid_ F77_FUNC_(ga_pgroup_nodeid,GA_PGROUP_NODEID)
Integer FATR ga_pgroup_nodeid_(Integer *grp)
{
  return wnga_pgroup_nodeid(grp);
}

#define nga_pgroup_nodeid_ F77_FUNC_(nga_pgroup_nodeid,NGA_PGROUP_NODEID)
Integer FATR nga_pgroup_nodeid_(Integer *grp)
{
  return wnga_pgroup_nodeid(grp);
}

#define ga_proc_topology_ F77_FUNC_(ga_proc_topology,GA_PROC_TOPOLOGY)
void FATR ga_proc_topology_(Integer* g_a, Integer* proc, Integer* pr,
                            Integer *pc)
{
  Integer subscript[2];
  wnga_proc_topology(g_a, proc, subscript);
  *pr = subscript[0];
  *pc = subscript[1];
}

#define nga_proc_topology_ F77_FUNC_(nga_proc_topology,NGA_PROC_TOPOLOGY)
void FATR nga_proc_topology_(Integer* g_a, Integer* proc, Integer* subscript)
{
  wnga_proc_topology(g_a, proc, subscript);
}

#define ga_set_debug_ F77_FUNC_(ga_set_debug,GA_SET_DEBUG)
void FATR ga_set_debug_(logical *flag)
{
  wnga_set_debug(flag);
}

#define nga_set_debug_ F77_FUNC_(nga_set_debug,NGA_SET_DEBUG)
void FATR nga_set_debug_(logical *flag)
{
  wnga_set_debug(flag);
}

/* Routines from onesided.c */

#define ga_nbput_ F77_FUNC_(ga_nbput,GA_NBPUT)
void FATR ga_nbput_(Integer *g_a, Integer *ilo, Integer *ihi,
                    Integer *jlo, Integer *jhi, void *buf,
                    Integer *ld, Integer *nbhandle)
{
    Integer lo[2], hi[2];
    lo[0]=*ilo;
    lo[1]=*jlo;
    hi[0]=*ihi;
    hi[1]=*jhi;
    wnga_nbput(g_a, lo, hi, buf, ld, nbhandle);
}

#define nga_nbput_ F77_FUNC_(nga_nbput,NGA_NBPUT)
void FATR nga_nbput_(Integer *g_a, Integer *lo,
                     Integer *hi, void *buf, Integer *ld,
                     Integer *nbhandle)
{
    wnga_nbput(g_a, lo, hi, buf, ld, nbhandle);
}

#define ga_put_ F77_FUNC_(ga_put,GA_PUT)
void FATR ga_put_(Integer *g_a, Integer *ilo, Integer *ihi,
                  Integer *jlo, Integer *jhi, void *buf, Integer *ld)
{
    Integer lo[2], hi[2];
    lo[0]=*ilo;
    lo[1]=*jlo;
    hi[0]=*ihi;
    hi[1]=*jhi;
    wnga_put(g_a, lo, hi, buf, ld);
}

#define nga_put_ F77_FUNC_(nga_put,NGA_PUT)
void FATR nga_put_(Integer *g_a, Integer *lo,
                   Integer *hi, void *buf, Integer *ld)
{
    wnga_put(g_a, lo, hi, buf, ld);
}

/* Routines from global.util.c */

#define ga_error_ F77_FUNC_(ga_error,GA_ERROR)
void FATR ga_error_(
#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
    char *string, Integer *icode, int slen
#else
    char *string, int slen, Integer *icode
#endif
    )
{
  char buf[FMSG];
  ga_f2cstring(string,slen, buf, FMSG);
  wnga_error(buf,*icode);
}

#define nga_error_ F77_FUNC_(nga_error,NGA_ERROR)
void FATR nga_error_(
#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
    char *string, Integer *icode, int slen
#else
    char *string, int slen, Integer *icode
#endif
    )
{
  char buf[FMSG];
  ga_f2cstring(string,slen, buf, FMSG);
  wnga_error(buf,*icode);
}

