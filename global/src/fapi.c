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

#include "c.names.h"
#include "farg.h"
#include "stdlib.h"

#if PROFILING_DEFINES
#   include "wapidefs.h"
#endif
#include "wapi.h"

#define FNAM 31

/* Routines from base.c */

logical FATR ga_allocate_(Integer *g_a)
{
  return wnga_allocate(g_a);
}

logical FATR nga_allocate_(Integer *g_a)
{
  return wnga_allocate(g_a);
}

logical FATR ga_compare_distr_(Integer *g_a, Integer *g_b)
{
  return wnga_compare_distr(g_a, g_b);
}

logical FATR nga_compare_distr_(Integer *g_a, Integer *g_b)
{
  return wnga_compare_distr(g_a, g_b);
}

#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
logical FATR ga_create_(type, dim1, dim2, array_name, chunk1, chunk2, g_a, slen)
#else
logical FATR ga_create_(type, dim1, dim2, array_name, slen, chunk1, chunk2,
    g_a)
#endif
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

#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
logical FATR nga_create_(Integer *type, Integer *ndim,
    Integer *dims, char* array_name, Integer *chunk,
    Integer *g_a, int slen)
#else
logical FATR nga_create_(Integer *type, Integer *ndim,
    Integer *dims, char* array_name, int slen,
    Integer *p_handle, Integer *g_a)
#endif
{
  char buf[FNAM];
  ga_f2cstring(array_name ,slen, buf, FNAM);

  return (wnga_create(*type, *ndim,  dims, buf, chunk, g_a));
}

#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
logical FATR nga_create_config_(Integer *type, Integer *ndim,
    Integer *dims, char* array_name, Integer *chunk,
    Integer *p_handle, Integer *g_a, int
    slen)
#else
logical FATR nga_create_config_(Integer *type, Integer *ndim,
    Integer *dims, char* array_name, int slen,
    Integer *chunk, Integer *p_handle,
    Integer *g_a)
#endif
{
  char buf[FNAM];
  ga_f2cstring(array_name ,slen, buf, FNAM);

  return (wnga_create_config(*type, *ndim,  dims, buf, chunk, *p_handle,
                             g_a));
}

#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
logical FATR nga_create_ghosts_(Integer *type, Integer *ndim, Integer *dims,
    Integer *width, char* array_name, Integer *chunk, Integer *g_a,
    int slen)
#else
logical FATR nga_create_ghosts_(Integer *type, Integer *ndim, Integer *dims,
    Integer *width, char* array_name, int slen,
    Integer *chunk, Integer *g_a)
#endif
{
  char buf[FNAM];
  ga_f2cstring(array_name ,slen, buf, FNAM);

  return (wnga_create_ghosts(*type, *ndim,  dims, width, buf, chunk, g_a));
}

#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
logical FATR nga_create_ghosts_config_(Integer *type, Integer *ndim,
    Integer *dims, Integer *width, char* array_name,
    Integer *chunk, Integer *p_handle,
    Integer *g_a,
    int slen)
#else
logical FATR nga_create_ghosts_config_(Integer *type, Integer *ndim,
    Integer *dims, Integer *width, char* array_name,
    int slen,
    Integer *chunk,
    Integer *p_handle,
    Integer *g_a)
#endif
{
  char buf[FNAM];
  ga_f2cstring(array_name ,slen, buf, FNAM);

  return (wnga_create_ghosts_config(*type, *ndim,  dims, width, buf, chunk,
                                    *p_handle, g_a));
}

#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
logical FATR nga_create_ghosts_irreg_(Integer *type, Integer *ndim,
    Integer *dims, Integer width[], char* array_name, Integer map[],
    Integer block[], Integer *g_a, int slen)
#else
logical FATR nga_create_ghosts_irreg_(Integer *type, Integer *ndim,
    Integer *dims, Integer width[], char* array_name, int slen,
    Integer map[], Integer block[], Integer *g_a)
#endif
{
  char buf[FNAM];
  Integer st;
  ga_f2cstring(array_name ,slen, buf, FNAM);

  st = wnga_create_ghosts_irreg(*type, *ndim,  dims, width,
                                buf, map, block, g_a);
  return st;
}

#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
logical FATR nga_create_ghosts_irreg_config_(Integer *type,
    Integer *ndim, Integer *dims, Integer width[], char* array_name,
    Integer map[], Integer block[], Integer *p_handle, Integer *g_a,
    int slen)
#else
logical FATR nga_create_ghosts_irreg_config_(Integer *type,
    Integer *ndim, Integer *dims, Integer width[], char* array_name,
    int slen, Integer map[], Integer block[],
    Integer *p_handle, Integer *g_a)
#endif
{
  char buf[FNAM];
  Integer st;
  ga_f2cstring(array_name ,slen, buf, FNAM);

  st = wnga_create_ghosts_irreg_config(*type, *ndim,  dims,
      width, buf, map, block, *p_handle, g_a);
  return st;
}

#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
logical FATR ga_create_irreg_(
    Integer *type, Integer *dim1, Integer *dim2, char *array_name,
    Integer *map1, Integer *nblock1, Integer *map2, Integer *nblock2,
    Integer *g_a, int slen)
#else
logical FATR ga_create_irreg_(
    Integer *type, Integer *dim1, Integer *dim2, char *array_name, int
    slen, Integer *map1, Integer *nblock1, Integer *map2, Integer
    *nblock2, Integer *g_a)
#endif
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

#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
logical FATR nga_create_irreg_(Integer *type, Integer *ndim, Integer *dims,
                 char* array_name, Integer map[], Integer block[],
                 Integer *g_a, int slen)
#else
logical FATR nga_create_irreg_(Integer *type, Integer *ndim, Integer *dims,
                 char* array_name, int slen,
                 Integer map[], Integer block[], Integer *g_a)
#endif
{
  char buf[FNAM];
  Integer st;
  ga_f2cstring(array_name ,slen, buf, FNAM);

  st = wnga_create_irreg(*type, *ndim,  dims, buf, map, block, g_a);
  return st;
}

#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
logical FATR nga_create_irreg_config_(Integer *type, Integer *ndim,
    Integer *dims, char* array_name, Integer map[],
    Integer block[], Integer *p_handle, Integer *g_a,
    int slen)
#else
logical FATR nga_create_irreg_config_(Integer *type, Integer *ndim,
    Integer *dims, char* array_name, int slen, Integer map[],
    Integer block[], Integer *p_handle, Integer *g_a)
#endif
{
  char buf[FNAM];
  Integer st;
  ga_f2cstring(array_name ,slen, buf, FNAM);

  st = wnga_create_irreg_config(*type, *ndim,  dims, buf, map,
      block, *p_handle, g_a);
  return st;
}

Integer FATR ga_create_handle_()
{
  return wnga_create_handle();
}

Integer FATR nga_create_handle_()
{
  return wnga_create_handle();
}

logical FATR ga_create_mutexes_(Integer *num)
{
  return wnga_create_mutexes(num);
}

logical FATR nga_create_mutexes_(Integer *num)
{
  return wnga_create_mutexes(num);
}

logical FATR ga_destroy_(Integer *g_a)
{
  return wnga_destroy(g_a);
}

logical FATR nga_destroy_(Integer *g_a)
{
  return wnga_destroy(g_a);
}

logical FATR ga_destroy_mutexes_()
{
  return wnga_destroy_mutexes();
}

logical FATR nga_destroy_mutexes_()
{
  return wnga_destroy_mutexes();
}

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

void FATR nga_distribution_(Integer *g_a, Integer *proc, Integer *lo, Integer *hi)
{
  wnga_distribution(g_a, proc, lo, hi);
}

logical FATR ga_duplicate_( Integer *g_a, Integer *g_b, char *array_name, int slen)
{
  char buf[FNAM];

  ga_f2cstring(array_name ,slen, buf, FNAM);
  return(wnga_duplicate(g_a, g_b, buf));
}

logical FATR nga_duplicate_( Integer *g_a, Integer *g_b, char *array_name, int slen)
{
  char buf[FNAM];

  ga_f2cstring(array_name ,slen, buf, FNAM);
  return(wnga_duplicate(g_a, g_b, buf));
}

void FATR ga_fill_(Integer *g_a, void* val)
{
  wnga_fill(g_a, val);
}

void FATR nga_fill_(Integer *g_a, void* val)
{
  wnga_fill(g_a, val);
}

void FATR ga_get_block_info_(Integer *g_a, Integer *num_blocks,
                             Integer *block_dims)
{
  wnga_get_block_info(g_a, num_blocks, block_dims);
}

void FATR nga_get_block_info_(Integer *g_a, Integer *num_blocks,
                             Integer *block_dims)
{
  wnga_get_block_info(g_a, num_blocks, block_dims);
}

logical FATR ga_get_debug_()
{
  return wnga_get_debug();
}

logical FATR nga_get_debug_()
{
  return wnga_get_debug();
}

Integer FATR ga_get_dimension_(Integer *g_a)
{
  return wnga_get_dimension(g_a);
}

Integer FATR nga_get_dimension_(Integer *g_a)
{
  return wnga_get_dimension(g_a);
}

void FATR ga_get_proc_grid_(Integer *g_a, Integer *dims)
{
  wnga_get_proc_grid(g_a, dims);
}

void FATR ga_get_proc_index_(Integer *g_a, Integer *iproc, Integer *index)
{
  wnga_get_proc_index(g_a, iproc, index);
}

void FATR nga_get_proc_index_(Integer *g_a, Integer *iproc, Integer *index)
{
  wnga_get_proc_index(g_a, iproc, index);
}

void FATR nga_get_proc_grid_(Integer *g_a, Integer *dims)
{
  wnga_get_proc_grid(g_a, dims);
}

logical FATR ga_has_ghosts_(Integer *g_a)
{
  return wnga_has_ghosts(g_a);
}

logical FATR nga_has_ghosts_(Integer *g_a)
{
  return wnga_has_ghosts(g_a);
}

void FATR ga_initialize_()
{
  wnga_initialize();
}

void FATR nga_initialize_()
{
  wnga_initialize();
}

void FATR ga_initialize_ltd_(Integer *limit)
{
  wnga_initialize_ltd(limit);
}

void FATR nga_initialize_ltd_(Integer *limit)
{
  wnga_initialize_ltd(limit);
}

void FATR ga_inquire_(Integer *g_a, Integer *type, Integer *dim1, Integer *dim2)
{
  Integer dims[2], ndim;
  wnga_inquire(g_a, type, &ndim, dims);
  if (ndim != 2) wnga_error("Wrong array dimension in ga_inquire",ndim);
  *type = pnga_type_c2f(*type);
  *dim1 = dims[0];
  *dim2 = dims[1];
}

void FATR nga_inquire_(Integer *g_a, Integer *type, Integer *ndim, Integer *dims)
{
  wnga_inquire(g_a, type, ndim, dims);
  *type = pnga_type_c2f(*type);
}

Integer FATR ga_inquire_memory_()
{
  return wnga_inquire_memory();
}

Integer FATR nga_inquire_memory_()
{
  return wnga_inquire_memory();
}

void FATR ga_inquire_name_(Integer *g_a, char *array_name, int len)
{
  char *c_name;
  wnga_inquire_name(g_a, &c_name);
  ga_c2fstring(c_name, array_name, len);
}

void FATR nga_inquire_name_(Integer *g_a, char *array_name, int len)
{
  char *c_name;
  wnga_inquire_name(g_a, &c_name);
  ga_c2fstring(c_name, array_name, len);
}

logical FATR ga_is_mirrored_(Integer *g_a)
{
  return wnga_is_mirrored(g_a);
}

logical FATR nga_is_mirrored_(Integer *g_a)
{
  return wnga_is_mirrored(g_a);
}

void FATR ga_list_nodeid_(Integer *list, Integer *nprocs)
{
  wnga_list_nodeid(list, nprocs);
}

void FATR nga_list_nodeid_(Integer *list, Integer *nprocs)
{
  wnga_list_nodeid(list, nprocs);
}

Integer FATR nga_locate_num_blocks_(Integer *g_a, Integer *lo, Integer *hi)
{
  return wnga_locate_num_blocks(g_a,lo,hi);
}

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

logical FATR nga_locate_region_( Integer *g_a,
                                 Integer *lo,
                                 Integer *hi,
                                 Integer *map,
                                 Integer *proclist,
                                 Integer *np)
{
  return wnga_locate_region(g_a, lo, hi, map, proclist, np);
}

void FATR ga_lock_(Integer *mutex)
{
  wnga_lock(mutex);
}

void FATR nga_lock_(Integer *mutex)
{
  wnga_lock(mutex);
}

logical FATR ga_locate_(Integer *g_a, Integer *i, Integer *j, Integer *owner)
{
  Integer subscript[2];
  subscript[0] = *i;
  subscript[1] = *j;
  return wnga_locate(g_a, subscript, owner);
}

logical FATR nga_locate_(Integer *g_a, Integer *subscript, Integer *owner)
{
  return wnga_locate(g_a, subscript, owner);
}

void FATR ga_mask_sync_(Integer *begin, Integer *end)
{
  wnga_mask_sync(begin, end);
}

void FATR nga_mask_sync_(Integer *begin, Integer *end)
{
  wnga_mask_sync(begin, end);
}

Integer FATR ga_memory_avail_()
{
  return wnga_memory_avail();
}

Integer FATR nga_memory_avail_()
{
  return wnga_memory_avail();
}

logical FATR ga_memory_limited_()
{
  return wnga_memory_limited();
}

logical FATR nga_memory_limited_()
{
  return wnga_memory_limited();
}

void FATR nga_merge_distr_patch_(Integer *g_a, Integer *alo, Integer *ahi,
                                 Integer *g_b, Integer *blo, Integer *bhi)
{
  wnga_merge_distr_patch(g_a, alo, ahi, g_b, blo, bhi);
}

void FATR ga_merge_mirrored_(Integer *g_a)
{
  wnga_merge_mirrored(g_a);
}

void FATR nga_merge_mirrored_(Integer *g_a)
{
  wnga_merge_mirrored(g_a);
}

void FATR ga_nblock_(Integer *g_a, Integer *nblock)
{
  wnga_nblock(g_a, nblock);
}

void FATR nga_nblock_(Integer *g_a, Integer *nblock)
{
  wnga_nblock(g_a, nblock);
}

Integer FATR ga_ndim_(Integer *g_a)
{
  return wnga_ndim(g_a);
}

Integer FATR nga_ndim_(Integer *g_a)
{
  return wnga_ndim(g_a);
}

Integer FATR ga_nnodes_()
{
  return wnga_nnodes();
}

Integer FATR nga_nnodes_()
{
  return wnga_nnodes();
}

Integer FATR ga_nodeid_()
{
  return wnga_nodeid();
}

Integer FATR nga_nodeid_()
{
  return wnga_nodeid();
}

Integer FATR ga_pgroup_absolute_id_(Integer *grp, Integer *pid)
{
  return wnga_pgroup_absolute_id(grp, pid);
}

Integer FATR nga_pgroup_absolute_id_(Integer *grp, Integer *pid)
{
  return wnga_pgroup_absolute_id(grp, pid);
}

Integer FATR ga_pgroup_create_(Integer *list, Integer *count)
{
  return wnga_pgroup_create(list, count);
}

Integer FATR nga_pgroup_create_(Integer *list, Integer *count)
{
  return wnga_pgroup_create(list, count);
}

logical FATR ga_pgroup_destroy_(Integer *grp)
{
  return wnga_pgroup_destroy(grp);
}

logical FATR nga_pgroup_destroy_(Integer *grp)
{
  return wnga_pgroup_destroy(grp);
}

Integer FATR ga_pgroup_get_default_()
{
  return wnga_pgroup_get_default();
}

Integer FATR nga_pgroup_get_default_()
{
  return wnga_pgroup_get_default();
}

Integer FATR ga_pgroup_get_mirror_()
{
  return wnga_pgroup_get_mirror();
}

Integer FATR nga_pgroup_get_mirror_()
{
  return wnga_pgroup_get_mirror();
}

Integer FATR ga_pgroup_get_world_()
{
  return wnga_pgroup_get_world();
}

Integer FATR nga_pgroup_get_world_()
{
  return wnga_pgroup_get_world();
}

void FATR ga_pgroup_set_default_(Integer *grp)
{
  wnga_pgroup_set_default(grp);
}

void FATR nga_pgroup_set_default_(Integer *grp)
{
  wnga_pgroup_set_default(grp);
}

Integer FATR ga_pgroup_split_(Integer *grp, Integer *grp_num)
{
  return wnga_pgroup_split(grp, grp_num);
}

Integer FATR nga_pgroup_split_(Integer *grp, Integer *grp_num)
{
  return wnga_pgroup_split(grp, grp_num);
}

Integer FATR ga_pgroup_split_irreg_(Integer *grp, Integer *mycolor)
{
  return wnga_pgroup_split_irreg(grp, mycolor);
}

Integer FATR nga_pgroup_split_irreg_(Integer *grp, Integer *mycolor)
{
  return wnga_pgroup_split_irreg(grp, mycolor);
}

Integer FATR ga_pgroup_nnodes_(Integer *grp)
{
  return wnga_pgroup_nnodes(grp);
}

Integer FATR nga_pgroup_nnodes_(Integer *grp)
{
  return wnga_pgroup_nnodes(grp);
}

Integer FATR ga_pgroup_nodeid_(Integer *grp)
{
  return wnga_pgroup_nodeid(grp);
}

Integer FATR nga_pgroup_nodeid_(Integer *grp)
{
  return wnga_pgroup_nodeid(grp);
}

void FATR ga_proc_topology_(Integer* g_a, Integer* proc, Integer* pr,
                            Integer *pc)
{
  Integer subscript[2];
  wnga_proc_topology(g_a, proc, subscript);
  *pr = subscript[0];
  *pc = subscript[1];
}


void FATR nga_proc_topology_(Integer* g_a, Integer* proc, Integer* subscript)
{
  wnga_proc_topology(g_a, proc, subscript);
}

void FATR ga_set_debug_(logical *flag)
{
  wnga_set_debug(flag);
}

void FATR nga_set_debug_(logical *flag)
{
  wnga_set_debug(flag);
}

/* Routines from onesided.c */

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

void FATR nga_nbput_(Integer *g_a, Integer *lo,
                     Integer *hi, void *buf, Integer *ld,
                     Integer *nbhandle)
{
    wnga_nbput(g_a, lo, hi, buf, ld, nbhandle);
}

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

void FATR nga_put_(Integer *g_a, Integer *lo,
                   Integer *hi, void *buf, Integer *ld)
{
    wnga_put(g_a, lo, hi, buf, ld);
}

/* Routines from global.util.c */

#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
void FATR ga_error_(char *string, Integer *icode, int slen)
#else
void FATR ga_error_(char *string, int slen, Integer *icode)
#endif
{
#define FMSG 256
  char buf[FMSG];
  ga_f2cstring(string,slen, buf, FMSG);
  wnga_error(buf,*icode);
}

#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
void FATR nga_error_(char *string, Integer *icode, int slen)
#else
void FATR nga_error_(char *string, int slen, Integer *icode)
#endif
{
#define FMSG 256
  char buf[FMSG];
  ga_f2cstring(string,slen, buf, FMSG);
  wnga_error(buf,*icode);
}

