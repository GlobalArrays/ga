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

/**
 *  Routines from base.c
 */

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

/**
 *  Routines from onesided.c
 */

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

/**
 *  Routines from global.util.c
 */

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
