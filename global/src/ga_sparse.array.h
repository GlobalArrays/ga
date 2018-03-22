/*$Id: ga_sparse.array.h,v 1.40.2.4 2007/12/18 18:41:27 d3g293 Exp $ */
#include "gaconfig.h"
#include "typesf2c.h"

/**
 * Struct containing all data needed to keep track of iterator state
 */
typedef struct {
  Integer idim,jdim;/* dimensions of sparse array */
  Integer g_data;   /* global array containing data values of sparse matrix */
  Integer g_j;      /* global array containing j indices of sparse matrix */
  Integer g_i;      /* global array containing first j index for row i */
  Integer grp;      /* handle for process group on which array is defined */
  Integer ilo, ihi; /* minimum and maximum rows indices contained on this process */
  Integer nprocs;   /* number of processors containing this array */
  Integer nblocks;  /* number of non-zero sparse blocks contained on this process */
  Integer *blkidx;  /* array containing indices of non-zero blocks */
  Integer *blksize; /* array containining sizes of non-zero blocks */
  Integer *offset;  /* array containing starting index in g_i for each block */
  Integer *idx;     /* local buffer containing i indices */
  Integer *jdx;     /* local buffer containing j indices */
  void    *val;     /* local buffer containing values */
  Integer nval;     /* number of values currently stored in local buffers */
  Ingeger maxval;   /* maximum number of values that can currently */
                    /* be stored in local buffers */
  Integer type;     /* type of data stored in array */
  Integer size;     /* size of data element */
  Integer active;   /* array is currently in use */
  Integer ready;    /* array data has been distributed */
} _sparse_array;

extern _sparse_array *SPA;

extern void sai_init_sparse_arrays();
extern void pnga_sprs_array_add_element(Integer s_a, Integer i, Integer j, void *val);
extern Integer pnga_sprs_array_create(Integer idim, Integer jdim, Integer type);
extern Integer pnga_sprs_array_assemble(Integer s_a);

