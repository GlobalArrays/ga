/*$Id: iterator.h,v 1.40.2.4 2007/12/18 18:41:27 d3g293 Exp $ */
#include "gaconfig.h"
#include "typesf2c.h"

/**
 * Struct containing all data needed to keep track of iterator state
 */
typedef struct {
  Integer g_a;
  Integer lo[MAXDIM];
  Integer hi[MAXDIM];
  Integer count;
  Integer *map;
  Integer *proclist;
  int     *proclistperm;
  Integer *mapc;
  Integer nproc;
  Integer offset;
  Integer iproc;
  Integer iblock;
  Integer new_proc;
  Integer oversize;
  Integer lobuf[MAXDIM];
  Integer hibuf[MAXDIM];
  /* ScalPACK distribution parameters*/
  Integer blk_num[MAXDIM];
  Integer blk_dim[MAXDIM];
  Integer blk_inc[MAXDIM];
  Integer blk_ld[MAXDIM];
  Integer hlf_blk[MAXDIM];
  Integer blk_size[MAXDIM];
  Integer proc_index[MAXDIM];
  Integer index[MAXDIM];
} _iterator_hdl;

#if 0
/* this macro finds the block indices for a given block */
#define gam_find_block_indices(ga_handle,nblock,index) {                       \
  int _itmp, _i;                                                       \
  int _ndim = GA[ga_handle].ndim;                                              \
  _itmp = nblock;                                                              \
  index[0] = _itmp%GA[ga_handle].num_blocks[0];                                \
  for (_i=1; _i<_ndim; _i++) {                                                 \
    _itmp = (_itmp-index[_i-1])/GA[ga_handle].num_blocks[_i-1];                \
    index[_i] = _itmp%GA[ga_handle].num_blocks[_i];                            \
  }                                                                            \
}

/* this macro finds the ScaLAPACK indices for a given processor */
#ifdef COMPACT_SCALAPACK
#define gam_find_proc_indices(ga_handle,proc,index) {                          \
  Integer _itmp, _i;                                                           \
  Integer _ndim = GA[ga_handle].ndim;                                          \
  _itmp = proc;                                                                \
  index[0] = _itmp%GA[ga_handle].nblock[0];                                    \
  for (_i=1; _i<_ndim; _i++) {                                                 \
    _itmp = (_itmp-index[_i-1])/GA[ga_handle].nblock[_i-1];                    \
    index[_i] = _itmp%GA[ga_handle].nblock[_i];                                \
  }                                                                            \
}
#else
#define gam_find_proc_indices(ga_handle,proc,index) {                          \
  Integer _itmp, _i;                                                           \
  Integer _ndim = GA[ga_handle].ndim;                                          \
  _itmp = proc;                                                                \
  index[_ndim-1] = _itmp%GA[ga_handle].nblock[_ndim-1];                        \
  for (_i=_ndim-2; _i>=0; _i--) {                                              \
    _itmp = (_itmp-index[_i+1])/GA[ga_handle].nblock[_i+1];                    \
    index[_i] = _itmp%GA[ga_handle].nblock[_i];                                \
  }                                                                            \
}
#endif

/* this macro finds the block index corresponding to a given set of indices */
#define gam_find_block_from_indices(ga_handle,nblock,index) {                  \
  int _ndim = GA[ga_handle].ndim;                                              \
  int _i;                                                                      \
  nblock = index[_ndim-1];                                                     \
  for (_i=_ndim-2; _i >= 0; _i--) {                                            \
    nblock  = nblock*GA[ga_handle].num_blocks[_i]+index[_i];                   \
  }                                                                            \
}

/* this macro finds the proc that owns a given set block indices
   using the ScaLAPACK data distribution */
#ifdef COMPACT_SCALAPACK
#define gam_find_proc_from_sl_indices(ga_handle,proc,index) {                  \
  int _ndim = GA[ga_handle].ndim;                                              \
  int _i;                                                                      \
  Integer _index2[MAXDIM];                                                     \
  for (_i=0; _i<_ndim; _i++) {                                                 \
    _index2[_i] = index[_i]%GA[ga_handle].nblock[_i];                          \
  }                                                                            \
  proc = _index2[_ndim-1];                                                     \
  for (_i=_ndim-2; _i >= 0; _i--) {                                            \
    proc = proc*GA[ga_handle].nblock[_i]+_index2[_i];                          \
  }                                                                            \
}
#else
#define gam_find_proc_from_sl_indices(ga_handle,proc,index) {                  \
  int _ndim = GA[ga_handle].ndim;                                              \
  int _i;                                                                      \
  Integer _index2[MAXDIM];                                                     \
  for (_i=0; _i<_ndim; _i++) {                                                 \
    _index2[_i] = index[_i]%GA[ga_handle].nblock[_i];                          \
  }                                                                            \
  proc = _index2[0];                                                           \
  for (_i=1; _i < _ndim; _i++) {                                               \
    proc = proc*GA[ga_handle].nblock[_i]+_index2[_i];                          \
  }                                                                            \
}
#endif
/* this macro computes the strides on both the remote and local
   processors that map out the data. ld and ldrem are the physical dimensions
   of the memory on both the local and remote processors. */
/* NEEDS C_INT64 CONVERSION */
#define gam_setstride(ndim, size, ld, ldrem, stride_rem, stride_loc){\
  int _i;                                                            \
  stride_rem[0]= stride_loc[0] = (int)size;                          \
  __CRAYX1_PRAGMA("_CRI novector");                                  \
  for(_i=0;_i<ndim-1;_i++){                                          \
    stride_rem[_i] *= (int)ldrem[_i];                                \
    stride_loc[_i] *= (int)ld[_i];                                   \
      stride_rem[_i+1] = stride_rem[_i];                             \
      stride_loc[_i+1] = stride_loc[_i];                             \
  }                                                                  \
}
#endif

extern void gai_iterator_init(Integer, Integer [], Integer [], _iterator_hdl *);
extern void gai_iterator_reset(_iterator_hdl *);
extern int gai_iterator_next(_iterator_hdl *, int *, Integer *[],
        Integer *[], char **, Integer []);
extern int gai_iterator_last(_iterator_hdl *);
extern void gai_iterator_destroy(_iterator_hdl *);

extern void pnga_local_iterator_init(Integer, _iterator_hdl*);
extern int pnga_local_iterator_next(_iterator_hdl*, Integer[],
                Integer[], char**, Integer[]);
