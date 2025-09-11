#ifndef _XGA_MACROS_HPP
#define _XGA_MACROS_HPP

/* find the proc grid indices for a block indexed by proc */
#define XGA_FIND_PROC_INDICES_M(proc, index) {                             \
  int64_t _itmp, _i;                                                       \
  _itmp = proc;                                                            \
  index[0] = _itmp%nblock[0];                                              \
  for (_i=1; _i<p_ndim; _i++) {                                            \
    _itmp = (_itmp-index[_i-1])/nblock[_i-1];                              \
    index[_i] = _itmp%nblock[_i];                                          \
  }                                                                        \
}

/* this macro finds the block index corresponding to a given set of indices */
#define XGA_FIND_BLOCK_FROM_INDICES_M(_iblock,index) {                     \
  int _i;                                                                  \
  _iblock = index[p_ndim-1];                                               \
  for (_i=p_ndim-2; _i >= 0; _i--) {                                       \
    _iblock  = _iblock*nblock[_i]+index[_i];                               \
  }                                                                        \
}

/* this macro finds the proc that owns a given set block indices
   using the ScaLAPACK data distribution */
#define XGA_FIND_PROC_FROM_SL_INDICES_M(proc,index) {                          \
  int _i;                                                                      \
  int64_t _index2[MAXDIM];                                                     \
  for (_i=0; _i<p_ndim; _i++) {                                                \
    _index2[_i] = index[_i]%nblock[_i];                                        \
  }                                                                            \
  proc = _index2[0];                                                           \
  for (_i=1; _i < p_ndim; _i++) {                                              \
    proc = proc*nblock[_i]+_index2[_i];                                        \
  }                                                                            \
}

/* find the tile indices for a tile indexed by proc */
#define XGA_FIND_TILE_PROC_INDICES_M(proc, index) {                        \
  int64_t _itmp, _i;                                                       \
  _itmp = proc;                                                            \
  index[0] = _itmp%nblock[0];                                              \
  for (_i=1; _i<p_ndim; _i++) {                                            \
    _itmp = (_itmp-index[_i-1])/nblock[_i-1];                              \
    index[_i] = _itmp%nblock[_i];                                          \
  }                                                                        \
}

/* this macro finds the proc that owns a given set block indices
   using the ScaLAPACK data distribution */
#define XGA_FIND_TILE_PROC_FROM_INDICES_M(proc,index) {                        \
  int _i;                                                                      \
  int64_t _index2[MAXDIM];                                                     \
  for (_i=0; _i<p_ndim; _i++) {                                                \
    _index2[_i] = index[_i]%nblock[_i];                                        \
  }                                                                            \
  proc = _index2[p_ndim-1];                                                    \
  for (_i=p_ndim-2; _i >= 0; _i--) {                                           \
    proc = proc*nblock[_i]+_index2[_i];                                        \
  }                                                                            \
}


#define XGA_FIND_TILE_PROC_FROM_SL_INDICES_M(proc,_index) {                    \
  int _i;                                                                      \
  int64_t _index2[MAXDIM];                                                     \
  for (_i=0; _i<p_ndim; _i++) {                                                \
    _index2[_i] = _index[_i]%nblock[_i];                                       \
  }                                                                            \
  proc = _index2[0];                                                           \
  for (_i=1; _i <p_ndim; _i++) {                                               \
    proc = proc*nblock[_i]+_index2[_i];                                        \
  }                                                                            \
}


/* this macro finds the block indices for a given block */
#define XGA_FIND_BLOCK_INDICES_M(nblock, index) {                              \
  int _itmp, _i;                                                               \
  int _ndim = p_ndim;                                                          \
  _itmp = nblock;                                                              \
  index[0] = _itmp%blk_num[0];                                                 \
  for (_i=1; _i<_ndim; _i++) {                                                 \
    _itmp = (_itmp-index[_i-1])/blk_num[_i-1];                                 \
    index[_i] = _itmp%blk_num[_i];                                             \
  }                                                                            \
}

/* this macro finds coordinates of the chunk of array owned by processor proc */
#define XGA_OWNS_NO_HANDLE_M(proc, lo, hi)                                     \
{                                                                              \
  int64_t _loc, _nb, _d, _index, _dim=p_ndim,_dimstart=0, _dimpos;             \
  for(_nb=1, _d=0; _d<_dim; _d++)_nb *= (int64_t)nblock[_d];                   \
  if((int64_t)proc > _nb - 1 || proc<0){                                       \
    for(_d=0; _d<_dim; _d++){                                                  \
      lo[_d] = (int64_t)0;                                                     \
      hi[_d] = (int64_t)-1;}                                                   \
  }                                                                            \
  else{                                                                        \
    _index = proc;                                                             \
    for(_d=0; _d<_dim; _d++){                                                  \
      _loc = _index% (int64_t)nblock[_d];                                      \
      _index  /= (int64_t)nblock[_d];                                          \
      _dimpos = _loc + _dimstart; /* correction to find place in p_mapc */     \
      _dimstart += (int64_t)nblock[_d];                                        \
      lo[_d] = (int64_t)p_mapc[_dimpos];                                       \
      if (_loc==nblock[_d]-1) hi[_d]=p_dims[_d]-1;                               \
      else hi[_d] = p_mapc[_dimpos+1]-1;                                       \
    }                                                                          \
  }                                                                            \
}


/* this macro finds cordinates of the chunk of array owned by processor proc
 * proc: processor (or block) index
 * lo: lower indices of elements owned by processor (or block)
 * hi: upper indices of elements owned by processor (or block)
 */
#define XGA_OWNS_M(proc, lo, hi)                                               \
{                                                                              \
  if (p_distr == REGULAR) {                                                    \
    XGA_OWNS_NO_HANDLE_M(proc,lo, hi)                                          \
  } else if (p_distr == TILED_IRREG) {                                         \
    int _index[MAXDIM];                                                        \
    int _i;                                                                    \
    int _ndim = p_ndim;                                                        \
    int _offset = 0;                                                           \
    XGA_FIND_BLOCK_INDICES_M(proc,_index);                                     \
    for (_i=0; _i<_ndim; _i++) {                                               \
      lo[_i] = p_mapc[_offset+_index[_i]];                                     \
      if (_index[_i] < blk_num[_i]-1) {                                        \
        hi[_i] = p_mapc[_offset+_index[_i]+1]-1;                               \
      } else {                                                                 \
        hi[_i] = p_dims[_i]-1;                                                 \
      }                                                                        \
      _offset += blk_num[_i];                                                  \
    }                                                                          \
  }                                                                            \
}

#define XGA_CHECKSUBSCRIPT_M(subscr, lo, hi)                                   \
{                                                                              \
  int _d;                                                                      \
  for(_d=0; _d<p_ndim; _d++)                                                   \
  if( subscr[_d]<  lo[_d] ||  subscr[_d]>  hi[_d]){                            \
    char err_string[512];                                                      \
    sprintf(err_string,"check subscript failed:%ld not in (%ld:%ld) dim=%d",   \
        (int64_t)subscr[_d],  (int64_t)lo[_d],  (int64_t)hi[_d], _d);          \
    p_env->error(err_string, _d);                                              \
  }                                                                            \
}


/**
 * Return pointer (ptr_loc) to location in memory of element with subscripts
 * (subscript). Also return physical dimensions of array in memory in ld.
 */
#define XGA_LOCATION_M(proc, subscript, ptr_loc, ld)                       \
{                                                                          \
  int64_t _offset=0, _d, _w, _factor=1, _last=p_ndim-1;                    \
  std::vector<int64_t> _lo(p_ndim), _hi(p_ndim);                           \
  int64_t _pinv, _p_handle;                                                \
                                                                           \
  XGA_OWNS_M(proc, _lo, _hi);                                              \
  XGA_CHECKSUBSCRIPT_M(subscript, _lo, _hi);                               \
  if(_last==0) ld[0]=_hi[0]- _lo[0]+1+2*(int64_t)width[0];                 \
  for(_d=_last; _d > 0; _d--)            {                                 \
    _w = width[_d];                                                        \
    _offset += (subscript[_d]-_lo[_d]+_w) * _factor;                       \
    ld[_d-1] = _hi[_d] - _lo[_d] + 1 + 2*_w;                                 \
    _factor *= ld[_d-1];                                                     \
  }                                                                        \
  _offset += (subscript[0]-_lo[0]                                          \
      + width[0]) * _factor;                                               \
  _pinv=p_group->getLocalRank(proc);                                       \
  *(ptr_loc) = static_cast<char*>(ptr[_pinv])+_offset*p_elemsize;          \
}

/**
 * map_ij: pointer to map array containing axis partitions
 * n: number of blocks along axis
 * scale: factor for coming up with an initial guess
 * elem: array element index that we are trying to find
 * block: index of block containing elem
 */
#define XGA_FINDBLOCK_M(map_ij,n,scale,elem,block)                  \
{                                                                   \
  int64_t candidate, b;                                             \
  bool found;                                                       \
  int64_t *map= (map_ij);                                           \
                                                                    \
  candidate = static_cast<int64_t>(scale*static_cast<double>(elem));\
  found = false;                                                    \
  if(map[candidate] <= (elem)){ /* search downward */               \
    b= candidate;                                                   \
    while(b<(n)-1){                                                 \
      found = (map[b+1]>(elem));                                    \
      if(found)break;                                               \
      b++;                                                          \
    }                                                               \
  } else { /* search upward */                                      \
    b= candidate-1;                                                 \
    while(b>=0){                                                    \
      found = (map[b]<=(elem));                                     \
      if(found)break;                                               \
      b--;                                                          \
    }                                                               \
  }                                                                 \
  if(!found)b=(n)-1;                                                \
  *(block) = b;                                                     \
}

/**
 * Find indices of block containing the array element at the location
 * in subscript.
 */
#define XGA_FIND_BLOCK_INDICES_FROM_SUBSCRIPT_M(subscript,index)    \
{                                                                   \
  int _type = p_distr;                                              \
  int64_t _offset;                                                  \
  int _i;                                                           \
  if (_type == REGULAR) {                                           \
    for (_i=0, _offset=0; _i<p_ndim; _i++) {                        \
      XGA_FINDBLOCK_M(p_mapc+_offset,nblock[_i],                    \
          scale[_i],subscript[_i],&index[_i]);                      \
      _offset += nblock[_i];                                        \
    }                                                               \
  } else if (_type == TILED_IRREG) {                                \
    for (_i=0, _offset=0; _i<p_ndim; _i++) {                        \
      XGA_FINDBLOCK_M(p_mapc+_offset, blk_num[_i],                  \
          scale[_i],subscript[_i],&index[_i]);                      \
      _offset += blk_num[_i];                                       \
    }                                                               \
  } else {                                                          \
    for (_i=0; _i<p_ndim; _i++) {                                   \
      index[_i] = (subscript[_i]-1)/blk_dims[_i];                   \
    }                                                               \
  }                                                                 \
}

/**
 * Find the pointer to the location in memory on process proc
 * of the element indexed by subscript
 */
#define XGA_LOCATE_PTR_M(proc, _subscript, _ptr_loc)                \
{                                                                   \
  int64_t _offset=0, _d, _w, _factor=1, _last=p_ndim-1;             \
  int64_t _lo[MAXDIM], _hi[MAXDIM], _iproc;                         \
  _iproc = proc;                                                    \
  XGA_OWNS_M(proc, _lo, _hi);                                       \
  XGA_CHECKSUBSCRIPT_M(_subscript, _lo, _hi);                       \
  for (_d=_last; _d>0;  _d--) {                                     \
     _w = width[_d];                                                \
    _offset += (_subscript[_d]-_lo[_d]+_w)*_factor;                 \
    _factor *= _hi[_d] - _lo[_d] + 1 + 2*_w;                        \
  }                                                                 \
  _offset += (_subscript[0]-_lo[0]+width[0])*_factor;               \
  *(_ptr_loc) = static_cast<void*>(static_cast<char*>(ptr[_iproc])  \
      + _offset*p_elemsize);                                        \
}

#define XGA_REGIONERROR_M(_ndim, lo, hi, val){                      \
  int _d, _l;                                                       \
  const char *str= "cannot locate region: ";                        \
  char err_string[512];                                             \
  sprintf(err_string, "%s", str);                                   \
  _d=0;                                                             \
  _l = strlen(str);                                                 \
  sprintf(err_string+_l, " [%ld:%ld ",lo[_d],hi[_d]);               \
  _l=strlen(err_string);                                            \
  for(_d=1; _d< _ndim; _d++){                                       \
    sprintf(err_string+_l, ",%ld:%ld ",lo[_d],hi[_d]);              \
    _l=strlen(err_string);                                          \
  }                                                                 \
  sprintf(err_string+_l, "%s", "]");                                \
  _l=strlen(err_string);                                            \
  p_env->error(err_string, val);                                    \
}
#endif
