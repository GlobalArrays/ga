/* XGA header file */
#ifndef _XGA_HEADER_H
#define _XGA_HEADER_H
#include "xga_types.hpp"
#include "xga_private.hpp"

namespace XGA {

template<typename _type>
class GlobalArray {

public:

  /**
   * Basic Constructor
   * @param[in] group home group for global array
   * @param[in] ndim dimension of global array
   * @param[in] dims dimensions of global array
   */
  GlobalArray(Group *group, int ndim, int64_t *dims)
  {
    p_ndim = ndim;
    if constexpr(std::is_same_v<_type,int>) {
      p_datatype = XGA_INT;
    } else if constexpr(std::is_same_v<_type,long>) {
      p_datatype = XGA_LONG;
    } else if constexpr(std::is_same_v<_type,long long>) {
      p_datatype = XGA_LONGLONG;
    } else if constexpr(std::is_same_v<_type,float>) {
      p_datatype = XGA_FLOAT;
    } else if constexpr(std::is_same_v<_type,double>) {
      p_datatype = XGA_DOUBLE;
    } else if constexpr(std::is_same_v<_type,std::complex<float> >) {
      p_datatype = XGA_COMPLEX;
    } else if constexpr(std::is_same_v<_type,std::complex<double> >) {
      p_datatype = XGA_DCOMPLEX;
    }

    p_group = group;

    p_Impl = new p_GA(group, ndim, dims, p_datatype);
  };

  GlobalArray(Group *group, int ndim, int *dims)
  {
    p_ndim = ndim;
    int64_t tdims[MAXDIM];
    int i;
    for (i=0; i<ndim; i++) tdims[i] = static_cast<int64_t>(dims[i]);
    if constexpr(std::is_same_v<_type,int>) {
      p_datatype = XGA_INT;
    } else if constexpr(std::is_same_v<_type,long>) {
      p_datatype = XGA_LONG;
    } else if constexpr(std::is_same_v<_type,long long>) {
      p_datatype = XGA_LONGLONG;
    } else if constexpr(std::is_same_v<_type,float>) {
      p_datatype = XGA_FLOAT;
    } else if constexpr(std::is_same_v<_type,double>) {
      p_datatype = XGA_DOUBLE;
    } else if constexpr(std::is_same_v<_type,std::complex<float> >) {
      p_datatype = XGA_COMPLEX;
    } else if constexpr(std::is_same_v<_type,std::complex<double> >) {
      p_datatype = XGA_DCOMPLEX;
    }

    p_group = group;

    p_Impl = new p_GA(group, ndim, tdims, p_datatype);
  }

  /**
   * Basic destructor
   */
  ~GlobalArray()
  {
    if (p_Impl) delete p_Impl;
  };

  /**
   * Set data distribution type
   * @param[in] distr data distribution type
   */
  void setDataDistribution(XGA::data_distribution distr)
  {
    p_Impl->setDataDistribution(distr);
  }

  /**
   * Set block sizes and processor grid for ScaLAPACK-style data
   * distribution. This is only strictly an ScaLAPACK distribution in
   * 2 dimensions but the generalization to higher dimensions is
   * straightforward
   * @param[in] dims dimensions of individual blocks
   * @param[in] prod_grid dimension of processor grid
   */
  void setBlockLayout(int64_t *dims, int *proc_grid)
  {
    p_Impl->setBlockLayout(dims, proc_grid);
  }
  void setBlockLayout(int *dims, int *proc_grid)
  {
    int64_t *tdims = new int64_t[p_ndim];
    int i;
    for (i=0; i<p_ndim; i++) tdims[i] = static_cast<int64_t>(dims[i]);
    p_Impl->setBlockLayout(tdims, proc_grid);
    delete [] tdims;
  }

  /**
   * @param[in] mapc array containing partitions along each axis
   * @param[in] nblock array containing processor decomposition
   */
  void setIrreglarDistribution(int64_t *mapc, int *nblock)
  {
    p_Impl->setIrregularDistribution(mapc, nblock);
  }
  void setIrreglarDistribution(int *mapc, int *nblock)
  {
    int ntot = 0;
    int i;
    for (i=0;  i<p_ndim; i++) ntot += nblock[i];
    int64_t *tmap = new int64_t[ntot];
    for (i=0; i<ntot; i++) tmap[i] = static_cast<int64_t>(mapc[i]);
    p_Impl->setIrregularDistribution(tmap, nblock);
    delete [] tmap;
  }

  /**
   * Allocate resources to create global array
   */
  void allocate()
  {
    p_Impl->allocate();
  }

  /**
   * Find block owned by processor proc
   * @param[in] proc processor being queried
   * @param[out] lo,hi lower and upper bounding indices of block
   *             owned by processor proc
   */
  void distribution(const int proc, int64_t *lo, int64_t *hi)
  {
    p_Impl->distribution(proc,lo,hi);
  }
  void distribution(const int proc, int *lo, int *hi)
  {
    int i;
    int64_t tlo[MAXDIM], thi[MAXDIM];
    p_Impl->distribution(proc,tlo,thi);
    for (i=0; i<p_ndim; i++) {
      lo[i] = static_cast<int>(tlo[i]);
      hi[i] = static_cast<int>(thi[i]);
    }
  }

  /**
   * Locate process that owns a particular array element
   * @param[in] subscript indices of element
   * @param[out] owner process that owns element indexed by subscript
   * @return false if subscript is not located in array
   */
  bool locate(const int64_t *subscript, int *owner)
  {
    return p_Impl->locate(subscript, owner);
  }
  bool locate(const int *subscript, int *owner)
  {
    int64_t tsub[MAXDIM];
    int i;
    for (i=0; i<p_ndim; i++) tsub[i] = static_cast<int64_t>(subscript[i]);
    return p_Impl->locate(tsub, owner);
  }

  /**
   * @param[in] lo,hi lower and upper indices of patch in global array
   * @param[out] map list of lower and upper indices for portion of
   *             patch the exists on each processor containing a portion
   *             of the patch. The map is constructed so that for a D
   *             dimensional global array, the first D elements are the
   *             lower indices on the first processor in proclist, the
   *             next D elements are the upper indices of the first
   *             processor in proclist, the next D elements are the
   *             lower indices for the second processor in proclist and
   *             so on.
   * @param[out] proclist list of processors containing some portion of
   *             patch
   * @param[out] np total number of processors containing some portion
   *             of the patch
   * @return false if bounds of patch are invalid
   *
   * For a block cyclic data distribution, this function returns a list
   * of blocks that cover the region, along with the lower and upper
   * indices of each block.
   */
  bool locateRegion(const int64_t *lo, const int64_t *hi,
      std::vector<int64_t> &map, std::vector<int> proclist, int *np)
  {
    return p_Impl->locateRegion(lo, hi, map, proclist, np);
  }
  bool locateRegion(const int *lo, const int *hi,
      std::vector<int> &map, std::vector<int> proclist, int *np)
  {
    int64_t tlo[MAXDIM], thi[MAXDIM];
    std::vector<int64_t> tmap;
    int i;
    for (i=0; i<p_ndim; i++) {
      tlo[i] = static_cast<int64_t>(lo[i]);
      thi[i] = static_cast<int64_t>(hi[i]);
    }
    bool ret = p_Impl->locateRegion(tlo, thi, tmap, proclist, np);
    map.clear();
    size_t size = tmap.size();
    for (i=0; i<size; i++) map.push_back(static_cast<int>(tmap[i]));
    return ret;
  }

  /**
   * Access data corresponding to a specific patch
   * @param[in] plo,phi lower and upper indices of patch
   * @param[out] rptr pointer to data
   * @param[out] ld array of strides for data
   */
  void accessPtr(int64_t *plo, int64_t *phi, void **rptr, int64_t *ld)
  {
    p_Impl->accessPtr(plo, phi, rptr, ld);
  }
  void accessPtr(int *plo, int *phi, void **rptr, int *ld)
  {
    int i;
    int64_t tlo[MAXDIM], thi[MAXDIM], tld[MAXDIM];
    tld[0] = 1;
    for (i=0; i<p_ndim; i++) {
      tlo[i] = static_cast<int64_t>(plo[i]);
      thi[i] = static_cast<int64_t>(phi[i]);
    }
    p_Impl->accessPtr(tlo, thi, rptr, tld);
    for (i=0; i<p_ndim-1; i++) {
      ld[i] = static_cast<int>(tld[i]);
    }
  }

  /**
   * Access data corresponding to a specific block
   * @param[in] index indices of block in proc grid
   * @param[out] rptr pointer to data
   * @param[out] ld array of strides for block
   */
  void accessBlockGridPtr(int *index, void **rptr, int64_t *ld)
  {
    p_Impl->accessBlockGridPtr(index, rptr, ld);
  }
  void accessBlockGridPtr(int *index, void **rptr, int *ld)
  {
    int i;
    int64_t tld[MAXDIM];
    p_Impl->accessBlockGridPtr(index, rptr, tld);
    for (i=0; i<p_ndim-1; i++) ld[i] = static_cast<int>(tld[i]);
  }

  /**
   * Release data corresponding to a specific patch
   * @param[in] plo,phi lower and upper indices of patch
   */
  void releasePtr(int64_t *plo, int64_t *phi)
  {
    p_Impl->releasePtr(plo, phi);
  }
  void releasePtr(int *plo, int *phi)
  {
    int i;
    int64_t tlo[MAXDIM], thi[MAXDIM];
    for (i=0; i<p_ndim; i++) {
      tlo[i] = static_cast<int64_t>(plo[i]);
      thi[i] = static_cast<int64_t>(phi[i]);
    }
    p_Impl->releasePtr(tlo, thi);
  }

  /**
   * Release data corresponding to a specific block
   * @param[in] index indices of block in proc grid
   */
  void releaseBlockGridPtr(int *index)
  {
    p_Impl->releaseBlockGridPtr(index);
  }

  /**
   * Synchronize global array across all processor that are hosting
   * the array
   */
  void sync()
  {
    p_Impl->sync();
  }

  /**
   * Copy data from local buffer to global array
   * @param[in] lo,hi bounding indices of block in global array
   * @param[in] buf pointer to first element in local buffer
   * @param[in] ld strides in local buffer
   */
  void put(int64_t *lo, int64_t *hi, _type* buf, int64_t *ld)
  {
    p_Impl->put(lo, hi, buf, ld);
  }
  void put(int *lo, int *hi, _type* buf, int *ld)
  {
    int i;
    int64_t tlo[MAXDIM], thi[MAXDIM], tld[MAXDIM];
    tld[0] = 1;
    for (i=0; i<p_ndim; i++) {
      tlo[i] = static_cast<int64_t>(lo[i]);
      thi[i] = static_cast<int64_t>(hi[i]);
      if (i<p_ndim-1) tld[i] = static_cast<int64_t>(ld[i]);
    }
    p_Impl->put(tlo, thi, buf, tld);
  }

  /**
   * Copy data from global array to local buffer
   * @param[in] lo,hi bounding indices of block in global array
   * @param[in] buf pointer to first element in local buffer
   * @param[in] ld strides in local buffer
   */
  void get(int64_t *lo, int64_t *hi, _type* buf, int64_t *ld)
  {
    p_Impl->get(lo, hi, buf, ld);
  }
  void get(int *lo, int *hi, _type* buf, int *ld)
  {
    int i;
    int64_t tlo[MAXDIM], thi[MAXDIM], tld[MAXDIM];
    tld[0] = 1;
    for (i=0; i<p_ndim; i++) {
      tlo[i] = static_cast<int64_t>(lo[i]);
      thi[i] = static_cast<int64_t>(hi[i]);
      if (i<p_ndim-1) tld[i] = static_cast<int64_t>(ld[i]);
    }
    p_Impl->get(tlo, thi, buf, tld);
  }

  /**
   * Accumulate data from local buffer to global array
   * @param[in] lo,hi bounding indices of block in global array
   * @param[in] buf pointer to first element in local buffer
   * @param[in] ld strides in local buffer
   * @param[in] alpha scale factor for adding contents of buffer
   *            to global array
   */
  void acc(int64_t *lo, int64_t *hi, _type* buf, int64_t *ld, _type alpha)
  {
    _type talpha = alpha;
    p_Impl->acc(lo, hi, buf, ld, &talpha);
  }
  void acc(int *lo, int *hi, _type* buf, int *ld, _type alpha)
  {
    _type talpha = alpha;
    int i;
    int64_t tlo[MAXDIM], thi[MAXDIM], tld[MAXDIM];
    tld[0] = 1;
    for (i=0; i<p_ndim; i++) {
      tlo[i] = static_cast<int64_t>(lo[i]);
      thi[i] = static_cast<int64_t>(hi[i]);
      if (i<p_ndim-1) tld[i] = static_cast<int64_t>(ld[i]);
    }
    p_Impl->acc(tlo, thi, buf, tld, &talpha);
  }

  /**
   * Scatter values to random locations in a global array
   * @param[in] v array containing values to be scattered to array. The
   *            type of values in v must match the type of values
   *            in the global array
   * @param[in] subscript array of indices representing locations of
   *            values in global array. Each ndim locations represents
   *            the index location of one value
   * @param[in] nv number of values to scattered
   */
  void scatter(_type *v, int64_t *subscript, int64_t nv)
  {
    p_Impl->scatter(v, subscript, nv, 1);
  }
  void scatter(_type *v, int *subscript, int nv)
  {
    int64_t tnv = static_cast<int64_t>(nv);
    p_Impl->scatter(v, subscript, nv, 0);
  }

  /**
   * Gather values from random locations in a global array
   * @param[in] v array containing values gathered from array. The
   *            type of values in v must match the type of values
   *            in the global array
   * @param[in] subscript array of indices representing locations of
   *            values in global array. Each ndim locations represents
   *            the index location of one value
   * @param[in] nv number of values to gathered 
   */
  void gather(_type *v, int64_t *subscript, int64_t nv)
  {
    p_Impl->gather(v, subscript, nv, 1);
  }
  void gather(_type *v, int *subscript, int nv)
  {
    int64_t tnv = static_cast<int64_t>(nv);
    p_Impl->gather(v, subscript, tnv, 0);
  }

  /**
   * Accumulate values to random locations in a global array
   * @param[in] v array containing values to be accumulated to array. The
   *            type of values in v must match the type of values
   *            in the global array
   * @param[in] subscript array of indices representing locations of
   *            values in global array. Each ndim locations represents
   *            the index location of one value
   * @param[in] nv number of values to scattered
   * @param[in] scale scale factor to multiply each value by before being
   *            accumulated
   */
  void scatterAcc(_type *v, int64_t *subscript, int64_t nv, _type alpha)
  {
    _type talpha = alpha;
    p_Impl->scatterAcc(v, subscript, nv, &talpha, 1);
  }
  void scatterAcc(_type *v, int *subscript, int nv, _type alpha)
  {
    _type talpha = alpha;
    int64_t tnv = static_cast<int64_t>(nv);
    p_Impl->scatterAcc(v, subscript, tnv, &talpha, 0);
  }

  /**
   * Read the value at location indicated by subscript and increment b
   * the amount inc. This operation is atomic with respect to other read
   * increment operations.
   * @param[in] subscript locate of element to be read and incremented
   * @param[in] inc amount to increment element
   * @return current value of element
   */
  _type readInc(int64_t *subscript, _type inc)
  {
    _type result;
    p_Impl->readInc(subscript, &inc, &result);
    return result;
  }
  _type readInc(int *subscript, _type inc)
  {
    int i;
    int64_t tsub[MAXDIM];
    _type result;
    for (i=0; i<p_ndim; i++) tsub[i] = static_cast<int64_t>(subscript[i]);
    p_Impl->readInc(tsub, &inc, &result);
    return result;
  }

  /**
   * Set all values in the array to zero
   */
  void zero()
  {
    p_Impl->zero();
  }

  /**
   * Fill array with a single value
   * @param value pointer to value being filled
   */
  void fill(_type value)
  {
    _type tvalue = value;
    p_Impl->fill(&tvalue);
  }

  /**
   * Clear internal data from allocation so code can exit cleanly
   */
  void clear()
  {
    delete p_Impl;
    p_Impl = NULL;
  }

private:

  xga_types p_datatype = XGA_UNKNOWN;

  int p_ndim;

  p_GA *p_Impl;

  Group *p_group;
};
}
#endif
