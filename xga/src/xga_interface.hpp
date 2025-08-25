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
    if constexpr(std::is_same_v<_type,int>) {
      p_datatype = XGA_INT;
    } else if constexpr(std::is_same_v<_type,long>) {
      p_datatype = XGA_LONG;
    } else if constexpr(std::is_same_v<_type,long>) {
      p_datatype = XGA_LONG;
    } else if constexpr(std::is_same_v<_type,float>) {
      p_datatype = XGA_FLOAT;
    } else if constexpr(std::is_same_v<_type,double>) {
      p_datatype = XGA_DOUBLE;
    } else if constexpr(std::is_same_v<_type,std::complex<float> >) {
      p_datatype = XGA_COMPLEX;
    } else if constexpr(std::is_same_v<_type,std::complex<double> >) {
      p_datatype = XGA_DCOMPLEX;
    }

    p_Impl = new p_GA(group, ndim, dims, p_datatype);
  };

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
   * @param[in] mapc array containing partitions along each axis
   * @param[in] nblock array containing processor decomposition
   */
  void setIrreglarDistribution(int64_t *mapc, int *nblock)
  {
    p_Impl->setIrregularDistribution(mapc, nblock);
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

  /**
   * Locate process that owns a particular array element
   * @param subscript indices of element
   * @param owner process that owns element indexed by subscript
   * @return false if subscript is not located in array
   */
  bool locate(const int64_t *subscript, int *owner)
  {
    return p_Impl->locate(subscript, owner);
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

  /**
   * Access data corresponding to a specific block
   * @param[in] index indices of block in proc grid or block cyclic layout
   * @param[out] rptr pointer to data
   * @param[out] ld array of strides for block
   */
  void accessBlockGridPtr(int *index, void **rptr, int64_t *ld)
  {
    p_Impl->accessBlockGridPtr(index, rptr, ld);
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
  void put(int64_t *lo, int64_t *hi, void* buf, int64_t *ld)
  {
    p_Impl->put(lo, hi, buf, ld);
  }

  /**
   * Copy data from global array to local buffer
   * @param[in] lo,hi bounding indices of block in global array
   * @param[in] buf pointer to first element in local buffer
   * @param[in] ld strides in local buffer
   */
  void get(int64_t *lo, int64_t *hi, void* buf, int64_t *ld)
  {
    p_Impl->get(lo, hi, buf, ld);
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

  p_GA *p_Impl;
};
}
#endif
