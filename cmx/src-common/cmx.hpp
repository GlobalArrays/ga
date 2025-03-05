/* cmx header file */
#ifndef _CMX_COMMON_H
#define _CMX_COMMON_H

#include <mpi.h>

#include <stdlib.h>
#include <vector>
#include <complex>
#include "defines.hpp"
//#include "cmx_impl.hpp"
#include "group.hpp"
#include "alloc.hpp"
#include "environment.hpp"
//#include "p_alloc.hpp"

namespace CMX {

enum cmx_types{CMX_UNKNOWN = 0,
               CMX_INT,
               CMX_LONG,
               CMX_FLOAT,
               CMX_DOUBLE,
               CMX_COMPLEX,
               CMX_DCOMPLEX};

template<typename _type>
class CMX {

public:

  /**
   * Basic constructor
   * @param[in] group home group for allocation
   * @param[in] size size of allocation, in bytes
   */
  CMX(Group *group, int64_t size)
  {
    /* Assign p_datatype parameter */
    if constexpr(std::is_same_v<_type,int>) {
      p_datatype=CMX_INT;
    } else if constexpr(std::is_same_v<_type,long>) {
      p_datatype=CMX_LONG;
    } else if constexpr(std::is_same_v<_type,float>) {
      p_datatype=CMX_FLOAT;
    } else if constexpr(std::is_same_v<_type,double>) {
      p_datatype=CMX_DOUBLE;
    } else if constexpr(std::is_same_v<_type,std::complex<float> >) {
      p_datatype=CMX_COMPLEX;
    } else if constexpr(std::is_same_v<_type,std::complex<double> >) {
      p_datatype=CMX_DCOMPLEX;
    }
    /* initialize implementation */
    p_Impl = new p_CMX<_type>(group->p_group, size);
  }

  /**
   * Basic destructor
   */
  CMX()
  {
  }

private:

  int p_datatype = CMX_UNKNOWN;

  //p_Allocation<_type> *p_Impl;
};
}
#endif /* _CMX_COMMON_H */
