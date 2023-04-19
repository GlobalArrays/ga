/* cmx header file */
#ifndef _CMX_IMPL_H
#define _CMX_IMPL_H

#include <mpi.h>

#include <stdlib.h>
#include <vector>
#include <complex>
#include "defines.hpp"
#include "reg_cache.hpp"

#define XDEBUG

namespace CMX {

enum _cmx_types{_CMX_UNKNOWN = 0,
  _CMX_INT,
  _CMX_LONG,
  _CMX_FLOAT,
  _CMX_DOUBLE,
  _CMX_COMPLEX,
  _CMX_DCOMPLEX};

class p_Environment;

template<typename _type>
class p_CMX {

public:

  /**
   * Basic constructor
   * @param[in] group home group for allocation
   * @param[in] size size of allocation, in bytes
   */
  p_CMX(p_Group *group, cmxInt size){
    if constexpr(std::is_same_v<_type,int>) {
      p_datatype=_CMX_INT;
    } else if constexpr(std::is_same_v<_type,long>) {
      p_datatype=_CMX_LONG;
    } else if constexpr(std::is_same_v<_type,float>) {
      p_datatype=_CMX_FLOAT;
    } else if constexpr(std::is_same_v<_type,double>) {
      p_datatype=_CMX_DOUBLE;
    } else if constexpr(std::is_same_v<_type,std::complex<float> >) {
      p_datatype=_CMX_COMPLEX;
    } else if constexpr(std::is_same_v<_type,std::complex<double> >) {
      p_datatype=_CMX_DCOMPLEX;
    }

    p_environment = p_Environment::instance();

#ifdef DEBUG
    printf("Initialize p_CMX p_datatype: %d\n",p_datatype);
    switch(p_datatype) {
      case _CMX_UNKNOWN:
        printf("UNKNOWN datatype\n");
        break;
      case _CMX_INT:
        printf("int datatype\n");
        break;
      case _CMX_LONG:
        printf("long datatype\n");
        break;
      case _CMX_FLOAT:
        printf("float datatype\n");
        break;
      case _CMX_DOUBLE:
        printf("double datatype\n");
        break;
      case _CMX_COMPLEX:
        printf("single complex datatype\n");
        break;
      case _CMX_DCOMPLEX:
        printf("double complex datatype\n");
        break;
      default:
        printf("UNASSIGNED datatype\n");
        break;
    }
#endif

  }

  /**
   * Simple destructor
   */
  ~p_CMX()
  {
  }

private:

  int p_datatype = _CMX_UNKNOWN;

  p_Environment *p_environment;

};
}
#endif /* _CMX_IMPL_H */
