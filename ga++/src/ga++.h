#ifndef _GAPP_H
#define _GAPP_H

#include <stdio.h>
#include "ga.h"
#include "macdecls.h"
#ifdef MPIPP
#include <mpi.h>
#else
#include "sndrcv.h"
#endif

#define _GA_USENAMESPACE_ 0

#if _GA_USENAMESPACE_
#define _GA_STATIC_ 
#define _GA_EXTERN_ extern
#else
#define _GA_STATIC_ static
#define _GA_EXTERN_ 
#endif

#if _GA_USENAMESPACE_
namespace GA {
#else
class GA {
 public:
#endif
  class GAServices;
  _GA_EXTERN_  _GA_STATIC_  GAServices SERVICES;

#include "init_term.h"
#include "GlobalArray.h"
#include "GAServices.h"

  //GAServices SERVICES;

#if ! _GA_USENAMESPACE_
 private:
  GA() { }
#endif
};

#endif // _GAPP_H
