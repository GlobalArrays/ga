#ifndef _GAPP_H
#define _GAPP_H

#if defined(__cplusplus) || defined(c_plusplus)

#ifdef MPIPP
#   include <mpi.h>
#else
#   include <tcgmsg.h>
#endif
#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#include "ga.h"
#include "macdecls.h"

#define GANbhdl ga_nbhdl_t   

#ifdef _GA_USENAMESPACE_
#define _GA_STATIC_ 
#define _GA_EXTERN_ extern
#else // ! _GA_USENAMESPACE_
#define _GA_STATIC_ static
#define _GA_EXTERN_ 
#endif // _GA_USENAMESPACE_

#ifdef _GA_USENAMESPACE_
namespace GA {
#else // ! _GA_USENAMESPACE_
class GA {
 public:
#endif // _GA_USENAMESPACE_
  class GAServices;
  _GA_EXTERN_  _GA_STATIC_  GAServices SERVICES;

#include "init_term.h"
#include "PGroup.h"
#include "GlobalArray.h"
#include "GAServices.h"

  //GAServices SERVICES;

#ifndef _GA_USENAMESPACE_
 private:
  GA() { }
#endif // ! _GA_USENAMESPACE_
};

#endif // _GAPP_H
#endif
