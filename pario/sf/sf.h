#ifndef _SF_INCLUDED
#  define _SF_INCLUDED
   typedef double SFsize_t;
#include "typesf2c.h"
#if defined(__STDC__) || defined(__cplusplus) || defined(WIN32)
# define _ARGS_(s) s
#else
# define _ARGS_(s) ()
#endif


extern Integer sf_create _ARGS_((char* fname, SFsize_t* size_hard_limit,
       SFsize_t* size_soft_limit, SFsize_t* req_size, Integer *handle));
 
extern void sf_errmsg _ARGS_((int code, char *msg));

#undef _ARGS_
#if defined(CRAY) || defined(WIN32)
#define sf_write_ SF_WRITE
#define sf_read_ SF_READ
#define sf_wait_ SF_WAIT
#define sf_waitall_ SF_WAITALL
#define sf_destroy_ SF_DESTROY
#define sf_errmsg SF_ERRMSG
#endif
#endif
