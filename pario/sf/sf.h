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
 
extern Integer sf_create_suffix _ARGS_((char* fname, SFsize_t* size_hard_limit,
       SFsize_t* size_soft_limit, SFsize_t* req_size, Integer *handle,
       Integer *suffix));

extern void sf_errmsg _ARGS_((int code, char *msg));

#undef _ARGS_
#if (defined(CRAY)&& !defined(__crayx1)) || defined(WIN32)
#define sf_write_ SF_WRITE
#define sf_read_ SF_READ
#define sf_wait_ SF_WAIT
#define sf_waitall_ SF_WAITALL
#define sf_destroy_ SF_DESTROY
#define sf_errmsg_ SF_ERRMSG
#define sf_open_ SF_OPEN
#define sf_close_ SF_CLOSE
#define sf_rwtor_ SF_RWTOR
#elif defined(F2C2_)
#define sf_errmsg   sf_errmsg__
#define sf_write_ sf_write__
#define sf_read_ sf_read__
#define sf_wait_ sf_wait__
#define sf_waitall_ sf_waitall__
#define sf_destroy_ sf_destroy__
#define sf_open_ sf_open__
#define sf_close_ sf_close__
#define sf_rwtor_ sf_rwtor__
#endif
#endif
