/* file name: elio.h */
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#if defined(PARAGON)
#  include <sys/mount.h>
#endif
#if defined(AIX)
#  include <piofs/piofs_ioctl.h> 
#endif

#include "chemio.h"


/* file descriptor type definition */
typedef struct {
  int   fd;
  int   fs;
} Fd_t;


/*asynchronous I/O request type */
typedef long io_request_t;

typedef long Size_t;


#ifndef _VOID_DEFINED_
#  if !defined(__STDC__) || !defined(__cplusplus)
      typedef char Void;
#  else
      typedef void Void;
#  endif
#  define _VOID_DEFINED_ 1 
#endif


#if defined(__STDC__) || defined(__cplusplus)
# define _ARGS_(s) s
#else
# define _ARGS_(s) ()
#endif


extern Size_t elio_read     _ARGS_((Fd_t *fd, off_t offset, Void *buf,
                                    Size_t bytes)); 
extern int    elio_aread    _ARGS_((Fd_t *fd, off_t offset, Void *buf,
                                    Size_t bytes, io_request_t *req_id));
extern Size_t elio_write    _ARGS_((Fd_t *fd, off_t offset, Void *buf,
                                    Size_t bytes)); 
extern int    elio_awrite   _ARGS_((Fd_t *fd, off_t offset, Void *buf,
                                    Size_t bytes, io_request_t *req_id));
extern int    elio_wait     _ARGS_((io_request_t *id));
extern int    elio_probe    _ARGS_((io_request_t *id, int* status));
extern int    elio_delete   _ARGS_((char *filename));
extern Fd_t  *elio_open     _ARGS_((char *fname, int type));
extern Fd_t  *elio_gopen    _ARGS_((char *fname, int type));
extern void   elio_close    _ARGS_((Fd_t *fd));
extern int    elio_stat     _ARGS_((char *fname));
       void   elio_init     _ARGS_(());
       void   elio_terminate _ARGS_(());

#undef _ARGS_


#define ELIO_FILENAME_MAX 1024

#define SDIRS_INIT_SIZE 1024

#define FS_UFS		0
#define FS_PFS		1
#define FS_PIOFS	2




#if !defined(PRINT_AND_ABORT)
#define PRINT_AND_ABORT(val) \
{ \
  fprintf(stderr, "ELIO Super-Fatal: PRINT_AND_ABORT not defined!\n"); \
  exit(val); \
}
#endif

#define ELIO_ABORT(msg, val) \
{ \
  fprintf(stderr, "ELIO Fatal -- Exiting with %d\n", val ); \
  fprintf(stderr, "ELIO Fatal -- Msg: %s\n", msg ); \
  elio_terminate(); \
  PRINT_AND_ABORT(val); \
}




