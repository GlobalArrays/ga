/* file name: elio.h  to be included by all apps that use ELIO */

/*include file that contains some common constants, also include by fortran */
#include "chemio.h" 
#include <sys/types.h>
#include <stdio.h>

/*********************** type definitions for ELIO interface *****************/
/*
GA, DRA and other libs use -DEXT_INT flag to switch definition of
Integer data type from int to long:

If you adopt it in EAF, we can use the same flags.

Jarek
*/
#ifdef EXT_INT
typedef long Integer;
#else
typedef int Integer;
#endif 

typedef struct {
  int   fd;
  int   fs;
} fd_struct;                      /* file descriptor type definition */
typedef fd_struct *elio_fd_t;/* Internal filedescriptor type  */
typedef int  Fd_t;           /* File descriptor handle type used by apps */
typedef Integer io_request_t;   /* asynchronous I/O request type */
typedef Integer Size_t;         /* size of I/O request type */ 


#ifndef _VOID_DEFINED_
#  define _VOID_DEFINED_ 1 
#  if defined(__STDC__) || defined(__cplusplus)
      typedef void Void;
#  else
      typedef char Void;
#  endif
#endif


/********************** ELIO function prototypes *****************************/
#if defined(__STDC__) || defined(__cplusplus)
# define _ARGS_(s) s
#else
# define _ARGS_(s) ()
#endif

extern Size_t elio_read     _ARGS_((Fd_t fd, off_t offset, Void *buf,
                                    Size_t bytes)); 
extern int    elio_aread    _ARGS_((Fd_t fd, off_t offset, Void *buf,
                                    Size_t bytes, io_request_t *req_id));
extern Size_t elio_write    _ARGS_((Fd_t fd, off_t offset, Void *buf,
                                    Size_t bytes)); 
extern int    elio_awrite   _ARGS_((Fd_t fd, off_t offset, Void *buf,
                                    Size_t bytes, io_request_t *req_id));
extern int    elio_wait     _ARGS_((io_request_t *id));
extern int    elio_probe    _ARGS_((io_request_t *id, int* status));
extern int    elio_delete   _ARGS_((char *filename));
extern Fd_t   elio_open     _ARGS_((char *fname, int type));
extern Fd_t   elio_gopen    _ARGS_((char *fname, int type));
extern void   elio_close    _ARGS_((Fd_t fd));
extern int    elio_stat     _ARGS_((char *fname));
       void   elio_init     _ARGS_(());

#undef _ARGS_

/* constants to indicate filesystem type */
#define FS_UFS		0     /* Unix filesystem type */
#define FS_PFS		1     /* PFS Intel parallel filesystem type */
#define FS_PIOFS	2     /* IBM SP parallel filesystem type */


/**************************** Error Macro ******************************/
/* ELIO defines error macro called in case of error
 * the macro can also use user-provided error routine PRINT_AND_ABORT
 * defined as macro to do some cleanup in the application before
 * aborting
 * The requirement is that PRINT_AND_ABORT is defined before
 * including ELIO header file - this file
 */
#if !defined(PRINT_AND_ABORT)
#if defined(SUN) && !defined(SOLARIS)
extern int fprintf();
extern void fflush();
#endif

#define PRINT_AND_ABORT(msg, val) \
{ \
  fprintf(stderr, "ELIO fatal error: %s %d\n", msg, (int) val); \
  fprintf(stdout, "ELIO fatal error: %s %d\n", msg, (int) val); \
  fflush(stdout);\
  perror("elio failed:");\
  exit(val); \
}
#endif

#define ELIO_ABORT(msg, val) PRINT_AND_ABORT(msg, val)

/************** this stuff is exported because EAF uses it **************/
#define ELIO_FILENAME_MAX 1024
#define ELIO_MAX_FILES    64
#define SDIRS_INIT_SIZE 1024


