/* file name: elio.h  to be included by all apps that use ELIO */

/*include file that contains some common constants, also include by fortran */
#include "chemio.h" 
#include <sys/types.h>
#include <stdio.h>

/*********************** type definitions for ELIO interface *****************/
typedef long Size_t;         /* size of I/O request type */ 
typedef struct {
  int   fd;
  int   fs;
} fd_struct;                      /* file descriptor type definition */
typedef struct{
  int   fs;
  long  avail;
} stat_t;
typedef fd_struct* Fd_t;
typedef long io_request_t;   /* asynchronous I/O request type */


#ifndef _VOID_DEFINED_
#  define _VOID_DEFINED_ 1 
#  if defined(__STDC__) || defined(__cplusplus)
      typedef void Void;
#  else
      typedef char Void;
#  endif
#endif

#define   ELIO_UFS	0	/* Unix filesystem type */
#define   ELIO_PFS      1	/* PFS Intel parallel filesystem type */
#define   ELIO_PIOFS    2	/* IBM SP parallel filesystem type */

extern Size_t elio_read(Fd_t fd, off_t offset, Void *buf, Size_t bytes); 
extern int    elio_aread(Fd_t fd, off_t offset, Void *buf,
                         Size_t bytes, io_request_t *req_id);
extern Size_t elio_write(Fd_t fd, off_t offset, const Void *buf, 
                         Size_t bytes); 
extern int    elio_awrite(Fd_t fd, off_t offset, const Void *buf,
                          Size_t bytes, io_request_t *req_id);
extern int    elio_wait(io_request_t *id);
extern int    elio_probe(io_request_t *id, int* status);
extern int    elio_delete(const char *filename);
extern Fd_t   elio_open(const char *fname, int type);
extern Fd_t   elio_gopen(const char *fname, int type);
extern int    elio_close(Fd_t fd);
extern int    elio_stat(char *fname, stat_t *statinfo);
extern int    elio_dirname(const char *fname, char *statinfo, int len);
       void   elio_init(void);
int elio_truncate(Fd_t fd, off_t length);
int elio_length(Fd_t fd, off_t *length);

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
do { \
  fprintf(stderr, "ELIO fatal error: %s %d\n", msg, (int) val); \
  fprintf(stdout, "ELIO fatal error: %s %d\n", msg, (int) val); \
  fflush(stdout);\
  perror("elio failed:");\
  exit(val); \
} while(0)
#endif

#define ELIO_ABORT PRINT_AND_ABORT

#define ELIO_ERROR(msg, val) do{ \
 if(_elio_Errors_Fatal) PRINT_AND_ABORT(msg, val);\
 else return(ELIO_FAIL);\
} while(0)


/************** this stuff is exported because EAF uses it **************/
#define ELIO_FILENAME_MAX 1024
#define SDIRS_INIT_SIZE 1024
