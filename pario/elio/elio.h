/* file name: elio.h  to be included by all apps that use ELIO */


/*include file that contains some common constants, also included by fortran */
#include "chemio.h" 
#include <sys/types.h>
#include <stdio.h>

#define   ELIO_UFS	     0	/* Unix filesystem type */
#define   ELIO_PFS           1	/* PFS Intel parallel filesystem type */
#define   ELIO_PIOFS         2	/* IBM SP parallel filesystem type */
#define   ELIO_PENDING_ERR -44  /* error code for failing elio_(g)open */


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


/********************** prototypes for elio functions ***********************/
extern Size_t elio_read(Fd_t fd, off_t offset, void *buf, Size_t bytes); 
extern int    elio_aread(Fd_t fd, off_t offset, void *buf,
                         Size_t bytes, io_request_t *req_id);
extern Size_t elio_write(Fd_t fd, off_t offset, const void *buf, 
                         Size_t bytes); 
extern int    elio_awrite(Fd_t fd, off_t offset, const void *buf,
                          Size_t bytes, io_request_t *req_id);
extern int    elio_wait(io_request_t *id);
extern int    elio_probe(io_request_t *id, int* status);
extern int    elio_delete(const char *filename);
extern Fd_t   elio_open(const char *fname, int type);
extern Fd_t   elio_gopen(const char *fname, int type);
extern int    elio_close(Fd_t fd);
extern int    elio_stat(char *fname, stat_t *statinfo);
extern int    elio_dirname(const char *fname, char *statinfo, int len);
extern int    elio_truncate(Fd_t fd, off_t length);
extern int    elio_length(Fd_t fd, off_t *length);
extern void   elio_errmsg(int code, char *msg);

