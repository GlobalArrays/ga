#ifndef EAF_H 
#define EAH_H

/* This section used by both C and Fortran */

#define   EAF_RW -1
#define   EAF_W  -2
#define   EAF_R  -3

#ifndef EAF_FORTRAN

/* This section used by only C */

/* This to ensure size_t is defined */
#include <stdio.h>
#include <sys/types.h>

typedef double eaf_off_t;

int eaf_write(int fd, eaf_off_t offset, const void *buf, size_t bytes);

int eaf_awrite(int fd, eaf_off_t offset, const void *buf, size_t bytes,
	       int *req_id);

int eaf_read(int fd, eaf_off_t offset, void *buf, size_t bytes);

int eaf_aread(int fd, eaf_off_t offset, void *buf, size_t bytes, 
	      int *req_id);

int eaf_wait(int fd, int id);

int eaf_probe(int id, int *status);

int eaf_open(const char *fname, int type, int *fd);

int eaf_close(int fd);

int eaf_delete(const char *fname);

int eaf_length(int fd, eaf_off_t *length);

int eaf_stat(const char *path, int *avail_kb, char *fstype, int fslen);

int eaf_eof(int code);

void eaf_errmsg(int code, char *msg);

void eaf_print_stats(int fd);

int eaf_truncate(int fd, eaf_off_t length);

int eaf_length(int fd, eaf_off_t *length);


#endif
#endif
