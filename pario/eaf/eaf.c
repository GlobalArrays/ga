/* DISCLAIMER
 *
 * This material was prepared as an account of work sponsored by an
 * agency of the United States Government.  Neither the United States
 * Government nor the United States Department of Energy, nor Battelle,
 * nor any of their employees, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY,
 * COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT,
 * SOFTWARE, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
 * INFRINGE PRIVATELY OWNED RIGHTS.
 *
 *
 * ACKNOWLEDGMENT
 *
 * This software and its documentation were produced with United States
 * Government support under Contract Number DE-AC06-76RLO-1830 awarded by
 * the United States Department of Energy.  The United States Government
 * retains a paid-up non-exclusive, irrevocable worldwide license to
 * reproduce, prepare derivative works, perform publicly and display
 * publicly by or for the US Government, including the right to
 * distribute to other US Government contractors.
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#define PATH_MAX _MAX_PATH
#define F_OK 2
#else
#include <unistd.h>
#endif
#include <string.h>
#include <limits.h>
#ifdef EAF_STATS
#include <sys/types.h>
#include <sys/time.h>
#endif
#include "elio.h"
#include "eaf.h"
#include "eafP.h"

#ifdef OPEN_MAX
#define EAF_MAX_FILES OPEN_MAX
#else
#define EAF_MAX_FILES 64
#endif

static struct {
    char *fname;		/* Filename --- if non-null is active*/
    Fd_t elio_fd;		/* ELIO file descriptor */
    int type;                   /* file type */
    int nwait;			/* #waits */
    int nwrite;			/* #synchronous writes */
    int nread;			/* #synchronous reads */
    int nawrite;		/* #asynchronous writes */
    int naread;			/* #asynchronous reads */
    double nb_write;		/* #synchronous bytes written */
    double nb_read;		/* #synchronous bytes read */
    double nb_awrite;		/* #asynchronous bytes written */
    double nb_aread;		/* #asynchronous bytes read */
    double t_write;		/* Wall seconds synchronous writing */
    double t_read;		/* Wall seconds synchronous reading */
    double t_awrite;		/* Wall seconds asynchronous writing */
    double t_aread;		/* Wall seconds asynchronous reading */
    double t_wait;		/* Wall seconds waiting */
} file[EAF_MAX_FILES];

static int valid_fd(int fd)
{
    return ( (fd >= 0) && (fd < EAF_MAX_FILES) && (file[fd].fname) );
}

static double wall_time(void)
/*
  Return wall_time in seconds as cheaply and as accurately as possible
  */
{
#ifdef EAF_STATS
  static int firstcall = 1;
  static unsigned firstsec, firstusec;
  double low, high;
  struct timeval tp;
  struct timezone tzp;

  if (firstcall) {
      (void) gettimeofday(&tp,&tzp);
      firstusec = tp.tv_usec;
      firstsec  = tp.tv_sec;
      firstcall = 0;
  }

  (void) gettimeofday(&tp,&tzp);

  low = (double) (tp.tv_usec>>1) - (double) (firstusec>>1);
  high = (double) (tp.tv_sec - firstsec);

  return high + 1.0e-6*(low+low);
#else
  return 0.0;
#endif
}

int eaf_open(const char *fname, int type, int *fd)
/*
  Open the named file returning the EAF file descriptor in fd.
  Return 0 on success, non-zero on failure
  */
{
    int i=0;

    while ((i<EAF_MAX_FILES) && file[i].fname) /* Find first empty slot */
	i++;
    if (i == EAF_MAX_FILES) return EAF_ERR_MAX_OPEN;
	
    if (!(file[i].fname = strdup(fname)))
	return EAF_ERR_MEMORY;

    if (!(file[i].elio_fd = elio_open(fname, type, ELIO_PRIVATE))) {
	free(file[i].fname);
	file[i].fname = 0;
	return ELIO_PENDING_ERR;
    }

    file[i].nwait = file[i].nread = file[i].nwrite = 
	file[i].naread = file[i].nawrite = 0;
    file[i].nb_read = file[i].nb_write = file[i].nb_aread = 
	file[i].nb_awrite = file[i].t_read = file[i].t_write = 
	file[i].t_wait = 0.0;

    file[i].type = type;
    *fd = i;
    return EAF_OK;
}


void eaf_print_stats(int fd)
/*
  Print performance statistics for this file to standard output
  */
{
    eaf_off_t len;
    double mbr, mbw, mbra, mbwa;
    if (!valid_fd(fd)) return;

    if (eaf_length(fd, &len)) len = -1;

    printf("\n");
    printf("------------------------------------------------------------\n");
    printf("EAF file %d: \"%s\" size=%lu bytes\n", 
	   fd, file[fd].fname, (unsigned long) len);
    printf("------------------------------------------------------------\n");
    printf("               write      read    awrite     aread      wait\n");
    printf("               -----      ----    ------     -----      ----\n");
    printf("     calls: %8d  %8d  %8d  %8d  %8d\n", 
	   file[fd].nwrite, file[fd].nread, file[fd].nawrite, 
	   file[fd].naread, file[fd].nwait);
    printf("   data(b): %.2e  %.2e  %.2e  %.2e\n",
	   file[fd].nb_write, file[fd].nb_read, file[fd].nb_awrite, 
	   file[fd].nb_aread);
    printf("   time(s): %.2e  %.2e  %.2e  %.2e  %.2e\n",
	   file[fd].t_write, file[fd].t_read, 
	   file[fd].t_awrite, file[fd].t_aread, 
	   file[fd].t_wait);
    mbr = 0.0;
    mbw = 0.0;
    mbwa= 0.0;
    mbra= 0.0;
    if (file[fd].t_write > 0.0) mbw = file[fd].nb_write/(1e6*file[fd].t_write);
    if (file[fd].t_read  > 0.0) mbr = file[fd].nb_read/(1e6*file[fd].t_read);
    if ((file[fd].t_wait + file[fd].t_aread) > 0.0) 
      mbra = 1e-6*file[fd].nb_aread / 
	(file[fd].t_wait + file[fd].t_aread);
    if ((file[fd].t_wait + file[fd].t_awrite) > 0.0) 
      mbwa = 1e-6*file[fd].nb_awrite / 
	(file[fd].t_wait + file[fd].t_awrite);
	
    /* Note that wait time does not distinguish between read/write completion 
       so that entire wait time is counted 
       in computing effective speed for async read & write */
    if (mbwa+mbra) {
      printf("rate(mb/s): %.2e  %.2e  %.2e* %.2e*\n", mbw, mbr, mbwa, mbra);
      printf("------------------------------------------------------------\n");
      printf("* = Effective rate.  Full wait time used for read and write.\n\n");
    }
    else {
      printf("rate(mb/s): %.2e  %.2e\n", mbw, mbr);
      printf("------------------------------------------------------------\n\n");
    }
    fflush(stdout);
}

int eaf_close(int fd)
/*
  Close the EAF file and return 0 on success, non-zero on failure
  */
{
    if (!valid_fd(fd)) return EAF_ERR_INVALID_FD;
    
    free(file[fd].fname);
    file[fd].fname = 0;

    return elio_close(file[fd].elio_fd);
}


int eaf_write(int fd, eaf_off_t offset, const void *buf, size_t bytes)
/*
  Write the buffer to the file at the specified offset.
  Return 0 on success, non-zero on failure
  */
{    
    double start = wall_time();
    Size_t rc;

    if (!valid_fd(fd)) return EAF_ERR_INVALID_FD;

    rc = elio_write(file[fd].elio_fd, (off_t) offset, buf, (Size_t) bytes);
    if (rc != bytes){
        if(rc < 0) return((int)rc); /* rc<0 means ELIO detected error */
 	else return EAF_ERR_WRITE;
    }else {
	file[fd].nwrite++;
	file[fd].nb_write += bytes;
	file[fd].t_write += wall_time() - start;

	return EAF_OK;
    }
}

int eaf_awrite(int fd, eaf_off_t offset, const void *buf, size_t bytes,
	       int *req_id)
/*
  Initiate an asynchronous write of the buffer to the file at the
  specified offset.  Return in *req_id the ID of the request for
  subsequent use in eaf_wait/probe.  The buffer may not be reused until
  the operation has completed.
  Return 0 on success, non-zero on failure
  */
{
    double start = wall_time();
    io_request_t req;
    int rc;

    if (!valid_fd(fd)) return EAF_ERR_INVALID_FD;

    rc = elio_awrite(file[fd].elio_fd, (off_t)offset, buf, (Size_t)bytes, &req);
    if(!rc){
	*req_id = req;
	file[fd].nawrite++;
	file[fd].nb_awrite += bytes;
    } 
    file[fd].t_awrite += wall_time() - start;
    return rc;
}

int eaf_read(int fd, eaf_off_t offset, void *buf, size_t bytes)
/*
  Read the buffer from the specified offset in the file.
  Return 0 on success, non-zero on failure
  */
{
    double start = wall_time();
    Size_t rc;
    
    if (!valid_fd(fd)) return EAF_ERR_INVALID_FD;
    
    rc = elio_read(file[fd].elio_fd, (off_t) offset, buf, (Size_t) bytes);
    if (rc != bytes){
        if(rc < 0) return((int)rc); /* rc<0 means ELIO detected error */
        else return EAF_ERR_READ;
    } else {
	file[fd].nread++;
	file[fd].nb_read += bytes;
	file[fd].t_read += wall_time() - start;
	return EAF_OK;
    }
}    

int eaf_aread(int fd, eaf_off_t offset, void *buf, size_t bytes, 
	      int *req_id)
/*
  Initiate an asynchronous read of the buffer from the file at the
  specified offset.  Return in *req_id the ID of the request for
  subsequent use in eaf_wait/probe.  The buffer may not be reused until
  the operation has completed.
  Return 0 on success, non-zero on failure
  */
{
    double start = wall_time();
    io_request_t req;
    int rc;

    if (!valid_fd(fd)) return EAF_ERR_INVALID_FD;

    rc = elio_aread(file[fd].elio_fd, (off_t) offset, buf, (Size_t)bytes, &req);

    if(!rc){
	*req_id = req;
	file[fd].naread++;
	file[fd].nb_aread += bytes;
    }
    file[fd].t_aread += wall_time() - start;
    return rc;
}

int eaf_wait(int fd, int req_id)
/*
  Wait for the I/O operation referred to by req_id to complete.
  Return 0 on success, non-zero on failure
  */
{
    double start = wall_time();
    int code;

    io_request_t req = req_id;
    code = elio_wait(&req);
    file[fd].t_wait += wall_time() - start;
    file[fd].nwait++;

    return code;
}

int eaf_probe(int req_id, int *status)
/*
 *status returns 0 if the I/O operation reffered to by req_id
  is complete, 1 otherwise. 
  Return 0 on success, non-zero on failure.
  */
{
    io_request_t req = req_id;
    int rc;

    rc = elio_probe(&req, status);
    if(!rc) *status = !(*status == ELIO_DONE);
    return rc;
}

int eaf_delete(const char *fname)
/*
  Delete the named file.  If the delete succeeds, or the file
  does not exist, return 0.  Otherwise return non-zero.
  */
{
    if (access(fname, F_OK) == 0)
	if (unlink(fname))
	    return EAF_ERR_UNLINK;

    return EAF_OK;
}

int eaf_stat(const char *path, int *avail_kb, char *fstype, int fslen)
/*
  Return in *avail_kb and *fstype the amount of free space (in Kb)
  and filesystem type (currenly UFS, PFS, or PIOFS) of the filesystem
  associated with path.  Path should be either a filename, or a directory
  name ending in a slash (/).  fslen should specify the size of the
  buffer pointed to by fstype.

  Return 0 on success, non-zero on failure.
  */
{
 char dirname[PATH_MAX];
 stat_t statinfo;
 int rc;

 if (rc = elio_dirname(path, dirname, sizeof(dirname))) return rc;
 if (rc = elio_stat(dirname, &statinfo)) return rc;
 if (fslen < 8) return EAF_ERR_TOO_SHORT;

 *avail_kb = statinfo.avail;
 if (statinfo.fs == ELIO_UFS)
     strcpy(fstype, "UFS");
 else if (statinfo.fs == ELIO_PFS)
     strcpy(fstype, "PFS");
 else if (statinfo.fs == ELIO_PIOFS)
     strcpy(fstype, "PIOFS");
 else
     strcpy(fstype, "UNKNOWN");

 return EAF_OK;
}

int eaf_eof(int code)
/*
  Return 0 if code corresponds to EOF, or non-zero.
  */
{
    return !(code == EAF_ERR_EOF);
}

void eaf_errmsg(int code, char *msg)
/*
  Return in msg (assumed to hold up to 80 characters)
  a description of the error code obtained from an EAF call,
  or an empty string if there is no such code
  */
{
    if (code == EAF_OK) 
	(void) strcpy(msg, "OK");
    else if (code == EAF_ERR_EOF) 
	(void) strcpy(msg, "end of file");
    else if (code == EAF_ERR_MAX_OPEN)
	(void) strcpy(msg, "too many open files");
    else if (code == EAF_ERR_MEMORY)
	(void) strcpy(msg, "memory allocation failed");
    else if (code == EAF_ERR_OPEN)
	(void) strcpy(msg, "failed opening file");
    else if (code == EAF_ERR_CLOSE)
	(void) strcpy(msg, "failed closing file");
    else if (code == EAF_ERR_INVALID_FD)
	(void) strcpy(msg, "invalid file descriptor");
    else if (code == EAF_ERR_WRITE)
	(void) strcpy(msg, "write failed");
    else if (code == EAF_ERR_AWRITE)
	(void) strcpy(msg, "asynchronous write failed");
    else if (code == EAF_ERR_READ)
	(void) strcpy(msg, "read failed");
    else if (code == EAF_ERR_AREAD)
	(void) strcpy(msg, "asynchronous read failed");
    else if (code == EAF_ERR_WAIT)
	(void) strcpy(msg, "wait failed");
    else if (code == EAF_ERR_PROBE)
	(void) strcpy(msg, "probe failed");
    else if (code == EAF_ERR_UNLINK)
	(void) strcpy(msg, "unlink failed");
    else if (code == EAF_ERR_UNIMPLEMENTED)
	(void) strcpy(msg, "unimplemented operation");
    else if (code == EAF_ERR_STAT)
	(void) strcpy(msg, "stat failed");
    else if (code == EAF_ERR_TOO_SHORT)
	(void) strcpy(msg, "an argument string/buffer is too short");
    else if (code == EAF_ERR_TOO_LONG)
	(void) strcpy(msg, "an argument string/buffer is too long");
    else if (code == EAF_ERR_NONINTEGER_OFFSET)
	(void) strcpy(msg, "offset is not an integer");
    else if (code == EAF_ERR_TRUNCATE)
	(void) strcpy(msg, "truncate failed");
    else 
	elio_errmsg(code, msg);
}


int eaf_truncate(int fd, eaf_off_t length)
/*
  Truncate the file to the specified length.
  Return 0 on success, non-zero otherwise.
  */
{

    if (!valid_fd(fd)) return EAF_ERR_INVALID_FD;

#ifdef CRAY 
   /* ftruncate does not work with Cray FFIO, we need to implement it
    * as a sequence of generic close, truncate, open calls 
    */

    rc = elio_close(file[fd].elio_fd);
    if(rc) return rc;
    if(truncate(file[fd].fname, (off_t) length)) return EAF_ERR_TRUNCATE;  
    if (!(file[fd].elio_fd = elio_open(file[fd].fname, file[fd].type, ELIO_PRIVATE))) {
        free(file[fd].fname);
        file[fd].fname = 0;
        return ELIO_PENDING_ERR;
    }
#else
    if(elio_truncate(file[fd].elio_fd, (off_t)length)) return EAF_ERR_TRUNCATE;
#endif

    return EAF_OK;


/*  return elio_truncate(file[fd].elio_fd, (off_t) length);*/
}


int eaf_length(int fd, eaf_off_t *length)
/*
  Return in length the length of the file.  
  Return 0 on success, nonzero on failure.
  */
{
    off_t len;
    int rc;

    if (!valid_fd(fd)) return EAF_ERR_INVALID_FD;

    rc = elio_length(file[fd].elio_fd, &len);
    if(!rc) *length = (eaf_off_t) len;
    return rc;
}
