/**********************************************************************\
 ELementary I/O (ELIO) disk operations for Chemio libraries   
\**********************************************************************/

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <unistd.h>
#include <fcntl.h>
#if defined(PARAGON)
#  include <sys/mount.h>
#  include <nx.h>
#endif
#if defined(SP) || defined(SP1)
#  include <piofs/piofs_ioctl.h>
#endif

#include "elio.h"
#include "pablo.h"

/****************** Internal Constants and Parameters **********************/

#define  MAX_AIO_REQ  4
#define  NULL_AIO    -123456
#define  FOPEN_MODE 0644
#define  MAX_ATTEMPTS 10


#if  defined(AIX) || defined(DECOSF) || defined(SGITFP) || defined(notKSR)
#   define AIO 1
#   include <aio.h>
#   if defined(KSR)||defined(AIX)
#      define INPROGRESS EINPROG
#   else 
#      define INPROGRESS EINPROGRESS
#   endif
      struct aiocb    cb_fout[MAX_AIO_REQ];
const struct aiocb   *cb_fout_arr[MAX_AIO_REQ];
#else
#   define INPROGRESS 1            /* I wish this didn't have to be here */
#endif

static int            aio_req[MAX_AIO_REQ]; /* array for AIO requests */
static int            first_elio_init = 1;  /* intialization status */

/****************************** Internal Macros *****************************/
#if defined(AIO) || defined(PARAGON)
#  define AIO_LOOKUP(aio_i) {\
      aio_i = 0;\
      while(aio_req[aio_i] != NULL_AIO && aio_i < MAX_AIO_REQ) aio_i++;\
}
#else
#  define AIO_LOOKUP(aio_i) aio_i = MAX_AIO_REQ
#endif


#define SYNC_EMMULATE(op) \
  if( elio_ ## op (fd, offset, buf, bytes) != bytes ) \
    { \
       *req_id = ELIO_DONE; \
       stat   = ELIO_FAIL;  \
       fprintf(stderr,"sync_emmulate:stat=%d bytes=%d\n",(int)stat,(int)bytes);\
    } \
  else \
    { \
       *req_id = ELIO_DONE; \
       stat    = 0; \
    }

/*****************************************************************************/

 

/*\ Blocking Write - returns number of bytes written or -1 if failed
\*/
Size_t elio_write(fd, offset, buf, bytes)
     Fd_t         fd;
     off_t        offset;
     Void        *buf;
     Size_t       bytes;
{
  Size_t stat           = -1;
  Size_t bytes_to_write = bytes;
  int    attempt        = 0;

      PABLO_start( PABLO_elio_write );

      if(offset != lseek(fd->fd, offset, SEEK_SET))
	ELIO_ERR("elio_write: lseek()", NULL, stat);

      /* interrupted write should be restarted */
      do 
	{
	  if(attempt == MAX_ATTEMPTS)
	    ELIO_ERR("elio_write: max attempts exceeded", NULL, stat);

	  stat = write(fd->fd, buf, bytes_to_write);
	  if(stat < bytes_to_write && stat >= -1)
	    {
	      bytes_to_write -= stat;
	      buf = stat + (char*)buf; /*advance pointer by # bytes written */
	    }
	  else
	    bytes_to_write = 0;
	  attempt++;
	}
      while(bytes_to_write && (errno == EINTR || errno == EAGAIN));

      if(stat != -1) stat = bytes -  bytes_to_write;

      if(stat != bytes)
	ELIO_ERR("elio_write: write failed", NULL, stat)

      PABLO_end(PABLO_elio_write);
      return(stat);
}



void elio_set_cb(fd, offset, reqn, buf, bytes)
     Fd_t  fd;
     off_t  offset;
     int    reqn;
     Void  *buf;
     Size_t bytes;
{
#if defined(AIO)
    cb_fout[reqn].aio_offset = offset;
    cb_fout_arr[reqn] = cb_fout+reqn;
#   if defined(KSR)
      cb_fout[reqn].aio_whence = SEEK_SET;
#   else
      cb_fout[reqn].aio_buf    = buf;
      cb_fout[reqn].aio_nbytes = bytes;
#     if defined(AIX)
        cb_fout[reqn].aio_whence = SEEK_SET;
#     else
        cb_fout[reqn].aio_sigevent.sigev_notify = SIGEV_NONE;
        cb_fout[reqn].aio_fildes    = fd->fd;
#     endif
#   endif
#endif
}


/*\ Asynchronous Write: returns 0 if succeded or -1 if failed
\*/
int elio_awrite(fd, offset, buf, bytes, req_id)
     Fd_t         fd;
     off_t        offset;
     Void         *buf;
     Size_t        bytes;
     io_request_t *req_id;
{
  Size_t stat = -1;
  int    aio_i;

  PABLO_start(PABLO_elio_awrite);

  *req_id = ELIO_DONE;
  AIO_LOOKUP(aio_i);

  if(aio_i >= MAX_AIO_REQ){
#     if defined(DEBUG) && (defined(AIO) || defined(PARAGON))
         fprintf(stderr, "elio_awrite: Warning- asynch overflow\n");
#     endif
      SYNC_EMMULATE(write);
  } else {
      *req_id = (io_request_t) aio_i;
      elio_set_cb(fd, offset, aio_i, buf, bytes);
#if defined(PARAGON)
      if(offset != lseek(fd->fd, offset, SEEK_SET))
	ELIO_ERR("elio_awrite: lseek()", NULL, stat);

      *req_id = _iwrite(fd->fd, buf, bytes);

      stat = (*req_id == (io_request_t)-1) ? (Size_t)-1: (Size_t)0;
#elif defined(KSR) && defined(AIO)
      stat = awrite(fd->fd, buf, bytes, cb_fout+aio_i);
#elif defined(AIX) && defined(AIO)
      stat = aio_write(fd->fd, cb_fout+aio_i);
#elif defined(AIO)
      stat = aio_write(cb_fout+aio_i);
#endif
      aio_req[aio_i] = (int) *req_id;
    }

  if(stat ==-1) 
    ELIO_ERR("elio_awrite: awrite failed", NULL, stat);

  PABLO_end(PABLO_elio_awrite);
  return(stat);
}



/*\ Blocking Read - returns number of bytes read or -1 if failed
\*/
Size_t elio_read(fd, offset, buf, bytes)
Fd_t         fd;
off_t        offset;
Void        *buf;
Size_t       bytes;
{
Size_t stat = -1;
Size_t bytes_to_read = bytes;
int    attempt=0;

  PABLO_start(PABLO_elio_read);

      /* lseek error is always fatal */
      if(offset != lseek(fd->fd, offset, SEEK_SET))
	ELIO_ERR("elio_read: lseek() failed", NULL, stat);

      /* interrupted read should be restarted */
      do {
             if(attempt == MAX_ATTEMPTS)
	       ELIO_ERR("elio_read: max attempts exceeded", NULL, stat);

             stat = read(fd->fd, buf, bytes_to_read);
             if(stat < bytes_to_read && stat >= -1){
                bytes_to_read -= stat;
                buf = stat + (char*)buf; /*advance pointer by # bytes written */
             }else
                bytes_to_read = 0;
             attempt++;
      }while(bytes_to_read && (errno == EINTR || errno == EAGAIN));

      if(stat != -1) stat = bytes -  bytes_to_read;

      if(stat != bytes) 
	ELIO_ERR("elio_read: read failed", NULL, stat);

  PABLO_end(PABLO_elio_read);
  return(stat);
}



/*\ Asynchronous Read: returns 0 if succeded or -1 if failed
\*/
int elio_aread(fd, offset, buf, bytes, req_id)
Fd_t          fd;
off_t         offset;
Void         *buf;
Size_t        bytes;
io_request_t *req_id;
{
  Size_t stat = -1;
  int    aio_i;

  PABLO_start(PABLO_elio_aread);

  *req_id = ELIO_DONE;
  AIO_LOOKUP(aio_i);

  if(aio_i >= MAX_AIO_REQ){
#     if defined(DEBUG) && (defined(AIO) || defined(PARAGON))
         fprintf(stderr, "elio_read: Warning- asynch overflow\n");
#     endif
      SYNC_EMMULATE(read);
  } else {

     *req_id = (io_request_t) aio_i;
      elio_set_cb(fd, offset, aio_i, buf, bytes);
#if defined(PARAGON)
      if(offset != lseek(fd->fd, offset, SEEK_SET))
	ELIO_ERR("elio_aread: lseek failed", NULL, stat);
       req_id = _iread(fd->fd, buf, bytes);

      stat = (*req_id == (io_request_t)-1) ? (Size_t)-1: (Size_t)0;
#elif defined(KSR) && defined(AIO)
      stat = aread(fd->fd, buf, bytes, cb_fout+aio_i);
#elif defined(AIX) && defined(AIO)
      stat = aio_read(fd->fd, cb_fout+aio_i);
#elif defined(AIO)
      stat = aio_read(cb_fout+aio_i);
#endif
     aio_req[aio_i] = *req_id;
    }
  if(stat ==-1) 
    ELIO_ERR("elio_aread: aread failed", NULL, stat);

  PABLO_end(PABLO_elio_aread);
  return(stat);
}


/*\ Wait for asynchronous I/O operation to complete. Invalidate id.
\*/
int elio_wait(req_id)
io_request_t *req_id;
{
  int  aio_i=0;
#ifdef AIX
  int  rc;
#endif

  PABLO_start(PABLO_elio_wait); 

  if(*req_id != ELIO_DONE ) { 

#if defined(PARAGON)
      iowait(*req_id);
#elif defined(KSR)
      if((int)iosuspend(1, cb_fout_arr+(int)*req_id) ==-1)
	ELIO_ERR("elio_wait: iosuspend failed", NULL, -1);
#elif defined(AIX)

      /* I/O can be interrupted on SP through rcvncall ! */
      do {
           rc =(int)aio_suspend(1, cb_fout_arr+(int)*req_id);
      } while(rc == -1 && errno == EINTR); 
      if(rc  == -1) 
	ELIO_ERR("elio_wait: aio_suspend failed", NULL, -1);

#elif defined(AIO)

      if((int)aio_suspend(cb_fout_arr+(int)*req_id, 1, NULL) != 0)
	ELIO_ERR("elio_wait:aio_suspend", NULL, -1);

    /* only on DEC aio_return is required to clean internal data structures */
      if(aio_return(cb_fout+(int)*req_id) == -1)
	ELIO_ERR("elio_wait:aio_return", NULL, -1);
#endif
      while(aio_req[aio_i] != *req_id && aio_i < MAX_AIO_REQ) aio_i++;
      if(aio_i >= MAX_AIO_REQ)
	ELIO_ERR("elio_wait: Handle is not in aio_req table", NULL, -1);
      aio_req[aio_i] = NULL_AIO;
      *req_id = ELIO_DONE;
   }

   PABLO_end(PABLO_elio_wait);
   return ELIO_OK;
}



/*\ Check if asynchronous I/O operation completed. If yes, invalidate id.
\*/
int elio_probe(req_id, status)
io_request_t *req_id;
int          *status;
{
  int    errval;
  int    aio_i = 0;
     
   PABLO_start(PABLO_elio_probe);
   if(*req_id != ELIO_DONE){

#if defined(PARAGON)
       if( iodone(*req_id)== (long) 0) errval = INPROGRESS;
       else errval = 0;
#elif defined(KSR)
       errval = cb_fout[(int)*req_id].aio_errno;
#elif defined(AIX)
       errval = aio_error(cb_fout[(int)*req_id].aio_handle);
#elif defined(AIO)
       errval = aio_error(cb_fout+(int)*req_id);
#endif

       switch (errval) {
       case 0:           while(aio_req[aio_i] != *req_id && aio_i < MAX_AIO_REQ) aio_i++;
			 if(aio_i >= MAX_AIO_REQ)
			   ELIO_ERR("elio_probe: id not in table", NULL, -1);
			 *req_id = ELIO_DONE; 
	                 *status = ELIO_DONE;
			 aio_req[aio_i] = NULL_AIO;
			 break;
       case INPROGRESS:  *status = ELIO_PENDING; 
			 break;
       default:          ELIO_ERR("elio_probe", NULL, -1);
       }
   }
   PABLO_end(PABLO_elio_probe);
   return ELIO_OK;
}




/*\ Stat a file (or path) to determine it's filesystem type
\*/
int  elio_stat(fname, avail)
char *fname;
Size_t *avail;
{
  struct  stat     ufs_stat;
  struct  statfs   ufs_statfs;
  char             tmp_pathname[ELIO_FILENAME_MAX];
  int              i;
  int              ret_fs = -1;
#if defined(PARAGON)
  struct statpfs  *statpfsbuf;
  struct estatfs   estatbuf;
  int              bufsz;
#endif
#if defined(SP) || defined(SP1)
  piofs_statfs_t piofs_stat;
#endif
 
  PABLO_start(PABLO_elio_stat); 

  if(fname[0] == '/')
     {
        i = strlen(fname);
        while(fname[i] != '/') i--;
        tmp_pathname[i+1] = (char) 0;
        while(i >= 0)
          {
             tmp_pathname[i] = fname[i];
             i--;
          }
      }
   else
     strcpy(tmp_pathname, "./");
#if defined(DEBUG)
   fprintf(stderr, "tmp_pathname =%s|    fname=%s|\n", tmp_pathname, fname);
#endif

#if defined(PARAGON)
   bufsz = sizeof(struct statpfs) + SDIRS_INIT_SIZE;
   if( (statpfsbuf = (struct statpfs *) malloc(bufsz)) == NULL)
     ELIO_ERR("elio_stat: unable to malloc statpfsbuf", tmp_pathname, -1);
   if(statpfs(tmp_pathname, &estatbuf, statpfsbuf, bufsz) == 0)
     {
       if(estatbuf.f_type == MOUNT_PFS)
         ret_fs = FS_PFS;
       else if(estatbuf.f_type == MOUNT_UFS || estatbuf.f_type == MOUNT_NFS)
         ret_fs = FS_UFS;
       else
	 ELIO_ERR("elio_stat: stat ok, Unable to determine filesystem type", 
		  tmp_pathname, -1);
       *avail = estatbuf.f_bavail * 1024;   /* blocks avail -- block == 1KB */
     }
   else
     ELIO_ERR("elio_stat: Unable to to stat path.", tmp_pathname, -1);
   free(statpfsbuf);
#else
#  if defined(SP) || defined(SP1)
   strcpy(piofs_stat.name, tmp_pathname);
   if(piofsioctl(i, PIOFS_STATFS, &piofs_stat) == 0)
     {
       *avail = piofs_stat->f_bsize * piofs_stat->f_bavail;
       ret_fs = FS_PIOFS;
     }
#  endif
   if(ret_fs == -1)
     {
       if(stat(tmp_pathname, &ufs_stat) != 0)
	 ELIO_ERR("elio_stat: Not able to stat UFS filesystem", 
		  tmp_pathname, -1)
	   else
	     ret_fs = FS_UFS;
/* SGI requires two additional arguments to statfs so as the structure grows
   in future OS revisions, old binaries continues to work.
*/
#if defined(SGITFP) || defined(CRAY)
       if(statfs(tmp_pathname, &ufs_statfs, sizeof(ufs_statfs), 0) != 0)
#else
       if(statfs(tmp_pathname, &ufs_statfs) != 0)
#endif
	 ELIO_ERR("elio_stat: Not able to statfs UFS filesystem",
		  tmp_pathname, -1)
           else
#if defined(SGITFP) || defined(CRAY) /* f_bfree == f_bavail -- naming changes */
	     *avail = ufs_statfs.f_bsize * ufs_statfs.f_bfree;
#else
	     *avail = ufs_statfs.f_bsize * ufs_statfs.f_bavail;
#endif
     }
#endif
#if defined(DEBUG)
   fprintf(stderr, "Determined filesystem: %d\n", ret_fs);
#endif
   PABLO_end(PABLO_elio_stat);
   return(ret_fs);
}




/*\ Noncollective File Open
\*/
Fd_t  elio_open(fname, type)
char* fname;
int   type;
{
  Fd_t fd;
  int ptype;
  Size_t avail;

  PABLO_start(PABLO_elio_open);
  if(first_elio_init) elio_init();

   switch(type){
     case ELIO_W:  ptype = O_CREAT | O_TRUNC | O_WRONLY;
                   break;
     case ELIO_R:  ptype = O_RDONLY;
                   break;
     case ELIO_RW: ptype = O_CREAT | O_RDWR;
                   break;
     default:      ELIO_ERR("elio_open: mode incorrect", fname, (Fd_t) -1);
   }


   if( (fd = (Fd_t) malloc(sizeof(fd_struct)) ) == NULL)
     ELIO_ERR("elio_open: Unable to malloc Fd_t structure", fname, (Fd_t) -1);

   fd->fs = elio_stat(fname, &avail);
   fd->fd = open(fname, ptype, FOPEN_MODE );

   if((int)fd->fd == -1)
     ELIO_ERR("elio_open: open failed", fname, (Fd_t) -1);

   PABLO_end(PABLO_elio_open);
   return(fd);
}




/*\ Collective File Open
\*/
Fd_t  elio_gopen(fname, type)
char* fname;
int   type;
{
  Fd_t fd;
  Size_t avail;

  PABLO_start(PABLO_elio_gopen);

  if(first_elio_init) elio_init();
  
#  if defined(PARAGON)
   {
      int ptype;

      switch(type){
        case ELIO_W:  ptype = O_CREAT | O_TRUNC | O_WRONLY;
                      break;
        case ELIO_R:  ptype = O_RDONLY;
                      break;
        case ELIO_RW: ptype = O_CREAT | O_RDWR;
                      break;
        default:      ELIO_ERR("elio_gopen: mode incorrect", fname, -1);
      }

     if( (fd = (Fd_t) malloc(sizeof(fd_struct)) ) == NULL)
       ELIO_ERR("elio_gopen: Unable to malloc Fd_t structure\n", fname, -1);

     fd->fs = elio_stat(fname, &avail);
     fd->fd = gopen(fname, ptype, M_ASYNC, FOPEN_MODE );
   }

   if( (int)fd->fd == -1)
     ELIO_ERR("elio_gopen: gopen failed",(Fd_t) -1);
#else
      ELIO_ERR("elio_gopen: Collective open only supported on Paragon", 
	       fname, (Fd_t) -1);
#  endif

   if((int)fd->fd == -1)
     ELIO_ERR("elio_gopen: gopen failed", fname, (Fd_t) -1);

   PABLO_end(PABLO_elio_gopen);
   return(fd);
}


/*\ Close File
\*/
int elio_close(fd)
Fd_t fd;
{
   PABLO_start(PABLO_elio_close);

   /* if close fails, it must be a fatal error */
   if((int) close(fd->fd)==-1) 
     ELIO_ERR("elio_close: close failed", NULL, -1);
   free(fd);

   PABLO_end(PABLO_elio_close);
   return(ELIO_OK);
}


/*\ Delete File
\*/
int elio_delete(filename)
char  *filename;
{
   int rc;

   PABLO_start(PABLO_elio_delete);

    rc = unlink(filename);
    if(rc ==-1) 
      ELIO_ERR("elio_delete: unlink failed", filename, rc);

    PABLO_end(PABLO_elio_delete);
    return(rc);
}



/*\ Initialize ELIO
\*/
void elio_init()
{
  int   i;

  PABLO_start(PABLO_elio_init);

  if(first_elio_init)
    {
      first_elio_init = 0;
#if defined(AIO) || defined(PARAGON)
      for(i=0; i < MAX_AIO_REQ; i++)
	aio_req[i] = NULL_AIO;
#endif
    }
}



/*\ Error handling routine
\*/
void elio_err(char *func, char *fname)
{
  fprintf(stderr, "ELIO: Error in routine %s", func);
  if(fname != NULL)
    fprintf(stderr, " on file: |%s|", fname);
  perror("\nELIO: ");
}
 
