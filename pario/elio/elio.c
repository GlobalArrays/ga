/**********************************************************************\
 ELementary I/O (ELIO) disk operations for Chemio libraries   
 Authors: Jarek Nieplocha (PNNL) and Jace Mogill (ANL)
\**********************************************************************/

#include "eliop.h"

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
int                   _elio_Errors_Fatal=1; /* sets mode of handling errors */


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


void elio_errors_fatal(onoff)
int onoff;
{
    _elio_Errors_Fatal = onoff;
}
 

/*\ Blocking Write - returns number of bytes written or -1 if failed
\*/
Size_t elio_write(fd, offset, buf, bytes)
     Fd_t         fd;
     off_t        offset;
     const Void        *buf;
     Size_t       bytes;
{
Size_t stat, bytes_to_write = bytes;
int    attempt=0;

      PABLO_start( PABLO_elio_write );

      /* lseek error is always fatal */
      if(offset != lseek(fd->fd, offset, SEEK_SET))
                         ELIO_ERROR("elio_write: seek broken:",0);

      /* interrupted write should be restarted */
      do {
             if(attempt == MAX_ATTEMPTS) 
                ELIO_ERROR("elio_write: num max attempts exceeded", attempt);

             stat = write(fd->fd, buf, bytes_to_write);
             if(stat < bytes_to_write && stat >= -1){
                bytes_to_write -= stat;
                buf = stat + (char*)buf; /*advance pointer by # bytes written */
             }else
                bytes_to_write = 0;
             attempt++;
      }while(bytes_to_write && (errno == EINTR || errno == EAGAIN));

      if(stat != -1) stat = bytes -  bytes_to_write;

      if(stat != bytes) ELIO_ERROR("elio_write: failed", 0);

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
     const Void   *buf;
     Size_t        bytes;
     io_request_t *req_id;
{
  Size_t stat;
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
                   ELIO_ERROR("elio_awrite: seek broken:",0);
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
  if(stat ==-1) ELIO_ERROR("elio_awrite: failed", aio_i);

  PABLO_end(PABLO_elio_awrite);
  return(stat);
}

int elio_truncate(Fd_t fd, off_t length)
/*
  Truncate the file at the specified length.
  */
{
    (void) lseek(fd->fd, 0L, SEEK_SET);
    if (ftruncate(fd->fd, length))
	return ELIO_FAIL;
    else {
	return ELIO_OK;
    }
}

int elio_length(Fd_t fd, off_t *length)
/*
 Return in length the length of the file
  */
{
    if ((*length = lseek(fd->fd, (off_t) 0, SEEK_END)) != -1)
	return ELIO_OK;
    else
	return ELIO_FAIL;
}

/*\ Blocking Read - returns number of bytes read or -1 if failed
\*/
Size_t elio_read(fd, offset, buf, bytes)
Fd_t         fd;
off_t        offset;
Void        *buf;
Size_t       bytes;
{
Size_t stat, bytes_to_read = bytes;
int    attempt=0;

  PABLO_start(PABLO_elio_read);

      /* lseek error is always fatal */
      if(offset != lseek(fd->fd, offset, SEEK_SET))
                         ELIO_ERROR("elio_read: seek broken:",0);

      /* interrupted read should be restarted */
      do {
             if(attempt == MAX_ATTEMPTS) 
                ELIO_ERROR("elio_read: num max attempts exceeded", attempt);

             stat = read(fd->fd, buf, bytes_to_read);
             if(stat < bytes_to_read && stat >= -1){
                bytes_to_read -= stat;
                buf = stat + (char*)buf; /*advance pointer by # bytes written */
             }else
                bytes_to_read = 0;
             attempt++;
      }while(bytes_to_read && (errno == EINTR || errno == EAGAIN));

      if(stat != -1) stat = bytes -  bytes_to_read;

      if(stat != bytes) ELIO_ERROR("elio_read: failed", 0);

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
  Size_t stat;
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
        	   ELIO_ERROR("elio_aread: seek broken:",0);
      *req_id = _iread(fd->fd, buf, bytes);
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
  if(stat ==-1) ELIO_ERROR("elio_aread: failed", 0);

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
	ELIO_ERROR("elio_wait: suspend error",0);
#elif defined(AIX)

      /* I/O can be interrupted on SP through rcvncall ! */
      do {
           rc =(int)aio_suspend(1, cb_fout_arr+(int)*req_id);
      } while(rc == -1 && errno == EINTR); 
      if(rc  == -1) ELIO_ERROR("elio_wait:  suspend error",0);

#elif defined(AIO)

      if((int)aio_suspend(cb_fout_arr+(int)*req_id, 1, NULL) != 0)
	      ELIO_ERROR("elio_wait: suspend error",0);

      /* only on DEC aio_return is required to clean internal data structures */
      if(aio_return(cb_fout+(int)*req_id) == -1)
	      ELIO_ERROR("elio_wait: suspend error",0);
#endif
      while(aio_req[aio_i] != *req_id && aio_i < MAX_AIO_REQ) aio_i++;
      if(aio_i >= MAX_AIO_REQ)
	ELIO_ERROR("elio_wait: Handle is not in aio_req table", 1);
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
  if(*req_id == ELIO_DONE){
      *status = ELIO_DONE;
  }
  else {
      
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
      case 0: 
          while(aio_req[aio_i] != *req_id && aio_i < MAX_AIO_REQ) aio_i++;
	  if(aio_i >= MAX_AIO_REQ)
	      ELIO_ERROR("elio_probe: id %d not in table", 1);
	  *req_id = ELIO_DONE; 
	  *status = ELIO_DONE;
	  aio_req[aio_i] = NULL_AIO;
	  break;
      case INPROGRESS:
	  *status = ELIO_PENDING; 
	  break;
      default:
          return ELIO_FAIL; /* Problem */
      }
  }
  PABLO_end(PABLO_elio_probe);
  return ELIO_OK;
}


/*\ Noncollective File Open
\*/
Fd_t  elio_open(fname, type)
const char* fname;
int   type;
{
  Fd_t fd=NULL;
  stat_t statinfo;
  int ptype;
  char dirname[ELIO_FILENAME_MAX];

  PABLO_start(PABLO_elio_open);
  if(first_elio_init) elio_init();

   switch(type){
     case ELIO_W:  ptype = O_CREAT | O_TRUNC | O_WRONLY;
                   break;
     case ELIO_R:  ptype = O_RDONLY;
                   break;
     case ELIO_RW: ptype = O_CREAT | O_RDWR;
                   break;
     default:      
	 return (Fd_t) 0;
	 /* ELIO_ABORT("elio_open: mode incorrect", 0); */
   }


  if( (fd = (Fd_t ) malloc(sizeof(fd_struct)) ) == NULL) {
      return (Fd_t) 0;
      /* ELIO_ABORT("elio_open: Unable to malloc Fd_t structure\n", 1); */
  }

  if( elio_dirname(fname, dirname, ELIO_FILENAME_MAX) != ELIO_OK) {
      free(fd);
      return (Fd_t) 0;
      /* ELIO_ABORT("elio_open: could not determine directory path", -1); */
  }

  if( elio_stat(dirname, &statinfo) != ELIO_OK) {
      free(fd);
      return (Fd_t) 0;
      /* ELIO_ABORT("elio_open: stat problem", -1); */
  }

  fd->fs = statinfo.fs;
  
  fd->fd = open(fname, ptype, FOPEN_MODE );
  if( (int)fd->fd == -1) {
      free(fd);
      return (Fd_t) 0;
      /* ELIO_ABORT("elio_open failed",0); */
  }
  
   PABLO_end(PABLO_elio_open);
   return(fd);
}



/*\ Collective File Open
\*/
Fd_t  elio_gopen(fname, type)
const char* fname;
int   type;
{
  Fd_t fd=NULL;

  PABLO_start(PABLO_elio_gopen);

# if defined(PARAGON)
  if(first_elio_init) elio_init();
  
   {
      char dirname[ELIO_FILENAME_MAX];
      stat_t statinfo;
      int ptype;

      switch(type){
        case ELIO_W:  ptype = O_CREAT | O_TRUNC | O_WRONLY;
                      break;
        case ELIO_R:  ptype = O_RDONLY;
                      break;
        case ELIO_RW: ptype = O_CREAT | O_RDWR;
                      break;
        default:      ELIO_ABORT("elio_open: mode incorrect", 0);
      }

     if( (fd = (Fd_t ) malloc(sizeof(fd_struct)) ) == NULL)
       ELIO_ABORT("elio_gopen: Unable to malloc Fd_t structure\n", 1);

     if( elio_dirname(fname, dirname, ELIO_FILENAME_MAX) != ELIO_OK) 
       ELIO_ABORT("elio_gopen: could not determine directory path", -1);

     if( elio_stat(dirname, &statinfo) != ELIO_OK)
       ELIO_ABORT("elio_gopen: stat problem", -1);

      fd->fs = statinfo.fs;
      fd->fd = gopen(fname, ptype, M_ASYNC, FOPEN_MODE );
   }

   if((int)fd->fd == -1)ELIO_ABORT("elio_gopen failed",0);
#  else
      ELIO_ABORT("elio_gopen: Collective open only supported on Paragon",0);
#  endif


   PABLO_end(PABLO_elio_gopen);
   return(fd);
}


/*\ Close File
\*/
int elio_close(fd)
Fd_t fd;
{
   PABLO_start(PABLO_elio_close);

   if(close(fd->fd)==-1) return 1;
   free(fd);

   PABLO_end(PABLO_elio_close);

   return ELIO_OK;
}


/*\ Delete File
\*/
int elio_delete(filename)
const char  *filename;
{
    int rc;

    PABLO_start(PABLO_elio_delete);
    rc = unlink(filename);
    if(rc ==-1) ELIO_ERROR("elio_delete failed",0);

    PABLO_end(PABLO_elio_delete);
    return(ELIO_OK);
}



/*\ Initialize ELIO
\*/
void elio_init()
{
  if(first_elio_init) {
#     if defined(AIO) || defined(PARAGON)
           int i;
           for(i=0; i < MAX_AIO_REQ; i++)
	     aio_req[i] = NULL_AIO;
#     endif
      first_elio_init = 0;
  }
}


