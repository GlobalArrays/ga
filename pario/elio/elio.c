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


#if  defined(AIX) || defined(DECOSF) || defined(SGITFP) || defined(SGI64)
#    ifndef NOAIO
#       define AIO 1
#    endif
#endif



#if defined(AIO)
#   include <aio.h>
#   if defined(KSR)||defined(AIX)
#      define INPROGRESS EINPROG
#   else 
#      define INPROGRESS EINPROGRESS
#   endif
    struct aiocb          cb_fout[MAX_AIO_REQ];
#ifndef AIX
    const
#endif
           struct aiocb   *cb_fout_arr[MAX_AIO_REQ];

#else
#   define INPROGRESS 1            /* I wish this didn't have to be here */
#endif

static long           aio_req[MAX_AIO_REQ]; /* array for AIO requests */
static int            first_elio_init = 1;  /* intialization status */
int                   _elio_Errors_Fatal=0; /* sets mode of handling errors */


/****************************** Internal Macros *****************************/
#if defined(AIO) || defined(PARAGON)
#  define AIO_LOOKUP(aio_i) {\
      aio_i = 0;\
      while(aio_req[aio_i] != NULL_AIO && aio_i < MAX_AIO_REQ) aio_i++;\
}
#else
#  define AIO_LOOKUP(aio_i) aio_i = MAX_AIO_REQ
#endif


#define SYNC_EMULATE(op) *req_id = ELIO_DONE; \
  if( elio_ ## op (fd, offset, buf, bytes) != bytes ) \
       stat   = -1;  \
  else \
       stat    = 0; 

/*****************************************************************************/


void elio_errors_fatal(int onoff)
{
    _elio_Errors_Fatal = onoff;
}
 


/*\ Blocking Write 
 *    - returns number of bytes written or error code (<0) if failed
\*/
Size_t elio_write(Fd_t fd, off_t  offset, const void* buf, Size_t bytes)
{
  Size_t stat, bytes_to_write = bytes;

  int pablo_code = PABLO_elio_write;
  PABLO_start( pablo_code );
  
  if(offset != lseek(fd->fd, offset, SEEK_SET)) ELIO_ERROR(SEEKFAIL,0);
  
  while (bytes_to_write) {
    stat = write(fd->fd, buf, bytes_to_write);
    if ((stat == -1) && ((errno == EINTR) || (errno == EAGAIN))) {
      ; /* interrupted write should be restarted */
    } else if (stat > 0) {
      bytes_to_write -= stat;
      buf = stat + (char*)buf; /*advance pointer by # bytes written*/
    } else {
      perror("elio_write");
      ELIO_ERROR(WRITFAIL, stat);
    }
  };

  /* Only get here if all has gone OK */
  
  PABLO_end(pablo_code);
  
  return bytes;
}



void elio_set_cb(Fd_t fd, off_t offset, int reqn, void *buf, Size_t bytes)
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


/*\ Asynchronous Write: returns 0 if succeded or err code if failed
\*/
int elio_awrite(Fd_t fd, off_t offset, const void* buf, Size_t bytes, io_request_t * req_id)
{
  Size_t stat;
  int    aio_i;

  int pablo_code = PABLO_elio_awrite;
  PABLO_start( pablo_code );

  *req_id = ELIO_DONE;
  AIO_LOOKUP(aio_i);

  if(aio_i >= MAX_AIO_REQ){
#     if defined(DEBUG) && (defined(AIO) || defined(PARAGON))
         fprintf(stderr, "elio_awrite: Warning- asynch overflow\n");
#     endif
      SYNC_EMULATE(write);
  } else {
      *req_id = (io_request_t) aio_i;
      elio_set_cb(fd, offset, aio_i, (void*) buf, bytes);
#if defined(PARAGON)
      if(offset != lseek(fd->fd, offset, SEEK_SET)) ELIO_ERROR(SEEKFAIL,0);
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
  if(stat ==-1) ELIO_ERROR(AWRITFAIL, aio_i);

  PABLO_end(pablo_code);

  return((int)stat);
}


/*\ Truncate the file at the specified length.
\*/
int elio_truncate(Fd_t fd, off_t length)
{
    int pablo_code = PABLO_elio_truncate;
    PABLO_start( pablo_code );

    (void) lseek(fd->fd, 0L, SEEK_SET);
    if (ftruncate(fd->fd, length))
	return TRUNFAIL;
    else {
	return ELIO_OK;
    }
    PABLO_end(pablo_code);
}


/*\ Return in length the length of the file
\*/
int elio_length(Fd_t fd, off_t *length)
{
    int pablo_code = PABLO_elio_length;
    PABLO_start( pablo_code );

    if ((*length = lseek(fd->fd, (off_t) 0, SEEK_END)) != -1)
	return ELIO_OK;
    else
	return SEEKFAIL;

    PABLO_end(pablo_code);
}


/*\ Blocking Read 
 *      - returns number of bytes read or error code (<0) if failed
\*/
Size_t elio_read(Fd_t fd, off_t  offset, void* buf, Size_t bytes)
{
Size_t stat, bytes_to_read = bytes;
int    attempt=0;

  int pablo_code = PABLO_elio_read;
  PABLO_start( pablo_code );

  if(offset != lseek(fd->fd,offset,SEEK_SET)) ELIO_ERROR(SEEKFAIL,0);
  
  while (bytes_to_read) {
    stat = read(fd->fd, buf, bytes_to_read);
    if (stat == 0) {
      ELIO_ERROR(EOFFAIL, stat);
    } else if ((stat == -1) && ((errno == EINTR) || (errno == EAGAIN))) {
      ; /* interrupted read should be restarted */
    } else if (stat > 0) {
      bytes_to_read -= stat;
      buf = stat + (char*)buf; /*advance pointer by # bytes read*/
    } else {
      perror("elio_read");
      ELIO_ERROR(READFAIL, stat);
    }
  };
  
  /* Only get here if all has gone OK */
  
  PABLO_end(pablo_code);
  
  return bytes;
}



/*\ Asynchronous Read: returns 0 if succeded or -1 if failed
\*/
int elio_aread(Fd_t fd, off_t offset, void* buf, Size_t bytes, io_request_t * req_id)
{
  Size_t stat;
  int    aio_i;

  int pablo_code = PABLO_elio_aread;
  PABLO_start( pablo_code );

  *req_id = ELIO_DONE;
  AIO_LOOKUP(aio_i);

  if(aio_i >= MAX_AIO_REQ){
#     if defined(DEBUG) && (defined(AIO) || defined(PARAGON))
         fprintf(stderr, "elio_read: Warning- asynch overflow\n");
#     endif
      SYNC_EMULATE(read);
  } else {

     *req_id = (io_request_t) aio_i;
      elio_set_cb(fd, offset, aio_i, buf, bytes);
#if defined(PARAGON)
      if(offset != lseek(fd->fd, offset, SEEK_SET)) ELIO_ERROR(SEEKFAIL,0);
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
  if(stat ==-1) ELIO_ERROR(AWRITFAIL, 0);

  PABLO_end(pablo_code);
  return((int)stat);
}


/*\ Wait for asynchronous I/O operation to complete. Invalidate id.
\*/
int elio_wait(io_request_t *req_id)
{
  int  aio_i=0;
#ifdef AIO
  int  rc;
#endif

  int pablo_code = PABLO_elio_wait;
  PABLO_start( pablo_code );

  if(*req_id != ELIO_DONE ) { 

#if defined(PARAGON)
      iowait(*req_id);
#elif defined(AIO)
#  if defined(AIX)

      do {    /* I/O can be interrupted on SP through rcvncall ! */
           rc =(int)aio_suspend(1, cb_fout_arr+(int)*req_id);
      } while(rc == -1 && errno == EINTR); 

#  elif defined(KSR)
      rc = iosuspend(1, cb_fout_arr+(int)*req_id);
#  else
      if((int)aio_suspend(cb_fout_arr+(int)*req_id, 1, NULL) != 0) rc =-1;
#  endif
      if(rc ==-1) ELIO_ERROR(SUSPFAIL,0);

#  ifndef KSR
      /* on DEC aio_return is required to clean internal data structures */
      if(aio_return(cb_fout+(int)*req_id) == -1) ELIO_ERROR(RETUFAIL,0);
#  endif
#endif
      while(aio_req[aio_i] != *req_id && aio_i < MAX_AIO_REQ) aio_i++;
      if(aio_i >= MAX_AIO_REQ) ELIO_ERROR(HANDFAIL, aio_i);
      aio_req[aio_i] = NULL_AIO;
      *req_id = ELIO_DONE;
   }

   PABLO_end(pablo_code);
   return ELIO_OK;
}



/*\ Check if asynchronous I/O operation completed. If yes, invalidate id.
\*/
int elio_probe(io_request_t *req_id, int* status)
{
  int    errval=-1;
  int    aio_i = 0;
     
  int pablo_code = PABLO_elio_probe;
  PABLO_start( pablo_code );

  if(*req_id == ELIO_DONE){
      *status = ELIO_DONE;
  }
  else {
      
#if defined(PARAGON)
      if( iodone(*req_id)== (long) 0) errval = INPROGRESS;
      else errval = 0;
#elif defined(AIO)
#  if defined(KSR)
      errval = cb_fout[(int)*req_id].aio_errno;
#  elif defined(AIX)
      errval = aio_error(cb_fout[(int)*req_id].aio_handle);
#  else
      errval = aio_error(cb_fout+(int)*req_id);
#  endif
#endif
      switch (errval) {
      case 0: 
          while(aio_req[aio_i] != *req_id && aio_i < MAX_AIO_REQ) aio_i++;
          if(aio_i >= MAX_AIO_REQ) ELIO_ERROR(HANDFAIL, aio_i);
	  *req_id = ELIO_DONE; 
	  *status = ELIO_DONE;
	  aio_req[aio_i] = NULL_AIO;
	  break;
      case INPROGRESS:
	  *status = ELIO_PENDING; 
	  break;
      default:
          return PROBFAIL;
      }
  }
  PABLO_end(pablo_code);
  return ELIO_OK;
}


/*\ Noncollective File Open
\*/
Fd_t  elio_open(const char* fname, int type)
{
  Fd_t fd=NULL;
  stat_t statinfo;
  int ptype, rc;
  char dirname[ELIO_FILENAME_MAX];

  int pablo_code = PABLO_elio_open;
  PABLO_start( pablo_code );

  if(first_elio_init) elio_init();

   switch(type){
     case ELIO_W:  ptype = O_CREAT | O_TRUNC | O_WRONLY;
                   break;
     case ELIO_R:  ptype = O_RDONLY;
                   break;
     case ELIO_RW: ptype = O_CREAT | O_RDWR;
                   break;
     default:      
                   ELIO_ERROR_NULL(MODEFAIL, type);
   }


  if((fd = (Fd_t ) malloc(sizeof(fd_struct)) ) == NULL) 
                   ELIO_ERROR_NULL(ALOCFAIL, 0);

  if( (rc = elio_dirname(fname, dirname, ELIO_FILENAME_MAX)) != ELIO_OK) {
                   free(fd);
                   ELIO_ERROR_NULL(rc, 0);
  }

  if( (rc = elio_stat(dirname, &statinfo)) != ELIO_OK) {
                   free(fd);
                   ELIO_ERROR_NULL(rc, 0);
  }

  fd->fs = statinfo.fs;
  
  fd->fd = open(fname, ptype, FOPEN_MODE );
  if( (int)fd->fd == -1) {
                   free(fd);
                   ELIO_ERROR_NULL(OPENFAIL, 0);
  }
  
  PABLO_end(pablo_code);
  return(fd);
}



/*\ Collective File Open
\*/
Fd_t  elio_gopen(const char* fname, int type)
{
  Fd_t fd=NULL;

  int pablo_code = PABLO_elio_gopen;
  PABLO_start( pablo_code );

# if defined(PARAGON)
  if(first_elio_init) elio_init();
  
   {
      char dirname[ELIO_FILENAME_MAX];
      stat_t statinfo;
      int ptype, rc;

      switch(type){
        case ELIO_W:  ptype = O_CREAT | O_TRUNC | O_WRONLY;
                      break;
        case ELIO_R:  ptype = O_RDONLY;
                      break;
        case ELIO_RW: ptype = O_CREAT | O_RDWR;
                      break;
        default:      ELIO_ERROR_NULL(MODEFAIL, 0);
      }

      if((fd = (Fd_t ) malloc(sizeof(fd_struct)) ) == NULL)
                   ELIO_ERROR_NULL(ALOCFAIL, 0);

      if( (rc = elio_dirname(fname, dirname, ELIO_FILENAME_MAX)) != ELIO_OK) {
                   free(fd);
                   ELIO_ERROR_NULL(rc, 0);
      }

      if( (rc = elio_stat(dirname, &statinfo)) != ELIO_OK) {
                   free(fd);
                   ELIO_ERROR_NULL(rc, 0);
      }


      fd->fs = statinfo.fs;
      fd->fd = gopen(fname, ptype, M_ASYNC, FOPEN_MODE );
   }

   if((int)fd->fd == -1){
                   free(fd);
                   ELIO_ERROR_NULL(OPENFAIL, 0);
   }
#  else
      ELIO_ERROR_NULL(UNSUPFAIL,0);
#  endif

   PABLO_end(pablo_code);
   return(fd);
}


/*\ Close File
\*/
int elio_close(Fd_t fd)
{
    int pablo_code = PABLO_elio_close;
    PABLO_start( pablo_code );

    if(close(fd->fd)==-1) ELIO_ERROR(CLOSFAIL, 0);
    free(fd);

    PABLO_end(pablo_code);
    return ELIO_OK;
}


/*\ Delete File
\*/
int elio_delete(const char* filename)
{
    int rc;

    int pablo_code = PABLO_elio_delete;
    PABLO_start( pablo_code );

    rc = unlink(filename);
    if(rc ==-1) ELIO_ERROR(DELFAIL,0);

    PABLO_end(pablo_code);
    return(ELIO_OK);
}



/*\ Initialize ELIO
\*/
void elio_init(void)
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


/*\ Return Error String Associated with Given Error Code 
\*/
void elio_errmsg(int code, char *msg)
{
     if(code==ELIO_OK){
         (void) strcpy(msg, ">OK");
         return;
     }
     else if(code == ELIO_PENDING_ERR) code = elio_pending_error;

     if(code<OFFSET || code >OFFSET+ERRLEN) *msg=(char)0;
     else (void) strcpy(msg, errtable[-OFFSET + code]);
}
      

int elio_pending_error=UNKNFAIL;

char *errtable[ERRLEN] ={
">Unable to Seek",
">Write Failed",
">Asynchronous Write Failed",
">Read Failed",
">Asynchronous Read Failed",
">Suspend Failed",
">I/O Request Handle not in Table",
">Incorrect File Mode",
">Unable to Determine Directory",
">Stat For Specified File or Directory Failed",
">Open Failed",
">Unable To Allocate Internal Data Structure",
">Unsupported Feature",
">Unlink Failed",
">Close Failed",
">Operation Interrupted Too Many Times",
">AIO Return Failed",
">Name String too Long",
">Unable to Determine Filesystem Type",
">Numeric Conversion Error", 
">Incorrect Filesystem/Device Type",
">Error in Probe",
">Unable to Truncate",
">End of File",
""};
