/**********************************************************************\
 ELementary I/O (ELIO) disk operations for Chemio libraries   
 Authors: Jarek Nieplocha (PNNL) and Jace Mogill (ANL)
\**********************************************************************/

#ifdef CRAY_T3E
#define FFIO 1
#endif

#include "eliop.h"


#if  defined(AIX) || defined(DECOSF) || defined(SGITFP) || defined(SGI64) || defined(SGI_N32) || defined(CRAY) || defined(PARAGON)
     /* systems with Asynchronous I/O */
#else
#    ifndef NOAIO
#      define NOAIO
#    endif
#endif

/****************** Internal Constants and Parameters **********************/

#define  MAX_AIO_REQ  4
#define  NULL_AIO    -123456
#define  FOPEN_MODE 0644
#define  MAX_ATTEMPTS 10


#ifndef NOAIO
#   define AIO 1
#endif


#ifdef FFIO
#  define WRITE  ffwrite
#  define WRITEA ffwritea
#  define READ   ffread
#  define READA  ffreada
#  define CLOSE  ffclose
#  define SEEK   ffseek
#  define OPEN   ffopens
#  define DEFARG FULL
#else
#  define WRITE  write
#  define WRITEA writea
#  define READ   read
#  define READA  reada
#  define CLOSE  close
#  define SEEK   lseek
#  define OPEN   open
#  define DEFARG 0
#endif


/* structure to emulate control block in Posix AIO */
#if defined (CRAY)
#   if defined(FFIO)
       typedef struct { struct ffsw stat; int filedes; }io_status_t;
#   else 
#      include <sys/iosw.h>
       typedef struct { struct iosw stat; int filedes; }io_status_t;
#   endif
    io_status_t cb_fout[MAX_AIO_REQ];
    io_status_t *cb_fout_arr[MAX_AIO_REQ];

#elif defined(AIO)
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
#endif

#ifndef INPROGRESS
#   define INPROGRESS 1
#endif

static long           aio_req[MAX_AIO_REQ]; /* array for AIO requests */
static int            first_elio_init = 1;  /* intialization status */
int                   _elio_Errors_Fatal=0; /* sets mode of handling errors */


/****************************** Internal Macros *****************************/
#if defined(AIO)
#  define AIO_LOOKUP(aio_i) {\
      aio_i = 0;\
      while(aio_req[aio_i] != NULL_AIO && aio_i < MAX_AIO_REQ) aio_i++;\
}
#else
#  define AIO_LOOKUP(aio_i) aio_i = MAX_AIO_REQ
#endif

#define SYNC_EMULATE(op) *req_id = ELIO_DONE; \
  if((stat= elio_ ## op (fd, offset, buf, bytes)) != bytes ){ \
       ELIO_ERROR(stat,0);  \
  }else \
       stat    = 0; 

#ifndef MIN 
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#endif

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
  
  if(offset != SEEK(fd->fd,offset,SEEK_SET)) ELIO_ERROR(SEEKFAIL,0);
  
  while (bytes_to_write) {
    stat = WRITE(fd->fd, buf, bytes_to_write);
    if ((stat == -1) && ((errno == EINTR) || (errno == EAGAIN))) {
      ; /* interrupted write should be restarted */
    } else if (stat > 0) {
      bytes_to_write -= stat;
      buf = stat + (char*)buf; /*advance pointer by # bytes written*/
    } else {
      ELIO_ERROR(WRITFAIL, stat);
    }
  }

  /* Only get here if all has gone OK */
  
  PABLO_end(pablo_code);
  
  return bytes;
}



int elio_set_cb(Fd_t fd, off_t offset, int reqn, void *buf, Size_t bytes)
{
#if defined(AIO)
#   if defined(PARAGON) || defined(CRAY)
       if(offset != SEEK(fd->fd, offset, SEEK_SET))return (SEEKFAIL);
#      if  defined(CRAY)
           cb_fout_arr[reqn] = cb_fout+reqn;
           cb_fout[reqn].filedes    = fd->fd;
#      endif
#   else
       cb_fout[reqn].aio_offset = offset;
       cb_fout_arr[reqn] = cb_fout+reqn;
#      if defined(KSR)
         cb_fout[reqn].aio_whence = SEEK_SET;
#      else
         cb_fout[reqn].aio_buf    = buf;
         cb_fout[reqn].aio_nbytes = bytes;
#        if defined(AIX)
           cb_fout[reqn].aio_whence = SEEK_SET;
#        else
           cb_fout[reqn].aio_sigevent.sigev_notify = SIGEV_NONE;
           cb_fout[reqn].aio_fildes    = fd->fd;
#        endif
#      endif
#   endif
#endif
    return ELIO_OK;
}


/*\ Asynchronous Write: returns 0 if succeded or err code if failed
\*/
int elio_awrite(Fd_t fd, off_t offset, const void* buf, Size_t bytes, io_request_t * req_id)
{
  Size_t stat;
  int    aio_i;
  int    rc;

  int pablo_code = PABLO_elio_awrite;
  PABLO_start( pablo_code );

  *req_id = ELIO_DONE;

#ifdef AIO
   AIO_LOOKUP(aio_i);

   /* blocking io when request table is full */
   if(aio_i >= MAX_AIO_REQ){
#     if defined(DEBUG) && defined(ASYNC)
         fprintf(stderr, "elio_awrite: Warning- asynch overflow\n");
#     endif
      SYNC_EMULATE(write);
   } else {
      *req_id = (io_request_t) aio_i;
      if((rc=elio_set_cb(fd, offset, aio_i, (void*) buf, bytes)))
                                                 ELIO_ERROR(rc,0);

#    if defined(PARAGON)
       *req_id = _iwrite(fd->fd, buf, bytes);
       stat = (*req_id == (io_request_t)-1) ? (Size_t)-1: (Size_t)0;
#    elif defined(CRAY)
       rc = WRITEA(fd->fd, (char*)buf, bytes, &cb_fout[aio_i].stat, DEFARG);
       stat = (rc < 0)? -1 : 0; 
#    elif defined(KSR)
       stat = awrite(fd->fd, buf, bytes, cb_fout+aio_i);
#    elif defined(AIX)
       stat = aio_write(fd->fd, cb_fout+aio_i);
#    else
       stat = aio_write(cb_fout+aio_i);
#    endif
     aio_req[aio_i] = *req_id;
  }

#else
      /* call blocking write when AIO not available */
      SYNC_EMULATE(write);
#endif

  if(stat ==-1) ELIO_ERROR(AWRITFAIL, 0);

  PABLO_end(pablo_code);

  return((int)stat);
}


/*\ Truncate the file at the specified length.
\*/
int elio_truncate(Fd_t fd, off_t length)
{
#ifdef WIN32
#   define ftruncate _chsize 
#endif

    int pablo_code = PABLO_elio_truncate;
    PABLO_start( pablo_code );

    (void) SEEK(fd->fd, 0L, SEEK_SET);
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

    if ((*length = SEEK(fd->fd, (off_t) 0, SEEK_END)) != -1)
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

  if(offset != SEEK(fd->fd,offset,SEEK_SET)) ELIO_ERROR(SEEKFAIL,0);
  
  while (bytes_to_read) {
    stat = READ(fd->fd, buf, bytes_to_read);
    if(stat==0){
      ELIO_ERROR(EOFFAIL, stat);
    } else if ((stat == -1) && ((errno == EINTR) || (errno == EAGAIN))) {
      ; /* interrupted read should be restarted */
    } else if (stat > 0) {
      bytes_to_read -= stat;
      buf = stat + (char*)buf; /*advance pointer by # bytes read*/
    } else {
      ELIO_ERROR(READFAIL, stat);
    }
    attempt++;
  }
  
  /* Only get here if all went OK */
  
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

#ifdef AIO
    AIO_LOOKUP(aio_i);

    /* blocking io when request table is full */
    if(aio_i >= MAX_AIO_REQ){
#       if defined(DEBUG)
           fprintf(stderr, "elio_read: Warning- asynch overflow\n");
#       endif
        SYNC_EMULATE(read);

    } else {

       int    rc; 

       *req_id = (io_request_t) aio_i;
        if((stat=elio_set_cb(fd, offset, aio_i, (void*) buf, bytes)))
                                                 ELIO_ERROR((int)stat,0);
#       if defined(PARAGON)
          *req_id = _iread(fd->fd, buf, bytes);
          stat = (*req_id == (io_request_t)-1) ? (Size_t)-1: (Size_t)0;
#       elif defined(CRAY)
          rc = READA(fd->fd, buf, bytes, &cb_fout[aio_i].stat, DEFARG);
          stat = (rc < 0)? -1 : 0;
#       elif defined(KSR)
          stat = aread(fd->fd, buf, bytes, cb_fout+aio_i);
#       elif defined(AIX)
          stat = aio_read(fd->fd, cb_fout+aio_i);
#       else
          stat = aio_read(cb_fout+aio_i);
#       endif
        aio_req[aio_i] = *req_id;
    }
#else

    /* call blocking write when AIO not available */
    SYNC_EMULATE(read);

#endif

    if(stat ==-1) ELIO_ERROR(AWRITFAIL, 0);

    PABLO_end(pablo_code);
    return((int)stat);
}


/*\ Wait for asynchronous I/O operation to complete. Invalidate id.
\*/
int elio_wait(io_request_t *req_id)
{
  int  aio_i=0;
  int  rc=0;

  int pablo_code = PABLO_elio_wait;
  PABLO_start( pablo_code );

  if(*req_id != ELIO_DONE ) { 

#    ifdef AIO
#      if defined(PARAGON)
          iowait(*req_id);

#    elif defined(CRAY)

#      if defined(FFIO)
       {
          struct ffsw dumstat, *prdstat=&(cb_fout[*req_id].stat);
          fffcntl(cb_fout[*req_id].filedes, FC_RECALL, prdstat, &dumstat);
          if (FFSTAT(*prdstat) == FFERR) ELIO_ERROR(SUSPFAIL,0);
       }
#      else
       {
          struct iosw *statlist[1];
          statlist[0] = &(cb_fout[*req_id].stat);
          recall(cb_fout[*req_id].filedes, 1, statlist); 
       }
#      endif

#  elif defined(AIX)

      do {    /* I/O can be interrupted on SP through rcvncall ! */
           rc =(int)aio_suspend(1, cb_fout_arr+(int)*req_id);
      } while(rc == -1 && errno == EINTR); 

#  elif defined(KSR)
      rc = iosuspend(1, cb_fout_arr+(int)*req_id);
#  else
      if((int)aio_suspend(cb_fout_arr+(int)*req_id, 1, NULL) != 0) rc =-1;
#  endif
      if(rc ==-1) ELIO_ERROR(SUSPFAIL,0);

#  if defined(DECOSF)
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
  } else {
      
#ifdef AIO
#    if defined(PARAGON)
        if( iodone(*req_id)== (long) 0) errval = INPROGRESS;
        else errval = 0;

#    elif defined(CRAY)

#     if defined(FFIO)
      {
         struct ffsw dumstat, *prdstat=&(cb_fout[*req_id].stat);
         fffcntl(cb_fout[*req_id].filedes, FC_ASPOLL, prdstat, &dumstat);
         errval = (FFSTAT(*prdstat) == 0) ? INPROGRESS: 0;
      }
#     else

         errval = ( IO_DONE(cb_fout[*req_id].stat) == 0)? INPROGRESS: 0;

#     endif

#   elif defined(KSR)
      errval = cb_fout[(int)*req_id].aio_errno;
#   elif defined(AIX)
      errval = aio_error(cb_fout[(int)*req_id].aio_handle);
#   else
      errval = aio_error(cb_fout+(int)*req_id);
#   endif
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


#if defined(CRAY) && defined(FFIO)
static int cray_part_info(char *dirname,long *pparts,long *sparts)
{
  struct statfs stats;
  long temp,count=0;

  if(statfs(dirname, &stats, sizeof(struct statfs), 0) == -1) return -1;

  temp = stats.f_priparts;
  while(temp != 0){
      count++;
      temp <<= 1;
  }
 *pparts = count;

 if(stats.f_secparts != 0){

    temp = (stats.f_secparts << count);
    count = 0;
    while(temp != 0){
           count++;
           temp <<= 1;
    }
    *sparts = count;
 }
 return ELIO_OK;

}

#endif


/*\ Noncollective File Open
\*/
Fd_t  elio_open(const char* fname, int type, int mode)
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

#ifdef WIN32
   ptype |= O_BINARY;
#endif

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
  fd->mode = mode;
  
#if defined(CRAY) && defined(FFIO)
  {
    struct ffsw ffstat;
    long pparts, sparts, cbits, cblocks;
    extern long _MPP_MY_PE;
    char *ffio_str="cache:256"; /*  intern I/O buffer/cache 256*4096 bytes */ 
                                /*  JN: we do not want read-ahead write-behind*/

    if(cray_part_info(dirname,&pparts,&sparts) != ELIO_OK){
                   free(fd);
                   ELIO_ERROR_NULL(STATFAIL, 0);
    }

    ptype |= ( O_BIG | O_PLACE | O_RAW );
    cbits = (sparts != 0) ? 1 : 0;

    if( sparts != 0) {

      /* stripe is set so we only select secondary partitions with cbits */
      if(mode == ELIO_SHARED){
         cbits = ~((~0L)<<MIN(32,sparts)); /* use all secondary partitions */
         cblocks = 100;
      }else{
         cbits = 1 << (_MPP_MY_PE%sparts);  /* round robin over s part */
      }

      cbits <<= pparts;        /* move us out of the primary partitions */

     }

     
/*     printf ("parts=%d cbits = %X\n",sparts,cbits);*/

     if(mode == ELIO_SHARED)
      fd->fd = OPEN(fname, ptype, FOPEN_MODE, cbits, cblocks, &ffstat,NULL);
     else
      fd->fd = OPEN(fname, ptype, FOPEN_MODE, 0L, 0,  &ffstat,ffio_str);

  }
#else
  fd->fd = OPEN(fname, ptype, FOPEN_MODE );
#endif

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
      fd->mode = ELIO_SHARED;
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

    if(CLOSE(fd->fd)==-1) ELIO_ERROR(CLOSFAIL, 0);
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
#     if defined(ASYNC)
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
