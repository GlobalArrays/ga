/******************************************************************************
Source File:    eaf_c2f.c

Description:    Fortran bindings to C EAF calls
                Parameter explanation on C calls
  
Author:         Jace A Mogill

Date Created:   16 May 1996

Modifications:

CVS: $Source: /tmp/hpctools/ga/pario/eaf/eaf_c2f.c,v $
CVS: $Date: 1996-07-17 15:28:21 $
CVS: $Revision: 1.1 $
CVS: $State: Exp $
******************************************************************************/

#if defined(CRAY)              /* This is needed to get the _fcd type for    */
# include <fortran.h>         /* da_fort_char_t since Fortran string pnters */
#endif                        /* are actually 128bit structures             */
/* #include "../ELIO/elio.h" */
#include "elio.h"
#include "eaf_c2f.h"
#include "eaf.h"


/*\
\*/
/*\ Close a file
\*/
eaf_fort_status_t      EAF_Close(eaf_fort_fd_t *fort_fd)
{
  EAF_CloseC(*fort_fd);
  return (eaf_fort_status_t) EAF_STAT_OK;
}




/*\
\*/
/*\ Open a persistent file
\*/
#if defined(CRAY)  /* Cray Fortran passes a 128bit structure, not pointer */
eaf_fort_fd_t        EAF_OpenPersist(eaf_fort_char_t  fn,
#else
eaf_fort_fd_t        EAF_OpenPersist(eaf_fort_char_t *fn,
#endif
				 eaf_fort_int_t  *type,
				 eaf_fort_strlen_t fn_len)
{
  char         tmp_fn[EAF_FILENAME_MAX];
  char        *tmp_str;

#if defined(CRAY)           /* NOTICE:                                      */
  fn_len = fn.fcd_len / 8;  /*    The 8 indicates the number of bits per    */
                            /*    byte, which should always work here, but  */
                            /*    the Cray scientists are hard at work on.  */
  strncpy(tmp_fn, fn.c_pointer, fn_len);
#else
  strncpy(tmp_fn, fn, fn_len);
#endif

  tmp_fn[(fn_len < EAF_FILENAME_MAX) ? fn_len : (EAF_FILENAME_MAX - 1)] = 0;
  if ((tmp_str = strchr(tmp_fn, ' ')) != NULL)
  {
    *tmp_str = 0;
  };

  return( (eaf_fort_fd_t) EAF_OpenPersistC(tmp_fn, *type) );
}



/*\
\*/
/*\ Open Scratch File
\*/
#if defined(CRAY)  /* Cray Fortran passes a 128bit structure, not pointer */
eaf_fort_fd_t        EAF_OpenScratch(eaf_fort_char_t  fn,
#else
eaf_fort_fd_t        EAF_OpenScratch(eaf_fort_char_t *fn,
#endif
				 eaf_fort_int_t  *type,
				 eaf_fort_strlen_t fn_len)
{
  char         tmp_fn[EAF_FILENAME_MAX];
  char        *tmp_str;

#if defined(CRAY)           /* NOTICE:                                      */
  fn_len = fn.fcd_len / 8;  /*    The 8 indicates the number of bits per    */
                            /*    byte, which should always work here, but  */
                            /*    the Cray scientists are hard at work on.  */
  strncpy(tmp_fn, fn.c_pointer, fn_len);
#else
  strncpy(tmp_fn, fn, fn_len);
#endif
  tmp_fn[(fn_len < EAF_FILENAME_MAX) ? fn_len : (EAF_FILENAME_MAX - 1)] = 0;
  if ((tmp_str = strchr(tmp_fn, ' ')) != NULL)
  {
    *tmp_str = 0;
  }

  return( (eaf_fort_fd_t) EAF_OpenScratchC(tmp_fn, *type) );
}




/*\
\*/
/*\ Synchronous Write
\*/
eaf_fort_size_t         EAF_Write(eaf_fort_fd_t   *fort_fd,
				   eaf_fort_size_t *fort_offset,
				   Void            *buf,
				   eaf_fort_size_t *fort_size)
{
  return( EAF_WriteC(*fort_fd, *fort_offset, buf, *fort_size) );
}




/*\
\*/
/*\ Aynchronously Write
\*/
eaf_fort_status_t         EAF_AWrite(eaf_fort_fd_t   *fort_fd,
				      eaf_fort_size_t *fort_offset,
				      Void            *buf,
				      eaf_fort_size_t *fort_size,
				      eaf_fort_req_t  *req_id)
{
  return( EAF_AWriteC(*fort_fd, *fort_offset, buf, *fort_size, req_id) );
}




/*\
\*/
/*\ Synchronously Read
\*/
eaf_fort_size_t         EAF_Read(eaf_fort_fd_t   *fort_fd,
				  eaf_fort_size_t *fort_offset,
				  Void            *buf,
				  eaf_fort_size_t *fort_size)
{
  Size_t        b_read;

  b_read = EAF_ReadC(*fort_fd, *fort_offset, buf, *fort_size);

  return (eaf_fort_size_t) b_read;
}




/*\
\*/
/*\ Asynchronously Read
\*/
eaf_fort_status_t         EAF_ARead(eaf_fort_fd_t   *fort_fd,
				     eaf_fort_size_t *fort_offset,
				     Void            *buf,
				     eaf_fort_size_t *fort_size,
				     eaf_fort_req_t  *req_id)
{
  int    stat;
  
  stat = EAF_AReadC(*fort_fd, *fort_offset, buf, *fort_size, req_id);
  
  return (eaf_fort_status_t) stat;
}




/*\
\*/
/*\ Block on an IO operation
\*/
eaf_fort_status_t          EAF_Wait(eaf_fort_req_t  *id)
{
  int    stat;
  
  stat = EAF_WaitC(id);
  
  return (eaf_fort_status_t) stat;
}




/*\
\*/
/*\ Check for DONE or PENDING
\*/
eaf_fort_status_t          EAF_Probe(eaf_fort_req_t   *id,
				      eaf_fort_status_t  *stat)
{
  int   ret_stat;
  
  ret_stat = EAF_ProbeC(id, stat);
  
  return (eaf_fort_status_t) ret_stat;
}

