/******************************************************************************
Source File:    eaf_c2f.c

Description:    Fortran bindings to C EAF calls
                Parameter explanation on C calls
  
Author:         Jace A Mogill

Date Created:   16 May 1996

Modifications:

CVS: $Source: /tmp/hpctools/ga/pario/eaf/eaf_c2f.c,v $
CVS: $Date: 1996-09-17 22:12:19 $
CVS: $Revision: 1.6 $
CVS: $State: Exp $
******************************************************************************/
#define EAF_FILENAME_MAX ELIO_FILENAME_MAX

#if defined(CRAY)              /* This is needed to get the _fcd type for    */
# include <fortran.h>         /* da_fort_char_t since Fortran string pnters */
#endif                        /* are actually 128bit structures             */
/* #include "../ELIO/elio.h" */
#include "elio.h"
#include "eaf_c2f.h"
#include "eaf.h"
#include <string.h>


/*\
\*/
/*\ Close a file
\*/
eaf_fort_status_t      EAF_Close(eaf_fort_fd_t *fort_fd)
{
  FD_IN_RANGE("EAF_Close", fort_fd);
  return (eaf_fort_status_t) EAF_CloseC(fd_table[*fort_fd]);
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
  Integer      ffd=0;       /* Index into array of C struct pointers --
			       This is Fortran's FD value                  */
  char         tmp_fn[EAF_FILENAME_MAX];
  char        *tmp_str;


#if defined(CRAY)
#  if 0           /* Use the Cray Fortran string format of the month        */
                            /* NOTICE:                                      */
  fn_len = fn.fcd_len / 8;  /*    The 8 indicates the number of bits per    */
                            /*    byte, which should always work here, but  */
                            /*    the Cray scientists are hard at work on.  */
  strncpy(tmp_fn, fn.c_pointer, fn_len);
#  else           /* Use Cray's handy new conversion primitives             */
  fn_len = _fcdlen(fn);
  strncpy(tmp_fn, _fcdtocp(fn), fn_len);
#  endif          /* Cray Fortran string conversion                         */

#else
  strncpy(tmp_fn, fn, fn_len);
#endif

  tmp_fn[(fn_len < EAF_FILENAME_MAX) ? fn_len : (EAF_FILENAME_MAX - 1)] = 0;
  if ((tmp_str = strchr(tmp_fn, ' ')) != NULL)
  {
    *tmp_str = 0;
  };


  while(ffd<EAF_MAX_FILES && fd_table[ffd]!=NULL) ffd++;
  if(ffd<EAF_MAX_FILES)
    {
      fd_table[ffd]=EAF_OpenPersistC(tmp_fn, *type);
      return((eaf_fort_fd_t) ffd);
    }
  else
    EAF_ERR("EAF_OpenPersist: No space in C's (Fd_t) fd_table[]",
	    tmp_fn, -1);
  
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
  Integer      ffd=0;     /* Index into array of C struct pointers --
		      	     This is Fortran's FD value                  */
  char         tmp_fn[EAF_FILENAME_MAX];
  char        *tmp_str;

#if defined(CRAY)
#  if 0           /* Use the Cray Fortran string format of the month        */
                            /* NOTICE:                                      */
  fn_len = fn.fcd_len / 8;  /*    The 8 indicates the number of bits per    */
                            /*    byte, which should always work here, but  */
                            /*    the Cray scientists are hard at work on.  */
  strncpy(tmp_fn, fn.c_pointer, fn_len);
#  else           /* Use Cray's handy new conversion primitives             */
  fn_len = _fcdlen(fn);
  strncpy(tmp_fn, _fcdtocp(fn), fn_len);
#  endif          /* Cray Fortran string conversion                         */

#else
  strncpy(tmp_fn, fn, fn_len);
#endif

  tmp_fn[(fn_len < EAF_FILENAME_MAX) ? fn_len : (EAF_FILENAME_MAX - 1)] = 0;
  if ((tmp_str = strchr(tmp_fn, ' ')) != NULL)
  {
    *tmp_str = 0;
  }

  while(ffd<EAF_MAX_FILES && fd_table[ffd]!=NULL) ffd++;
  if(ffd<EAF_MAX_FILES)
    {
      fd_table[ffd]=EAF_OpenScratchC(tmp_fn, *type);
      return((eaf_fort_fd_t) ffd);
    }
  else
    EAF_ERR("EAF_OpenPersist: No space in C's (Fd_t) fd_table[]", tmp_fn, -1);

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
  FD_IN_RANGE("EAF_Write", fort_fd);
  return( EAF_WriteC(fd_table[*fort_fd], *fort_offset, buf, *fort_size) );
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
  FD_IN_RANGE("EAF_AWrite", fort_fd);
  return( EAF_AWriteC(fd_table[*fort_fd], *fort_offset, buf, *fort_size, req_id) );
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
  FD_IN_RANGE("EAF_Read", fort_fd);
  return (eaf_fort_size_t) EAF_ReadC(fd_table[*fort_fd], *fort_offset, buf, *fort_size);
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
  FD_IN_RANGE("EAF_ARead", fort_fd);
  return (eaf_fort_status_t) EAF_AReadC(fd_table[*fort_fd], *fort_offset, buf, *fort_size, req_id);
}




/*\
\*/
/*\ Block on an IO operation
\*/
eaf_fort_status_t          EAF_Wait(eaf_fort_req_t  *id)
{
  return (eaf_fort_status_t) EAF_WaitC(id);
}




/*\
\*/
/*\ Check for DONE or PENDING
\*/
eaf_fort_status_t          EAF_Probe(eaf_fort_req_t   *id,
				      eaf_fort_status_t  *stat)
{
  int   s, ret;

#if 0
  return (eaf_fort_status_t) EAF_ProbeC(id, stat);
#else
  ret = EAF_ProbeC(id, &s);
  *stat = (eaf_fort_status_t) s;
  return (eaf_fort_status_t) ret;
#endif
}





#if defined(CRAY)
Integer        EAF_STAT(_fcd  path, Integer *avail, Integer *fstype)
#else
Integer        eaf_stat_(char *path, Integer *avail, Integer *fstype, int flen)
#endif
{
 char cpath[EAF_FILENAME_MAX], dirname[EAF_FILENAME_MAX];
 stat_t statinfo;
 int rc;


 if(flen>EAF_FILENAME_MAX) return((Integer)CHEMIO_FAIL);


#ifdef CRAY
  strncpy(cpath, _fcdtocp(path), _fcdlen(path));
#else
  strncpy(cpath, path, flen);
#endif

 elio_dirname(cpath, dirname, EAF_FILENAME_MAX);
 rc = elio_stat(dirname, &statinfo);
 *fstype = (statinfo.fs == FS_UFS)? 0 : 1;
 *avail = (Integer)statinfo.avail;
 return((Integer)rc);
}

