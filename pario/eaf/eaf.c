/******************************************************************************
Source File:    eaf.c

Description:    General EAF library

Author:         Jace A Mogill

Date Created:   16 May 1996

Modifications:

CVS: $Source: /tmp/hpctools/ga/pario/eaf/eaf.c,v $
CVS: $Date: 1996-09-17 22:12:18 $
CVS: $Revision: 1.7 $
CVS: $State: Exp $
******************************************************************************/

#include "elio.h"
#include "eaf.h"

/*\
|*|      Blocking Write
|*| 
|*|      Returns:
|*|                Number of bytes written
|*|                -1 if failed
\*/
Size_t EAF_WriteC(fd, offset, buf, bytes)
Fd_t         fd;
off_t        offset;
Void        *buf;
Size_t       bytes;
{    
  return( elio_write(fd, offset, buf, bytes) );
}




/*\
|*|      Asynchronous Write
|*| 
|*|      Returns:
|*|                 0 on success
|*|                -1 if failed
\*/
int EAF_AWriteC(fd, offset, buf, bytes, req_id)
Fd_t         fd;
off_t        offset;
Void        *buf;
Size_t       bytes;
io_request_t *req_id;
{
  return (elio_awrite(fd, offset, buf, bytes, req_id));
}




/*\
|*|      Blocking Read
|*| 
|*|      Returns:
|*|                Number of bytes written
|*|                -1 if failed
\*/
Size_t EAF_ReadC(fd, offset, buf, bytes)
Fd_t         fd;
off_t        offset;
Void        *buf;
Size_t       bytes;
{
  return(elio_read(fd, offset, buf, bytes));
}




/*\
|*|      Asynchronous Write
|*| 
|*|      Returns:
|*|                 0 on success
|*|                -1 if failed
\*/
int EAF_AReadC(fd, offset, buf, bytes, req_id)
Fd_t          fd;
off_t         offset;
Void         *buf;
Size_t        bytes;
io_request_t *req_id;
{
  return (elio_aread(fd, offset, buf, bytes, req_id));
}




/*\
|*|      Block waiting for asyncronous IO <id>
|*| 
|*|      Returns:
|*|                 0 on success
|*|                 Invalidated <id>
\*/
int EAF_WaitC(id)
io_request_t *id;
{
  return (elio_wait(id));
}





/*\
|*|      Test asyncronous IO <id> for completion
|*| 
|*|      Returns:
|*|                 0 if pending, 1 if completed
|*|                 Invalidated <id>
\*/
int EAF_ProbeC(id, status)
io_request_t *id;
int          *status;
{
  return (elio_probe(id, status));
}




/*\
|*|      (Noncollective) Persistent File Open
|*| 
|*|      Returns:
|*|                 File descriptor
\*/
Fd_t  EAF_OpenPersistC(fname, type)
char* fname;
int   type;
{
  if(first_eaf_init) EAF_InitC();

  return (elio_open(fname, type));
}




/*\
|*|      (Noncollective) Scratch File Open
|*| 
|*|      Returns:
|*|                 File descriptor
\*/
Fd_t  EAF_OpenScratchC(fname, type)
char *fname;
int   type;
{
  Fd_t  fd;
  int   i=0;
  
  if(first_eaf_init) EAF_InitC();
  
  fd = elio_open(fname, type);
  while(i< EAF_MAX_FILES && eaf_fd[i] != NULL) i++;
  eaf_fd[i] = fd;
  if((eaf_fname[i]=(char*) malloc(strlen(fname)+1)) == NULL)
    EAF_ERR("EAF_OpenSF: Unable to malloc scratch file name", fname, (Fd_t) -1);
  strcpy(eaf_fname[i], fname);

  return(fd);
}

  



/*\
|*|      Close File <fd>
|*|          If the <fd> is in the scratch file (eaf_fd) table
|*|          also unlink/delete it.
\*/
int  EAF_CloseC(fd)
Fd_t  fd;
{
  int i=0;
  int r;
 
  elio_close(fd);
  while(i< EAF_MAX_FILES && eaf_fd[i] != fd) i++;
  if(eaf_fd[i] == fd && i < EAF_MAX_FILES)
    {
      if(elio_delete(eaf_fname[i]) != NULL)
	EAF_ERR("EAF_Close: Unable to delete scratch file on close.",
		NULL, -1);
      eaf_fd[i] = NULL;
      free(eaf_fname[i]);
    } 
  return( CHEMIO_OK);
}





/*\
|*|      Initialize EAF
|*| 
\*/
void EAF_InitC()
{
  int i;

  if(first_eaf_init)
    {
      first_eaf_init = 0;
      /* Initialize Scratch File table */
      for(i=0; i < EAF_MAX_FILES; i++)
	{
	  fd_table[i] = NULL;
	  eaf_fd[i] = NULL;
	};
    };
}




/*\
|*|
|*|  Terminate EAF
\*/
void EAF_TerminateC()
{
  int i=0;
  
  /* Close all files that I know about */
  /* NOTE:  This is currently restricted to Scratch files  */
  while(i < EAF_MAX_FILES)
    {
      if(eaf_fd[i] != NULL)
	{
	  fprintf(stderr,"Terminating with SF |%s| still open\n", eaf_fname[i]);
	  EAF_CloseC(eaf_fd[i]);
	}
      i++;
    };
}



/*\ Error handling routine
\*/
void eaf_err(char *func, char *fname)
{
  fprintf(stderr, "EAF: Error in routine %s", func);
  if(fname != NULL)
    fprintf(stderr, " on file: |%s|", fname);
  perror("\nEAF: ");
}




int EAF_Stat(path, statinfo)
char *path;
stat_t *statinfo;
{
 char dirname[ELIO_FILENAME_MAX];

 if(strlen(path)>ELIO_FILENAME_MAX) 
    EAF_ERR("eaf_stat: internal param. LEN too small", path, ELIO_FILENAME_MAX);

 elio_dirname(path, dirname, ELIO_FILENAME_MAX);
 return(elio_stat(dirname, statinfo));
}
