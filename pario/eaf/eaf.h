/******************************************************************************
Source File:    eaf.h

Description:    General header for C bindings
  
Author:         Jace A Mogill

Date Created:   16 May 1996

Modifications:

CVS: $Source: /tmp/hpctools/ga/pario/eaf/eaf.h,v $
CVS: $Date: 1996-07-17 15:58:04 $
CVS: $Revision: 1.2 $
CVS: $State: Exp $
******************************************************************************/
#if defined(__STDC__) || defined(__cplusplus)
# define _ARGS_(s) s
#else
# define _ARGS_(s) ()
#endif

extern Size_t EAF_ReadC     _ARGS_((Fd_t *fd, off_t offset, Void *buf,
                                    Size_t bytes)); 
extern int    EAF_AReadC    _ARGS_((Fd_t *fd, off_t offset, Void *buf,
                                    Size_t bytes, io_request_t *req_id));
extern Size_t EAF_WriteC    _ARGS_((Fd_t *fd, off_t offset, Void *buf,
                                    Size_t bytes)); 
extern int    EAF_AWriteC   _ARGS_((Fd_t *fd, off_t offset, Void *buf,
                                    Size_t bytes, io_request_t *req_id));
extern int    EAF_WaitC     _ARGS_((io_request_t *id));
extern int    EAF_ProbeC    _ARGS_((io_request_t *id, int* status));
extern Fd_t  *EAF_OpenScratchC    _ARGS_((char *fname, int type));
extern Fd_t  *EAF_OpenPersistC    _ARGS_((char *fname, int type));
extern void   EAF_CloseC     _ARGS_((Fd_t *fd));
       void   EAF_InitC      _ARGS_(());
       void   EAF_TerminateC _ARGS_(());

#undef _ARGS_

/******************************************************************/
static Fd_t *eaf_fd[EAF_MAX_FILES];
static char *eaf_fname[EAF_MAX_FILES];
static int   first_eaf_init = 1;

#if !defined(PRINT_AND_ABORT)
#define PRINT_AND_ABORT(val) \
{ \
  fprintf(stderr, "EAF Super-Fatal: PRINT_AND_ABORT not defined!\n"); \
  exit(val); \
}
#endif

#define EAF_ABORT(msg, val) \
{ \
  fprintf(stderr, "EAF Fatal -- Exiting with %d\n", val ); \
  fprintf(stderr, "EAF Fatal -- Msg: %s\n", msg ); \
  EAF_TerminateC(); \
  PRINT_AND_ABORT(val); \
}
