/******************************************************************************
Source File:    eaf.h

Description:    General header for C bindings
  
Author:         Jace A Mogill

Date Created:   16 May 1996

Modifications:

CVS: $Source: /tmp/hpctools/ga/pario/eaf/eaf.h,v $
CVS: $Date: 1996-08-05 15:38:07 $
CVS: $Revision: 1.5 $
CVS: $State: Exp $
******************************************************************************/
#if defined(__STDC__) || defined(__cplusplus)
# define _ARGS_(s) s
#else
# define _ARGS_(s) ()
#endif

extern Size_t EAF_ReadC     _ARGS_((Fd_t fd, off_t offset, Void *buf,
                                    Size_t bytes)); 
extern int    EAF_AReadC    _ARGS_((Fd_t fd, off_t offset, Void *buf,
                                    Size_t bytes, io_request_t *req_id));
extern Size_t EAF_WriteC    _ARGS_((Fd_t fd, off_t offset, Void *buf,
                                    Size_t bytes)); 
extern int    EAF_AWriteC   _ARGS_((Fd_t fd, off_t offset, Void *buf,
                                    Size_t bytes, io_request_t *req_id));
extern int    EAF_WaitC     _ARGS_((io_request_t *id));
extern int    EAF_ProbeC    _ARGS_((io_request_t *id, int* status));
extern Fd_t   EAF_OpenScratchC    _ARGS_((char *fname, int type));
extern Fd_t   EAF_OpenPersistC    _ARGS_((char *fname, int type));
extern void   EAF_CloseC     _ARGS_((Fd_t fd));
       void   EAF_InitC      _ARGS_(());
       void   EAF_TerminateC _ARGS_(());

#undef _ARGS_

/******************************************************************/
#define  EAF_MAX_FILES ELIO_MAX_FILES

static Fd_t eaf_fd[EAF_MAX_FILES];
static char *eaf_fname[EAF_MAX_FILES];
static int   first_eaf_init = 1;
/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .*/
static Fd_t    fd_table[EAF_MAX_FILES];/* The pointers to the Fd structure
                                          pointer can't be passed to Fortran
                                          as an integer, so we save an array
                                          of the pointers, and give Fortran
                                          the integer index into that array */



/**************************** Error Macro ******************************/
/* ELIO defines error macro called in case of error
 * the macro can also use user-provided error routine PRINT_AND_ABORT
 * defined as macro to do some cleanup in the application before
 * aborting
 * The requirement is that PRINT_AND_ABORT is defined before
 * including ELIO header file - this file
 */
#if !defined(PRINT_AND_ABORT)
#if defined(SUN) && !defined(SOLARIS)
extern int fprintf();
extern void fflush();
#endif

#define PRINT_AND_ABORT(msg, val) \
{ \
  fprintf(stderr, "EAF fatal error: %s %d\n", msg, (int) val); \
  fprintf(stdout, "EAF fatal error: %s %d\n", msg, (int) val); \
  fflush(stdout);\
  exit(val); \
}
#endif

#define EAF_ABORT(msg, val) \
{ \
  fprintf(stderr, "EAF Fatal -- Exiting with %d\n", val ); \
  fprintf(stderr, "EAF Fatal -- Msg: %s\n", msg ); \
  EAF_TerminateC(); \
  PRINT_AND_ABORT(msg, val); \
}
