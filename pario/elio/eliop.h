#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#define PRINT_AND_ABORT(msg, val) ga_error(msg, (long)val)
#ifndef GLOBAL_H
extern void ga_error(char*, long);
#endif

#if (defined(SP) || defined(SP1)) && !defined(NOPIOFS)
#define PIOFS 1
#endif


#if (defined(SUN) && !defined(SOLARIS)) || defined(LINUX)
#        include <sys/vfs.h>
#        define  STATVFS statfs
#elif defined(CRAY) || defined(AIX)
#        include <sys/statfs.h>
#        define  STATVFS statfs
#elif defined(KSR)
#        include <sys/mount.h>
#        define  STATVFS statfs
#elif !defined(PARAGON)
#        include <sys/statvfs.h>
#        define  STATVFS statvfs
#endif

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

extern int                   _elio_Errors_Fatal;
extern void                  elio_init(void);

#if !defined(PRINT_AND_ABORT)
#   if defined(SUN) && !defined(SOLARIS)
      extern int fprintf();
      extern void fflush();
#   endif
#   define PRINT_AND_ABORT(msg, val){\
     fprintf(stderr, "ELIO fatal error: %s %ld\n", msg,  val);\
     fprintf(stdout, "ELIO fatal error: %s %ld\n", msg,  val);\
     fflush(stdout);\
     exit(val);\
   }
#endif

/**************************** Error Macro ******************************/
/* ELIO defines error macro called in case of error
 * the macro can also use user-provided error routine PRINT_AND_ABORT
 * defined as macro to do some cleanup in the application before
 * aborting
 * The requirement is that PRINT_AND_ABORT is defined before
 * including ELIO header file - this file
 */
#define ELIO_ABORT PRINT_AND_ABORT

#define ELIO_ERROR(msg, val) { \
 if(! _elio_Errors_Fatal) return(ELIO_FAIL);\
 else PRINT_AND_ABORT(msg, val);\
}

