#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

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
