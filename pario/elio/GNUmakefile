#$Id: GNUmakefile,v 1.6 2007-10-30 02:05:01 manoj Exp $
#
#  Initial version Jace A Mogill (12 June 1996), ANL

 LIB_DISTRIB = ../../lib
     LIBRARY = libpario.a
LIBRARY_SHARED= libpario.so
         OBJ = elio.o stat.o
     HEADERS = chemio.h elio.h ../sf/coms.h


INCDIR = ../../include
LIB_INCLUDES = -I$(INCDIR)

ifdef NOAIO
   LIB_DEFINES += -DNOAIO
else
ifeq ($(AIO_LARGE_FILES),NO)
   LIB_DEFINES += -DNOAIO
endif
endif

ifdef PABLO
   LIB_DEFINES = -DPABLO
endif

include ../makefile.h
include ../../config/makefile.h
include ../../config/makelib.h

# on platforms with Posix AIO you can choose not to use it by defining NOAIO


$(FULL_LIBRARY_PATH)(elio.o):	./elio.h ./eliop.h ./chemio.h ../sf/coms.h
$(FULL_LIBRARY_PATH)(stat.o):	./elio.h ./eliop.h ./chemio.h
