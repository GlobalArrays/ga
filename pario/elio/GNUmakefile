#$Id: GNUmakefile,v 1.4 2000-03-28 00:48:15 d3g681 Exp $
#
#  Initial version Jace A Mogill (12 June 1996), ANL

 LIB_DISTRIB = ../../lib
     LIBRARY = libpario.a
         OBJ = elio.o stat.o
     HEADERS = chemio.h elio.h


INCDIR = ../../include

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


$(LIBRARY_PATH)(elio.o):	./elio.h ./eliop.h ./chemio.h
$(LIBRARY_PATH)(stat.o):	./elio.h ./eliop.h ./chemio.h
