#$Id: GNUmakefile,v 1.1 1999-05-27 16:38:06 d3h325 Exp $
#
#  Initial version Jace A Mogill (12 June 1996), ANL

 LIB_DISTRIB = ../../lib
     LIBRARY = libchemio.a
         OBJ = elio.o stat.o
     HEADERS = chemio.h elio.h


INCDIR = ../../include
include ../../config/makefile.h
include ../makefile.h

# on platforms with Posix AIO you can choose not to use it by defining NOAIO
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


include ../../config/makelib.h

$(LIBRARY_PATH)(elio.o):	./elio.h ./eliop.h ./chemio.h
$(LIBRARY_PATH)(stat.o):	./elio.h ./eliop.h ./chemio.h
