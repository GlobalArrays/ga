#$Id: makefile,v 1.8 1996-10-01 23:30:02 d3h325 Exp $
#
#  Initial version Jace A Mogill (12 June 1996), ANL

 LIB_DISTRIB = ../../lib
     LIBRARY = libchemio.a
         OBJ = elio.o stat.o
     HEADERS = chemio.h

ifdef PABLO
   LIB_DEFINES = -DPABLO
endif
ifdef NOPIOFS
   LIB_DEFINES = -DNOPIOFS
endif
# on platforms with Posix AIO you can choose not to use it by defining NOAIO
ifdef NOAIO
   LIB_DEFINES += -DNOAIO
endif

include ../../config/makefile.h
include ../../config/makelib.h

$(LIBRARY_PATH)(elio.o):	./elio.h ./eliop.h ./chemio.h

# DO NOT EDIT BENEATH THIS LINE ... GENERATED AUTOMATICALLY
$(LIBRARY_PATH)(stat.o):	./eliop.h ./chemio.h 
