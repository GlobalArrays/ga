#$Id: makefile,v 1.11 1997-03-13 22:07:04 d3h325 Exp $
#
#  Initial version Jace A Mogill (12 June 1996), ANL

 LIB_DISTRIB = ../../lib
     LIBRARY = libchemio.a
         OBJ = elio.o stat.o
     HEADERS = chemio.h

ifdef PABLO
   LIB_DEFINES = -DPABLO
endif

NOPIOFS = 1

ifdef NOPIOFS
   LIB_DEFINES = -DNOPIOFS
endif

#under AIX, there are problems with AIO and large files
ifdef LARGE_FILES
LIB_DEFINES += $(shell uname -a | awk \
              '{ if ($$1 == "AIX" && ($$4 > 4 || ($$4 == 4 && $$3 > 1)))\
               print "-D_LARGE_FILES -D_LARGE_FILE_API" }')
endif

# on platforms with Posix AIO you can choose not to use it by defining NOAIO
ifdef NOAIO
   LIB_DEFINES += -DNOAIO
endif


include ../../config/makefile.h
include ../../config/makelib.h

$(LIBRARY_PATH)(elio.o):	./elio.h ./eliop.h ./chemio.h

