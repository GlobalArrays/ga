#$Id: makefile,v 1.5 1996-09-26 00:01:39 d3g681 Exp $
#
#  Initial version Jace A Mogill (12 June 1996), ANL

     LIBRARY = libchemio.a
         OBJ = elio.o stat.o
     HEADERS = chemio.h

ifdef PABLO
   LIB_DEFINES = -DPABLO
endif

ifdef NOPIOFS
   LIB_DEFINES = -DNOPIOFS
endif

include ../../config/makefile.h
include ../../config/makelib.h

$(LIBRARY_PATH)(elio.o):	./elio.h ./eliop.h ./chemio.h

# DO NOT EDIT BENEATH THIS LINE ... GENERATED AUTOMATICALLY
$(LIBRARY_PATH)(stat.o):	./eliop.h ./chemio.h 
