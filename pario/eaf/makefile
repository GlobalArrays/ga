#$Id: makefile,v 1.8 1996-10-09 17:42:20 d3g681 Exp $

          OBJ = eaf.o eaf_f2c.o
      LIBRARY = libchemio.a
  LIB_DISTRIB = ../../lib
      HEADERS = eaf.fh eaf.h
  LIB_DEFINES = -DEAF_STATS
  LIB_TARGETS = test.x
 LIB_INCLUDES = -I../elio -I../../global/src

include ../../config/makefile.h
include ../../config/makelib.h

test.x:	test.o $(LIBRARY_PATH)
	$(LINK.f) -o $@ test.o $(LIBS)

# DO NOT EDIT BENEATH THIS LINE ... GENERATED AUTOMATICALLY
$(LIBRARY_PATH)(eaf_f2c.o):	../../global/src/types.f2c.h ./eafP.h ./eaf.h 
$(LIBRARY_PATH)(eaf.o):	../elio/elio.h ./eafP.h ./eaf.h 
$(LIBRARY_PATH)(test.o):	./eaf.fh 
