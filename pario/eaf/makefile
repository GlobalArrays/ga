#$Id: makefile,v 1.7 1996-10-08 23:33:45 d3h325 Exp $

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
#JN - edited to point to types.f2c.h in GA distribution
$(LIBRARY_PATH)(eaf.o):	../elio/elio.h ./eafP.h ./eaf.h 
$(LIBRARY_PATH)(test.o):	./eaf.fh 
$(LIBRARY_PATH)(eaf_f2c.o):	../../global/src/types.f2c.h ./eafP.h ./eaf.h 
