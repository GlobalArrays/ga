#$Id: makefile,v 1.6 1996-10-01 21:02:36 d3e129 Exp $

          OBJ = eaf.o eaf_f2c.o
      LIBRARY = libchemio.a
      HEADERS = eaf.fh eaf.h
  LIB_DEFINES = -DEAF_STATS
  LIB_TARGETS = test.x
 LIB_INCLUDES = -I../elio

include ../../config/makefile.h

FOPTIMIZE = -g
FDEBUG = -g
CDEBUG = -g
COPTIMIZE = -g


include ../../config/makelib.h


test.x:	test.o $(LIBRARY_PATH)
	$(LINK.f) -o $@ test.o $(LIBS)

# DO NOT EDIT BENEATH THIS LINE ... GENERATED AUTOMATICALLY
$(LIBRARY_PATH)(eaf.o):	../elio/elio.h ./eafP.h ./eaf.h 
$(LIBRARY_PATH)(test.o):	./eaf.fh 
$(LIBRARY_PATH)(eaf_f2c.o):	$(INCDIR)/types.f2c.h ./eafP.h ./eaf.h 
