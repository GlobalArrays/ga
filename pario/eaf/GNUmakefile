#$Id: GNUmakefile,v 1.2 1999-08-25 20:04:24 edo Exp $

          OBJ = eaf.o eaf_f2c.o
      LIBRARY = libchemio.a
  LIB_DISTRIB = ../../lib
      HEADERS = eaf.fh eaf.h
  LIB_DEFINES = -DEAF_STATS
  LIB_TARGETS = test.x
       INCDIR = ../../include
 LIB_INCLUDES = -I$(INCDIR)

include ../makefile.h
include ../../config/makefile.h
include ../../config/makelib.h

test.x:	test.o $(LIBRARY_PATH)
	$(LINK.f) -o $@ test.o $(LIBS) $(LIBS)
