#$Id: GNUmakefile,v 1.4 2000-04-17 22:41:05 d3h325 Exp $

          OBJ = eaf.o eaf_f2c.o
      LIBRARY = libpario.a
  LIB_DISTRIB = ../../lib
      HEADERS = eaf.fh eaf.h
  LIB_DEFINES = -DEAF_STATS
  LIB_TARGETS = test.x
       INCDIR = ../../include
 LIB_INCLUDES = -I$(INCDIR)

ifndef OLD_GA
include ../../armci/config/makecoms.h
endif
include ../makefile.h
include ../../config/makefile.h
include ../../config/makelib.h

test.x:	test.o $(LIBRARY_PATH)
	$(LINK.f) $(FLDOPT) -o $@ test.o $(LIBRARY_PATH) $(LIBS) $(FLIBS)
