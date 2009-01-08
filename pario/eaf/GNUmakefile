#$Id: GNUmakefile,v 1.5.10.1 2007-06-12 20:33:35 manoj Exp $

          OBJ = eaf.o eaf_f2c.o
      LIBRARY = libpario.a
LIBRARY_SHARED= libpario.so
  LIB_DISTRIB = ../../lib
      HEADERS = eaf.fh eaf.h
  LIB_DEFINES = -DEAF_STATS
  LIB_TARGETS = test.x
       INCDIR = ../../include
 LIB_INCLUDES = -I$(INCDIR)

ifeq ($(ARMCI_NETWORK), PORTALS)
	LIB_DEFINES += -DXT3
endif

ifndef OLD_GA
include ../../armci/config/makecoms.h
endif
include ../makefile.h
include ../../config/makefile.h
include ../../config/makelib.h

test.x:	test.o $(FULL_LIBRARY_PATH)
	$(LINK.f) $(FLDOPT) -o $@ test.o $(FULL_LIBRARY_PATH) $(LIBS) $(FLIBS)
